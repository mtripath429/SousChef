import json
from openai_utils import get_openai_client, response_text

from db import SessionLocal, Item
from rag import query_recipes_by_ingredients


def tool_get_pantry():
    session = SessionLocal()
    items = session.query(Item).all()
    session.close()
    return [
        {
            "name": i.name,
            "category": i.category,
            "quantity": i.quantity,
            "unit": i.unit,
            "best_buy_date": i.best_buy_date.isoformat() if i.best_buy_date else None,
        }
        for i in items
    ]


def tool_query_local_recipes(ingredients):
    return query_recipes_by_ingredients(ingredients)


def recommend_recipes_with_agent(extra_candidates=None):
    pantry = tool_get_pantry()
    if not pantry:
        return {"recipes": []}

    pantry_names = list({item["name"].lower() for item in pantry})
    candidate_recipes = tool_query_local_recipes(pantry_names)
    # Merge in extra/web candidates if provided (they should be list of recipe dicts)
    if extra_candidates:
        # normalize incoming candidates to expected dict keys
        for c in extra_candidates:
            candidate_recipes.append(c)

    # Provide a strict JSON schema and a short example to encourage machine-parsable output.
    json_example = {
        "recipes": [
            {
                "title": "Spinach Chickpea Curry",
                "used_items": ["spinach", "chickpeas"],
                "missing_items": ["rice"],
                "explanation": "Uses spinach and chickpeas that are near best-by...",
                "ingredients": [
                    {"name": "spinach", "amount": 200, "unit": "g"},
                    {"name": "chickpeas", "amount": 400, "unit": "g"}
                ],
                "source": "https://example.com/recipe",
                "detailed_steps": "1) Do this. 2) Do that.",
                "servings": 4,
                "prep_time": "10 mins",
                "cook_time": "20 mins",
                "tags": ["vegetarian", "quick"]
            }
        ]
    }

    system = (
        "You are a meal planning assistant called SousChef. "
        "You receive the user's pantry items (with optional best_buy_date) and a list of candidate recipes. "
        "Each candidate recipe has a structured 'ingredients' field: a list of objects with 'name', 'amount', and 'unit'. "
        "Pick 3–5 recipes that maximize usage of pantry items, prioritizing items that are expired or closest to their best_buy_date. "
        "Explain briefly why each recipe is chosen with respect to waste reduction. "
        "For each chosen recipe, infer which pantry items it uses and which extra ingredients are missing. "
        "For each chosen recipe, include a 'source' field (URL or attribution) if known, otherwise set it to null. "
        "Also provide a 'detailed_steps' field with step-by-step, user-friendly instructions (more detailed than the brief 'steps' field). "
        "Additionally include 'servings', 'prep_time', 'cook_time', and a 'tags' list (e.g., vegetarian, vegan, gluten-free, one-pan, quick) when available. "
        "In your output, for each recipe, you MUST copy the exact 'ingredients' list from the original candidate recipe without changing amounts or units. "
        "Return ONLY JSON matching this schema. Follow this example exactly (do not add narration):\n"
        + json.dumps(json_example)
    )

    user = json.dumps(
        {
            "pantry": pantry,
            "candidate_recipes": candidate_recipes,
        }
    )

    client = get_openai_client()

    def call_model(system_prompt: str, user_prompt: str):
        """Call the available OpenAI client shape and return raw text content."""
        if hasattr(client, "responses"):
            resp = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            return response_text(resp)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                response_format={"type": "json_object"},
            )
            return response_text(resp)

        # Legacy completions fallback
        prompt = system_prompt + "\n" + user_prompt + "\nReturn ONLY JSON matching the example."
        resp = client.completions.create(model="gpt-3.5-turbo-instruct", prompt=prompt)
        return response_text(resp)


    def validate_parsed(data_obj):
        """Basic validation for the agent's parsed JSON structure.

        Returns (valid: bool, reason: str).
        """
        if not isinstance(data_obj, dict):
            return False, "top-level JSON is not an object"
        if "recipes" not in data_obj or not isinstance(data_obj["recipes"], list):
            return False, "missing 'recipes' list"
        if len(data_obj["recipes"]) == 0:
            return False, "'recipes' list is empty"
        # Check each recipe for minimal required fields
        for idx, r in enumerate(data_obj["recipes"]):
            if not isinstance(r, dict):
                return False, f"recipe at index {idx} is not an object"
            if not r.get("title"):
                return False, f"recipe at index {idx} missing 'title'"
            if not isinstance(r.get("ingredients", []), list):
                return False, f"recipe at index {idx} has non-list 'ingredients'"
            # Ensure ingredients have at least 'name'
            for ing in r.get("ingredients", []):
                if not isinstance(ing, dict) or not ing.get("name"):
                    return False, f"recipe at index {idx} has ingredient missing 'name'"
        return True, "ok"


    max_retries = 3
    last_raw = None
    for attempt in range(max_retries):
        raw = call_model(system, user)
        last_raw = raw
        # Try to parse the JSON robustly
        try:
            parsed = json.loads(raw)
        except Exception:
            import re

            m = re.search(r"\{.*\}|\[.*\]", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None
            else:
                parsed = None

        valid, reason = (False, "no parse")
        if parsed is not None:
            valid, reason = validate_parsed(parsed)

        if valid:
            data = parsed
            break

        # If invalid and we have retries left, ask the model to regenerate
        if attempt < max_retries - 1:
            # Provide the model with the previous raw response and a brief reason
            followup_user = (
                user
                + "\n\nThe previous response was invalid: "
                + reason
                + "\nHere is the raw response the assistant gave:\n" + raw
                + "\nPlease respond again with ONLY JSON matching the example schema."
            )
            # next loop will call the model again with same system + followup_user
            user = followup_user
            continue
        # Exhausted retries
        raise RuntimeError(f"Agent failed to produce valid JSON after {max_retries} attempts. Last raw response:\n{last_raw}")

    # Basic validation: top-level 'recipes' list
    if not isinstance(data, dict) or "recipes" not in data or not isinstance(data["recipes"], list):
        raise RuntimeError(f"Agent response JSON missing 'recipes' list. Raw: {data}")

    # Ensure each recipe contains required keys, add fallbacks if possible
    for rec in data["recipes"]:
        rec.setdefault("title", None)
        rec.setdefault("ingredients", [])
        rec.setdefault("used_items", [])
        rec.setdefault("missing_items", [])
        rec.setdefault("explanation", "")
        rec.setdefault("source", None)
        rec.setdefault("detailed_steps", None)
        rec.setdefault("servings", None)
        rec.setdefault("prep_time", None)
        rec.setdefault("cook_time", None)
        rec.setdefault("tags", [])

    # Post-process: if the agent omitted `source` or other metadata, try to
    # backfill from the candidate_recipes provided by our local RAG index.
    try:
        # Build a title->meta map from candidate_recipes (case-insensitive)
        title_map = { (c.get("title") or "").strip().lower(): c for c in candidate_recipes }
        for rec in data["recipes"]:
            title = (rec.get("title") or "").strip().lower()
            meta = title_map.get(title)
            if meta:
                if not rec.get("source") and meta.get("source"):
                    rec["source"] = meta.get("source")
                if not rec.get("detailed_steps") and meta.get("detailed_steps"):
                    rec["detailed_steps"] = meta.get("detailed_steps")
                if not rec.get("servings") and meta.get("servings"):
                    rec["servings"] = meta.get("servings")
                if not rec.get("prep_time") and meta.get("prep_time"):
                    rec["prep_time"] = meta.get("prep_time")
                if not rec.get("cook_time") and meta.get("cook_time"):
                    rec["cook_time"] = meta.get("cook_time")
                if (not rec.get("tags") or len(rec.get("tags") or []) == 0) and meta.get("tags"):
                    rec["tags"] = meta.get("tags")
                # If agent omitted ingredients, copy from meta (shouldn't usually happen)
                if (not rec.get("ingredients") or len(rec.get("ingredients") or []) == 0) and meta.get("ingredients"):
                    rec["ingredients"] = meta.get("ingredients")
    except Exception:
        # Non-fatal; if backfilling fails, continue with original data
        pass

    return data  # expected {"recipes": [...]} 
