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


def recommend_recipes_with_agent():
    pantry = tool_get_pantry()
    if not pantry:
        return {"recipes": []}

    pantry_names = list({item["name"].lower() for item in pantry})
    candidate_recipes = tool_query_local_recipes(pantry_names)

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
    # Support multiple SDK shapes: prefer `responses`, then `chat.completions`,
    # then legacy `completions` as a last resort.
    if hasattr(client, "responses"):
        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        content = response_text(resp)
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            # Use chat completions and ask for strict JSON
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = response_text(resp)
        else:
            # Legacy completions fallback: include the JSON example in the prompt
            prompt = system + "\n" + user + "\nReturn ONLY JSON matching the example."
            resp = client.completions.create(model="gpt-3.5-turbo-instruct", prompt=prompt)
            content = response_text(resp)

    # Try to parse the JSON robustly and validate expected keys
    try:
        data = json.loads(content)
    except Exception:
        # Try to extract the first JSON object/array substring
        import re

        m = re.search(r"\{.*\}|\[.*\]", content, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception as e:
                raise RuntimeError(f"Failed to parse agent JSON response: {e}\nRaw response:\n{content}")
        else:
            raise RuntimeError(f"Agent returned non-JSON response:\n{content}")

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
