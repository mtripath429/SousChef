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

    system = (
        "You are a meal planning assistant called SousChef. "
        "You receive the user's pantry items (with optional best_buy_date) and a list of candidate recipes. "
        "Each candidate recipe has a structured 'ingredients' field: a list of objects with 'name', 'amount', and 'unit'. "
        "Pick 3–5 recipes that maximize usage of pantry items, prioritizing items that are expired or closest to their best_buy_date. "
        "Explain briefly why each recipe is chosen with respect to waste reduction. "
        "For each chosen recipe, infer which pantry items it uses and which extra ingredients are missing. "
        "In your output, for each recipe, you MUST copy the exact 'ingredients' list from the original candidate recipe without changing amounts or units. "
        "Return ONLY JSON with key 'recipes', which is a list of objects with keys: title, used_items, missing_items, explanation, ingredients."
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
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = response_text(resp)
        else:
            # Legacy completions fallback: send a JSON-only instruction in the prompt
            prompt = system + "\n" + user + "\nReturn ONLY JSON with key 'recipes'."
            resp = client.completions.create(model="gpt-3.5-turbo-instruct", prompt=prompt)
            content = response_text(resp)

    data = json.loads(content)
    return data  # expected {"recipes": [...]} 
