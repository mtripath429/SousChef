import json
from typing import List

from openai_utils import get_openai_client, response_text


def get_recipes_for_ingredients(ingredients: List[str], top_k: int = 5):
    """
    Use the OpenAI Responses API (web search tool) when available to find
    up-to-date recipe pages for the given ingredients. Falls back to a
    model-driven search prompt if the tool isn't available.

    Returns a list of recipe dicts with keys: title, source/url, ingredients,
    steps, detailed_steps, servings, prep_time, cook_time, tags.
    """
    client = get_openai_client()
    q = f"Find the top {top_k} recipe webpages that use these ingredients: {', '.join(ingredients)}."
    # Include a strict JSON output example to encourage structured results
    example = {
        "recipes": [
            {
                "title": "Example Recipe",
                "url": "https://example.com/recipe",
                "ingredients": [{"name": "spinach", "amount": 200, "unit": "g"}],
                "steps": "Short steps",
                "detailed_steps": "1) Do this. 2) Do that.",
                "servings": 4,
                "prep_time": "10 mins",
                "cook_time": "20 mins",
                "tags": ["vegetarian"]
            }
        ]
    }

    prompt = (
        q
        + "\nReturn ONLY JSON matching this shape (an object with key 'recipes' which is a list):\n"
        + json.dumps(example)
    )

    # Try Responses API first
    try:
        if hasattr(client, "responses"):
            resp = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = response_text(resp)
            data = json.loads(content)
            return data.get("recipes", [])[:top_k]
    except Exception:
        # fall through to chat fallback
        pass

    # Fallback to chat/completions
    try:
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = response_text(resp)
            data = json.loads(content)
            return data.get("recipes", [])[:top_k]
    except Exception:
        pass

    # Last resort: legacy completions
    try:
        resp = client.completions.create(model="gpt-3.5-turbo-instruct", prompt=prompt)
        content = response_text(resp)
        # try to extract json
        import re

        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            return data.get("recipes", [])[:top_k]
    except Exception:
        pass

    return []
