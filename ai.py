from datetime import date
import json
from openai_utils import get_openai_client, response_text


def estimate_best_buy(item_name: str, category: str, purchase_date: date) -> dict:
    """
    Returns {"best_buy_date": "YYYY-MM-DD", "reason": "..."}.
    """
    system = (
        "You are a food safety assistant. "
        "Given an ingredient, storage type (pantry/fridge/freezer), and purchase date, "
        "estimate a conservative 'best by' date in ISO format (YYYY-MM-DD). "
        "Use typical US guidance and err on the side of safety. "
        "Respond ONLY in JSON with keys best_buy_date and reason."
    )

    user = (
        f"Ingredient: {item_name}\n"
        f"Storage: {category}\n"
        f"Purchase date: {purchase_date.isoformat()}\n"
    )

    client = get_openai_client()
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )

    content = response_text(resp)
    data = json.loads(content)
    return data
