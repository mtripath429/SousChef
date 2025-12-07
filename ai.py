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
    # Newer SDKs expose `client.responses.create`. Older/newer variants
    # may expose `client.chat.completions.create` instead. Support both.
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
        # Fallback to chat completions API available on some SDK versions
        # Build messages list similar to Responses input
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # Use chat.completions.create if available
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = response_text(resp)
        else:
            # As a last resort, try the legacy completions API with a JSON-only
            # instruction in the prompt.
            prompt = system + "\n" + user + "\nRespond ONLY in JSON with keys best_buy_date and reason."
            resp = client.completions.create(model="gpt-3.5-turbo-instruct", prompt=prompt)
            content = response_text(resp)
    data = json.loads(content)
    return data
