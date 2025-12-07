# SousChef

An AI-powered meal planning app that:

- Manages your pantry/fridge/freezer inventory
- Recommends recipes that maximize use of what you already have (and reduce waste)
- Builds a grocery list of missing ingredients
- Flags items to toss or use ASAP based on best‑by dates

## Features

1) Inventory
- Add items with name, category, quantity, unit, and purchase date
- Optional AI estimation of a best‑buy date
- Edit quantities in-place; delete items

2) Recipe Recommender
- Retrieves candidate recipes via a local RAG index (ChromaDB with OpenAI embeddings)
- Uses an agent to pick recipes that maximize pantry usage, prioritizing items nearing best‑by
- “Cook this” button applies a recipe to your pantry (decrements quantities)

3) Grocery List
- Aggregates missing ingredients across selected recipes
- Computes quantities required by comparing to pantry amounts (basic unit conversions for g↔kg, ml↔L, and items)

4) Toss‑Out / Expiring
- Shows items that are expired or expiring soon (<= 2 days)
- “Expired” includes items whose best‑by date is today or earlier

## Requirements

- Python 3.10+
- An OpenAI API key with access to the specified models

Install dependencies:

```
pip install -r requirements.txt
```

## OpenAI credentials

Set your API key via Streamlit secrets (preferred):

Create `.streamlit/secrets.toml` in the project root:

```
[general]
OPENAI_API_KEY = "sk-..."
```

Alternatively, set environment variable `OPENAI_API_KEY`.

## Running the app

```
streamlit run streamlit_app.py
```

On first recipe search, a Chroma index will be built under `chroma_db/` (ensure the process has write permissions).

## Models used

- Responses API: `gpt-4.1-mini` (JSON mode)
- Embeddings: `text-embedding-3-small`

You can change these in `ai.py`, `agent.py`, and `rag.py` if desired.

## Notes on Units

Basic conversions supported:
- Mass: `g` ↔ `kg`
- Volume: `ml` ↔ `l`
- Count: `item` (aliases: `items`, `pcs`, `piece`, `pieces`)

Unit conversions are intentionally simple and may not cover all cases. Extend `_normalize_unit` and `_convert_amount` in `streamlit_app.py` as needed.
