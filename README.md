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
- Retrieves candidate recipes via a lightweight in‑memory embeddings index (OpenAI embeddings + NumPy cosine similarity)
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

On first recipe search, an in‑memory index of the bundled seed recipes is built using OpenAI embeddings. No external database is required.

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

## Troubleshooting

- ModuleNotFoundError: No module named `sqlalchemy`
  - Make sure dependencies are installed in the active environment:
    - `pip install -r requirements.txt`
    - Verify with `python -c "import sqlalchemy, pkgutil; print(sqlalchemy.__version__)"`
  - Ensure you’re using the same interpreter where you installed packages (activate your virtualenv/conda env).
  - On Streamlit Cloud/Codespaces, trigger a rebuild/restart so requirements are applied.

- OPENAI_API_KEY errors
  - Ensure `.streamlit/secrets.toml` contains `OPENAI_API_KEY` or export it in your shell before running the app.

- RAG retrieval issues
  - The app now uses an in‑memory embeddings index and no longer depends on ChromaDB or SQLite. If searches return nothing, ensure your OpenAI API key is set and reachable (see above). A first search will perform several embedding calls.
