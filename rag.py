import json
import os
import numpy as np
from openai_utils import get_openai_client


_INDEX = None  # lazy in-memory index of recipe embeddings


def load_seed_recipes():
    with open("recipes/seed_recipes.json", "r") as f:
        return json.load(f)


def embed_texts(texts):
    client = get_openai_client()
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    vectors = [d.embedding for d in resp.data]
    return np.array(vectors).tolist()  # Chroma wants plain Python lists


def _normalize_ingredient_name(s: str) -> str:
    return (s or "").strip().lower()


def build_index():
    global _INDEX
    recipes = load_seed_recipes()
    # Create simple textual representation focused on ingredients and title
    docs = []
    metas = []
    for r in recipes:
        ingredient_names = [ing["name"] for ing in r["ingredients"]]
        doc = f"{r['title']} | ingredients: {', '.join(ingredient_names)}"
        docs.append(doc)
        metas.append(
            {
                "id": r["id"],
                "title": r["title"],
                "ingredients": r["ingredients"],
                "steps": r.get("steps"),
                # Optional source URL or attribution; may be missing for bundled recipes
                "source": r.get("source"),
                "detailed_steps": r.get("detailed_steps"),
                "servings": r.get("servings"),
                "prep_time": r.get("prep_time"),
                "cook_time": r.get("cook_time"),
                "tags": r.get("tags"),
            }
        )

    vectors = np.array(embed_texts(docs))  # shape: (N, D)
    # Normalize for cosine similarity (avoid div by zero)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    _INDEX = {
        "vectors": vectors,  # numpy array
        "metas": metas,
    }


def ensure_index():
    global _INDEX
    if _INDEX is None:
        build_index()


def query_recipes_by_ingredients(ingredients, top_k=5):
    ensure_index()
    query_text = "ingredients: " + ", ".join(_normalize_ingredient_name(i) for i in ingredients)
    q_vec = np.array(embed_texts([query_text])[0])
    # cosine similarity with pre-normalized doc vectors
    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        q_norm = 1.0
    q_vec = q_vec / q_norm

    mats = _INDEX["vectors"]  # (N, D)
    sims = np.dot(mats, q_vec)  # (N,)
    # Get top_k indices
    top_idx = np.argsort(-sims)[:top_k]

    metas = _INDEX["metas"]
    results = []
    for idx in top_idx:
        meta = metas[int(idx)]
        results.append(
            {
                "id": meta["id"],
                "title": meta["title"],
                "ingredients": meta["ingredients"],
                "steps": meta["steps"],
            }
        )
    return results
