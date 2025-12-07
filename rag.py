import json
import os
import numpy as np
import chromadb
from openai_utils import get_openai_client


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "recipes"


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


def get_chroma_collection():
    client_chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client_chroma.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def build_index():
    collection = get_chroma_collection()
    if collection.count() > 0:
        return

    recipes = load_seed_recipes()
    docs = []
    ids = []
    metadatas = []

    for r in recipes:
        ingredient_names = [ing["name"] for ing in r["ingredients"]]
        doc = (
            f"{r['title']}\n"
            f"Ingredients: {', '.join(ingredient_names)}\n"
            f"Steps: {r['steps']}"
        )
        docs.append(doc)
        ids.append(str(r["id"]))
        metadatas.append(
            {
                "id": r["id"],
                "title": r["title"],
                # full structured ingredients with amount + unit
                "ingredients": r["ingredients"],
                "steps": r["steps"],
            }
        )

    embeddings = embed_texts(docs)

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def ensure_index():
    collection = get_chroma_collection()
    if collection.count() == 0:
        build_index()


def query_recipes_by_ingredients(ingredients, top_k=5):
    ensure_index()
    collection = get_chroma_collection()

    query_text = "Recipes that use: " + ", ".join(ingredients)
    query_embedding = embed_texts([query_text])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    recipes = []
    for meta in results["metadatas"][0]:
        recipes.append(
            {
                "id": meta["id"],
                "title": meta["title"],
                "ingredients": meta["ingredients"],  # keep structured data
                "steps": meta["steps"],
            }
        )
    return recipes
