import streamlit as st
from datetime import date, timedelta
from sqlalchemy import func

from db import init_db, SessionLocal, Item
from ai import estimate_best_buy
from rag import query_recipes_by_ingredients
from agent import recommend_recipes_with_agent


def main():
    init_db()

    st.set_page_config(page_title="SousChef", layout="wide")

    st.sidebar.title("SousChef")
    page = st.sidebar.radio(
        "Go to",
        ["Inventory", "Recipe Recommender", "Grocery List", "Toss-Out / Expiring"]
    )

    if page == "Inventory":
        inventory_page()
    elif page == "Recipe Recommender":
        recipe_page()
    elif page == "Grocery List":
        grocery_page()
    elif page == "Toss-Out / Expiring":
        tossout_page()


# ---------- Helper: apply recipe to pantry ----------

def apply_recipe_to_pantry(recipe):
    """
    Decrements pantry/fridge quantities based on a recipe's ingredients.
    Assumes recipe['ingredients'] is a list of {name, amount, unit}.
    """
    session = SessionLocal()

    for ing in recipe.get("ingredients", []):
        name = (ing.get("name") or "").strip().lower()
        amount = ing.get("amount")
        unit = (ing.get("unit") or "").strip().lower()

        if not name or amount is None:
            continue

        # Case-insensitive name match
        item = (
            session.query(Item)
            .filter(func.lower(Item.name) == name)
            .first()
        )
        if not item:
            continue

        # Simple unit check: only subtract if units match (you can get fancier later)
        if item.unit and unit and item.unit.lower() != unit:
            continue

        # Subtract quantity and clamp at zero
        current_qty = float(item.quantity or 0.0)
        item.quantity = max(0.0, current_qty - float(amount))

    session.commit()
    session.close()


# ---------- Inventory ----------

def inventory_page():
    st.header("Inventory")

    st.subheader("Add Item")

    with st.form("add_item_form"):
        name = st.text_input("Item name")
        category = st.selectbox("Category", ["pantry", "fridge", "freezer"])
        quantity = st.number_input("Quantity", min_value=0.0, value=1.0)
        unit = st.text_input("Unit (e.g., g, oz, cup, item)", value="item")
        purchase_date = st.date_input("Purchase date", value=date.today())
        use_ai = st.checkbox("Use AI to estimate best-buy date", value=True)
        submitted = st.form_submit_button("Add item")

    if submitted and name:
        session = SessionLocal()
        item = Item(
            name=name,
            category=category,
            quantity=quantity,
            unit=unit,
            purchase_date=purchase_date,
        )

        if use_ai:
            try:
                result = estimate_best_buy(name, category, purchase_date)
                from datetime import date as dcls
                item.best_buy_date = dcls.fromisoformat(result["best_buy_date"])
                item.best_buy_source = "ai"
                st.info(
                    f"AI-estimated best-buy date for {name}: "
                    f"{item.best_buy_date} ({result.get('reason', '')})"
                )
            except Exception as e:
                st.error(f"AI estimation failed: {e}")
        else:
            item.best_buy_date = None
            item.best_buy_source = "user"

        session.add(item)
        session.commit()
        session.close()
        st.success(f"Added {name}")

    st.subheader("Current Inventory")
    session = SessionLocal()
    items = session.query(Item).all()
    session.close()

    if items:
        st.table([
            {
                "Name": i.name,
                "Category": i.category,
                "Qty": i.quantity,
                "Unit": i.unit,
                "Purchase": i.purchase_date,
                "Best Buy": i.best_buy_date,
                "Source": i.best_buy_source,
            }
            for i in items
        ])
    else:
        st.info("No items yet. Add some above.")


# ---------- Recipe Recommender ----------

def recipe_page():
    st.header("Recipe Recommender")

    # Debug / explicit RAG demo
    with st.expander("Debug: RAG-only recipe search"):
        ing_input = st.text_input("Enter ingredients (comma-separated)", "spinach, chickpeas")
        if st.button("Search recipes (RAG only)"):
            ingredients = [s.strip() for s in ing_input.split(",") if s.strip()]
            results = query_recipes_by_ingredients(ingredients)
            for r in results:
                st.markdown(f"### {r['title']}")
                st.write("Ingredients:")
                for ing in r["ingredients"]:
                    st.write(f"- {ing['name']} — {ing['amount']} {ing['unit']}")
                st.write(r["steps"])
                st.write("---")

    st.subheader("SousChef agent recommendations")

    if "recommended_recipes" not in st.session_state:
        st.session_state["recommended_recipes"] = []

    if st.button("Suggest recipes from my pantry"):
        with st.spinner("SousChef is thinking..."):
            result = recommend_recipes_with_agent()
        recipes = result.get("recipes", [])
        st.session_state["recommended_recipes"] = recipes

    recipes = st.session_state["recommended_recipes"]

    selected_titles = []
    for idx, r in enumerate(recipes):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"#### {r['title']}")
            st.write("Uses:", ", ".join(r.get("used_items", [])))
            st.write("Missing:", ", ".join(r.get("missing_items", [])))
            st.write(r.get("explanation", ""))

            with st.expander("View ingredient amounts"):
                for ing in r.get("ingredients", []):
                    st.write(f"- {ing['name']} — {ing['amount']} {ing['unit']}")

            st.write("---")

        with col2:
            if st.checkbox("Select", key=f"select_{idx}"):
                selected_titles.append(r["title"])

        with col3:
            if st.button("Cook this", key=f"cook_{idx}"):
                apply_recipe_to_pantry(r)
                st.success(f"Updated pantry based on '{r['title']}'")

    st.session_state["selected_recipe_titles"] = selected_titles


# ---------- Grocery List ----------

def grocery_page():
    st.header("Grocery List")

    recipes = st.session_state.get("recommended_recipes", [])
    selected_titles = st.session_state.get("selected_recipe_titles", [])

    selected_recipes = [r for r in recipes if r["title"] in selected_titles]

    if not selected_recipes:
        st.info("No recipes selected yet. Go to Recipe Recommender and select some.")
        return

    # Simple: union of missing_items across selected recipes
    to_buy = set()
    for r in selected_recipes:
        for item in r.get("missing_items", []):
            to_buy.add(item.lower())

    st.markdown("### Recipes selected")
    for r in selected_recipes:
        st.write(f"- {r['title']}")

    st.markdown("### Items to buy")
    if to_buy:
        for item in sorted(to_buy):
            st.write(f"- {item}")
    else:
        st.success("You already have everything you need for these recipes.")


# ---------- Toss-Out / Expiring ----------

def tossout_page():
    st.header("Toss-Out / Expiring Items")

    today = date.today()
    soon = today + timedelta(days=2)

    session = SessionLocal()
    items = session.query(Item).all()
    session.close()

    expired = []
    expiring_soon = []
    for i in items:
        if not i.best_buy_date:
            continue
        if i.best_buy_date < today:
            expired.append(i)
        elif today <= i.best_buy_date <= soon:
            expiring_soon.append(i)

    st.markdown("### Expired (consider tossing)")
    if expired:
        for i in expired:
            st.write(f"- {i.name} ({i.category}), best by {i.best_buy_date}")
    else:
        st.success("No expired items.")

    st.markdown("### Expiring soon (use ASAP)")
    if expiring_soon:
        for i in expiring_soon:
            st.write(f"- {i.name} ({i.category}), best by {i.best_buy_date}")
    else:
        st.info("Nothing expiring in the next 2 days.")


if __name__ == "__main__":
    main()
