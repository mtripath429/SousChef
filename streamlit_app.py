import streamlit as st
from datetime import date, timedelta
from sqlalchemy import func

from db import init_db, SessionLocal, Item
from ai import estimate_best_buy
from rag import query_recipes_by_ingredients
from agent import recommend_recipes_with_agent


st.set_page_config(page_title="SousChef", layout="wide")


def main():
    init_db()

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

def _normalize_unit(u: str) -> str:
    u = (u or "").strip().lower()
    aliases = {
        "g": "g",
        "gram": "g",
        "grams": "g",
        "kg": "kg",
        "kilogram": "kg",
        "kilograms": "kg",
        "ml": "ml",
        "milliliter": "ml",
        "milliliters": "ml",
        "l": "l",
        "liter": "l",
        "liters": "l",
        "item": "item",
        "items": "item",
        "pcs": "item",
        "piece": "item",
        "pieces": "item",
    }
    return aliases.get(u, u)


def _convert_amount(amount: float, from_unit: str, to_unit: str):
    from_u = _normalize_unit(from_unit)
    to_u = _normalize_unit(to_unit)
    if from_u == to_u:
        return amount, True
    # mass
    if from_u == "kg" and to_u == "g":
        return amount * 1000.0, True
    if from_u == "g" and to_u == "kg":
        return amount / 1000.0, True
    # volume
    if from_u == "l" and to_u == "ml":
        return amount * 1000.0, True
    if from_u == "ml" and to_u == "l":
        return amount / 1000.0, True
    # items
    if from_u == "item" and to_u == "item":
        return amount, True
    # not convertible
    return amount, False


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

        # Try to convert recipe amount to the stored item's unit (basic conversions)
        target_unit = (item.unit or "").strip().lower()
        amt_to_subtract = float(amount)
        ok_unit = True
        if unit and target_unit:
            converted, ok_unit = _convert_amount(float(amount), unit, target_unit)
            if ok_unit:
                amt_to_subtract = converted
        elif unit and not target_unit:
            # If item has no unit stored, assume direct subtraction
            amt_to_subtract = float(amount)
        elif not unit and target_unit:
            # Recipe unit missing but item has unit; only subtract if item unit is 'item'
            if _normalize_unit(target_unit) != "item":
                ok_unit = False

        if not ok_unit:
            continue

        # Subtract quantity and clamp at zero
        current_qty = float(item.quantity or 0.0)
        item.quantity = max(0.0, current_qty - float(amt_to_subtract))

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
        st.markdown("#### Manage Inventory")
        for i in items:
            cols = st.columns([3, 2, 2, 2, 3, 2])
            with cols[0]:
                st.write(i.name)
            with cols[1]:
                st.write(i.category)
            with cols[2]:
                new_qty = st.number_input(
                    f"Qty_{i.id}", min_value=0.0, value=float(i.quantity or 0.0), key=f"qty_{i.id}"
                )
            with cols[3]:
                st.write(i.unit or "")
            with cols[4]:
                st.write(f"Best by: {i.best_buy_date or '-'}")
            with cols[5]:
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Save", key=f"save_{i.id}"):
                        s = SessionLocal()
                        itm = s.query(Item).get(i.id)
                        if itm:
                            itm.quantity = float(new_qty)
                            s.commit()
                        s.close()
                        st.success(f"Saved {i.name}")
                with col_b:
                    if st.button("Delete", key=f"del_{i.id}"):
                        s = SessionLocal()
                        itm = s.query(Item).get(i.id)
                        if itm:
                            s.delete(itm)
                            s.commit()
                        s.close()
                        st.warning(f"Deleted {i.name}")
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
            try:
                result = recommend_recipes_with_agent()
                recipes = result.get("recipes", [])
                st.session_state["recommended_recipes"] = recipes
            except Exception as e:
                st.error(f"Failed to get recommendations: {e}")

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

    # Build an aggregated grocery list with quantities using pantry comparison
    # Fetch pantry
    session = SessionLocal()
    pantry_items = session.query(Item).all()
    session.close()

    # Build a map of pantry availability: name -> {unit, quantity}
    pantry_map = {}
    for p in pantry_items:
        key = (p.name or "").strip().lower()
        pantry_map.setdefault(key, []).append({
            "unit": (p.unit or "").strip().lower(),
            "quantity": float(p.quantity or 0.0),
        })

    def available_amount(name: str, unit: str) -> float:
        key = (name or "").strip().lower()
        total = 0.0
        for entry in pantry_map.get(key, []):
            qty = entry["quantity"]
            ent_unit = entry["unit"]
            converted, ok = _convert_amount(qty, ent_unit or unit, unit)
            if ok:
                total += float(converted)
        return total

    # Aggregate required amounts per ingredient
    needed = {}
    for r in selected_recipes:
        for ing in r.get("ingredients", []):
            name = (ing.get("name") or "").strip().lower()
            amount = float(ing.get("amount") or 0.0)
            unit = (ing.get("unit") or "").strip().lower()
            if not name or amount <= 0:
                continue
            have = available_amount(name, unit)
            short = max(0.0, amount - have)
            if short > 0:
                key = (name, _normalize_unit(unit))
                needed[key] = needed.get(key, 0.0) + short

    st.markdown("### Recipes selected")
    for r in selected_recipes:
        st.write(f"- {r['title']}")

    st.markdown("### Items to buy")
    if needed:
        for (name, unit), amt in sorted(needed.items()):
            st.write(f"- {name}: {amt:.2f} {unit}")
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
        if i.best_buy_date <= today:
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
