import sqlite_compat  # ensure modern sqlite before any other imports (SQLAlchemy may import sqlite3)
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
        "oz": "oz",
        "ounce": "oz",
        "ounces": "oz",
        "lb": "lb",
        "lbs": "lb",
        "pound": "lb",
        "pounds": "lb",
        "ml": "ml",
        "milliliter": "ml",
        "milliliters": "ml",
        "l": "l",
        "liter": "l",
        "liters": "l",
        "tsp": "tsp",
        "teaspoon": "tsp",
        "teaspoons": "tsp",
        "tbsp": "tbsp",
        "tablespoon": "tbsp",
        "tablespoons": "tbsp",
        "cup": "cup",
        "cups": "cup",
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
    # mass (g, kg, oz, lb)
    if from_u == "kg" and to_u == "g":
        return amount * 1000.0, True
    if from_u == "g" and to_u == "kg":
        return amount / 1000.0, True
    if from_u == "oz" and to_u == "g":
        return amount * 28.3495, True
    if from_u == "g" and to_u == "oz":
        return amount / 28.3495, True
    if from_u == "lb" and to_u == "kg":
        return amount * 0.453592, True
    if from_u == "kg" and to_u == "lb":
        return amount / 0.453592, True
    if from_u == "lb" and to_u == "g":
        return amount * 453.592, True
    if from_u == "g" and to_u == "lb":
        return amount / 453.592, True
    # volume
    if from_u == "l" and to_u == "ml":
        return amount * 1000.0, True
    if from_u == "ml" and to_u == "l":
        return amount / 1000.0, True
    # US kitchen measures approximations via mL
    if from_u == "tsp" and to_u == "ml":
        return amount * 4.92892, True
    if from_u == "ml" and to_u == "tsp":
        return amount / 4.92892, True
    if from_u == "tbsp" and to_u == "ml":
        return amount * 14.7868, True
    if from_u == "ml" and to_u == "tbsp":
        return amount / 14.7868, True
    if from_u == "cup" and to_u == "ml":
        return amount * 240.0, True
    if from_u == "ml" and to_u == "cup":
        return amount / 240.0, True
    # Cross conversions among tsp/tbsp/cup using ml as intermediary
    if from_u == "tsp" and to_u == "tbsp":
        return (amount * 4.92892) / 14.7868, True
    if from_u == "tbsp" and to_u == "tsp":
        return (amount * 14.7868) / 4.92892, True
    if from_u == "tsp" and to_u == "cup":
        return (amount * 4.92892) / 240.0, True
    if from_u == "cup" and to_u == "tsp":
        return (amount * 240.0) / 4.92892, True
    if from_u == "tbsp" and to_u == "cup":
        return (amount * 14.7868) / 240.0, True
    if from_u == "cup" and to_u == "tbsp":
        return (amount * 240.0) / 14.7868, True
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

    # Use immediate controls (not a Streamlit form) so toggles are reactive instantly
    name = st.text_input("Item name", key="add_name")
    category = st.selectbox("Category", ["pantry", "fridge", "freezer"], key="add_category")
    quantity = st.number_input("Quantity", min_value=0.0, value=1.0, key="add_quantity")
    unit = st.text_input("Unit (e.g., g, oz, cup, item)", value="item", key="add_unit")
    purchase_date = st.date_input("Purchase date", value=date.today(), key="add_purchase_date")
    use_ai = st.checkbox("Use AI to estimate best-buy date", value=True, key="add_use_ai")
    manual_best_by = None
    if not use_ai:
        st.markdown("**Manual best‑by** — since AI is disabled, you can optionally set a best‑by date")
        manual_best_by = st.date_input("Manual best-by date", value=purchase_date, key="add_manual_best_by")

    if st.button("Add item", key="add_item_button") and name:
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
            # If the user supplied a manual best-by date, use it; otherwise leave blank
            if manual_best_by:
                item.best_buy_date = manual_best_by
                item.best_buy_source = "user"
            else:
                item.best_buy_date = None
                item.best_buy_source = "user"

        session.add(item)
        session.commit()
        session.close()
        st.success(f"Added {name}")
        # Clear inputs for convenience
        try:
            st.session_state["add_name"] = ""
            st.session_state["add_category"] = "pantry"
            st.session_state["add_quantity"] = 1.0
            st.session_state["add_unit"] = "item"
            st.session_state["add_purchase_date"] = date.today()
            st.session_state["add_use_ai"] = True
            st.session_state["add_manual_best_by"] = st.session_state.get("add_purchase_date", date.today())
        except Exception:
            pass
        # Refresh to show the new item in the list immediately
        try:
            st.experimental_rerun()
        except Exception:
            pass

    st.subheader("Current Inventory")
    session = SessionLocal()
    items = session.query(Item).all()
    session.close()

    if items:
        st.markdown("#### Manage Inventory")
        # Render header row
        header_cols = st.columns([3, 2, 2, 2, 3, 2])
        headers = ["Item", "Location", "Quantity", "Measurement", "Best By", "Actions"]
        for c, h in zip(header_cols, headers):
            c.markdown(f"**{h}**")

        # Render rows
        for i in items:
            cols = st.columns([3, 2, 2, 2, 3, 2])
            with cols[0]:
                st.write(i.name)
            with cols[1]:
                st.write(i.category)
            with cols[2]:
                # Show only the numeric input control (no visible label), keep internal key
                new_qty = st.number_input(
                    "", min_value=0.0, value=float(i.quantity or 0.0), key=f"qty_{i.id}"
                )
            with cols[3]:
                st.write(i.unit or "")
            with cols[4]:
                st.write(f"{i.best_buy_date or '-'}")
            with cols[5]:
                # Compact two small buttons in-line so they fit cleanly
                btn_a, btn_b = st.columns([1, 1], gap="small")
                if btn_a.button("Save", key=f"save_{i.id}"):
                    s = SessionLocal()
                    itm = s.get(Item, i.id)
                    if itm:
                        itm.quantity = float(new_qty)
                        s.commit()
                    s.close()
                    st.success(f"Saved {i.name}")
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
                if btn_b.button("Delete", key=f"del_{i.id}"):
                    s = SessionLocal()
                    itm = s.get(Item, i.id)
                    if itm:
                        s.delete(itm)
                        s.commit()
                    s.close()
                    st.warning(f"Deleted {i.name}")
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
    else:
        st.info("No items yet. Add some above.")


# ---------- Recipe Recommender ----------

def recipe_page():
    st.header("Recipe Recommender")

    # Mode selector: RAG (local) or Online (web search via Responses API)
    mode = st.radio("Search mode", ["RAG (local)", "Online (web)"], index=0)

    # Build pantry names for web search / RAG queries
    session = SessionLocal()
    pantry_items = session.query(Item).all()
    session.close()
    pantry_names = [ (p.name or "").strip().lower() for p in pantry_items if p.name ]

    st.subheader("SousChef agent recommendations")

    if "recommended_recipes" not in st.session_state:
        st.session_state["recommended_recipes"] = []

    if st.button("Suggest recipes from my pantry"):
        with st.spinner("SousChef is thinking..."):
            try:
                # If Online mode, fetch web candidates via the web_search tool
                extra = None
                if mode.startswith("Online"):
                    try:
                        from web_search import get_recipes_for_ingredients

                        extra = get_recipes_for_ingredients(pantry_names, top_k=5)
                    except Exception as e:
                        st.warning(f"Online search failed: {e}")

                result = recommend_recipes_with_agent(extra_candidates=extra)
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
            # Show source link if available
            src = r.get("source") or r.get("url")
            if src:
                try:
                    st.markdown(f"**Source:** [{src}]({src})")
                except Exception:
                    st.write("Source:", src)

            # Show metadata: servings, prep/cook time, tags
            servings = r.get("servings")
            prep = r.get("prep_time")
            cook = r.get("cook_time")
            tags = r.get("tags") or []
            meta_parts = []
            if servings:
                meta_parts.append(f"Servings: {servings}")
            if prep:
                meta_parts.append(f"Prep: {prep}")
            if cook:
                meta_parts.append(f"Cook: {cook}")
            if meta_parts:
                st.write(" — ".join(meta_parts))
            if tags:
                st.write("Tags:", ", ".join(tags))

            # Add to cookbook button (save web-sourced recipes into local seed_recipes.json)
            if st.button("Add to cookbook", key=f"add_{idx}"):
                try:
                    import json as _json
                    from pathlib import Path as _Path
                    import rag as _rag

                    recipes_file = _Path("recipes/seed_recipes.json")
                    data = _json.load(open(recipes_file))
                    max_id = max((r.get("id") or 0) for r in data) if data else 0
                    new_id = max_id + 1
                    # Normalize fields for storage
                    store = {
                        "id": new_id,
                        "title": r.get("title"),
                        "ingredients": r.get("ingredients", []),
                        "steps": r.get("steps") or r.get("detailed_steps") or "",
                        "source": r.get("source") or r.get("url"),
                        "detailed_steps": r.get("detailed_steps"),
                        "servings": r.get("servings"),
                        "prep_time": r.get("prep_time"),
                        "cook_time": r.get("cook_time"),
                        "tags": r.get("tags", []),
                    }
                    data.append(store)
                    _json.dump(data, open(recipes_file, "w"), indent=2)
                    # Rebuild the in-memory index
                    try:
                        _rag.build_index()
                    except Exception:
                        _rag._INDEX = None
                        _rag.build_index()
                    st.success(f"Added '{r.get('title')}' to cookbook")
                except Exception as e:
                    st.error(f"Failed to add to cookbook: {e}")

            with st.expander("View ingredient amounts"):
                ings = r.get("ingredients", [])
                if ings:
                    # Show ingredients as a small table
                    try:
                        import pandas as _pd

                        df = _pd.DataFrame(ings)
                        st.table(df)
                    except Exception:
                        for ing in ings:
                            st.write(f"- {ing.get('name')} — {ing.get('amount')} {ing.get('unit')}")
                else:
                    st.write("No ingredients available.")
            # Detailed step-by-step instructions (preferred over the short `steps`)
            with st.expander("Detailed recipe / Steps"):
                detailed = r.get("detailed_steps") or r.get("steps") or "No detailed steps provided."
                # Render numbered steps as markdown to preserve formatting
                try:
                    st.markdown(detailed)
                except Exception:
                    st.write(detailed)
            # Source link and metadata area
            src = r.get("source") or r.get("url")
            if src:
                try:
                    st.markdown(f"<a href=\"{src}\" target=\"_blank\">View source</a>", unsafe_allow_html=True)
                except Exception:
                    st.write("Source:", src)

            # Show metadata: servings, prep/cook time, tags
            servings = r.get("servings")
            prep = r.get("prep_time")
            cook = r.get("cook_time")
            tags = r.get("tags") or []
            meta_parts = []
            if servings:
                meta_parts.append(f"Servings: {servings}")
            if prep:
                meta_parts.append(f"Prep: {prep}")
            if cook:
                meta_parts.append(f"Cook: {cook}")
            if meta_parts:
                st.write(" — ".join(meta_parts))
            if tags:
                # Render tags as small badges using HTML
                badges = " ".join([
                    f'<span style="background:#eee;border-radius:6px;padding:3px 8px;margin-right:6px;display:inline-block">{t}</span>' for t in tags
                ])
                st.markdown(badges, unsafe_allow_html=True)

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
