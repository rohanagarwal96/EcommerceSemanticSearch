"""
Enrich the genericized e-commerce catalog: flatten the nested item_info and
sizing_comp JSON blobs into real columns, normalize the two different tags
formats, derive boolean attribute flags, and build a clean search_text field
that concatenates everything useful for embedding.

Input:  ecommerce_catalog_generic.csv  (output of genericize_catalog.py)
Output: ecommerce_catalog_enriched.csv (ready for embedding / indexing)

Usage:
    python enrich_catalog.py input.csv output.csv
"""

import ast
import json
import re
import sys

import pandas as pd

# item_info keys that are internal pipeline identifiers, not useful for search
DROP_ITEM_INFO_KEYS = {"ext_id", "ic_item_id", "ic_product_id"}
# planogram = physical in-store aisle/shelf location. Operational metadata
# specific to one store's floor plan, not a product attribute - dropped.

PRICE_RE = re.compile(r"\$?([\d.]+)\s*/\s*([\w.\s]+)")

# tag keywords -> derived boolean column. Matched against the normalized tag
# list (lowercased, underscores and spaces both stripped for comparison).
DIETARY_FLAG_KEYWORDS = {
    "is_organic": ["organic"],
    "is_vegan": ["vegan"],
    "is_gluten_free": ["gluten free", "gluten_free"],
    "is_kosher": ["kosher"],
    "is_lactose_free": ["lactose free", "lactose_free"],
}


def parse_tags(tags_str):
    """Two formats appear in the source data:
    - Python-list-repr string:  "['gluten_free', 'store_brand']"
    - Postgres array string:    '{"Store Brand","Family Pack"}'  or '{}'
    Both normalize to a clean list of lowercase, space-separated tag strings,
    with internal pipeline tokens (the '_internal_*' prefix) dropped.
    """
    if not isinstance(tags_str, str) or tags_str.strip() in ("", "{}"):
        return []

    raw_tags = []
    s = tags_str.strip()
    if s.startswith("["):
        try:
            raw_tags = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            raw_tags = []
    elif s.startswith("{"):
        inner = s[1:-1]
        if inner:
            # split on commas that aren't inside quotes
            raw_tags = [t.strip().strip('"') for t in re.findall(r'"[^"]*"|[^,]+', inner)]

    cleaned = []
    for t in raw_tags:
        t = str(t).strip().strip('"').replace("_", " ").strip()
        if not t or t.lower() in ("no tags", "internal any gluten free"):
            continue
        cleaned.append(t.lower())
    return sorted(set(cleaned))


def parse_item_info(info_str):
    """Flatten item_info JSON into category levels, ingredients, and a
    joined category_path used both as a filter field and inside search_text."""
    try:
        obj = json.loads(info_str)
    except (json.JSONDecodeError, TypeError):
        obj = {}

    levels = [obj.get(f"category_{i}") for i in range(4)]
    levels = [lvl for lvl in levels if lvl]
    return {
        "category_l0": obj.get("category_0"),
        "category_l1": obj.get("category_1"),
        "category_l2": obj.get("category_2"),
        "category_l3": obj.get("category_3"),
        "category_path": " > ".join(levels) if levels else None,
        "ingredients": obj.get("ingredients"),
    }


def parse_price(price_str):
    """'$0.06/fl. oz.' -> (0.06, 'fl. oz.')"""
    if not isinstance(price_str, str):
        return None, None
    m = PRICE_RE.match(price_str.strip())
    if not m:
        return None, None
    amount, uom = m.groups()
    try:
        return float(amount), uom.strip()
    except ValueError:
        return None, uom.strip()


def parse_sizing(sizing_str):
    try:
        obj = json.loads(sizing_str)
    except (json.JSONDecodeError, TypeError):
        obj = {}

    price_amount, price_uom = parse_price(obj.get("unit_price"))
    return {
        "unit_price_usd": price_amount,
        "unit_price_uom": price_uom or obj.get("uom_unit_price"),
        "package_size": obj.get("size_user_friendly"),
        "package_size_numeric": obj.get("size_from_unit_price") or obj.get("size"),
        "num_servings": obj.get("num_servings_nutrition"),
        "serving_size": obj.get("serving_size_nutrition"),
        "serving_size_uom": obj.get("serving_size_uom_nutrition"),
        "billed_by_weight": bool(obj.get("billed_by_weight")),
        "ordered_by_weight": bool(obj.get("ordered_by_weight")),
    }


def derive_dietary_flags(tags_list):
    tagset = " | ".join(tags_list)
    return {flag: any(kw in tagset for kw in kws) for flag, kws in DIETARY_FLAG_KEYWORDS.items()}


def build_search_text(row):
    """Single concatenated field for embedding. Every component is optional
    and skipped cleanly if missing, so no 'None'/'nan' artifacts leak in."""
    parts = [row["name"]]
    if pd.notna(row.get("brand")) and row["brand"] not in ("Store Brand",):
        parts.append(f"by {row['brand']}")
    if pd.notna(row.get("category_path")):
        parts.append(f"Category: {row['category_path']}.")
    if pd.notna(row.get("description")):
        parts.append(str(row["description"]))
    if pd.notna(row.get("ingredients")):
        parts.append(f"Ingredients: {row['ingredients']}.")
    if row.get("tags_list"):
        parts.append(f"Attributes: {', '.join(row['tags_list'])}.")
    text = " ".join(str(p) for p in parts)
    text = re.sub(r"<[^>]+>", " ", text)  # strip stray HTML tags seen in ingredients
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def enrich_catalog(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- item_info: category taxonomy + ingredients ---
    info = df["item_info"].apply(parse_item_info).apply(pd.Series)
    df = pd.concat([df, info], axis=1)

    # --- sizing_comp: price, package size, servings ---
    sizing = df["sizing_comp"].apply(parse_sizing).apply(pd.Series)
    df = pd.concat([df, sizing], axis=1)

    # drop dead legacy columns (entirely null in source) BEFORE deriving new
    # columns of the same name below (is_organic in particular is reused)
    dead_cols = [
        "name_clean",
        "category",
        "department",
        "subcategory",
        "size_raw",
        "is_organic",
        "item_info",
        "sizing_comp",
    ]
    df = df.drop(columns=[c for c in dead_cols if c in df.columns])

    # --- tags: normalize both formats, derive dietary flags ---
    df["tags_list"] = df["tags"].apply(parse_tags)
    flags = df["tags_list"].apply(derive_dietary_flags).apply(pd.Series)
    df = pd.concat([df, flags], axis=1)
    df["is_store_brand"] = (df["brand_raw"] == "Store Brand") | df["tags_list"].apply(
        lambda tl: "store brand" in tl
    )
    df["tags_str"] = df["tags_list"].apply(lambda tl: "|".join(tl))

    # --- brand cleanup ---
    df["brand"] = df["brand_raw"].fillna("Unknown")
    df = df.drop(columns=["tags", "brand_raw", "tags_list"])

    # --- final embedding-ready text field ---
    df["search_text"] = df.apply(build_search_text, axis=1)

    # column order: identity -> content -> structured attributes -> search_text
    col_order = [
        "item_id",
        "name",
        "brand",
        "category_l0",
        "category_l1",
        "category_l2",
        "category_l3",
        "category_path",
        "description",
        "ingredients",
        "tags_str",
        "is_organic",
        "is_vegan",
        "is_gluten_free",
        "is_kosher",
        "is_lactose_free",
        "is_store_brand",
        "unit_price_usd",
        "unit_price_uom",
        "package_size",
        "package_size_numeric",
        "num_servings",
        "serving_size",
        "serving_size_uom",
        "billed_by_weight",
        "ordered_by_weight",
        "search_text",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    return df


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "ecommerce_catalog_generic.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "ecommerce_catalog_enriched.csv"

    df = pd.read_csv(in_path)
    enriched = enrich_catalog(df)
    enriched.to_csv(out_path, index=False)
    print(f"Wrote {len(enriched)} rows, {len(enriched.columns)} columns to {out_path}")
    print(f"Columns: {list(enriched.columns)}")
