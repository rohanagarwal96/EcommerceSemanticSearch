"""
Genericize an e-commerce product catalog by removing retailer-specific
identifiers (store name, private-label programs, addresses, loyalty
program, URLs, phone numbers) while preserving all data useful for
semantic search: manufacturer brand names, product descriptions,
ingredients, and the category taxonomy.

Usage:
    python genericize_catalog.py input.csv output.csv
"""

import json
import re
import sys

import pandas as pd

STORE_NAME = "Wegmans"
GENERIC_BRAND = "Store Brand"  # replacement for private-label brand_raw
GENERIC_LINE = "Value Choice"  # replacement token inside product names
GENERIC_QUALITY_BANNER = "our house quality standard"  # replaces "Food You Feel Good About"
GENERIC_LOYALTY = "the store loyalty program"  # replaces "Shoppers Club"
GENERIC_BAKESHOP = "our bakery"  # replaces "Wegmans Bakeshop"

# Columns that only exist for internal pipeline tracking, not useful for search
DROP_COLS = ["url", "datapoint_id", "raw_data_id", "created_at_utc", "updated_at_utc"]

# Company legal name + address block, e.g.
# "Wegmans Food Markets, Inc., 1500 Brooks Ave, Rochester, NY 14603"
ADDRESS_BLOCK_RE = re.compile(
    r"Wegmans Food Markets,?\s*Inc\.?,?\s*\d+\s+[\w\s]+?,\s*Rochester,?\s*NY\s*\d{5}",
    re.IGNORECASE,
)
# Any sentence that mentions the retailer's hometown (bakery/restaurant "our hometown of
# Rochester, NY" style copy) - these are store-specific provenance claims, drop the sentence
ROCHESTER_SENTENCE_RE = re.compile(r"[^.!?]*\bRochester\b[^.!?]*(?:[.!?]|$)", re.IGNORECASE)
URL_RE = re.compile(r"\b(?:www\.)?[a-zA-Z0-9-]+\.(?:com|net|org)\b", re.IGNORECASE)
PHONE_RE_1 = re.compile(r"1-?8\d{2}[-.\s]?\d{3}[-.\s]?\d{4}")
PHONE_RE_2 = re.compile(r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}")


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.,])", r"\1", text)
    text = re.sub(r"^[\.\s]+", "", text)
    return text.strip()


def strip_store_identity(text):
    """Remove all retailer-identifying content from a free-text field."""
    if not isinstance(text, str):
        return text

    text = ADDRESS_BLOCK_RE.sub("", text)
    text = ROCHESTER_SENTENCE_RE.sub("", text)
    # \s+ (not a literal single space) since source text has inconsistent double-spacing
    text = re.sub(
        r"Food\s+You\s+Feel\s+Good\s+About", GENERIC_QUALITY_BANNER, text, flags=re.IGNORECASE
    )
    text = re.sub(r"Shoppers\s?Club", GENERIC_LOYALTY, text, flags=re.IGNORECASE)
    text = re.sub(rf"{STORE_NAME}\s+Bakeshop", GENERIC_BAKESHOP, text, flags=re.IGNORECASE)
    text = re.sub(rf"The\s+{STORE_NAME}\s+Family", "our team", text, flags=re.IGNORECASE)
    text = re.sub(r"Visit us at [\w.]+\.?", "", text, flags=re.IGNORECASE)

    # generic noise removal: URLs and phone numbers carry no semantic search value
    # regardless of whether they belong to the retailer or a manufacturer
    text = URL_RE.sub("", text)
    text = PHONE_RE_1.sub("", text)
    text = PHONE_RE_2.sub("", text)

    # catch-all: any remaining mentions of the retailer name. No \b word-boundary
    # requirement, since source text has cases where a missing space runs the
    # store name directly into an adjacent word (e.g. "BOWLWegmans")
    text = re.sub(STORE_NAME, GENERIC_LINE, text, flags=re.IGNORECASE)

    return _collapse_whitespace(text)


def genericize_brand(brand):
    """Any brand_raw value that names the retailer - exact match ('Wegmans') or
    a private-label variant ('Designed By Wegmans', 'Wegmans Organic', etc.) -
    collapses to the single generic store-brand label."""
    if not isinstance(brand, str):
        return brand
    return GENERIC_BRAND if STORE_NAME.lower() in brand.lower() else brand


def genericize_name(name):
    if not isinstance(name, str):
        return name
    return re.sub(rf"\b{STORE_NAME}\b", GENERIC_LINE, name, flags=re.IGNORECASE)


def genericize_tags(tags_str):
    """tags is stored as a brace-delimited list, e.g. {Organic,"Wegmans Brand",
    "Food You Feel Good About"}. Handle both this literal-phrase format and the
    legacy underscore-token format ('wegmans_brand') seen in older exports."""
    if not isinstance(tags_str, str):
        return tags_str
    tags_str = re.sub(rf"{STORE_NAME}\s+Brand", GENERIC_BRAND, tags_str, flags=re.IGNORECASE)
    tags_str = re.sub(
        r"Food\s+You\s+Feel\s+Good\s+About", GENERIC_QUALITY_BANNER, tags_str, flags=re.IGNORECASE
    )
    tags_str = re.sub(rf"{STORE_NAME.lower()}_brand", "store_brand", tags_str, flags=re.IGNORECASE)
    # catch-all for any other stray mention
    tags_str = re.sub(STORE_NAME, GENERIC_LINE, tags_str, flags=re.IGNORECASE)
    return tags_str


def genericize_item_info(info_str):
    """item_info is a JSON string. Category taxonomy (category_0..3) is left
    untouched since it is generic retail vocabulary, not retailer-specific.
    Only free-text nested values (e.g. ingredients) get the identity sweep."""
    if not isinstance(info_str, str):
        return info_str
    try:
        obj = json.loads(info_str)
    except (json.JSONDecodeError, TypeError):
        return strip_store_identity(info_str)

    # category_0..3 taxonomy labels are preserved as-is (generic retail vocabulary),
    # except the store name is still swapped out if a label happens to embed it
    # (e.g. "Brie & Wegmans Cave Ripened" -> "Brie & Value Choice Cave Ripened")
    PRESERVE_KEYS = {"category_0", "category_1", "category_2", "category_3"}

    def clean(k, v):
        if isinstance(v, str):
            if k in PRESERVE_KEYS:
                return re.sub(STORE_NAME, GENERIC_LINE, v, flags=re.IGNORECASE)
            return strip_store_identity(v)
        if isinstance(v, dict):
            return {kk: clean(kk, vv) for kk, vv in v.items()}
        if isinstance(v, list):
            return [clean(None, x) for x in v]
        return v

    cleaned = {k: clean(k, v) for k, v in obj.items()}
    return json.dumps(cleaned)


def genericize_catalog(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    if "name" in df.columns:
        df["name"] = df["name"].apply(genericize_name)
    if "description" in df.columns:
        df["description"] = df["description"].apply(strip_store_identity)
    if "brand_raw" in df.columns:
        df["brand_raw"] = df["brand_raw"].apply(genericize_brand)
    if "tags" in df.columns:
        df["tags"] = df["tags"].apply(genericize_tags)
    if "item_info" in df.columns:
        df["item_info"] = df["item_info"].apply(genericize_item_info)

    return df


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "sample_items.xlsx"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "sample_items_generic.csv"

    df = pd.read_excel(in_path) if in_path.endswith((".xlsx", ".xls")) else pd.read_csv(in_path)
    cleaned = genericize_catalog(df)
    cleaned.to_csv(out_path, index=False)
    print(f"Wrote {len(cleaned)} rows to {out_path}")
    print(f"Columns: {list(cleaned.columns)}")
