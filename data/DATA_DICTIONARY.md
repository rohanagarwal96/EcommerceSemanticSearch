# Enriched Catalog: Data Dictionary

Source: `ecommerce_catalog_generic.csv` (55,516 rows, genericized retail catalog)
Output: `ecommerce_catalog_enriched.csv` (55,516 rows, 27 columns)
Pipeline: `genericize_catalog.py` -> `enrich_catalog.py`

## What changed in this step

The source catalog stored two rich fields as nested JSON strings
(`item_info`, `sizing_comp`) and a `tags` field that mixed two different
serialization formats. Several columns (`name_clean`, `category`,
`department`, `subcategory`, `size_raw`, `is_organic`) were present in the
schema but entirely null throughout all 55,516 rows, so a real category
taxonomy and dietary flags had to be derived rather than read directly.

This step flattens both JSON blobs into typed columns, normalizes the tags
field, derives boolean attribute flags from tags, parses price/size strings
into numeric fields, and builds a single `search_text` column that
concatenates everything useful for an embedding model into one clean string
per item.

## Column reference

| Column | Type | Description | Null rate |
|---|---|---|---|
| `item_id` | int | Unique product identifier | 0% |
| `name` | str | Product name | 0% |
| `brand` | str | Manufacturer brand, or "Store Brand" for private label, or "Unknown" | 0% |
| `category_l0` .. `category_l3` | str | Taxonomy levels, coarsest to most specific | l0/l1: 0%, l2: 2%, l3: 50% |
| `category_path` | str | Levels joined as `"Grocery > International Foods > Latino Foods"` | 0% |
| `description` | str | Marketing/product description text | 6.5% |
| `ingredients` | str | Ingredient list, where available (mostly food/cosmetic items) | 47% |
| `tags_str` | str | Pipe-delimited normalized tags, e.g. `"gluten free\|organic"` | 67% (no tags recorded) |
| `is_organic`, `is_vegan`, `is_gluten_free`, `is_kosher`, `is_lactose_free` | bool | Derived from tags_str | 0% (False where absent) |
| `is_store_brand` | bool | True if private label | 0% |
| `unit_price_usd` | float | Parsed numeric price per unit | 71.5% (price data only recorded for ~28% of catalog) |
| `unit_price_uom` | str | Unit of the price (e.g. "fl. oz.", "lb", "ea") | 71.5% |
| `package_size` | str | Human-readable size, e.g. "16 fl. oz." | 4.4% |
| `package_size_numeric` | float | Numeric package size | 70.3% |
| `num_servings`, `serving_size`, `serving_size_uom` | float/str | Nutrition serving info, food items only | 56-60% |
| `billed_by_weight`, `ordered_by_weight` | bool | Fulfillment attributes (e.g. deli, produce) | 0% |
| `search_text` | str | **Concatenated field for embedding**: name + brand + category + description + ingredients + tags | 0% |

## Category distribution (top level)

More Departments (35%, mostly personal care/household/general merchandise),
Grocery (33%), Wine/Beer/Spirits (9%), Frozen (7%), Dairy (5%), Produce &
Floral (3%), Bakery (2%), Meat (2%), and smaller categories rounding out
the remainder.

## Known data characteristics to account for downstream

- **Price/size fields are sparse by nature**, not a parsing defect — only
  about 28% of the source catalog had per-unit pricing populated at all.
  Don't rely on `unit_price_usd` for full-catalog filtering; treat it as an
  optional facet.
- **`category_l3` is null for about half the catalog** — many products
  only have a 3-level taxonomy. Use `category_path` for display/embedding
  rather than assuming l3 is always populated.
- **No duplicate `item_id` and no fully duplicate rows.** Some products
  share the same `name` + `brand` but have distinct `item_id`, price, and
  package size — these are legitimate separate SKUs (different pack sizes
  or variable-weight listings), not data quality issues, and were
  intentionally not deduplicated.
- **`search_text` length varies widely** (55 to ~8,600 characters, median
  ~430) since ingredient lists and descriptions vary enormously by
  category. Worth capping/truncating at the embedding-model's max token
  length during the indexing step rather than assuming uniform length.
