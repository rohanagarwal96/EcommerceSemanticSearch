# Phase 2: Multimodal (CLIP) Module — Design

## Context

This is the second of eight sequential build phases for the E-Commerce
Semantic Product Search project. Phase 1 (text embedding baseline: FAISS +
`bge-small-en-v1.5`, a working `ecomsearch search` CLI over the full
55,516-row catalog) is complete and merged.

The core catalog (`data/ecommerce_catalog_enriched.csv`) has **no product
images** — the source retailer's Terms of Service prohibit image scraping,
and this was excluded deliberately during data sourcing. Phase 2 therefore
demonstrates cross-modal (text query → image results) search on a
**separate, properly licensed public dataset**, not derived from the main
catalog. The README must state this separation plainly.

## Dataset selection

**Chosen:** `nirmalsankalana/mini-product-image-and-text-dataset` on Kaggle.

- **License: MIT**, confirmed directly via `kaggle datasets metadata`
  (not inferred from the Kaggle web page, which is JS-rendered and doesn't
  expose license text to a simple fetch).
- 44,441 fashion products (corrected from an initial 44,671 estimate made
  during brainstorming, before download — that figure came from `wc -l`
  on `data.csv`, which overcounts because ~230 quoted `description`
  fields contain embedded newlines; the true row count, confirmed via
  proper CSV parsing after downloading in Task 6, is 44,441, with a
  perfect 1:1 match to 44,441 image files), each with a small thumbnail
  image (~3-15KB JPEG) plus real text: a `display name` (short,
  caption-style — e.g. "Puma Men Black 65CC Lo Ducati Sports Shoes"), a
  longer marketing
  `description`, and a `category` label.
- ~362MB total download — the smallest of the viable candidates
  considered (others: `paramaggarwal/fashion-product-images-small`, MIT,
  565MB, short text only; `nirmalsankalana/fashion-product-text-images-dataset`,
  MIT, same content family at 3.2GB, no advantage over the mini version;
  `bhavikjikadara/e-commerce-products-images`, CC BY 4.0, 297MB, text
  richness unverified; `ronakbokaria/myntra-products-dataset`, CC0-1.0,
  image content unconfirmed).
- Explicitly preprocessed "for efficient multimodal model training" by its
  uploader — a good fit for this use case as-is.

Source and license (MIT, dataset owner `nirmalsankalana`) must be
documented in the README alongside the Phase 1 disclosure that this is a
separate dataset from the main catalog.

## Scope: 5,000-image subset, not the full 44,441

Per the Phase 1 lesson (the dev machine's CPU, an Intel i7-8650U 15W
laptop chip, is weak — a full bge-base-en-v1.5 embed run projected to
~14 hours before a model switch), Phase 2 embeds a **5,000-image subset**,
stratified-sampled by `category` so the demo covers diverse product types
rather than risking an unbalanced random sample. This is a demonstration
of cross-modal search per the original brief, not a requirement to cover
every item. The full dataset remains available as a documented future
option if CLIP throughput on this subset turns out to be fast in
practice. As in Phase 1, actual throughput will be measured early (via
`py-spy`, same technique used in Phase 1) before committing to the full
sample run, not assumed from a generic estimate.

## Architecture

```
src/ecomsearch/multimodal/
  config.py          # dataset paths, artifacts/multimodal/ paths, CLIP_MODEL_NAME,
                      # SUBSET_SIZE=5000, DEMO_RESULTS_DIR
  clip_embedder.py    # ClipEmbedder: embed_images(paths), embed_text(texts)
  cli.py             # `search` command; separate console-script entry point
                      # `ecomsearch-images` (NOT a subcommand of the existing
                      # `ecomsearch` CLI, to keep "search the real 55k catalog"
                      # and "search the demo image dataset" unambiguous)

scripts/
  download_multimodal_dataset.py  # kaggle API download+unzip -> data/multimodal/
  build_multimodal_index.py       # sample -> embed -> build -> save

data/multimodal/     # gitignored (new .gitignore entry); raw downloaded
                      # dataset, regenerable via download_multimodal_dataset.py
  data.csv
  data/<id>.jpg

artifacts/multimodal/  # gitignored (existing artifacts/ pattern already covers
                        # this); regenerable via build_multimodal_index.py
  catalog.faiss
  item_ids.npy
  subset_metadata.csv   # the 5,000-row sampled subset's display name/category/path

demo_results/<query-slug>/  # gitignored; top-k matched images copied here per query
```

**Reuse, not duplication:** `ecomsearch.index.ProductIndex` (Phase 1) is
reused as-is — it was already embedding-agnostic (dimension inferred at
build time, generic int64 IDs), so no changes to `index.py` are needed.
Only a new `ClipEmbedder` is needed, since CLIP's dual image/text towers
have a fundamentally different interface (`transformers.CLIPModel`) than
`sentence-transformers`' `SentenceTransformer` used in Phase 1.
`transformers` is already an installed dependency (via
`sentence-transformers`), so `Pillow` (image loading) is the only new
dependency.

Unlike Phase 1's `bge-small-en-v1.5` (which needs an asymmetric query
instruction prefix), CLIP's text and image encoders share one joint
embedding space by construction — no prefix is applied to query text.

## Data flow

1. `scripts/download_multimodal_dataset.py`: if `data/multimodal/data.csv`
   doesn't already exist, downloads and unzips the Kaggle dataset via the
   `kaggle` API into `data/multimodal/`. Errors clearly (not a raw
   stack trace) if `~/.kaggle/kaggle.json` credentials aren't found.
2. `scripts/build_multimodal_index.py`:
   - Loads `data/multimodal/data.csv`.
   - `stratified_sample(df, "category", 5000)` — a standalone, unit-testable
     function — samples proportionally by category down to 5,000 rows.
   - `ClipEmbedder.embed_images(image_paths)` batch-encodes the sampled
     images, L2-normalized (cosine similarity via `IndexFlatIP`, same
     pattern as Phase 1).
   - Builds a `ProductIndex`, keyed by the real numeric product ID parsed
     from each image's filename (e.g. `10000.jpg` → `10000`).
   - Saves the index + item-id mapping to `artifacts/multimodal/`, plus
     `subset_metadata.csv` (the sampled 5,000 rows' display name,
     category, image path) so the CLI never needs to load the full
     44,441-row source CSV at query time.
3. `ecomsearch-images search "<query>"`:
   - `ClipEmbedder.embed_text([query])` — no prefix.
   - Loads the index, searches top-k, joins against `subset_metadata.csv`.
   - Prints a rich table (rank, score, display name, category, image
     filename).
   - Copies the top-k matched image files into
     `demo_results/<query-slug>/` (slug = sanitized query text) so results
     can actually be viewed, not just read as text — these are product
     photos, and that's the point of this phase.

## Testing (TDD)

- `ClipEmbedder`: tiny PIL-generated synthetic images (e.g. a solid red
  square vs. a solid blue square) plus a real model (session-scoped
  fixture, mirroring Phase 1's `embedder` fixture pattern in
  `tests/conftest.py`). Tests verify: unit-norm output vectors (same
  check as Phase 1); and a genuine cross-modal alignment sanity check —
  does the text "a red square" embed closer (higher cosine similarity) to
  the red-square image's embedding than to the blue-square one? This is
  the actual point of CLIP, so it's asserted directly rather than only
  checking shapes.
- `stratified_sample`: fast unit tests with a small synthetic DataFrame
  (no model or images involved) — verifying proportional category
  representation in the output.
- Integration test mirroring Phase 1's `test_integration.py`: build a
  tiny index from a few synthetic image/text pairs via the real
  `ClipEmbedder` + `ProductIndex`, run a query, assert the expected match
  ranks first.
- `ProductIndex` itself needs no new tests (unchanged, already covered by
  Phase 1's `tests/test_index.py`).

## Error handling

Same pattern and tone as Phase 1 — plain, actionable `SystemExit`
messages, no raw stack traces, for each foreseeable failure:

- `download_multimodal_dataset.py`: `~/.kaggle/kaggle.json` credentials
  not found.
- `build_multimodal_index.py`: `data/multimodal/data.csv` missing (run
  the download script first).
- `ecomsearch-images search`: no index found in `artifacts/multimodal/`
  (run the build script first).

## Dependencies

- New: `kaggle` (dataset download), `Pillow` (image loading for CLIP).
- Already present: `transformers` (via `sentence-transformers`), `faiss-cpu`,
  `pandas`, `numpy`, `rich`, `pytest`.

## Out of scope for Phase 2 (later phases / explicitly deferred)

- Embedding the full 44,441-item dataset (documented as a future option,
  not a Phase 2 requirement).
- Any integration between the multimodal module and the main
  `ecomsearch` catalog CLI or FastAPI backend — Phase 2 is a standalone
  demonstration per the original brief, kept clearly separate.
- A visual/web UI for browsing cross-modal results (the Streamlit
  frontend is Phase 5; Phase 2's "UI" is the CLI + copied image files).
- Hybrid retrieval, reranking, evaluation harness, deployment (Phases 3-6,
  unchanged from the original brief, and not affected by Phase 2).
