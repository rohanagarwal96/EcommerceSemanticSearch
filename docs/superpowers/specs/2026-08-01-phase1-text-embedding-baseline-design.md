# Phase 1: Text Embedding Baseline — Design

## Context

This is the first of eight sequential build phases for the E-Commerce
Semantic Product Search project (see project brief in repo history / the
brainstorming conversation that produced this spec). Phase 1 builds the
foundational text-embedding + FAISS retrieval loop that later phases
(BM25 hybrid search, cross-encoder reranking, the FastAPI backend) will
import and extend — it is not a throwaway prototype.

Catalog: `data/ecommerce_catalog_enriched.csv`, 55,516 rows. Key facts
from `data/DATA_DICTIONARY.md` that shape this design:

- `search_text` (the field to embed) ranges 55–8,600 characters, median
  ~430, with name/brand/category first and long ingredient lists trailing.
- `category_l3` is null for ~50% of rows; `unit_price_usd` populated for
  only ~28% — both are optional facets, not required fields.
- No nulls in `item_id`, `name`, or `search_text`.

## Goal

A working CLI: given a free-text query, return the top-k most semantically
relevant products from the full 55,516-row catalog, using
`BAAI/bge-base-en-v1.5` embeddings and a FAISS index.

## Architecture

```
src/ecomsearch/
  config.py       # paths, model name, MAX_TOKENS=512, DEFAULT_TOP_K=10
  embeddings.py    # bge-base-en-v1.5 wrapper: truncation, query prefix, normalization
  index.py         # FAISS IndexFlatIP build / save / load / search
  cli.py           # argparse CLI: `search` command

scripts/
  build_index.py   # batch job: catalog CSV -> embeddings -> FAISS index -> artifacts/

artifacts/          # gitignored; regenerable from source via build_index.py
  catalog.faiss      # matched by existing *.faiss gitignore pattern
  item_ids.npy       # index position -> item_id mapping
```

`artifacts/` is a new directory added to `.gitignore` (the existing
`*.faiss`/`*.index` patterns already cover the FAISS file itself; the
directory entry additionally covers the accompanying `item_ids.npy`
mapping file, which isn't covered by an extension-based pattern).

Rationale for a `src/` package over flat scripts: Phase 3 (hybrid fusion),
Phase 4 (eval harness), and Phase 5 (FastAPI backend) all need to import
this same embedding/search logic rather than shell out to a script.

At 55,516 rows x 768 dims (~164MB resident), a flat brute-force
`IndexFlatIP` is fast enough (sub-10ms search) with no approximation error.
IVF/HNSW/quantization are out of scope here — revisit only if Phase 4
latency benchmarking shows a need.

## Data flow

1. `scripts/build_index.py` loads the CSV with pandas, reading `item_id`
   and `search_text` for all rows.
2. `embeddings.py` batches `search_text` through the bge tokenizer/model
   (batch size 64), truncating each input at 512 tokens (the model's max
   context). No chunking or pooling — truncation only, since `search_text`
   is built name/brand/category-first, so truncation only ever cuts into
   the tail (long ingredient lists), not the most salient fields.
3. Each output embedding is L2-normalized, so FAISS inner product is
   equivalent to cosine similarity.
4. `index.py` builds an `IndexFlatIP`, adds vectors in `item_id` order,
   and persists both the index and the parallel `item_id` array to
   `artifacts/`.
5. `cli.py search "<query>"`:
   - Prepends the bge-recommended query instruction prefix
     (`"Represent this sentence for searching relevant passages: "`) to
     the query text. This prefix is applied **only** at query time, never
     to catalog `search_text` at index time — matches bge's documented
     asymmetric usage and is exercised by a dedicated unit test to guard
     against future regression (e.g. someone reusing the same embed
     function for both without noticing the asymmetry).
   - Embeds and normalizes the query the same way (truncation logic
     applies equally, though queries are expected to be short).
   - Runs FAISS top-k search, maps returned positions back to `item_id`
     via the persisted mapping.
   - Looks up display columns (`name`, `brand`, `category_path`,
     `unit_price_usd`) from the catalog and prints a pretty table (rank,
     score, item_id, name, brand, category_path) via `rich` or `tabulate`.

## Testing (TDD)

Tests written first, per project convention. Small synthetic fixtures
(5-10 fake products), not the full 55k catalog — keeps the suite fast.

- `embeddings.py`:
  - text longer than 512 tokens is truncated, not errored
  - query prefix is applied to queries and only to queries (not to
    document/passage embedding calls)
  - output vectors are unit-norm (L2 norm == 1 within floating-point
    tolerance)
- `index.py`:
  - build -> save -> load round-trip preserves vectors and the
    `item_id` mapping
  - top-k search on a small known set returns the expected nearest
    neighbor first
- One integration test: build a tiny index from synthetic products with
  one obvious best match for a given query string, assert it ranks first
  end-to-end through the same code path the CLI uses.

## Error handling

Two foreseeable failure modes, both handled with a plain, actionable
message (no raw stack traces):

- Catalog CSV missing at the configured path.
- `search` invoked before `build_index` has ever run (no index file in
  `artifacts/` yet) — message tells the user to run
  `python scripts/build_index.py` first.

## Dependencies & environment

- `requirements.txt` + `venv` (not Poetry/pip-tools) — simplest for an
  interviewer to clone and run.
- Core new dependencies this phase: `sentence-transformers`, `faiss-cpu`,
  `pandas`, `numpy`, `rich` (or `tabulate`), `pytest`.
- Model weights download from Hugging Face on first run and cache under
  the standard `~/.cache/huggingface` location — no special handling
  needed.

## Out of scope for Phase 1 (later phases)

- BM25 keyword search, hybrid fusion, cross-encoder reranking (Phase 3)
- Formal evaluation harness / metrics (Phase 4)
- Latency benchmarking and index optimization beyond the flat-index
  default above (Phase 4)
- FastAPI/Streamlit serving layer (Phase 5)
- Qdrant Cloud deployment (Phase 6)

## Amendment (during Task 6 execution): switched to `bge-small-en-v1.5`

Tasks 1-5 were implemented and reviewed against `BAAI/bge-base-en-v1.5` as
originally specified above. During Task 6 (the full 55,516-row batch
embed), that model proved impractical on the development machine's CPU
(Intel i7-8650U, a 15W 4-core/8-thread laptop chip): a live progress check
via `py-spy` at the 2h54m mark showed only 11,456/55,516 rows (20.6%)
complete, projecting a ~14 hour total run. `torch` threading (4 threads,
matching physical cores) and the BLAS backend (MKL + oneDNN, AVX2) were
confirmed correctly configured — the bottleneck is the hardware itself,
not misconfiguration, so there was no quick fix available at the
`bge-base-en-v1.5` model size.

**Decision:** switch `MODEL_NAME` in `src/ecomsearch/config.py` to
`BAAI/bge-small-en-v1.5` (~33M params vs. base's ~110M, same bge family
and query-prefix convention, 384-dim output instead of 768-dim). This is
expected to cut CPU inference cost roughly 3x, bringing the full-catalog
embed into a practical range on this hardware, at a modest, well-known
retrieval-quality tradeoff versus the base model (small underperforms
base by a few points on standard embedding benchmarks, but remains a
strong general-purpose text embedding model).

**Consequences:**
- `src/ecomsearch/config.py`'s `MODEL_NAME` constant is the single source
  of truth for which model is used; `embeddings.py`, `build_index.py`,
  and `cli.py` all reference it indirectly and needed no code changes.
- Tasks 3-5's tests were written generically (no hardcoded model name or
  embedding dimension) and continue to pass unchanged against the new
  model.
- The FAISS index dimension is inferred from the embedding output shape
  at build time, so `index.py` needed no changes either.
- Any future comparison/eval work (Phase 4) should note results reflect
  `bge-small-en-v1.5`, not `bge-base-en-v1.5` as originally scoped here.
