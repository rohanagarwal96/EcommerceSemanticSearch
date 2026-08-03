# E-Commerce Semantic Product Search

A semantic product search engine over a real e-commerce catalog: text and
image (multimodal/CLIP) search that goes beyond exact keyword matching,
targeting sub-200ms latency, deployed live at $0 infrastructure cost.

## Status

Phases 1-4 complete — a working semantic search CLI (dense, BM25 keyword,
and hybrid+reranked modes) over the full catalog, plus a cross-modal
(text-to-image) search demo. Retrieval quality has been evaluated across
all 4 modes — see [Evaluation Results](docs/eval_results.md). Latency was
benchmarked in a warmed, cached process and optimized via request-level
caching and search parallelization; `dense` and `bm25` modes meet a
<200ms p95 target, `hybrid` improved substantially (229.6ms → 214.3ms)
but doesn't fully clear it due to a documented architectural constraint
in the keyword-search library — see
[Latency Results](docs/latency_results.md) for the full investigation.
Phases 5-8 in progress; this section will be updated as each phase lands.

- [x] Phase 1 — Text embedding baseline (FAISS + bge-small-en-v1.5)
- [x] Phase 2 — Multimodal (CLIP) module
- [x] Phase 3 — Hybrid retrieval + reranking
- [x] Phase 4 — Evaluation and latency engineering
- [ ] Phase 5 — Serving layer (FastAPI + Streamlit)
- [ ] Phase 6 — Deployment (Qdrant Cloud + Hugging Face Spaces)
- [ ] Phase 7 — Production hygiene (CI, logging, rate limiting)
- [ ] Phase 8 — Documentation finalization

## Data

The catalog (`data/ecommerce_catalog_enriched.csv`, 55,516 rows) has been
sourced, genericized, and enriched already; see `data/DATA_DICTIONARY.md`
for the full schema. The source retailer's Terms of Service prohibit image
scraping, so this catalog has no product images.

The multimodal (CLIP) module (Phase 2) is demonstrated on a separate,
properly licensed public dataset:
[Mini Fashion Product Images and Text Dataset](https://www.kaggle.com/datasets/nirmalsankalana/mini-product-image-and-text-dataset)
by nirmalsankalana on Kaggle, MIT licensed, 44,441 fashion product
image/text pairs. Phase 2 embeds a ~5,000-item subset (stratified by
category) via CLIP for a cross-modal (text-to-image) search demo — this
is entirely separate from the main 55,516-row catalog used everywhere
else in this project.

## Stack

| Layer | Choice |
|---|---|
| Text embeddings | `BAAI/bge-small-en-v1.5` |
| Image embeddings | `openai/clip-vit-base-patch32` |
| Vector index (dev) | FAISS |
| Vector index (deployed) | Qdrant Cloud (free tier) |
| Keyword search | `rank_bm25` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Backend | FastAPI (Docker) |
| Frontend | Streamlit (Docker) |
| Hosting | Hugging Face Spaces (free) |
| CI/CD | GitHub Actions |

## Evaluation

Retrieval quality was measured across all 4 modes on 35 hand-labeled
queries (binary relevance, pooled candidates). Full methodology in
[docs/eval_results.md](docs/eval_results.md).

| Mode | Recall@10 | NDCG@10 | MRR |
|---|---|---|---|
| dense | 0.4441 | 0.9145 | 0.9486 |
| bm25 | 0.4120 | 0.8967 | 0.9429 |
| hybrid | 0.4437 | 0.9360 | 0.9857 |
| hybrid-rerank | 0.4285 | 0.9125 | 0.9357 |

## Latency

Latency was measured over 350 timed calls per mode in a warmed, cached
process, against a <200ms p95 target for `dense`/`bm25`/`hybrid`
(`hybrid-rerank` is not gated — the cross-encoder pass is inherently the
dominant cost there). `dense` and `bm25` pass with real margin; `hybrid`
improved from 229.6ms to 214.3ms via search parallelization but doesn't
fully clear the bar, due to `rank_bm25`'s brute-force full-corpus scoring
cost. Full methodology and investigation in
[docs/latency_results.md](docs/latency_results.md).

| Mode | p50 (ms) | p95 (ms) | p99 (ms) | Verdict |
|---|---|---|---|---|
| dense | 53.7 | 106.9 | 142.9 | PASS |
| bm25 | 83.9 | 146.3 | 150.4 | PASS |
| hybrid | 162.1 | 214.3 | 225.6 | FAIL |
| hybrid-rerank | 4735.2 | 6207.1 | 6662.9 | not gated |

## Setup

```bash
python -m venv venv
source venv/Scripts/activate   # on Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Build the search index once (embeds all 55,516 products; took a few
hours on a low-power laptop CPU in development, but should be much
faster — likely under 30 minutes — on a typical desktop or server CPU):

```bash
python scripts/build_index.py
```

Build the BM25 keyword index once (fast — pure term-frequency counting,
no neural network, typically well under a minute):

```bash
python scripts/build_bm25_index.py
```

Then choose a retrieval mode:

```bash
ecomsearch search "organic almond milk" --top-k 5 --mode hybrid-rerank
```

`--mode` accepts `dense`, `bm25`, `hybrid`, or `hybrid-rerank` (the
default) — useful for comparing retrieval strategies.

### Multimodal (CLIP) demo

Requires a Kaggle API token at `~/.kaggle/kaggle.json`
([setup instructions](https://www.kaggle.com/docs/api)).

```bash
python scripts/download_multimodal_dataset.py
python scripts/build_multimodal_index.py
ecomsearch-images search "something warm for rainy weather" --top-k 5
```

Matched images are copied to `demo_results/<query-slug>/` for viewing.

## Known limitations

To be documented as they arise. Note in advance: the Qdrant free-tier
cluster auto-suspends after about a week of inactivity, so a demo visitor
may see a cold-start delay on first query.

## License

MIT — see [LICENSE](LICENSE).
