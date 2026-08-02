# E-Commerce Semantic Product Search

A semantic product search engine over a real e-commerce catalog: text and
image (multimodal/CLIP) search that goes beyond exact keyword matching,
targeting sub-200ms latency, deployed live at $0 infrastructure cost.

## Status

Phase 1 complete — a working semantic search CLI over the full catalog.
Phases 2-8 in progress; this section will be updated as each phase lands.

- [x] Phase 1 — Text embedding baseline (FAISS + bge-small-en-v1.5)
- [ ] Phase 2 — Multimodal (CLIP) module
- [ ] Phase 3 — Hybrid retrieval + reranking
- [ ] Phase 4 — Evaluation and latency engineering
- [ ] Phase 5 — Serving layer (FastAPI + Streamlit)
- [ ] Phase 6 — Deployment (Qdrant Cloud + Hugging Face Spaces)
- [ ] Phase 7 — Production hygiene (CI, logging, rate limiting)
- [ ] Phase 8 — Documentation finalization

## Data

The catalog (`data/ecommerce_catalog_enriched.csv`, 55,516 rows) has been
sourced, genericized, and enriched already; see `data/DATA_DICTIONARY.md`
for the full schema. The source retailer's Terms of Service prohibit image
scraping, so this catalog has no product images — the multimodal/CLIP
module (Phase 2) uses a separately sourced, properly licensed public
product-image dataset instead. Details and license attribution will be
added here once that dataset is selected.

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

Then search:

```bash
ecomsearch search "organic almond milk" --top-k 5
```

## Known limitations

To be documented as they arise. Note in advance: the Qdrant free-tier
cluster auto-suspends after about a week of inactivity, so a demo visitor
may see a cold-start delay on first query.

## License

MIT — see [LICENSE](LICENSE).
