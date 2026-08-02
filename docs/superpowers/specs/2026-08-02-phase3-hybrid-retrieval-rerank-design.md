# Phase 3: Hybrid Retrieval + Reranking — Design

## Context

This is the third of eight sequential build phases for the E-Commerce
Semantic Product Search project. Phase 1 (text embedding baseline: FAISS
+ `bge-small-en-v1.5`, a working `ecomsearch search` CLI over the full
55,516-row catalog) and Phase 2 (multimodal/CLIP module, a standalone
demo on a separate dataset) are both complete and merged.

Phase 3 adds keyword (BM25) search over the same 55,516-row catalog used
in Phase 1, fuses it with the existing dense vector search via
Reciprocal Rank Fusion, and adds cross-encoder reranking on the fused
candidates — matching the original project brief's Phase 3 scope
exactly.

## Forward-looking requirement: mode selection

Phase 4 (evaluation, not yet built) needs to compute Recall@10, NDCG@10,
and MRR separately for **keyword-only, dense-only, hybrid, and
hybrid+rerank** and produce a comparison table. Phase 3 must therefore
expose each of these four retrieval modes as independently callable,
non-CLI-coupled functions — not just as one fixed end-to-end pipeline —
so Phase 4's eval harness can import and call them directly without
duplicating retrieval logic or reaching into CLI internals.

## Architecture

```
src/ecomsearch/
  bm25.py        # BM25Index: build/search/save/load, mirrors ProductIndex's shape
  fusion.py       # reciprocal_rank_fusion(ranked_id_lists, k=60) -- pure function
  reranker.py     # CrossEncoderReranker: wraps cross-encoder/ms-marco-MiniLM-L-6-v2
  search.py       # orchestration: dense_search, bm25_search, hybrid_search
  cli.py          # gains --mode {dense,bm25,hybrid,hybrid-rerank} flag (Modify)

scripts/
  build_bm25_index.py   # batch job: catalog CSV -> tokenize -> BM25Okapi -> artifacts/bm25.pkl
```

- `BM25Index` mirrors Phase 1's `ProductIndex` (`build`/`search`/`save`/
  `load`), keeping the codebase's established "index wrapper" shape
  consistent across keyword and dense retrieval.
- `rank_bm25`'s `BM25Okapi` has no native persistence, so `BM25Index.save`/
  `.load` use `pickle` — the pickled object plus the parallel `item_id`
  array are written to a single `artifacts/bm25.pkl` (gitignored, same
  `artifacts/` pattern as the FAISS index).
- No new heavy dependency for tokenization: a small internal
  lowercase/strip-punctuation/whitespace-split tokenizer is used
  consistently at both build time and query time — full NLP tokenization
  (e.g. spaCy) is unnecessary for BM25 and would be scope creep.
- The cross-encoder reranker uses `sentence_transformers.CrossEncoder`,
  which is already available via the installed `sentence-transformers`
  dependency — no new package required.
- `search.py` contains zero printing/table-rendering logic; `cli.py`
  calls into it and handles presentation only, matching the separation
  Phase 4 needs.

## Data flow

1. `scripts/build_bm25_index.py`: loads `item_id` + `search_text` from
   the catalog CSV, tokenizes every row, builds a `BM25Okapi` instance
   over the whole 55,516-row corpus, and pickles it (with the `item_id`
   mapping) to `artifacts/bm25.pkl`. Pure Python/numpy term-frequency
   work, no neural network — expected to be fast (likely well under a
   minute) even on this project's CPU-constrained dev machine, unlike
   Phase 1/2's embedding jobs.
2. `dense_search(query, top_k)` — thin wrapper around Phase 1's existing
   `Embedder` + `ProductIndex`, unchanged behavior.
3. `bm25_search(query, top_k)` — tokenizes the query with the same
   tokenizer used at build time, scores via `BM25Okapi.get_scores`,
   returns the top-k `(item_id, score)` pairs.
4. `hybrid_search(query, top_k, use_rerank=True)`:
   - Retrieves the top 100 candidates from each of `dense_search` and
     `bm25_search` (wide pools so RRF has good material to fuse).
   - `reciprocal_rank_fusion([dense_ids, bm25_ids], k=60)` combines the
     two rankings by rank position, not raw score — this deliberately
     avoids needing to normalize BM25 scores (roughly 0-30+,
     corpus-dependent) against cosine similarities (0-1), which aren't
     directly comparable. `k=60` is the standard default from the
     original RRF paper.
   - If `use_rerank` (the "hybrid+rerank" mode): take the fused top 50,
     look up each candidate's `search_text`, score `(query, search_text)`
     pairs with the cross-encoder, re-sort by that score, truncate to
     `top_k`.
   - If not `use_rerank` (the "hybrid" mode): truncate the fused RRF list
     directly to `top_k`.
5. `cli.py search "<query>" --mode {dense,bm25,hybrid,hybrid-rerank}`
   (default `hybrid-rerank`): dispatches to the matching function above,
   then renders the existing rich table unchanged — `--mode` only
   changes which candidates are returned, not how results are displayed.

## Testing (TDD)

- `BM25Index`: tests mirror Phase 1's `ProductIndex` tests exactly in
  spirit — small synthetic corpus, build→save→load round-trip, correct
  top result for an obvious keyword match. Fast, no model involved.
- `reciprocal_rank_fusion`: pure-function unit tests on small synthetic
  ranked ID lists — e.g. an item ranked #1 in both input lists must score
  higher than one ranked #1 in only one list. No model, very fast.
- `CrossEncoderReranker`: TDD test using the real (small, ~22M-param)
  cross-encoder model on a tiny synthetic query + 2-3 candidate texts,
  asserting the obviously-relevant text ranks first — same "exercise the
  real model" philosophy as Phase 1/2's embedder tests, not a mock.
- `search.py`: integration tests building tiny synthetic BM25 + dense
  indices together (mirroring Phase 1/2's integration-test pattern),
  verifying `hybrid_search` correctly fuses and (optionally) reranks
  end-to-end through the real component classes.

## Error handling

Same established pattern and tone as Phases 1-2: plain, actionable
`SystemExit` messages, no raw stack traces:

- `bm25_search`/`hybrid_search` invoked before `build_bm25_index.py` has
  ever run (no `artifacts/bm25.pkl` yet) — message tells the user to run
  the build script first, mirroring the existing FAISS-index-missing
  message exactly in tone.

## Dependencies

No new dependencies. `rank_bm25` needs to be added to `requirements.txt`
(not yet present); `sentence-transformers` (already installed) provides
`CrossEncoder`.

## Out of scope for Phase 3 (later phases)

- Formal evaluation harness / metrics computation (Phase 4) — Phase 3
  only needs to make the four modes independently callable; actually
  computing Recall@10/NDCG@10/MRR and the comparison table is Phase 4's
  job.
- Latency benchmarking and optimization (Phase 4).
- Any change to the multimodal (Phase 2) module — Phase 3 touches only
  the main catalog search path.
- FastAPI/Streamlit serving layer (Phase 5), Qdrant deployment (Phase 6).
