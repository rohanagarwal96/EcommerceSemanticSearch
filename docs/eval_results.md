# Evaluation Results

## Methodology

- 35 hand-labeled queries, binary relevance.
- Relevant items identified via pooling: the deduplicated union of the
  top-10 results from all 4 modes, judged relevant/not by hand.
  "Relevant" therefore means "relevant among pooled candidates", not
  an exhaustive ground truth over the full 55,516-row catalog.

## Results

| Mode | Recall@10 | NDCG@10 | MRR |
|---|---|---|---|
| dense | 0.4441 | 0.9145 | 0.9486 |
| bm25 | 0.4120 | 0.8967 | 0.9429 |
| hybrid | 0.4437 | 0.9360 | 0.9857 |
| hybrid-rerank | 0.4285 | 0.9125 | 0.9357 |
