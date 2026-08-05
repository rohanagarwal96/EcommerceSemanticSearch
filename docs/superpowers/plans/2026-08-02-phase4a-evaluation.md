# Phase 4a: Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A hand-labeled evaluation set of ~35 domain-appropriate queries, and `scripts/run_eval.py` producing a Recall@10/NDCG@10/MRR comparison table across all 4 retrieval modes (`dense`/`bm25`/`hybrid`/`hybrid-rerank`), written to `docs/eval_results.md`.

**Architecture:** Pure, independently-testable metric functions in `src/ecomsearch/eval.py`; a hand-curated `eval/eval_queries.json` (built via a small pooling helper script, then reviewed by the user before being treated as ground truth); a batch script that runs all 4 modes against the eval set and writes results.

**Tech Stack:** Existing `search.py` functions from Phase 3, plain Python/`math` for metrics (no new dependencies).

**Spec:** `docs/superpowers/specs/2026-08-02-phase4a-evaluation-design.md`

**⚠️ Critical process note for whoever executes this plan:** Task 4 produces `eval/eval_queries.json` — the ground-truth relevance judgments everything else depends on. Per the spec and the original project brief, **the user must review and approve this file before Task 5 (which runs the real evaluation against it) begins.** If executing via subagent-driven-development, the controller must pause after Task 4's review and get explicit user sign-off — this is not optional and is not satisfied by the normal spec-compliance/code-quality review alone.

---

### Task 1: Config additions

**Files:**
- Modify: `src/ecomsearch/config.py`

- [ ] **Step 1: Append these lines to the end of `src/ecomsearch/config.py`**

The file currently ends with the Phase 3 additions (`RERANK_POOL_SIZE = 50`). Append:
```python
EVAL_QUERIES_PATH = REPO_ROOT / "eval" / "eval_queries.json"
EVAL_RESULTS_PATH = REPO_ROOT / "docs" / "eval_results.md"
EVAL_TOP_K = 10
```

- [ ] **Step 2: Verify it imports**

Run:
```bash
python -c "from ecomsearch.config import EVAL_QUERIES_PATH, EVAL_RESULTS_PATH, EVAL_TOP_K; print(EVAL_QUERIES_PATH); print(EVAL_RESULTS_PATH); print(EVAL_TOP_K)"
```
Expected: prints the path to `eval/eval_queries.json`, then `docs/eval_results.md`, then `10`.

- [ ] **Step 3: Commit**

```bash
git add src/ecomsearch/config.py
git commit -m "feat: add Phase 4a eval config constants"
git push origin main
```

---

### Task 2: Evaluation metrics (TDD)

**Files:**
- Create: `src/ecomsearch/eval.py`
- Test: `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests in `tests/test_eval.py`**

```python
import math

import pytest

from ecomsearch.eval import mrr, ndcg_at_k, recall_at_k


def test_recall_at_k_full_recall():
    retrieved = [10, 20, 30]
    relevant = {10, 30}
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_recall_at_k_partial_recall():
    retrieved = [10, 20, 30]
    relevant = {10, 30, 40}
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_respects_k_cutoff():
    retrieved = [10, 20, 30, 40]
    relevant = {40}
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)


def test_ndcg_at_k_matches_formula():
    retrieved = [10, 20, 30]
    relevant = {10, 30}
    k = 3

    dcg = 1 / math.log2(2) + 0 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    expected = dcg / idcg

    assert ndcg_at_k(retrieved, relevant, k) == pytest.approx(expected)


def test_ndcg_at_k_perfect_ranking_scores_one():
    retrieved = [10, 30, 20]
    relevant = {10, 30}
    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_no_relevant_items_scores_zero():
    retrieved = [10, 20, 30]
    relevant = set()
    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)


def test_mrr_first_relevant_at_rank_two():
    retrieved = [20, 10, 30]
    relevant = {10}
    assert mrr(retrieved, relevant) == pytest.approx(0.5)


def test_mrr_first_relevant_at_rank_one():
    retrieved = [10, 20]
    relevant = {10, 30}
    assert mrr(retrieved, relevant) == pytest.approx(1.0)


def test_mrr_no_relevant_found():
    retrieved = [20, 30]
    relevant = {10}
    assert mrr(retrieved, relevant) == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.eval'`

- [ ] **Step 3: Write `src/ecomsearch/eval.py`**

```python
"""Evaluation metrics: Recall@k, NDCG@k, and MRR over ranked item_id lists."""

import math


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_at_k = set(retrieved_ids[:k])
    return len(retrieved_at_k & relevant_ids) / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    dcg = 0.0
    for i, item_id in enumerate(retrieved_ids[:k], start=1):
        if item_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval.py -v`
Expected: PASS (9 passed, should take well under a second — pure math, no model)

- [ ] **Step 5: Commit**

```bash
git add src/ecomsearch/eval.py tests/test_eval.py
git commit -m "feat: add Recall@k/NDCG@k/MRR evaluation metrics with TDD tests"
git push origin main
```

---

### Task 3: Eval-pooling helper script (TDD)

**Files:**
- Create: `scripts/pool_eval_candidates.py`
- Test: `tests/test_pool_eval_candidates.py`

- [ ] **Step 1: Write the failing test in `tests/test_pool_eval_candidates.py`**

```python
import pytest

import pool_eval_candidates


def test_main_exits_with_usage_message_when_no_queries_given(monkeypatch):
    monkeypatch.setattr("sys.argv", ["pool_eval_candidates.py"])

    with pytest.raises(SystemExit) as excinfo:
        pool_eval_candidates.main()

    assert "Usage" in str(excinfo.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pool_eval_candidates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pool_eval_candidates'`

- [ ] **Step 3: Write `scripts/pool_eval_candidates.py`**

```python
"""Helper: for a list of candidate queries, pool the top-10 results from all
4 retrieval modes and print each unique candidate's details for manual
relevance judging. Not part of the automated eval pipeline -- a one-time
tool used to help draft eval/eval_queries.json.

Usage:
    python scripts/pool_eval_candidates.py "organic almond milk" "gluten free pasta"
"""

import sys

import pandas as pd

from ecomsearch.config import CATALOG_PATH
from ecomsearch.search import bm25_search, dense_search, hybrid_search

MODES = {
    "dense": lambda query: dense_search(query, 10),
    "bm25": lambda query: bm25_search(query, 10),
    "hybrid": lambda query: hybrid_search(query, 10, use_rerank=False),
    "hybrid-rerank": lambda query: hybrid_search(query, 10, use_rerank=True),
}


def main() -> None:
    queries = sys.argv[1:]
    if not queries:
        raise SystemExit(
            'Usage: python scripts/pool_eval_candidates.py "query one" "query two" ...'
        )

    catalog = pd.read_csv(
        CATALOG_PATH, usecols=["item_id", "name", "brand", "category_path", "description"]
    ).set_index("item_id")

    for query in queries:
        pooled_ids = set()
        for search_fn in MODES.values():
            results = search_fn(query)
            pooled_ids.update(item_id for item_id, _ in results)

        print(f"\n=== Query: {query!r} ({len(pooled_ids)} pooled candidates) ===")
        for item_id in sorted(pooled_ids):
            row = catalog.loc[item_id]
            description = str(row["description"])[:120] if pd.notna(row["description"]) else ""
            print(
                f"  {item_id}\t{row['name']}\t{row['brand']}\t{row['category_path']}\t{description}"
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pool_eval_candidates.py -v`
Expected: PASS

- [ ] **Step 5: Sanity-check it against the real catalog**

Run: `python scripts/pool_eval_candidates.py "organic almond milk"`
Expected: prints a `=== Query: 'organic almond milk' ===` header followed by a list of pooled candidates with item_id/name/brand/category/description columns — should include almond milk products (e.g. Mooala, Malk, Califia Farms brands, based on prior manual testing in this project).

- [ ] **Step 6: Commit**

```bash
git add scripts/pool_eval_candidates.py tests/test_pool_eval_candidates.py
git commit -m "feat: add eval candidate pooling helper script"
git push origin main
```

---

### Task 4: Draft the eval query set (real curation work + required user review)

**Files:**
- Create: `eval/eval_queries.json`

This task has no pre-written final output — it requires running real searches against the real catalog and making relevance judgments based on what comes back. Follow this process:

- [ ] **Step 1: Draft ~35 candidate queries covering the catalog's actual composition**

Per `data/DATA_DICTIONARY.md`, the catalog's top-level category distribution is roughly: More Departments (35%, personal care/household/general merchandise), Grocery (33%), Wine/Beer/Spirits (9%), Frozen (7%), Dairy (5%), Produce & Floral (3%), Bakery (2%), Meat (2%), with smaller categories rounding out the rest. Draft queries that reflect this — grocery/personal-care/drugstore-centric, not generic e-commerce (no electronics, apparel, etc. — confirmed sparse/absent in this catalog during earlier manual testing).

Draft across these 4 buckets (~35 total):
- **~10 exact/near-exact product-type queries** — e.g. "gluten free pasta", "sparkling water", "greek yogurt", "organic almond milk" (a style, not a literal list to copy verbatim — pick your own based on what's actually in the catalog).
- **~8 brand-based queries** — pick real brand names you find while exploring the catalog (e.g. via `python -c "import pandas as pd; print(pd.read_csv('data/ecommerce_catalog_enriched.csv', usecols=['brand']).value_counts().head(30))"` to see common brands), not fictional ones.
- **~8 dietary/attribute-based queries** — e.g. "vegan protein bar", "kosher wine", "sugar free candy", "lactose free milk" (the catalog has `is_organic`/`is_vegan`/`is_gluten_free`/`is_kosher`/`is_lactose_free` boolean columns per the data dictionary — queries should exercise these).
- **~9 vague/conceptual but in-domain queries** — e.g. "healthy breakfast options", "something for a summer BBQ", "quick weeknight dinner ingredients" (realistic shopper intent, not literal keyword lists).

Write this candidate query list to a scratch file first (not `eval/eval_queries.json` yet) — you'll refine it in the next step.

- [ ] **Step 2: Build pooled candidates for each query**

Run (activate venv first: `source venv/Scripts/activate`):
```bash
python scripts/pool_eval_candidates.py "query one" "query two" "query three" ...
```
You can pass all ~35 queries in one invocation (each argument is one query) or batch them — either works. This will take a while since each call reloads models with no caching (Phase 3's known, deliberately-deferred issue) — expect multiple minutes, not seconds, for ~35 queries × 4 modes.

If a query's pooled candidates all look poor/irrelevant (e.g. the catalog just doesn't have good matches), consider swapping that query for a better-fitting one rather than forcing a judgment on bad candidates.

- [ ] **Step 3: Judge relevance and write `eval/eval_queries.json`**

For each query, read through its pooled candidates' name/brand/category_path/description and decide which ones a real shopper searching that exact query would consider a relevant result. Not every pooled candidate is relevant — pooling casts a wide net on purpose (see spec's "Relevance judgment methodology" section); judging is what narrows it back down.

Write the results to `eval/eval_queries.json` in this exact format:
```json
[
  {
    "query": "organic almond milk",
    "relevant_item_ids": [952903, 92137, 98504, 92671, 954690]
  },
  {
    "query": "gluten free pasta",
    "relevant_item_ids": [12345, 67890]
  }
]
```
(The `"organic almond milk"` example above uses real item_ids and products already confirmed relevant during Phase 3's manual CLI testing — Mooala and Malk organic almond milk products — as a worked example of the expected format and judgment standard. Every other query's `relevant_item_ids` must come from your own judging of that query's real pooled candidates, not copied.)

Create the `eval/` directory if it doesn't exist yet (`mkdir -p eval` or equivalent).

- [ ] **Step 4: Self-review before handoff**

- Does every query have at least 1 relevant item? (A query with zero relevant items in its pool is either badly chosen or the catalog genuinely lacks it — reconsider or replace it.)
- Is the ~35-query set reasonably balanced across the 4 buckets from Step 1, not e.g. 30 exact-product-type queries and 1 of everything else?
- Does `eval/eval_queries.json` parse as valid JSON (`python -c "import json; print(len(json.load(open('eval/eval_queries.json'))))"` should print the query count)?

- [ ] **Step 5: STOP — do not proceed to Task 5 yet**

Report back with the full contents of `eval/eval_queries.json` (or a clear summary: total query count, one example per bucket with its judged relevant items) for review. **The controller must present this to the user and get explicit approval before Task 5 runs the real evaluation against it.** This is a hard requirement from the original project brief, not a normal code-review checkpoint — don't skip it or treat a spec-compliance/code-quality pass as satisfying it.

- [ ] **Step 6: Commit (only after user approval from Step 5)**

```bash
git add eval/eval_queries.json
git commit -m "feat: add hand-labeled eval query set"
git push origin main
```

---

### Task 5: Eval runner script (TDD + real run)

**Files:**
- Create: `scripts/run_eval.py`
- Test: `tests/test_run_eval.py`

**Do not start this task until Task 4's `eval/eval_queries.json` has been reviewed and approved by the user.**

- [ ] **Step 1: Write the failing test in `tests/test_run_eval.py`**

```python
import pytest

import run_eval


def test_main_exits_with_clear_message_when_eval_queries_missing(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(run_eval, "EVAL_QUERIES_PATH", missing_path)

    with pytest.raises(SystemExit) as excinfo:
        run_eval.main()

    assert "does_not_exist.json" in str(excinfo.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_eval'`

- [ ] **Step 3: Write `scripts/run_eval.py`**

```python
"""Batch job: run all 4 retrieval modes against the hand-labeled eval set,
compute Recall@10/NDCG@10/MRR per mode, and write a comparison table.

Usage:
    python scripts/run_eval.py
"""

import json

from ecomsearch.config import EVAL_QUERIES_PATH, EVAL_RESULTS_PATH, EVAL_TOP_K
from ecomsearch.eval import mrr, ndcg_at_k, recall_at_k
from ecomsearch.search import bm25_search, dense_search, hybrid_search

MODES = {
    "dense": lambda query, top_k: dense_search(query, top_k),
    "bm25": lambda query, top_k: bm25_search(query, top_k),
    "hybrid": lambda query, top_k: hybrid_search(query, top_k, use_rerank=False),
    "hybrid-rerank": lambda query, top_k: hybrid_search(query, top_k, use_rerank=True),
}


def main() -> None:
    if not EVAL_QUERIES_PATH.exists():
        raise SystemExit(
            f"Eval query set not found at {EVAL_QUERIES_PATH}. "
            "Draft eval/eval_queries.json before running this script."
        )

    with open(EVAL_QUERIES_PATH, encoding="utf-8") as f:
        eval_queries = json.load(f)

    print(f"Loaded {len(eval_queries)} eval queries.")

    scores = {mode: {"recall": [], "ndcg": [], "mrr": []} for mode in MODES}

    for entry in eval_queries:
        query = entry["query"]
        relevant_ids = set(entry["relevant_item_ids"])
        print(f"Evaluating: {query!r}")

        for mode, search_fn in MODES.items():
            results = search_fn(query, EVAL_TOP_K)
            retrieved_ids = [item_id for item_id, _ in results]

            scores[mode]["recall"].append(recall_at_k(retrieved_ids, relevant_ids, EVAL_TOP_K))
            scores[mode]["ndcg"].append(ndcg_at_k(retrieved_ids, relevant_ids, EVAL_TOP_K))
            scores[mode]["mrr"].append(mrr(retrieved_ids, relevant_ids))

    lines = [
        "# Evaluation Results",
        "",
        "## Methodology",
        "",
        f"- {len(eval_queries)} hand-labeled queries, binary relevance.",
        "- Relevant items identified via pooling: the deduplicated union of the",
        "  top-10 results from all 4 modes, judged relevant/not by hand.",
        '  "Relevant" therefore means "relevant among pooled candidates", not',
        "  an exhaustive ground truth over the full 55,516-row catalog.",
        "",
        "## Results",
        "",
        "| Mode | Recall@10 | NDCG@10 | MRR |",
        "|---|---|---|---|",
    ]

    for mode in MODES:
        n = len(eval_queries)
        avg_recall = sum(scores[mode]["recall"]) / n
        avg_ndcg = sum(scores[mode]["ndcg"]) / n
        avg_mrr = sum(scores[mode]["mrr"]) / n
        lines.append(f"| {mode} | {avg_recall:.4f} | {avg_ndcg:.4f} | {avg_mrr:.4f} |")
        print(f"{mode}: Recall@10={avg_recall:.4f} NDCG@10={avg_ndcg:.4f} MRR={avg_mrr:.4f}")

    EVAL_RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote results to {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run_eval.py -v`
Expected: PASS

- [ ] **Step 5: Run the real evaluation**

Run: `python scripts/run_eval.py`
Expected: prints progress per query, then per-mode summary lines, then "Wrote results to ...". This will be slow — roughly 10-20 minutes for ~35 queries × 4 modes, since `search.py` reloads indices/models on every call (Phase 3's known, deliberately-deferred issue — not something to fix in this task). If it's still running after 20 minutes, check `Get-Process python | Select Id,CPU,StartTime` to confirm it's still actively computing (CPU time climbing) rather than stuck, same technique used in earlier phases. Consider launching it detached (`nohup python scripts/run_eval.py > run_eval.log 2>&1 & disown`) if you're concerned about a tool-level timeout on a single long-running foreground command.

Confirm afterward: `docs/eval_results.md` exists and contains a 4-row comparison table with plausible-looking numbers (all metrics between 0 and 1; `hybrid-rerank` should generally score at or near the top given Phase 3's manual testing already showed it producing the cleanest results, though don't force this — report the real numbers whatever they are).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_eval.py tests/test_run_eval.py
git commit -m "feat: add run_eval batch script with real evaluation results"
git push origin main
```

(`docs/eval_results.md` IS committed — unlike `artifacts/`, this is a human-readable results doc, not a regenerable binary artifact, and belongs in git.)

---

### Task 6: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass — Phase 1-3's 39 plus Phase 4a's new tests (eval x9, pool_eval_candidates x1, run_eval x1 = 11) = 50 total.

- [ ] **Step 2: Update `README.md`**

Read the current file first, then:
- Add a Phase 4a note to the Status section. The brief's original Phase 4 is "Evaluation and latency engineering" and isn't fully done yet (only the evaluation half is) — do NOT check off `- [ ] Phase 4 — Evaluation and latency engineering` yet, since Phase 4b (latency) is still outstanding. Instead, update the status intro sentence to mention evaluation results now exist, e.g. append a sentence: "Retrieval quality has been evaluated across all 4 modes — see [Evaluation Results](docs/eval_results.md)." placed after the existing Phases 1-3 summary sentence.
- Add the condensed comparison table (copy the `## Results` table from `docs/eval_results.md`) into a new `## Evaluation` section in README.md, with a link to `docs/eval_results.md` for the full methodology notes. Place this section after "Stack" and before "Setup".

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "docs: add Phase 4a evaluation results to README"
git push origin main
```
