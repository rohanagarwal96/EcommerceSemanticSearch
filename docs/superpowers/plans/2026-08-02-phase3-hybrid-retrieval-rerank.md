# Phase 3: Hybrid Retrieval + Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ecomsearch search "<query>" --mode {dense,bm25,hybrid,hybrid-rerank}` — BM25 keyword search, Reciprocal Rank Fusion with the existing dense search, and cross-encoder reranking, all independently callable so Phase 4's eval harness can compare the four modes directly.

**Architecture:** New `src/ecomsearch/{bm25,fusion,reranker,search}.py` modules plus `scripts/build_bm25_index.py`, with `cli.py` refactored to delegate all retrieval to `search.py`'s mode-dispatching functions instead of talking to `ProductIndex`/`Embedder` directly.

**Tech Stack:** `rank_bm25` (new), `sentence_transformers.CrossEncoder` (already installed), existing FAISS/`ProductIndex`/`Embedder` from Phase 1.

**Spec:** `docs/superpowers/specs/2026-08-02-phase3-hybrid-retrieval-rerank-design.md`

---

### Task 1: Add `rank_bm25` dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dependency**

Append this line to `requirements.txt` (keep the existing 9 lines unchanged):
```
rank_bm25>=0.2.2
```

- [ ] **Step 2: Install it**

Run:
```bash
source venv/Scripts/activate
pip install -r requirements.txt
```
Expected: `rank_bm25` installs successfully.

- [ ] **Step 3: Verify it imports**

Run: `python -c "from rank_bm25 import BM25Okapi; print('ok')"`
Expected: prints `ok`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add rank_bm25 dependency"
git push origin main
```

---

### Task 2: Config additions

**Files:**
- Modify: `src/ecomsearch/config.py`

- [ ] **Step 1: Add these lines to the end of `src/ecomsearch/config.py`**

The file currently ends with `DEFAULT_TOP_K = 10`. Append:
```python

BM25_INDEX_PATH = ARTIFACTS_DIR / "bm25.pkl"

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
CANDIDATE_POOL_SIZE = 100
RERANK_POOL_SIZE = 50
```

- [ ] **Step 2: Verify it imports**

Run:
```bash
python -c "from ecomsearch.config import BM25_INDEX_PATH, RERANKER_MODEL_NAME, RRF_K, CANDIDATE_POOL_SIZE, RERANK_POOL_SIZE; print(BM25_INDEX_PATH); print(RERANKER_MODEL_NAME); print(RRF_K, CANDIDATE_POOL_SIZE, RERANK_POOL_SIZE)"
```
Expected: prints the path to `artifacts/bm25.pkl`, then `cross-encoder/ms-marco-MiniLM-L-6-v2`, then `60 100 50`.

- [ ] **Step 3: Commit**

```bash
git add src/ecomsearch/config.py
git commit -m "feat: add Phase 3 config constants"
git push origin main
```

---

### Task 3: BM25 index (TDD)

**Files:**
- Create: `src/ecomsearch/bm25.py`
- Test: `tests/test_bm25.py`

- [ ] **Step 1: Write the failing tests in `tests/test_bm25.py`**

```python
import numpy as np

from ecomsearch.bm25 import BM25Index


def test_search_returns_best_keyword_match_first():
    texts = [
        "organic almond milk unsweetened",
        "wireless bluetooth headphones noise cancelling",
        "store brand paper towels six rolls",
    ]
    item_ids = np.array([101, 202, 303])

    index = BM25Index()
    index.build(texts, item_ids)

    results = index.search("almond milk", top_k=2)

    assert results[0][0] == 101


def test_save_and_load_round_trip(tmp_path):
    texts = ["organic almond milk", "wireless bluetooth headphones"]
    item_ids = np.array([11, 22])

    index = BM25Index()
    index.build(texts, item_ids)

    path = tmp_path / "bm25.pkl"
    index.save(path)

    loaded = BM25Index.load(path)
    results = loaded.search("almond milk", top_k=1)

    assert results[0][0] == 11
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bm25.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.bm25'`

- [ ] **Step 3: Write `src/ecomsearch/bm25.py`**

```python
"""BM25 keyword search index over product search_text."""
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self):
        self._bm25 = None
        self._item_ids: np.ndarray = np.empty((0,), dtype="int64")

    @property
    def item_ids(self) -> np.ndarray:
        return self._item_ids

    def build(self, texts: list[str], item_ids: np.ndarray) -> None:
        tokenized_corpus = [_tokenize(text) for text in texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._item_ids = item_ids.astype("int64")

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(self._item_ids[i]), float(scores[i])) for i in top_indices]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "item_ids": self._item_ids}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls()
        instance._bm25 = data["bm25"]
        instance._item_ids = data["item_ids"]
        return instance
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bm25.py -v`
Expected: PASS (2 passed, should take well under a second — pure Python/numpy, no model)

- [ ] **Step 5: Commit**

```bash
git add src/ecomsearch/bm25.py tests/test_bm25.py
git commit -m "feat: add BM25 keyword search index with TDD tests"
git push origin main
```

---

### Task 4: Reciprocal Rank Fusion (TDD)

**Files:**
- Create: `src/ecomsearch/fusion.py`
- Test: `tests/test_fusion.py`

- [ ] **Step 1: Write the failing tests in `tests/test_fusion.py`**

```python
from ecomsearch.fusion import reciprocal_rank_fusion


def test_item_ranked_first_in_both_lists_scores_highest():
    dense = [1, 2, 3]
    bm25 = [1, 3, 2]

    fused = reciprocal_rank_fusion([dense, bm25])

    assert fused[0][0] == 1


def test_item_only_in_one_list_still_included():
    dense = [1, 2]
    bm25 = [3, 4]

    fused = reciprocal_rank_fusion([dense, bm25])

    fused_ids = [item_id for item_id, _ in fused]
    assert set(fused_ids) == {1, 2, 3, 4}


def test_fusion_score_matches_rrf_formula():
    dense = [1, 2]
    bm25 = [2, 1]

    fused = reciprocal_rank_fusion([dense, bm25], k=60)
    scores = dict(fused)

    expected_item_1 = 1 / (60 + 1) + 1 / (60 + 2)
    expected_item_2 = 1 / (60 + 2) + 1 / (60 + 1)

    assert scores[1] == expected_item_1
    assert scores[2] == expected_item_2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.fusion'`

- [ ] **Step 3: Write `src/ecomsearch/fusion.py`**

```python
"""Reciprocal Rank Fusion for combining multiple ranked result lists."""
from ecomsearch.config import RRF_K


def reciprocal_rank_fusion(
    ranked_id_lists: list[list[int]], k: int = RRF_K
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fusion.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ecomsearch/fusion.py tests/test_fusion.py
git commit -m "feat: add reciprocal rank fusion with TDD tests"
git push origin main
```

---

### Task 5: Cross-encoder reranker (TDD)

**Files:**
- Create: `src/ecomsearch/reranker.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_reranker.py`

- [ ] **Step 1: Add a `cross_encoder` fixture to `tests/conftest.py`**

Add this fixture to the end of the existing `tests/conftest.py` (after the `clip_embedder` fixture — do not remove or change the existing `embedder`/`clip_embedder` fixtures):

```python


@pytest.fixture(scope="session")
def cross_encoder():
    from ecomsearch.reranker import CrossEncoderReranker

    return CrossEncoderReranker()
```

- [ ] **Step 2: Write the failing test in `tests/test_reranker.py`**

```python
def test_reranker_ranks_relevant_text_first(cross_encoder):
    candidates = [
        (1, "Organic almond milk, unsweetened, dairy-free beverage"),
        (2, "Wireless bluetooth headphones with noise cancelling"),
    ]

    results = cross_encoder.rerank("almond milk", candidates)

    assert results[0][0] == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_reranker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.reranker'`

- [ ] **Step 4: Write `src/ecomsearch/reranker.py`**

```python
"""Cross-encoder reranking for search result candidates."""
from sentence_transformers import CrossEncoder

from ecomsearch.config import RERANKER_MODEL_NAME


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[tuple[int, str]]) -> list[tuple[int, float]]:
        pairs = [(query, text) for _, text in candidates]
        scores = self._model.predict(pairs)
        item_ids = [item_id for item_id, _ in candidates]
        ranked = sorted(zip(item_ids, scores), key=lambda pair: pair[1], reverse=True)
        return [(int(item_id), float(score)) for item_id, score in ranked]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_reranker.py -v`
Expected: PASS (first run downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` from Hugging Face, a small ~22M-param model — should be quick even on this CPU-constrained machine, well under the time bge/CLIP took; subsequent runs use the cached model)

- [ ] **Step 6: Commit**

```bash
git add src/ecomsearch/reranker.py tests/conftest.py tests/test_reranker.py
git commit -m "feat: add cross-encoder reranker with TDD tests"
git push origin main
```

---

### Task 6: Search orchestration (TDD, integration-style)

**Files:**
- Create: `src/ecomsearch/search.py`
- Test: `tests/test_search.py`

- [ ] **Step 1: Write the failing tests in `tests/test_search.py`**

```python
import numpy as np
import pandas as pd
import pytest

from ecomsearch import search
from ecomsearch.bm25 import BM25Index
from ecomsearch.index import ProductIndex


@pytest.fixture
def synthetic_catalog(tmp_path, monkeypatch, embedder):
    texts = [
        "Organic almond milk unsweetened dairy free beverage",
        "Wireless bluetooth headphones noise cancelling",
        "Store brand paper towels six rolls",
    ]
    item_ids = np.array([101, 202, 303])

    vectors = embedder.embed_documents(texts)
    dense_index = ProductIndex(dim=vectors.shape[1])
    dense_index.add(vectors, item_ids)
    index_path = tmp_path / "catalog.faiss"
    item_ids_path = tmp_path / "item_ids.npy"
    dense_index.save(index_path, item_ids_path)

    bm25_index = BM25Index()
    bm25_index.build(texts, item_ids)
    bm25_path = tmp_path / "bm25.pkl"
    bm25_index.save(bm25_path)

    catalog_path = tmp_path / "catalog.csv"
    pd.DataFrame({"item_id": item_ids, "search_text": texts}).to_csv(catalog_path, index=False)

    monkeypatch.setattr(search, "INDEX_PATH", index_path)
    monkeypatch.setattr(search, "ITEM_IDS_PATH", item_ids_path)
    monkeypatch.setattr(search, "BM25_INDEX_PATH", bm25_path)
    monkeypatch.setattr(search, "CATALOG_PATH", catalog_path)

    return item_ids


def test_dense_search_returns_best_semantic_match(synthetic_catalog):
    results = search.dense_search("organic dairy free milk", top_k=1)
    assert results[0][0] == 101


def test_bm25_search_returns_best_keyword_match(synthetic_catalog):
    results = search.bm25_search("almond milk", top_k=1)
    assert results[0][0] == 101


def test_hybrid_search_without_rerank_returns_fused_top_result(synthetic_catalog):
    results = search.hybrid_search("almond milk", top_k=1, use_rerank=False)
    assert results[0][0] == 101


def test_hybrid_search_with_rerank_returns_relevant_top_result(synthetic_catalog):
    results = search.hybrid_search("almond milk", top_k=1, use_rerank=True)
    assert results[0][0] == 101


def test_dense_search_exits_with_clear_message_when_index_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "INDEX_PATH", tmp_path / "catalog.faiss")
    monkeypatch.setattr(search, "ITEM_IDS_PATH", tmp_path / "item_ids.npy")

    with pytest.raises(SystemExit) as excinfo:
        search.dense_search("anything", top_k=1)

    assert "build_index.py" in str(excinfo.value)


def test_bm25_search_exits_with_clear_message_when_index_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "BM25_INDEX_PATH", tmp_path / "bm25.pkl")

    with pytest.raises(SystemExit) as excinfo:
        search.bm25_search("anything", top_k=1)

    assert "build_bm25_index.py" in str(excinfo.value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.search'`

- [ ] **Step 3: Write `src/ecomsearch/search.py`**

```python
"""Retrieval orchestration: dense, keyword (BM25), and hybrid (RRF + rerank) search."""
import pandas as pd

from ecomsearch.bm25 import BM25Index
from ecomsearch.config import (
    BM25_INDEX_PATH,
    CANDIDATE_POOL_SIZE,
    CATALOG_PATH,
    INDEX_PATH,
    ITEM_IDS_PATH,
    RERANK_POOL_SIZE,
)
from ecomsearch.embeddings import Embedder
from ecomsearch.fusion import reciprocal_rank_fusion
from ecomsearch.index import ProductIndex
from ecomsearch.reranker import CrossEncoderReranker


def load_dense_index() -> ProductIndex:
    if not INDEX_PATH.exists() or not ITEM_IDS_PATH.exists():
        raise SystemExit(
            f"No dense index found at {INDEX_PATH}. "
            "Run `python scripts/build_index.py` first to build it."
        )
    return ProductIndex.load(INDEX_PATH, ITEM_IDS_PATH)


def load_bm25_index() -> BM25Index:
    if not BM25_INDEX_PATH.exists():
        raise SystemExit(
            f"No BM25 index found at {BM25_INDEX_PATH}. "
            "Run `python scripts/build_bm25_index.py` first to build it."
        )
    return BM25Index.load(BM25_INDEX_PATH)


def dense_search(query: str, top_k: int) -> list[tuple[int, float]]:
    index = load_dense_index()
    embedder = Embedder()
    query_vector = embedder.embed_query(query)
    return index.search(query_vector, top_k)


def bm25_search(query: str, top_k: int) -> list[tuple[int, float]]:
    index = load_bm25_index()
    return index.search(query, top_k)


def hybrid_search(query: str, top_k: int, use_rerank: bool = True) -> list[tuple[int, float]]:
    dense_results = dense_search(query, CANDIDATE_POOL_SIZE)
    bm25_results = bm25_search(query, CANDIDATE_POOL_SIZE)

    dense_ids = [item_id for item_id, _ in dense_results]
    bm25_ids = [item_id for item_id, _ in bm25_results]
    fused = reciprocal_rank_fusion([dense_ids, bm25_ids])

    if not use_rerank:
        return fused[:top_k]

    candidate_ids = [item_id for item_id, _ in fused[:RERANK_POOL_SIZE]]
    catalog = pd.read_csv(
        CATALOG_PATH, usecols=["item_id", "search_text"]
    ).set_index("item_id")
    candidates = [(item_id, catalog.loc[item_id, "search_text"]) for item_id in candidate_ids]

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, candidates)
    return reranked[:top_k]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_search.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ecomsearch/search.py tests/test_search.py
git commit -m "feat: add hybrid search orchestration with TDD tests"
git push origin main
```

---

### Task 7: Batch BM25 index-build script (TDD + real run)

**Files:**
- Create: `scripts/build_bm25_index.py`
- Test: `tests/test_build_bm25_index.py`

- [ ] **Step 1: Write the failing test in `tests/test_build_bm25_index.py`**

```python
import pytest

import build_bm25_index


def test_main_exits_with_clear_message_when_catalog_missing(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.csv"
    monkeypatch.setattr(build_bm25_index, "CATALOG_PATH", missing_path)

    with pytest.raises(SystemExit) as excinfo:
        build_bm25_index.main()

    assert "does_not_exist.csv" in str(excinfo.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_bm25_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_bm25_index'`

- [ ] **Step 3: Write `scripts/build_bm25_index.py`**

```python
"""Batch job: tokenize the full catalog and build the BM25 index.

Usage:
    python scripts/build_bm25_index.py
"""
import pandas as pd

from ecomsearch.bm25 import BM25Index
from ecomsearch.config import ARTIFACTS_DIR, BM25_INDEX_PATH, CATALOG_PATH


def main() -> None:
    if not CATALOG_PATH.exists():
        raise SystemExit(
            f"Catalog not found at {CATALOG_PATH}. "
            "Make sure data/ecommerce_catalog_enriched.csv is present before building the index."
        )

    print(f"Loading catalog from {CATALOG_PATH}...")
    catalog = pd.read_csv(CATALOG_PATH, usecols=["item_id", "search_text"])

    print(f"Building BM25 index over {len(catalog)} products...")
    index = BM25Index()
    index.build(catalog["search_text"].tolist(), catalog["item_id"].to_numpy())

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    index.save(BM25_INDEX_PATH)
    print(f"Saved BM25 index to {BM25_INDEX_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_bm25_index.py -v`
Expected: PASS

- [ ] **Step 5: Run the real batch job against the full catalog**

Run: `python scripts/build_bm25_index.py`
Expected: prints loading/building/saved messages. Unlike Phase 1/2's neural embedding jobs, this is pure Python/numpy term-frequency counting over 55,516 rows — expected to complete quickly (likely under a couple of minutes even on this CPU-constrained machine), but measure rather than assume: if it runs past ~5 minutes with no output, check `Get-Process python | Select Id,CPU,StartTime` to confirm it's still actively computing (CPU time climbing) rather than stuck. Confirm afterward: `artifacts/bm25.pkl` exists.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_bm25_index.py tests/test_build_bm25_index.py
git commit -m "feat: add build_bm25_index batch script"
git push origin main
```

(`artifacts/bm25.pkl` itself is gitignored by the existing `artifacts/` pattern — only the script and test are committed.)

---

### Task 8: CLI `--mode` flag (TDD + manual verification)

**Files:**
- Modify: `src/ecomsearch/cli.py`
- Modify: `tests/test_cli.py`

This task replaces `cli.py`'s direct use of `ProductIndex`/`Embedder` with calls into the new `search.py` orchestration functions, and replaces the now-obsolete `load_index`-based test (that function is being removed from `cli.py` — its error-handling responsibility has moved into `search.py`'s `load_dense_index`/`load_bm25_index`, already tested in Task 6) with a new test that verifies `--mode` dispatches correctly.

- [ ] **Step 1: Write the failing tests, replacing the full contents of `tests/test_cli.py`**

```python
import pandas as pd
import pytest

from ecomsearch import cli


@pytest.fixture
def fake_catalog(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "item_id": [101],
            "name": ["Organic Almond Milk"],
            "brand": ["Test Brand"],
            "category_path": ["Dairy > Milk Alternatives"],
        }
    ).to_csv(catalog_path, index=False)
    monkeypatch.setattr(cli, "CATALOG_PATH", catalog_path)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid-rerank"])
def test_search_dispatches_to_correct_mode(mode, fake_catalog, monkeypatch, capsys):
    calls = {}

    def fake_dense_search(query, top_k):
        calls["dense"] = (query, top_k)
        return [(101, 0.9)]

    def fake_bm25_search(query, top_k):
        calls["bm25"] = (query, top_k)
        return [(101, 5.0)]

    def fake_hybrid_search(query, top_k, use_rerank):
        calls["hybrid"] = (query, top_k, use_rerank)
        return [(101, 0.5)]

    monkeypatch.setattr(cli, "dense_search", fake_dense_search)
    monkeypatch.setattr(cli, "bm25_search", fake_bm25_search)
    monkeypatch.setattr(cli, "hybrid_search", fake_hybrid_search)

    cli.search("almond milk", top_k=1, mode=mode)

    captured = capsys.readouterr()
    assert "Organic Almond Milk" in captured.out

    if mode == "dense":
        assert calls["dense"] == ("almond milk", 1)
    elif mode == "bm25":
        assert calls["bm25"] == ("almond milk", 1)
    elif mode == "hybrid":
        assert calls["hybrid"] == ("almond milk", 1, False)
    elif mode == "hybrid-rerank":
        assert calls["hybrid"] == ("almond milk", 1, True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `cli.py` doesn't accept a `mode` argument yet, and doesn't import `dense_search`/`bm25_search`/`hybrid_search`, so this will fail with a `TypeError` (wrong number of arguments to `search()`) or `AttributeError` (no such attribute to monkeypatch).

- [ ] **Step 3: Replace the full contents of `src/ecomsearch/cli.py`**

```python
"""CLI entrypoint for semantic product search."""
import argparse

import pandas as pd
from rich.console import Console
from rich.table import Table

from ecomsearch.config import CATALOG_PATH, DEFAULT_TOP_K
from ecomsearch.search import bm25_search, dense_search, hybrid_search


def search(query: str, top_k: int, mode: str) -> None:
    if mode == "dense":
        results = dense_search(query, top_k)
    elif mode == "bm25":
        results = bm25_search(query, top_k)
    elif mode == "hybrid":
        results = hybrid_search(query, top_k, use_rerank=False)
    elif mode == "hybrid-rerank":
        results = hybrid_search(query, top_k, use_rerank=True)
    else:
        raise SystemExit(f"Unknown mode: {mode}")

    catalog = pd.read_csv(
        CATALOG_PATH,
        usecols=["item_id", "name", "brand", "category_path"],
    ).set_index("item_id")

    table = Table(title=f'Top {len(results)} results for "{query}" (mode={mode})')
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Item ID", justify="right")
    table.add_column("Name")
    table.add_column("Brand")
    table.add_column("Category")

    for rank, (item_id, score) in enumerate(results, start=1):
        row = catalog.loc[item_id]
        table.add_row(
            str(rank),
            f"{score:.4f}",
            str(item_id),
            str(row["name"]),
            str(row["brand"]),
            str(row["category_path"]),
        )

    Console().print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic product search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Search the catalog")
    search_parser.add_argument("query", help="Free-text search query")
    search_parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Number of results to return"
    )
    search_parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid", "hybrid-rerank"],
        default="hybrid-rerank",
        help="Retrieval mode",
    )

    args = parser.parse_args()

    if args.command == "search":
        search(args.query, args.top_k, args.mode)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (4 passed — one per mode, via `@pytest.mark.parametrize`)

- [ ] **Step 5: Manually exercise all four modes against the real indices**

Run each of:
```bash
ecomsearch search "organic almond milk" --top-k 5 --mode dense
ecomsearch search "organic almond milk" --top-k 5 --mode bm25
ecomsearch search "organic almond milk" --top-k 5 --mode hybrid
ecomsearch search "organic almond milk" --top-k 5 --mode hybrid-rerank
```
Expected: all four print a rendered table (title showing the mode used) with plausible, relevant almond-milk products. It's fine — expected, even — for the four modes to return somewhat different rankings/scores; that's the point of Phase 3. If `bm25`/`hybrid`/`hybrid-rerank` raise a "no BM25 index found" error, that means Task 7's `build_bm25_index.py` hasn't been run yet in this environment — run it first.

- [ ] **Step 6: Commit**

```bash
git add src/ecomsearch/cli.py tests/test_cli.py
git commit -m "feat: add --mode flag to search CLI for dense/bm25/hybrid/hybrid-rerank"
git push origin main
```

---

### Task 9: Full test suite check and README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass — 38 total. Breakdown: Phase 1 (10) + Phase 2 (12) = 22, plus Phase 3's entirely-new test files (bm25 x2, fusion x3, reranker x1, search x6, build_bm25_index x1 = 13), plus `test_cli.py` growing from 1 test (Phase 1's version) to 4 (this phase's replacement) — a net +3. Total: 22 + 13 + 3 = 38.

- [ ] **Step 2: Update `README.md`**

- Change the Phase 3 checklist line from `- [ ] Phase 3 — Hybrid retrieval + reranking` to `- [x] Phase 3 — Hybrid retrieval + reranking`.
- Update the Status section's intro sentence to include Phase 3, e.g. change "Phases 1-2 complete — ..." to "Phases 1-3 complete — a working semantic search CLI (dense, BM25 keyword, and hybrid+reranked modes) over the full catalog, plus a cross-modal (text-to-image) search demo. Phases 4-8 in progress; this section will be updated as each phase lands." Read the current README first and adapt precisely rather than assuming exact prior wording.
- Add a row to the "Stack" table for "Keyword search" (`rank_bm25`) and "Reranker" (`cross-encoder/ms-marco-MiniLM-L-6-v2`) if not already present — check the current table first, it may already list these from the original scaffolding.
- Update the Phase 1 CLI usage example in "Setup" to mention the new `--mode` flag, e.g. add a line after the existing `ecomsearch search "organic almond milk" --top-k 5` example:

```markdown
Build the BM25 keyword index once (fast — pure term-frequency counting,
no neural network):

```bash
python scripts/build_bm25_index.py
```

Then choose a retrieval mode:

```bash
ecomsearch search "organic almond milk" --top-k 5 --mode hybrid-rerank
```

`--mode` accepts `dense`, `bm25`, `hybrid`, or `hybrid-rerank` (the
default) — useful for comparing retrieval strategies.
```

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "docs: update README for Phase 3 completion"
git push origin main
```
