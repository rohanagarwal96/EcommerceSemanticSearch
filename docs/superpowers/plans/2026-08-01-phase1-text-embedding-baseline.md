# Phase 1: Text Embedding Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working CLI that takes a free-text query and returns the top-k semantically nearest products from the 55,516-row catalog, built on `bge-base-en-v1.5` embeddings and a FAISS index.

**Architecture:** An installable `src/ecomsearch` package (`config`, `embeddings`, `index`, `cli` modules) that later phases (BM25 hybrid fusion, reranking, FastAPI backend) import directly, plus a one-time `scripts/build_index.py` batch job that embeds the full catalog and persists the index to a gitignored `artifacts/` directory.

**Tech Stack:** Python 3.10+, `sentence-transformers` (bge-base-en-v1.5), `faiss-cpu`, `pandas`, `numpy`, `rich`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-08-01-phase1-text-embedding-baseline-design.md`

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/ecomsearch/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "ecomsearch"
version = "0.1.0"
requires-python = ">=3.10"

[project.scripts]
ecomsearch = "ecomsearch.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `requirements.txt`**

```
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
pandas>=2.0.0
numpy>=1.26.0
rich>=13.0.0
pytest>=8.0.0
```

- [ ] **Step 3: Create empty package `src/ecomsearch/__init__.py`**

```python
```

- [ ] **Step 4: Add `artifacts/` to `.gitignore`**

Add this line under the "data / model artifacts too large for git" section of `.gitignore` (the existing `*.faiss`/`*.index` patterns cover the index file by extension; this covers the whole directory, including the `item_ids.npy` mapping file that has no dedicated pattern):

```
artifacts/
```

- [ ] **Step 5: Create venv and install**

Run:
```bash
python -m venv venv
source venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
Expected: all packages install without error; `pip show ecomsearch` shows the package installed in editable mode.

- [ ] **Step 6: Verify the package imports**

Run: `python -c "import ecomsearch; print('ok')"`
Expected: prints `ok`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml requirements.txt src/ecomsearch/__init__.py .gitignore
git commit -m "chore: scaffold ecomsearch package"
git push origin main
```

---

### Task 2: Config module

**Files:**
- Create: `src/ecomsearch/config.py`

- [ ] **Step 1: Write `config.py`**

```python
"""Shared configuration constants for the ecomsearch package."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = REPO_ROOT / "data" / "ecommerce_catalog_enriched.csv"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
INDEX_PATH = ARTIFACTS_DIR / "catalog.faiss"
ITEM_IDS_PATH = ARTIFACTS_DIR / "item_ids.npy"

MODEL_NAME = "BAAI/bge-base-en-v1.5"
MAX_TOKENS = 512
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_TOP_K = 10
```

- [ ] **Step 2: Verify paths resolve correctly**

Run: `python -c "from ecomsearch.config import CATALOG_PATH; print(CATALOG_PATH); print(CATALOG_PATH.exists())"`
Expected: prints the absolute path to `data/ecommerce_catalog_enriched.csv` and `True`

- [ ] **Step 3: Commit**

```bash
git add src/ecomsearch/config.py
git commit -m "feat: add ecomsearch config module"
git push origin main
```

---

### Task 3: Embeddings module (TDD)

**Files:**
- Create: `src/ecomsearch/embeddings.py`
- Create: `tests/conftest.py`
- Test: `tests/test_embeddings.py`

- [ ] **Step 1: Write `tests/conftest.py` with a session-scoped embedder fixture**

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(scope="session")
def embedder():
    from ecomsearch.embeddings import Embedder

    return Embedder()
```

- [ ] **Step 2: Write the failing tests in `tests/test_embeddings.py`**

```python
import numpy as np

from ecomsearch.config import QUERY_PREFIX


def test_embed_documents_returns_unit_norm_vectors(embedder):
    texts = ["Organic whole milk", "Store brand paper towels, 6 rolls"]
    vectors = embedder.embed_documents(texts)
    norms = np.linalg.norm(vectors, axis=1)
    assert vectors.shape[0] == 2
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_embed_documents_truncates_long_text_without_error(embedder):
    long_text = "ingredient " * 2000  # far beyond 512 tokens
    vectors = embedder.embed_documents([long_text])
    assert vectors.shape[0] == 1
    assert np.isfinite(vectors).all()


def test_embed_query_differs_from_raw_document_embedding(embedder):
    query = "warm winter jacket"
    query_vector = embedder.embed_query(query)
    raw_vector = embedder.embed_documents([query])[0]
    assert not np.allclose(query_vector, raw_vector)


def test_embed_query_applies_configured_prefix(embedder, monkeypatch):
    captured = {}
    original_embed = embedder._embed

    def spy(texts):
        captured["texts"] = texts
        return original_embed(texts)

    monkeypatch.setattr(embedder, "_embed", spy)
    embedder.embed_query("wireless headphones")
    assert captured["texts"] == [QUERY_PREFIX + "wireless headphones"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.embeddings'`

- [ ] **Step 4: Write `src/ecomsearch/embeddings.py`**

```python
"""Text embedding utilities wrapping BAAI/bge-base-en-v1.5."""
from sentence_transformers import SentenceTransformer
import numpy as np

from ecomsearch.config import MAX_TOKENS, MODEL_NAME, QUERY_PREFIX


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self._model = SentenceTransformer(model_name)
        self._model.max_seq_length = MAX_TOKENS

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([QUERY_PREFIX + text])[0]

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_embeddings.py -v`
Expected: PASS (first run downloads `bge-base-en-v1.5` from Hugging Face, ~440MB — this can take a few minutes depending on bandwidth; subsequent runs use the cached model and are fast)

- [ ] **Step 6: Commit**

```bash
git add src/ecomsearch/embeddings.py tests/conftest.py tests/test_embeddings.py
git commit -m "feat: add bge-base-en-v1.5 embedding wrapper with TDD tests"
git push origin main
```

---

### Task 4: FAISS index module (TDD)

**Files:**
- Create: `src/ecomsearch/index.py`
- Test: `tests/test_index.py`

- [ ] **Step 1: Write the failing tests in `tests/test_index.py`**

```python
import numpy as np
import pytest

from ecomsearch.index import ProductIndex


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_search_returns_nearest_neighbor_first():
    vectors = _normalize(np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],
    ], dtype="float32"))
    item_ids = np.array([101, 202, 303])

    index = ProductIndex(dim=2)
    index.add(vectors, item_ids)

    query = _normalize(np.array([[1.0, 0.0]], dtype="float32"))[0]
    results = index.search(query, top_k=2)

    assert results[0][0] == 101
    assert results[1][0] == 303


def test_add_rejects_mismatched_lengths():
    index = ProductIndex(dim=2)
    with pytest.raises(ValueError):
        index.add(np.zeros((2, 2), dtype="float32"), np.array([1]))


def test_save_and_load_round_trip(tmp_path):
    vectors = _normalize(np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype="float32"))
    item_ids = np.array([11, 22])

    index = ProductIndex(dim=2)
    index.add(vectors, item_ids)

    index_path = tmp_path / "catalog.faiss"
    item_ids_path = tmp_path / "item_ids.npy"
    index.save(index_path, item_ids_path)

    loaded = ProductIndex.load(index_path, item_ids_path)
    results = loaded.search(vectors[0], top_k=1)

    assert results[0][0] == 11
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.index'`

- [ ] **Step 3: Write `src/ecomsearch/index.py`**

```python
"""FAISS-backed nearest neighbor index over product embeddings."""
from pathlib import Path

import faiss
import numpy as np


class ProductIndex:
    def __init__(self, dim: int):
        self._index = faiss.IndexFlatIP(dim)
        self._item_ids: np.ndarray = np.empty((0,), dtype="int64")

    @property
    def item_ids(self) -> np.ndarray:
        return self._item_ids

    def add(self, vectors: np.ndarray, item_ids: np.ndarray) -> None:
        if vectors.shape[0] != item_ids.shape[0]:
            raise ValueError("vectors and item_ids must have the same length")
        self._index.add(vectors.astype("float32"))
        self._item_ids = np.concatenate([self._item_ids, item_ids.astype("int64")])

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        query = np.expand_dims(query_vector.astype("float32"), axis=0)
        scores, positions = self._index.search(query, top_k)
        results = []
        for position, score in zip(positions[0], scores[0]):
            if position == -1:
                continue
            results.append((int(self._item_ids[position]), float(score)))
        return results

    def save(self, index_path: Path, item_ids_path: Path) -> None:
        faiss.write_index(self._index, str(index_path))
        np.save(item_ids_path, self._item_ids)

    @classmethod
    def load(cls, index_path: Path, item_ids_path: Path) -> "ProductIndex":
        index = faiss.read_index(str(index_path))
        instance = cls(dim=index.d)
        instance._index = index
        instance._item_ids = np.load(item_ids_path)
        return instance
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_index.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ecomsearch/index.py tests/test_index.py
git commit -m "feat: add FAISS ProductIndex with TDD tests"
git push origin main
```

---

### Task 5: End-to-end integration test

**Files:**
- Test: `tests/test_integration.py`

- [ ] **Step 1: Write the integration test**

```python
import numpy as np

from ecomsearch.index import ProductIndex


def test_end_to_end_search_ranks_obvious_match_first(embedder):
    products = [
        (1, "Organic whole milk, 1 gallon, dairy"),
        (2, "Wireless bluetooth headphones, noise cancelling"),
        (3, "Store brand paper towels, 6 rolls"),
        (4, "Organic almond milk, unsweetened, 1 quart"),
    ]
    item_ids = np.array([p[0] for p in products])
    texts = [p[1] for p in products]

    vectors = embedder.embed_documents(texts)
    index = ProductIndex(dim=vectors.shape[1])
    index.add(vectors, item_ids)

    query_vector = embedder.embed_query("organic dairy milk")
    results = index.search(query_vector, top_k=2)

    top_ids = [item_id for item_id, _ in results]
    assert top_ids[0] in (1, 4)
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_integration.py -v`
Expected: PASS. If it fails, inspect the printed `top_ids` — a mismatch here means the embedding+index code paths compose incorrectly (not that the model chose a "wrong" answer), since milk vs. headphones/paper towels is not a close call.

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end embed+index integration test"
git push origin main
```

---

### Task 6: Batch index-build script

**Files:**
- Create: `scripts/build_index.py`
- Test: `tests/test_build_index.py`

- [ ] **Step 1: Write the failing test in `tests/test_build_index.py`**

```python
import pytest

import build_index


def test_main_exits_with_clear_message_when_catalog_missing(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.csv"
    monkeypatch.setattr(build_index, "CATALOG_PATH", missing_path)

    with pytest.raises(SystemExit) as excinfo:
        build_index.main()

    assert "does_not_exist.csv" in str(excinfo.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_index'`

- [ ] **Step 3: Write `scripts/build_index.py`**

```python
"""Batch job: embed the full catalog and build the FAISS index.

Usage:
    python scripts/build_index.py
"""
import pandas as pd

from ecomsearch.config import ARTIFACTS_DIR, CATALOG_PATH, INDEX_PATH, ITEM_IDS_PATH
from ecomsearch.embeddings import Embedder
from ecomsearch.index import ProductIndex


def main() -> None:
    if not CATALOG_PATH.exists():
        raise SystemExit(
            f"Catalog not found at {CATALOG_PATH}. "
            "Make sure data/ecommerce_catalog_enriched.csv is present before building the index."
        )

    print(f"Loading catalog from {CATALOG_PATH}...")
    catalog = pd.read_csv(CATALOG_PATH, usecols=["item_id", "search_text"])

    print(f"Embedding {len(catalog)} products with bge-base-en-v1.5...")
    embedder = Embedder()
    vectors = embedder.embed_documents(catalog["search_text"].tolist())

    print("Building FAISS index...")
    index = ProductIndex(dim=vectors.shape[1])
    index.add(vectors, catalog["item_id"].to_numpy())

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    index.save(INDEX_PATH, ITEM_IDS_PATH)
    print(f"Saved index to {INDEX_PATH} and id mapping to {ITEM_IDS_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_index.py -v`
Expected: PASS

- [ ] **Step 5: Run the real batch job against the full catalog**

Run: `python scripts/build_index.py`
Expected: prints loading/embedding/building/saved messages; embedding 55,516 rows on CPU takes roughly 10-25 minutes depending on hardware. Confirm `artifacts/catalog.faiss` and `artifacts/item_ids.npy` now exist.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_index.py tests/test_build_index.py
git commit -m "feat: add build_index batch script"
git push origin main
```

(`artifacts/` itself is gitignored — only the script and test are committed.)

---

### Task 7: Search CLI

**Files:**
- Create: `src/ecomsearch/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test in `tests/test_cli.py`**

```python
import pytest

from ecomsearch import cli


def test_load_index_exits_with_clear_message_when_index_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "INDEX_PATH", tmp_path / "catalog.faiss")
    monkeypatch.setattr(cli, "ITEM_IDS_PATH", tmp_path / "item_ids.npy")

    with pytest.raises(SystemExit) as excinfo:
        cli.load_index()

    assert "build_index.py" in str(excinfo.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.cli'`

- [ ] **Step 3: Write `src/ecomsearch/cli.py`**

```python
"""CLI entrypoint for semantic product search."""
import argparse

import pandas as pd
from rich.console import Console
from rich.table import Table

from ecomsearch.config import CATALOG_PATH, DEFAULT_TOP_K, INDEX_PATH, ITEM_IDS_PATH
from ecomsearch.embeddings import Embedder
from ecomsearch.index import ProductIndex


def load_index() -> ProductIndex:
    if not INDEX_PATH.exists() or not ITEM_IDS_PATH.exists():
        raise SystemExit(
            f"No index found at {INDEX_PATH}. "
            "Run `python scripts/build_index.py` first to build it."
        )
    return ProductIndex.load(INDEX_PATH, ITEM_IDS_PATH)


def search(query: str, top_k: int) -> None:
    index = load_index()
    embedder = Embedder()
    query_vector = embedder.embed_query(query)
    results = index.search(query_vector, top_k)

    catalog = pd.read_csv(
        CATALOG_PATH,
        usecols=["item_id", "name", "brand", "category_path"],
    ).set_index("item_id")

    table = Table(title=f'Top {len(results)} results for "{query}"')
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

    args = parser.parse_args()

    if args.command == "search":
        search(args.query, args.top_k)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Manually exercise the CLI against the real index**

Run: `ecomsearch search "organic almond milk" --top-k 5`
Expected: a rendered table with 5 rows, ranked by score, showing plausible organic dairy/milk-alternative products at or near the top.

- [ ] **Step 6: Commit**

```bash
git add src/ecomsearch/cli.py tests/test_cli.py
git commit -m "feat: add search CLI"
git push origin main
```

---

### Task 8: Full test suite check and README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass (embeddings, index, integration, build_index, cli).

- [ ] **Step 2: Update `README.md`**

Replace the "Status" checklist item for Phase 1 (`- [ ] Phase 1 — Text embedding baseline...`) with `- [x] Phase 1 — Text embedding baseline (FAISS + bge-base-en-v1.5)`, and replace the "Setup" section placeholder with:

```markdown
## Setup

```bash
python -m venv venv
source venv/Scripts/activate   # on Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Build the search index once (embeds all 55,516 products; takes roughly
10-25 minutes on CPU):

```bash
python scripts/build_index.py
```

Then search:

```bash
ecomsearch search "organic almond milk" --top-k 5
```
```

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "docs: update README for Phase 1 completion"
git push origin main
```
