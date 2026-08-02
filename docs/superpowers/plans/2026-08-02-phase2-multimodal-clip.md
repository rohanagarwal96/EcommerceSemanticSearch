# Phase 2: Multimodal (CLIP) Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working `ecomsearch-images search "<text query>"` CLI that returns the top-k most visually/semantically relevant product images from a 5,000-item stratified sample of a public Kaggle fashion dataset, using CLIP embeddings and the existing FAISS `ProductIndex`.

**Architecture:** A new `src/ecomsearch/multimodal/` subpackage (config, `ClipEmbedder`, `stratified_sample`, a separate `cli.py`) plus two one-time `scripts/` jobs (download the Kaggle dataset, build the sampled index), reusing Phase 1's `ecomsearch.index.ProductIndex` unchanged.

**Tech Stack:** Python 3.10+, `transformers` (CLIP), `Pillow`, `kaggle`, `faiss-cpu`, `pandas`, `numpy`, `rich`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-08-02-phase2-multimodal-clip-design.md`

---

### Task 1: Project scaffolding

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Create: `src/ecomsearch/multimodal/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add the new console-script entry to `pyproject.toml`**

Change the `[project.scripts]` section from:
```toml
[project.scripts]
ecomsearch = "ecomsearch.cli:main"
```
to:
```toml
[project.scripts]
ecomsearch = "ecomsearch.cli:main"
ecomsearch-images = "ecomsearch.multimodal.cli:main"
```

- [ ] **Step 2: Add new dependencies to `requirements.txt`**

Add these three lines (keep the existing six lines unchanged):
```
transformers>=4.40.0
Pillow>=10.0.0
kaggle>=1.6.0
```

- [ ] **Step 3: Create empty package `src/ecomsearch/multimodal/__init__.py`**

```python
```

- [ ] **Step 4: Add new gitignore entries**

Add these two lines under the "data / model artifacts too large for git" section of `.gitignore` (alongside the existing `artifacts/` line):
```
data/multimodal/
demo_results/
```

- [ ] **Step 5: Install new dependencies and refresh the editable install**

Run:
```bash
source venv/Scripts/activate
pip install -r requirements.txt
pip install -e .
```
Expected: `transformers`, `Pillow`, and `kaggle` install successfully (transformers may already be present as a transitive dependency of `sentence-transformers` — pip will just confirm the version satisfies the new explicit constraint). `pip install -e .` re-registers console scripts.

- [ ] **Step 6: Verify the console-script wrapper was generated**

Run: `ls venv/Scripts/ecomsearch-images*`
Expected: lists a generated wrapper file (e.g. `ecomsearch-images.exe` and/or `ecomsearch-images-script.py`). This only confirms `pip install -e .` registered the entry point — do NOT try to run `ecomsearch-images` yet, since `src/ecomsearch/multimodal/cli.py` doesn't exist until Task 8 and invoking it now would fail with `ModuleNotFoundError`. If `ls` finds nothing, stop and report BLOCKED.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml requirements.txt src/ecomsearch/multimodal/__init__.py .gitignore
git commit -m "chore: scaffold multimodal subpackage"
git push origin main
```

---

### Task 2: Multimodal config module

**Files:**
- Create: `src/ecomsearch/multimodal/config.py`

- [ ] **Step 1: Write `config.py`**

```python
"""Shared configuration constants for the multimodal (CLIP) module."""
from pathlib import Path

from ecomsearch.config import REPO_ROOT

DATASET_DIR = REPO_ROOT / "data" / "multimodal"
DATASET_CSV_PATH = DATASET_DIR / "data.csv"
DATASET_IMAGES_DIR = DATASET_DIR / "data"

KAGGLE_DATASET_REF = "nirmalsankalana/mini-product-image-and-text-dataset"
KAGGLE_CREDENTIALS_PATH = Path.home() / ".kaggle" / "kaggle.json"

ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "multimodal"
INDEX_PATH = ARTIFACTS_DIR / "catalog.faiss"
ITEM_IDS_PATH = ARTIFACTS_DIR / "item_ids.npy"
SUBSET_METADATA_PATH = ARTIFACTS_DIR / "subset_metadata.csv"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SUBSET_SIZE = 5000
DEFAULT_TOP_K = 10

DEMO_RESULTS_DIR = REPO_ROOT / "demo_results"
```

- [ ] **Step 2: Verify it imports and paths resolve as expected**

Run:
```bash
python -c "from ecomsearch.multimodal.config import DATASET_CSV_PATH, ARTIFACTS_DIR, CLIP_MODEL_NAME; print(DATASET_CSV_PATH); print(ARTIFACTS_DIR); print(CLIP_MODEL_NAME)"
```
Expected: prints three lines — the (not-yet-existing) path to `data/multimodal/data.csv`, the path to `artifacts/multimodal`, and `openai/clip-vit-base-patch32`. No error.

- [ ] **Step 3: Commit**

```bash
git add src/ecomsearch/multimodal/config.py
git commit -m "feat: add multimodal config module"
git push origin main
```

---

### Task 3: CLIP embedder module (TDD)

**Files:**
- Create: `src/ecomsearch/multimodal/clip_embedder.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_clip_embedder.py`

- [ ] **Step 1: Add a session-scoped `clip_embedder` fixture to `tests/conftest.py`**

Replace the full contents of `tests/conftest.py` with:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(scope="session")
def embedder():
    from ecomsearch.embeddings import Embedder

    return Embedder()


@pytest.fixture(scope="session")
def clip_embedder():
    from ecomsearch.multimodal.clip_embedder import ClipEmbedder

    return ClipEmbedder()
```

- [ ] **Step 2: Write the failing tests in `tests/test_clip_embedder.py`**

```python
import numpy as np
from PIL import Image


def _make_solid_image(path, color):
    image = Image.new("RGB", (64, 64), color=color)
    image.save(path)


def test_embed_images_returns_unit_norm_vectors(clip_embedder, tmp_path):
    red_path = tmp_path / "red.jpg"
    blue_path = tmp_path / "blue.jpg"
    _make_solid_image(red_path, (220, 20, 20))
    _make_solid_image(blue_path, (20, 20, 220))

    vectors = clip_embedder.embed_images([red_path, blue_path])
    norms = np.linalg.norm(vectors, axis=1)
    assert vectors.shape[0] == 2
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_embed_text_returns_unit_norm_vectors(clip_embedder):
    vectors = clip_embedder.embed_text(["a red square", "a blue square"])
    norms = np.linalg.norm(vectors, axis=1)
    assert vectors.shape[0] == 2
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_text_embedding_is_closer_to_matching_image(clip_embedder, tmp_path):
    red_path = tmp_path / "red.jpg"
    blue_path = tmp_path / "blue.jpg"
    _make_solid_image(red_path, (220, 20, 20))
    _make_solid_image(blue_path, (20, 20, 220))

    image_vectors = clip_embedder.embed_images([red_path, blue_path])
    text_vectors = clip_embedder.embed_text(["a solid red square", "a solid blue square"])

    red_image, blue_image = image_vectors
    red_text, blue_text = text_vectors

    assert np.dot(red_text, red_image) > np.dot(red_text, blue_image)
    assert np.dot(blue_text, blue_image) > np.dot(blue_text, red_image)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_clip_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.multimodal.clip_embedder'`

- [ ] **Step 4: Write `src/ecomsearch/multimodal/clip_embedder.py`**

```python
"""Image/text embedding utilities wrapping OpenAI CLIP via transformers."""
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ecomsearch.multimodal.config import CLIP_MODEL_NAME


class ClipEmbedder:
    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        self._model = CLIPModel.from_pretrained(model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model.eval()

    def embed_images(self, image_paths: list) -> np.ndarray:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = self._processor(images=images, return_tensors="pt")
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
        return self._normalize(features)

    def embed_text(self, texts: list) -> np.ndarray:
        inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
        return self._normalize(features)

    @staticmethod
    def _normalize(features) -> np.ndarray:
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy().astype("float32")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_clip_embedder.py -v`
Expected: PASS (first run downloads `openai/clip-vit-base-patch32` from Hugging Face, ~600MB — can take a few minutes; subsequent runs use the cached model)

- [ ] **Step 6: Commit**

```bash
git add src/ecomsearch/multimodal/clip_embedder.py tests/conftest.py tests/test_clip_embedder.py
git commit -m "feat: add CLIP embedder with TDD tests"
git push origin main
```

---

### Task 4: Category-stratified sampling (TDD)

**Files:**
- Create: `src/ecomsearch/multimodal/sampling.py`
- Test: `tests/test_sampling.py`

- [ ] **Step 1: Write the failing tests in `tests/test_sampling.py`**

```python
import pandas as pd

from ecomsearch.multimodal.sampling import stratified_sample


def test_stratified_sample_preserves_category_proportions():
    df = pd.DataFrame({
        "category": ["a"] * 80 + ["b"] * 20,
        "value": range(100),
    })

    sampled = stratified_sample(df, "category", 10)

    counts = sampled["category"].value_counts()
    assert len(sampled) == 10
    assert counts.get("a", 0) == 8
    assert counts.get("b", 0) == 2


def test_stratified_sample_returns_full_df_when_n_exceeds_length():
    df = pd.DataFrame({"category": ["a", "b", "c"], "value": [1, 2, 3]})

    sampled = stratified_sample(df, "category", 10)

    assert len(sampled) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sampling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.multimodal.sampling'`

- [ ] **Step 3: Write `src/ecomsearch/multimodal/sampling.py`**

```python
"""Category-stratified sampling utilities for the multimodal dataset."""
import pandas as pd


def stratified_sample(df: pd.DataFrame, category_col: str, n: int) -> pd.DataFrame:
    if n >= len(df):
        return df.reset_index(drop=True)

    fraction = n / len(df)
    sampled = df.groupby(category_col, group_keys=False).apply(
        lambda group: group.sample(frac=fraction, random_state=42)
    )
    return sampled.reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sampling.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ecomsearch/multimodal/sampling.py tests/test_sampling.py
git commit -m "feat: add category-stratified sampling with TDD tests"
git push origin main
```

---

### Task 5: End-to-end cross-modal integration test

**Files:**
- Test: `tests/test_multimodal_integration.py`

- [ ] **Step 1: Write the integration test**

```python
import numpy as np
from PIL import Image

from ecomsearch.index import ProductIndex


def _make_solid_image(path, color):
    image = Image.new("RGB", (64, 64), color=color)
    image.save(path)


def test_end_to_end_cross_modal_search_ranks_matching_image_first(clip_embedder, tmp_path):
    red_path = tmp_path / "1.jpg"
    blue_path = tmp_path / "2.jpg"
    green_path = tmp_path / "3.jpg"
    _make_solid_image(red_path, (220, 20, 20))
    _make_solid_image(blue_path, (20, 20, 220))
    _make_solid_image(green_path, (20, 180, 20))

    item_ids = np.array([1, 2, 3])
    image_paths = [red_path, blue_path, green_path]

    vectors = clip_embedder.embed_images(image_paths)
    index = ProductIndex(dim=vectors.shape[1])
    index.add(vectors, item_ids)

    query_vector = clip_embedder.embed_text(["a solid red square"])[0]
    results = index.search(query_vector, top_k=1)

    assert results[0][0] == 1
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_multimodal_integration.py -v`
Expected: PASS. If it fails, inspect which item ranked first — red vs. blue vs. green is not a close call for CLIP, so a mismatch means the `ClipEmbedder`+`ProductIndex` composition is wired incorrectly, not that the model made a borderline call.

- [ ] **Step 3: Commit**

```bash
git add tests/test_multimodal_integration.py
git commit -m "test: add end-to-end cross-modal embed+index integration test"
git push origin main
```

---

### Task 6: Kaggle dataset download script (TDD + real run)

**Files:**
- Create: `scripts/download_multimodal_dataset.py`
- Test: `tests/test_download_multimodal_dataset.py`

- [ ] **Step 1: Write the failing tests in `tests/test_download_multimodal_dataset.py`**

```python
import pytest

import download_multimodal_dataset


def test_main_skips_when_dataset_already_present(tmp_path, monkeypatch, capsys):
    existing_csv = tmp_path / "data.csv"
    existing_csv.write_text("image,description,display name,category\n")
    monkeypatch.setattr(download_multimodal_dataset, "DATASET_CSV_PATH", existing_csv)

    download_multimodal_dataset.main()

    captured = capsys.readouterr()
    assert "already present" in captured.out


def test_main_exits_with_clear_message_when_credentials_missing(tmp_path, monkeypatch):
    missing_csv = tmp_path / "does_not_exist.csv"
    missing_credentials = tmp_path / "does_not_exist_kaggle.json"
    monkeypatch.setattr(download_multimodal_dataset, "DATASET_CSV_PATH", missing_csv)
    monkeypatch.setattr(download_multimodal_dataset, "KAGGLE_CREDENTIALS_PATH", missing_credentials)

    with pytest.raises(SystemExit) as excinfo:
        download_multimodal_dataset.main()

    assert "does_not_exist_kaggle.json" in str(excinfo.value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_download_multimodal_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'download_multimodal_dataset'`

- [ ] **Step 3: Write `scripts/download_multimodal_dataset.py`**

```python
"""One-time job: download the multimodal product image/text dataset from Kaggle.

Usage:
    python scripts/download_multimodal_dataset.py
"""
from ecomsearch.multimodal.config import (
    DATASET_CSV_PATH,
    DATASET_DIR,
    KAGGLE_CREDENTIALS_PATH,
    KAGGLE_DATASET_REF,
)


def main() -> None:
    if DATASET_CSV_PATH.exists():
        print(f"Dataset already present at {DATASET_CSV_PATH}, skipping download.")
        return

    if not KAGGLE_CREDENTIALS_PATH.exists():
        raise SystemExit(
            f"Kaggle credentials not found at {KAGGLE_CREDENTIALS_PATH}. "
            "Set up your Kaggle API token (https://www.kaggle.com/docs/api) "
            "before running this script."
        )

    from kaggle.api.kaggle_api_extended import KaggleApi

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET_REF} to {DATASET_DIR}...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET_REF, path=str(DATASET_DIR), unzip=True)
    print(f"Downloaded and extracted to {DATASET_DIR}")


if __name__ == "__main__":
    main()
```

Note: the `from kaggle.api.kaggle_api_extended import KaggleApi` import is deliberately placed *inside* `main()`, after the credentials check, not at module level — the `kaggle` package's own top-level import can raise an unfriendly error if credentials are missing, and we want our own clear `SystemExit` message to fire first.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_download_multimodal_dataset.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the real download**

Run: `python scripts/download_multimodal_dataset.py`
Expected: downloads and extracts to `data/multimodal/`. Should take a few minutes (~362MB). Confirm afterward: `data/multimodal/data.csv` exists, and `data/multimodal/data/` contains thousands of `.jpg` files.

- [ ] **Step 6: Commit**

```bash
git add scripts/download_multimodal_dataset.py tests/test_download_multimodal_dataset.py
git commit -m "feat: add Kaggle multimodal dataset download script"
git push origin main
```

(`data/multimodal/` itself is gitignored — only the script and test are committed.)

---

### Task 7: Batch multimodal index-build script (TDD + real run)

**Files:**
- Create: `scripts/build_multimodal_index.py`
- Test: `tests/test_build_multimodal_index.py`

- [ ] **Step 1: Write the failing test in `tests/test_build_multimodal_index.py`**

```python
import pytest

import build_multimodal_index


def test_main_exits_with_clear_message_when_dataset_missing(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.csv"
    monkeypatch.setattr(build_multimodal_index, "DATASET_CSV_PATH", missing_path)

    with pytest.raises(SystemExit) as excinfo:
        build_multimodal_index.main()

    assert "does_not_exist.csv" in str(excinfo.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_multimodal_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_multimodal_index'`

- [ ] **Step 3: Write `scripts/build_multimodal_index.py`**

```python
"""Batch job: stratified-sample the multimodal dataset, embed images with CLIP,
build the FAISS index.

Usage:
    python scripts/build_multimodal_index.py
"""
import pandas as pd

from ecomsearch.index import ProductIndex
from ecomsearch.multimodal.clip_embedder import ClipEmbedder
from ecomsearch.multimodal.config import (
    ARTIFACTS_DIR,
    DATASET_CSV_PATH,
    DATASET_IMAGES_DIR,
    INDEX_PATH,
    ITEM_IDS_PATH,
    SUBSET_METADATA_PATH,
    SUBSET_SIZE,
)
from ecomsearch.multimodal.sampling import stratified_sample


def main() -> None:
    if not DATASET_CSV_PATH.exists():
        raise SystemExit(
            f"Dataset not found at {DATASET_CSV_PATH}. "
            "Run `python scripts/download_multimodal_dataset.py` first."
        )

    print(f"Loading dataset from {DATASET_CSV_PATH}...")
    df = pd.read_csv(DATASET_CSV_PATH)

    print(f"Sampling {SUBSET_SIZE} rows stratified by category...")
    subset = stratified_sample(df, "category", SUBSET_SIZE)
    print(f"Sampled {len(subset)} rows across {subset['category'].nunique()} categories.")

    item_ids = subset["image"].apply(lambda name: int(name.split(".")[0])).to_numpy()
    image_paths = [DATASET_IMAGES_DIR / name for name in subset["image"]]

    print(f"Embedding {len(subset)} images with CLIP...")
    embedder = ClipEmbedder()
    vectors = embedder.embed_images(image_paths)

    print("Building FAISS index...")
    index = ProductIndex(dim=vectors.shape[1])
    index.add(vectors, item_ids)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    index.save(INDEX_PATH, ITEM_IDS_PATH)
    subset.assign(item_id=item_ids).to_csv(SUBSET_METADATA_PATH, index=False)
    print(
        f"Saved index to {INDEX_PATH}, id mapping to {ITEM_IDS_PATH}, "
        f"metadata to {SUBSET_METADATA_PATH}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_multimodal_index.py -v`
Expected: PASS

- [ ] **Step 5: Run the real batch job**

Run: `python scripts/build_multimodal_index.py`
Expected: prints sampling/embedding/building/saved messages. This embeds 5,000 small thumbnail images with CLIP — expected to be substantially faster than Phase 1's text embedding run, but measure actual progress rather than assuming: if it runs past ~15 minutes without finishing, check real throughput with `py-spy dump --pid <PID> --locals` (same technique used in Phase 1) rather than waiting blind. If a naive foreground/backgrounded run risks being killed by a ~10-minute tool-level timeout, launch it detached instead: `nohup python scripts/build_multimodal_index.py > build_multimodal_run.log 2>&1 & disown`. Confirm afterward: `artifacts/multimodal/catalog.faiss`, `artifacts/multimodal/item_ids.npy`, and `artifacts/multimodal/subset_metadata.csv` all exist, and `subset_metadata.csv` has roughly 5,000 rows (`wc -l artifacts/multimodal/subset_metadata.csv`).

- [ ] **Step 6: Commit**

```bash
git add scripts/build_multimodal_index.py tests/test_build_multimodal_index.py
git commit -m "feat: add build_multimodal_index batch script"
git push origin main
```

(`artifacts/multimodal/` itself is gitignored — only the script and test are committed.)

---

### Task 8: Cross-modal search CLI (TDD + manual verification)

**Files:**
- Create: `src/ecomsearch/multimodal/cli.py`
- Test: `tests/test_multimodal_cli.py`

- [ ] **Step 1: Write the failing tests in `tests/test_multimodal_cli.py`**

```python
import pytest

from ecomsearch.multimodal import cli


def test_load_index_exits_with_clear_message_when_index_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "INDEX_PATH", tmp_path / "catalog.faiss")
    monkeypatch.setattr(cli, "ITEM_IDS_PATH", tmp_path / "item_ids.npy")

    with pytest.raises(SystemExit) as excinfo:
        cli.load_index()

    assert "build_multimodal_index.py" in str(excinfo.value)


def test_slugify_converts_query_to_safe_directory_name():
    assert cli._slugify("Something warm for rainy weather!") == "something-warm-for-rainy-weather"


def test_slugify_falls_back_to_default_when_nothing_remains():
    assert cli._slugify("???") == "query"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multimodal_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.multimodal.cli'`

- [ ] **Step 3: Write `src/ecomsearch/multimodal/cli.py`**

```python
"""CLI entrypoint for cross-modal (text-to-image) product search."""
import argparse
import re
import shutil

import pandas as pd
from rich.console import Console
from rich.table import Table

from ecomsearch.index import ProductIndex
from ecomsearch.multimodal.clip_embedder import ClipEmbedder
from ecomsearch.multimodal.config import (
    DATASET_IMAGES_DIR,
    DEFAULT_TOP_K,
    DEMO_RESULTS_DIR,
    INDEX_PATH,
    ITEM_IDS_PATH,
    SUBSET_METADATA_PATH,
)


def load_index() -> ProductIndex:
    if not INDEX_PATH.exists() or not ITEM_IDS_PATH.exists():
        raise SystemExit(
            f"No index found at {INDEX_PATH}. "
            "Run `python scripts/build_multimodal_index.py` first to build it."
        )
    return ProductIndex.load(INDEX_PATH, ITEM_IDS_PATH)


def _slugify(query: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    return slug or "query"


def search(query: str, top_k: int) -> None:
    index = load_index()
    embedder = ClipEmbedder()
    query_vector = embedder.embed_text([query])[0]
    results = index.search(query_vector, top_k)

    metadata = pd.read_csv(SUBSET_METADATA_PATH).set_index("item_id")

    table = Table(title=f'Top {len(results)} image results for "{query}"')
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Item ID", justify="right")
    table.add_column("Display Name")
    table.add_column("Category")

    output_dir = DEMO_RESULTS_DIR / _slugify(query)
    output_dir.mkdir(parents=True, exist_ok=True)

    for rank, (item_id, score) in enumerate(results, start=1):
        row = metadata.loc[item_id]
        table.add_row(
            str(rank),
            f"{score:.4f}",
            str(item_id),
            str(row["display name"]),
            str(row["category"]),
        )
        source_image = DATASET_IMAGES_DIR / row["image"]
        shutil.copy(source_image, output_dir / f"{rank:02d}_{row['image']}")

    Console().print(table)
    print(f"Copied {len(results)} images to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-modal (text-to-image) product search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Search the image dataset by text query")
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_multimodal_cli.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Manually exercise the CLI against the real index**

Run: `ecomsearch-images search "something warm for rainy weather" --top-k 5`
Expected: a rendered table with 5 rows, plus a printed confirmation that images were copied to `demo_results/something-warm-for-rainy-weather/`. Open a couple of the copied image files and confirm they look like plausible matches for the query (e.g. jackets, sweaters, raincoats — not, say, sandals or swimwear).

- [ ] **Step 6: Commit**

```bash
git add src/ecomsearch/multimodal/cli.py tests/test_multimodal_cli.py
git commit -m "feat: add cross-modal search CLI"
git push origin main
```

---

### Task 9: Full test suite check and README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass (Phase 1's 10 tests plus Phase 2's new ones: CLIP embedder x3, sampling x2, multimodal integration x1, download script x2, build script x1, multimodal cli x3 — 22 total).

- [ ] **Step 2: Update `README.md`**

- Change the Phase 2 checklist line from `- [ ] Phase 2 — Multimodal (CLIP) module` to `- [x] Phase 2 — Multimodal (CLIP) module`.
- Add a row to the "Stack" table's Image embeddings line if not already accurate (it should already read `openai/clip-vit-base-patch32` — verify this matches what was actually implemented).
- Replace the existing "Data" section's multimodal placeholder sentence ("Details and license attribution will be added here once that dataset is selected.") with:

```markdown
The multimodal (CLIP) module (Phase 2) is demonstrated on a separate,
properly licensed public dataset:
[Mini Fashion Product Images and Text Dataset](https://www.kaggle.com/datasets/nirmalsankalana/mini-product-image-and-text-dataset)
by nirmalsankalana on Kaggle, MIT licensed, 44,671 fashion product
image/text pairs. Phase 2 embeds a 5,000-item subset (stratified by
category) via CLIP for a cross-modal (text-to-image) search demo — this
is entirely separate from the main 55,516-row catalog used everywhere
else in this project.
```

- Add a new subsection under "Setup" for the multimodal module:

```markdown
### Multimodal (CLIP) demo

Requires a Kaggle API token at `~/.kaggle/kaggle.json`
([setup instructions](https://www.kaggle.com/docs/api)).

```bash
python scripts/download_multimodal_dataset.py
python scripts/build_multimodal_index.py
ecomsearch-images search "something warm for rainy weather" --top-k 5
```

Matched images are copied to `demo_results/<query-slug>/` for viewing.
```

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "docs: update README for Phase 2 completion"
git push origin main
```
