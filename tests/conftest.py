import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(scope="session")
def embedder():
    from ecomsearch.embeddings import Embedder

    return Embedder()
