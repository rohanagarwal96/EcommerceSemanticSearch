import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    from ecomsearch.api.limiter import limiter

    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture(scope="session")
def embedder():
    from ecomsearch.embeddings import Embedder

    return Embedder()


@pytest.fixture(scope="session")
def clip_embedder():
    from ecomsearch.multimodal.clip_embedder import ClipEmbedder

    return ClipEmbedder()


@pytest.fixture(scope="session")
def cross_encoder():
    from ecomsearch.reranker import CrossEncoderReranker

    return CrossEncoderReranker()
