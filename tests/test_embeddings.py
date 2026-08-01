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
