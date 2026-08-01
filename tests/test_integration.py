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
