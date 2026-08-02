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
