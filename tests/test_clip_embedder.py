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
