"""Unit tests for embedding models, mock implementations, and error handling."""

import numpy as np
import pytest
from PIL import Image

from embedding_store.embedders import (
    FakeTextEmbedder,
    FakeVisualEmbedder,
    MiniLMTextEmbedder,
    SigLIPVisualEmbedder,
)


def test_fake_visual_embedder_shapes_and_normalization():
    """Verify FakeVisualEmbedder returns deterministic, normalized float32 vectors."""
    embedder = FakeVisualEmbedder(dim=768)
    assert embedder.dim == 768
    assert embedder.model_name == "mock/siglip-fake"

    img1 = Image.new("RGB", (32, 32), color=(255, 0, 0))
    img2 = Image.new("RGB", (32, 32), color=(0, 255, 0))

    vecs = embedder.embed_images([img1, img2])
    assert vecs.shape == (2, 768)
    assert vecs.dtype == np.float32

    # L2 normalized
    norm1 = np.linalg.norm(vecs[0])
    norm2 = np.linalg.norm(vecs[1])
    assert np.isclose(norm1, 1.0, atol=1e-5)
    assert np.isclose(norm2, 1.0, atol=1e-5)

    # Determinism: same image produces same vector
    vecs_repeat = embedder.embed_images([img1])
    assert np.allclose(vecs[0], vecs_repeat[0])

    # Query text projection
    q_vec = embedder.embed_query_text("a red apple")
    assert q_vec.shape == (768,)
    assert q_vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(q_vec), 1.0, atol=1e-5)


def test_fake_text_embedder_shapes_and_normalization():
    """Verify FakeTextEmbedder returns deterministic, normalized float32 vectors."""
    embedder = FakeTextEmbedder(dim=384)
    assert embedder.dim == 384

    texts = ["First transcript segment", "Second transcript segment"]
    vecs = embedder.embed_texts(texts)
    assert vecs.shape == (2, 384)
    assert vecs.dtype == np.float32

    assert np.isclose(np.linalg.norm(vecs[0]), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(vecs[1]), 1.0, atol=1e-5)

    # Query embedding
    q_vec = embedder.embed_query_text("semantic query")
    assert q_vec.shape == (384,)
    assert np.isclose(np.linalg.norm(q_vec), 1.0, atol=1e-5)


def test_missing_model_offline_fails_with_clear_message(tmp_path):
    """Verify loading an uncached model in offline mode raises an error directing to download-models."""
    embedder = SigLIPVisualEmbedder(
        model_name="nonexistent/fake-visual-model-xyz",
        device="cpu",
        models_dir=tmp_path,
        offline_mode=True,
    )

    with pytest.raises(RuntimeError) as exc_info:
        embedder.embed_images([Image.new("RGB", (10, 10))])

    assert "download-models" in str(exc_info.value)

    txt_embedder = MiniLMTextEmbedder(
        model_name="nonexistent/fake-text-model-xyz",
        device="cpu",
        models_dir=tmp_path,
        offline_mode=True,
    )
    with pytest.raises(RuntimeError) as exc_txt_info:
        txt_embedder.embed_texts(["hello"])

    assert "download-models" in str(exc_txt_info.value)
