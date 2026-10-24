"""Visual and Text Embedding Models and Deterministic Test Mocks.

Implements:
1. SigLIPVisualEmbedder: Google SigLIP/SigLIP2 multimodal model for frame images
   and cross-modal text query projection via the SigLIP text tower.
2. MiniLMTextEmbedder: SentenceTransformers all-MiniLM-L6-v2 for speech transcripts.
3. FakeVisualEmbedder / FakeTextEmbedder: Deterministic, hash-seeded mock embedders
   enabling ultra-fast, zero-network unit and integration testing.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence, Union

import numpy as np
from PIL import Image

from .config import (
    DEFAULT_MINILM_DIM,
    DEFAULT_SIGLIP_DIM,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISUAL_FALLBACK_MODEL,
    DEFAULT_VISUAL_MODEL,
    Module2Config,
)
from .logging_config import get_logger
from .offline_env import configure_offline_env

logger = get_logger("embedders")


class VisualEmbedderProtocol(Protocol):
    """Protocol for visual frame and cross-modal query embedding."""

    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def load_count(self) -> int: ...

    def embed_images(
        self, images: Sequence[Union[Image.Image, Path, str]]
    ) -> np.ndarray: ...

    def embed_query_text(self, text: str) -> np.ndarray: ...


class TextEmbedderProtocol(Protocol):
    """Protocol for transcript text segment and text query embedding."""

    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def load_count(self) -> int: ...

    def embed_texts(self, texts: List[str]) -> np.ndarray: ...

    def embed_query_text(self, text: str) -> np.ndarray: ...


def resolve_local_model_path(
    model_name: str,
    cache_dir: Optional[str] = None,
    offline: bool = True,
) -> str:
    """Resolve a model repo_id or path to a local directory snapshot when running offline.

    If model_name is already an existing directory, returns it directly.
    Otherwise, uses huggingface_hub.snapshot_download with local_files_only=True
    to locate the exact cached snapshot directory on disk, ensuring subsequent
    calls to transformers / sentence_transformers never trigger network activity.
    """
    p = Path(model_name)
    if p.is_dir():
        return str(p.resolve())

    if offline:
        try:
            from huggingface_hub import snapshot_download

            snapshot_path = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            logger.debug(
                "Resolved model '%s' to local snapshot directory: %s",
                model_name,
                snapshot_path,
            )
            return str(snapshot_path)
        except Exception as exc:
            logger.debug(
                "snapshot_download offline resolution for '%s' returned: %s",
                model_name,
                exc,
            )
            return model_name

    return model_name


def _extract_pooled_tensor(features: Any) -> Any:
    """Handle diverse transformers return types for get_*_features.

    Supports plain torch.Tensor, ModelOutput objects exposing pooler_output,
    last_hidden_state, or tuple index 0.
    """
    import torch

    if isinstance(features, torch.Tensor):
        return features

    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output

    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds

    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds

    if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
        # Mean pool or take CLS/first token if 3D
        lhs = features.last_hidden_state
        if len(lhs.shape) == 3:
            return lhs[:, 0, :]
        return lhs

    if hasattr(features, "__getitem__"):
        first_item = features[0]
        if isinstance(first_item, torch.Tensor):
            if len(first_item.shape) == 3:
                return first_item[:, 0, :]
            return first_item

    raise ValueError(f"Unrecognized features object type: {type(features)}")


class SigLIPVisualEmbedder:
    """Local SigLIP/SigLIP2 visual embedder and cross-modal query projecter."""

    def __init__(
        self,
        model_name: str = DEFAULT_VISUAL_MODEL,
        device: str = "cpu",
        batch_size: int = 16,
        models_dir: Optional[Path] = None,
        offline_mode: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._models_dir = str(models_dir) if models_dir else None
        self._offline_mode = offline_mode
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._dim = DEFAULT_SIGLIP_DIM
        self._load_count: int = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def load_count(self) -> int:
        return self._load_count

    def _load_model(self) -> None:
        """Lazily and atomically load SigLIP model, processor, and tokenizer."""
        if self._model is not None and self._processor is not None:
            return

        import traceback
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        if self._offline_mode:
            configure_offline_env()

        logger.info(
            "Loading visual model '%s' (device=%s, offline=%s)...",
            self._model_name,
            self._device,
            self._offline_mode,
        )

        local_model_path = resolve_local_model_path(
            self._model_name,
            cache_dir=self._models_dir,
            offline=self._offline_mode,
        )

        loaded_model = None
        loaded_processor = None
        loaded_tokenizer = None
        derived_dim = self._dim

        try:
            loaded_model = AutoModel.from_pretrained(
                local_model_path,
                cache_dir=self._models_dir,
                local_files_only=self._offline_mode,
            ).to(self._device)
            loaded_model.eval()

            loaded_processor = AutoProcessor.from_pretrained(
                local_model_path,
                cache_dir=self._models_dir,
                local_files_only=self._offline_mode,
            )

            # Some SigLIP models bundle tokenizer inside processor; fallback to AutoTokenizer
            try:
                loaded_tokenizer = AutoTokenizer.from_pretrained(
                    local_model_path,
                    cache_dir=self._models_dir,
                    local_files_only=self._offline_mode,
                )
            except Exception:
                loaded_tokenizer = getattr(loaded_processor, "tokenizer", None)

            # Derive embedding dimension from model config
            config = getattr(loaded_model, "config", None)
            if config:
                derived_dim = getattr(
                    config,
                    "projection_dim",
                    getattr(
                        config,
                        "text_config",
                        None,
                    )
                    and getattr(config.text_config, "hidden_size", DEFAULT_SIGLIP_DIM)
                    or DEFAULT_SIGLIP_DIM,
                )

            # Atomic assignment: only assign when everything has loaded successfully
            self._model = loaded_model
            self._processor = loaded_processor
            self._tokenizer = loaded_tokenizer
            self._dim = derived_dim
            self._load_count += 1

            logger.info(
                "Visual model '%s' loaded successfully (dim=%d, load_count=%d).",
                self._model_name,
                self._dim,
                self._load_count,
            )

        except Exception as exc:
            # Leave in clean "not loaded" state so subsequent calls can genuinely retry
            self._model = None
            self._processor = None
            self._tokenizer = None
            tb = traceback.format_exc()
            err_msg = (
                f"Failed to load visual model '{self._model_name}' (offline_mode={self._offline_mode}).\n"
                f"Details: {repr(exc)}\n"
                f"Traceback:\n{tb}\n"
                "If running offline, please execute 'python main.py download-models' while online first."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg) from exc

    def embed_images(
        self, images: Sequence[Union[Image.Image, Path, str]]
    ) -> np.ndarray:
        """Encode a collection of frame images into L2-normalized float32 vectors.

        Supports batching and handles potential CUDA OOM by falling back to smaller batches.
        """
        if not images:
            return np.empty((0, self._dim), dtype=np.float32)

        self._load_model()
        import torch

        all_vectors: List[np.ndarray] = []
        batch_size = self._batch_size

        for i in range(0, len(images), batch_size):
            batch_slice = images[i : i + batch_size]
            pil_batch: List[Image.Image] = []
            for item in batch_slice:
                if isinstance(item, Image.Image):
                    pil_batch.append(item.convert("RGB"))
                else:
                    with Image.open(item) as img:
                        pil_batch.append(img.convert("RGB"))

            try:
                batch_vecs = self._encode_image_batch(pil_batch)
                all_vectors.append(batch_vecs)
            except torch.cuda.OutOfMemoryError:
                logger.warning("CUDA OOM encountered on batch size %d; retrying item by item.", len(pil_batch))
                torch.cuda.empty_cache()
                for single_img in pil_batch:
                    single_vec = self._encode_image_batch([single_img])
                    all_vectors.append(single_vec)

        concatenated = np.vstack(all_vectors).astype(np.float32)
        if concatenated.shape[1] != self._dim:
            self._dim = concatenated.shape[1]
        return concatenated

    def _encode_image_batch(self, pil_images: List[Image.Image]) -> np.ndarray:
        """Process and encode a single batch of PIL images."""
        if self._processor is None or self._model is None:
            raise RuntimeError("SigLIPVisualEmbedder processor or model is not loaded. Call _load_model() first.")
        import torch
        import torch.nn.functional as F

        with torch.inference_mode():
            inputs = self._processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            raw_features = self._model.get_image_features(**inputs)
            tensor_features = _extract_pooled_tensor(raw_features)

            # L2 normalize
            normalized = F.normalize(tensor_features, p=2, dim=-1)
            return normalized.cpu().numpy().astype(np.float32)

    def embed_query_text(self, text: str) -> np.ndarray:
        """Project a text query into the visual embedding space via SigLIP's text tower.

        Uses lower-cased text, max_length=64, truncation=True, and padding='max_length'.
        """
        self._load_model()
        if self._processor is None or self._model is None:
            raise RuntimeError("SigLIPVisualEmbedder processor or model is not loaded. Call _load_model() first.")
        import torch
        import torch.nn.functional as F

        query_text = text.strip().lower()
        with torch.inference_mode():
            if self._tokenizer:
                tok_inputs = self._tokenizer(
                    [query_text],
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt",
                )
            else:
                tok_inputs = self._processor(
                    text=[query_text],
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt",
                )

            tok_inputs = {k: v.to(self._device) for k, v in tok_inputs.items()}
            raw_features = self._model.get_text_features(**tok_inputs)
            tensor_features = _extract_pooled_tensor(raw_features)

            # L2 normalize
            normalized = F.normalize(tensor_features, p=2, dim=-1)
            vec = normalized.cpu().numpy().astype(np.float32)[0]
            return vec


class MiniLMTextEmbedder:
    """Local SentenceTransformers all-MiniLM-L6-v2 text embedder for transcript segments."""

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        device: str = "cpu",
        batch_size: int = 64,
        models_dir: Optional[Path] = None,
        offline_mode: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._models_dir = str(models_dir) if models_dir else None
        self._offline_mode = offline_mode
        self._model = None
        self._dim = DEFAULT_MINILM_DIM
        self._load_count: int = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def load_count(self) -> int:
        return self._load_count

    def _load_model(self) -> None:
        """Lazily and atomically load SentenceTransformer model."""
        if self._model is not None:
            return

        import traceback
        from sentence_transformers import SentenceTransformer

        if self._offline_mode:
            configure_offline_env()

        logger.info(
            "Loading text model '%s' (device=%s, offline=%s)...",
            self._model_name,
            self._device,
            self._offline_mode,
        )

        local_model_path = resolve_local_model_path(
            self._model_name,
            cache_dir=self._models_dir,
            offline=self._offline_mode,
        )

        loaded_model = None
        try:
            loaded_model = SentenceTransformer(
                local_model_path,
                device=self._device,
                cache_folder=self._models_dir,
                local_files_only=self._offline_mode,
            )
            # Retrieve embedding dim
            test_emb = loaded_model.encode(["probe"], normalize_embeddings=True)
            self._dim = int(test_emb.shape[1])
            self._model = loaded_model
            self._load_count += 1
            logger.info(
                "Text model '%s' loaded successfully (dim=%d, load_count=%d).",
                self._model_name,
                self._dim,
                self._load_count,
            )
        except Exception as exc:
            self._model = None
            tb = traceback.format_exc()
            err_msg = (
                f"Failed to load text model '{self._model_name}' (offline_mode={self._offline_mode}).\n"
                f"Details: {repr(exc)}\n"
                f"Traceback:\n{tb}\n"
                "If running offline, please execute 'python main.py download-models' while online first."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg) from exc

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a collection of transcript texts into L2-normalized float32 vectors."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        self._load_model()
        if self._model is None:
            raise RuntimeError("MiniLMTextEmbedder model is not loaded. Call _load_model() first.")
        clean_texts = [t.strip() for t in texts]
        embeddings = self._model.encode(
            clean_texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query_text(self, text: str) -> np.ndarray:
        """Encode a search query string into a 1D normalized float32 vector."""
        vecs = self.embed_texts([text])
        return vecs[0]


class FakeVisualEmbedder:
    """Deterministic, hash-seeded mock visual embedder for fast offline unit testing."""

    def __init__(
        self,
        dim: int = DEFAULT_SIGLIP_DIM,
        model_name: str = "mock/siglip-fake",
    ) -> None:
        self._dim = dim
        self._model_name = model_name
        self._load_count: int = 1

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def load_count(self) -> int:
        return self._load_count

    def _hash_to_vector(self, data: bytes) -> np.ndarray:
        """Seed a pseudorandom generator from data hash and return normalized float32 vector."""
        h = hashlib.sha256(data).digest()
        seed = int.from_bytes(h[:4], byteorder="big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_images(
        self, images: Sequence[Union[Image.Image, Path, str]]
    ) -> np.ndarray:
        if not images:
            return np.empty((0, self._dim), dtype=np.float32)

        vectors: List[np.ndarray] = []
        for img_item in images:
            if isinstance(img_item, Image.Image):
                raw = img_item.tobytes()
            else:
                p = Path(img_item)
                if p.is_file():
                    raw = p.read_bytes()
                else:
                    raw = str(img_item).encode("utf-8")
            vectors.append(self._hash_to_vector(raw))

        return np.vstack(vectors).astype(np.float32)

    def embed_query_text(self, text: str) -> np.ndarray:
        raw = text.strip().lower().encode("utf-8")
        return self._hash_to_vector(raw)


class FakeTextEmbedder:
    """Deterministic, hash-seeded mock text embedder for fast offline unit testing."""

    def __init__(
        self,
        dim: int = DEFAULT_MINILM_DIM,
        model_name: str = "mock/minilm-fake",
    ) -> None:
        self._dim = dim
        self._model_name = model_name
        self._load_count: int = 1

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def load_count(self) -> int:
        return self._load_count

    def _hash_to_vector(self, text: str) -> np.ndarray:
        raw = text.strip().encode("utf-8")
        h = hashlib.sha256(raw).digest()
        seed = int.from_bytes(h[:4], byteorder="big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        vectors = [self._hash_to_vector(t) for t in texts]
        return np.vstack(vectors).astype(np.float32)

    def embed_query_text(self, text: str) -> np.ndarray:
        return self._hash_to_vector(text)


def download_models(config: Module2Config) -> None:
    """One-time cache function to download configured HuggingFace models to disk."""
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoProcessor

    print("==================================================")
    print("Initiating one-time model downloads to local cache...")
    print(f"Visual Model: {config.visual_model_name}")
    print(f"Text Model:   {config.text_model_name}")
    print(f"Target Cache: {config.models_dir or 'Default HuggingFace Hub'}")
    print("==================================================")

    # 1. Download Text Model
    print(f"\n[1/2] Downloading Text Model '{config.text_model_name}'...")
    SentenceTransformer(
        config.text_model_name,
        cache_folder=str(config.models_dir) if config.models_dir else None,
        local_files_only=False,
    )
    print("Text model downloaded successfully.")

    # 2. Download Visual Model
    print(f"\n[2/2] Downloading Visual Model '{config.visual_model_name}'...")
    try:
        AutoModel.from_pretrained(
            config.visual_model_name,
            cache_dir=str(config.models_dir) if config.models_dir else None,
            local_files_only=False,
        )
        AutoProcessor.from_pretrained(
            config.visual_model_name,
            cache_dir=str(config.models_dir) if config.models_dir else None,
            local_files_only=False,
        )
        print(f"Visual model '{config.visual_model_name}' downloaded successfully.")
    except Exception as exc:
        logger.warning(
            "Primary visual model '%s' download encountered error: %s",
            config.visual_model_name,
            exc,
        )
        print(f"Warning: Primary model download had issue: {exc}")
        print(f"Attempting fallback visual model '{DEFAULT_VISUAL_FALLBACK_MODEL}'...")
        AutoModel.from_pretrained(
            DEFAULT_VISUAL_FALLBACK_MODEL,
            cache_dir=str(config.models_dir) if config.models_dir else None,
            local_files_only=False,
        )
        AutoProcessor.from_pretrained(
            DEFAULT_VISUAL_FALLBACK_MODEL,
            cache_dir=str(config.models_dir) if config.models_dir else None,
            local_files_only=False,
        )
        print(f"Fallback visual model '{DEFAULT_VISUAL_FALLBACK_MODEL}' downloaded successfully.")

    print("\nAll models are locally cached. The system can now operate fully offline.")
