"""FAISS Vector Index Management and Automatic Cache Rebuilding.

Maintains modality-isolated FAISS IndexIDMap2(IndexFlatIP) indices.
Vectors are strictly L2-normalized to implement exact cosine similarity ranking.
Persistent storage is atomic (temp file + os.replace). Index verification checks
dimension, element count, and full ID set equality against SQLite. On any
detected corruption or discrepancy, the FAISS index is autonomously rebuilt
from SQLite float32 BLOBs without invoking any embedding models.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np

from .logging_config import get_logger
from .metadata_store import MetadataStore

logger = get_logger("vector_index")

INDEX_SCHEMA_VERSION = "2.0.0"


class VectorIndexManager:
    """Manages an individual FAISS IndexIDMap2 instance for a single modality."""

    def __init__(self, dim: int, modality: str) -> None:
        self.dim = dim
        self.modality = modality
        self.index: Any = self.build_index(dim)

    @staticmethod
    def build_index(dim: int) -> faiss.IndexIDMap2:
        """Create a fresh IndexIDMap2 wrapped around IndexFlatIP."""
        base_index = faiss.IndexFlatIP(dim)
        return faiss.IndexIDMap2(base_index)

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def get_id_set(self) -> Set[int]:
        """Extract the set of vector IDs currently stored in the FAISS index."""
        if self.index.ntotal == 0:
            return set()
        arr = faiss.vector_to_array(self.index.id_map)
        return set(int(x) for x in arr)

    def add(self, vectors: np.ndarray, vector_ids: np.ndarray | List[int]) -> None:
        """Add normalized vectors with specific integer IDs."""
        if len(vectors) == 0:
            return

        vec_arr = np.ascontiguousarray(vectors, dtype=np.float32)
        if vec_arr.ndim == 1:
            vec_arr = vec_arr.reshape(1, -1)

        if vec_arr.shape[1] != self.dim:
            raise ValueError(
                f"Dimension mismatch for {self.modality} index: expected {self.dim}, got {vec_arr.shape[1]}"
            )

        id_arr = np.ascontiguousarray(vector_ids, dtype=np.int64)
        if len(id_arr) != len(vec_arr):
            raise ValueError(
                f"Count mismatch: {len(vec_arr)} vectors vs {len(id_arr)} IDs"
            )

        self.index.add_with_ids(vec_arr, id_arr)

    def remove(self, vector_ids: List[int] | np.ndarray) -> int:
        """Remove vector IDs from the index using IDSelectorBatch."""
        if len(vector_ids) == 0:
            return 0

        id_arr = np.ascontiguousarray(vector_ids, dtype=np.int64)
        try:
            selector = getattr(faiss, "IDSelectorBatch")(id_arr)
        except Exception:
            try:
                selector = faiss.IDSelectorArray(
                    len(id_arr), faiss.cast_integer_to_idx_t_ptr(id_arr.ctypes.data)
                )
            except Exception:
                selector = faiss.IDSelectorBatch(len(id_arr), id_arr)
        removed_count = self.index.remove_ids(selector)
        return removed_count

    def search(
        self, query_vec: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute cosine similarity search.

        Args:
            query_vec: 1D or 2D L2-normalized float32 vector.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            Tuple of (scores: 1D ndarray, vector_ids: 1D ndarray).
        """
        if self.index.ntotal == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

        q = np.ascontiguousarray(query_vec, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        k = min(top_k, self.index.ntotal)
        scores, ids = self.index.search(q, k)
        return scores[0], ids[0]

    def save_index(self, target_path: Path) -> None:
        """Atomically persist FAISS index to disk via temporary file rename."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            prefix=f"{target_path.stem}_",
            suffix=".tmp",
            dir=str(target_path.parent),
        )
        os.close(tmp_fd)

        try:
            faiss.write_index(self.index, tmp_path_str)
            if os.path.exists(target_path):
                os.replace(tmp_path_str, target_path)
            else:
                os.rename(tmp_path_str, target_path)
            logger.debug("Saved %s FAISS index to %s (ntotal=%d)", self.modality, target_path.name, self.index.ntotal)
        except Exception as exc:
            if os.path.exists(tmp_path_str):
                os.remove(tmp_path_str)
            logger.error("Failed to save %s FAISS index to %s: %s", self.modality, target_path, exc)
            raise

    @classmethod
    def load_index(cls, path: Path, dim: int, modality: str) -> VectorIndexManager:
        """Load an existing FAISS index from disk."""
        if not path.is_file():
            raise FileNotFoundError(f"FAISS index file not found: {path}")

        loaded = faiss.read_index(str(path))
        # Ensure it is wrapped in IndexIDMap2
        if not isinstance(loaded, faiss.IndexIDMap2):
            loaded = faiss.IndexIDMap2(loaded)

        mgr = cls(dim=dim, modality=modality)
        mgr.index = loaded
        return mgr


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Read index manifest JSON if available."""
    if not manifest_path.is_file():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read index manifest %s: %s", manifest_path, exc)
        return {}


def save_manifest(manifest_path: Path, manifest_data: Dict[str, Any]) -> None:
    """Atomically save index manifest JSON."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        prefix="manifest_",
        suffix=".tmp",
        dir=str(manifest_path.parent),
    )
    os.close(tmp_fd)

    try:
        with open(tmp_path_str, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        if os.path.exists(manifest_path):
            os.replace(tmp_path_str, manifest_path)
        else:
            os.rename(tmp_path_str, manifest_path)
    except Exception as exc:
        if os.path.exists(tmp_path_str):
            os.remove(tmp_path_str)
        raise exc


def load_or_rebuild_index(
    modality: str,
    index_file: Path,
    manifest_file: Path,
    store: MetadataStore,
    expected_dim: int,
    expected_model: str,
) -> VectorIndexManager:
    """Load FAISS index with rigorous integrity checks; rebuild from SQLite on any defect.

    Integrity checks:
      1. Index file exists and is readable by FAISS.
      2. Dimension matches configured/stored model dimension.
      3. Total vector count matches matching SQLite table row count.
      4. Stored FAISS ID set matches SQLite vector_id set.

    On mismatch, failure, or file absence, completely rebuilds the index from
    SQLite float32 BLOBs without invoking any machine learning models.
    """
    rebuild_reason: Optional[str] = None
    mgr: Optional[VectorIndexManager] = None

    if not index_file.is_file():
        rebuild_reason = f"Index file does not exist: {index_file}"
    else:
        try:
            mgr = VectorIndexManager.load_index(index_file, expected_dim, modality)
            if mgr.index.d != expected_dim:
                rebuild_reason = (
                    f"Dimension mismatch: index has {mgr.index.d}, expected {expected_dim}"
                )
            else:
                # Compare against SQLite
                cur = store.connection.cursor()
                table = "visual_vectors" if modality == "visual" else "text_vectors"
                cur.execute(f"SELECT vector_id FROM {table} ORDER BY vector_id ASC")
                db_ids = set(int(r["vector_id"]) for r in cur.fetchall())

                if mgr.ntotal != len(db_ids):
                    rebuild_reason = (
                        f"Count mismatch: index has {mgr.ntotal}, SQLite has {len(db_ids)}"
                    )
                else:
                    faiss_ids = mgr.get_id_set()
                    if faiss_ids != db_ids:
                        rebuild_reason = "ID set mismatch between FAISS and SQLite"
        except Exception as exc:
            rebuild_reason = f"Error reading index file: {exc}"

    if rebuild_reason is not None:
        logger.warning(
            "FAISS %s index verification failed (%s). Rebuilding from SQLite embedding BLOBs...",
            modality,
            rebuild_reason,
        )
        mgr = VectorIndexManager(dim=expected_dim, modality=modality)
        vector_ids, embeddings = store.get_all_embeddings(modality)

        if len(vector_ids) > 0:
            mgr.add(embeddings, vector_ids)
            logger.info(
                "Successfully rebuilt %s FAISS index with %d vectors directly from SQLite.",
                modality,
                mgr.ntotal,
            )
        mgr.save_index(index_file)

        # Update manifest file atomically
        manifest_data = load_manifest(manifest_file)
        if not manifest_data or "counts" not in manifest_data:
            manifest_data = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "models": {},
                "dims": {},
                "counts": {},
                "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        models_dict = manifest_data.get("models")
        if not isinstance(models_dict, dict):
            models_dict = {}
            manifest_data["models"] = models_dict
        models_dict[modality] = expected_model

        dims_dict = manifest_data.get("dims")
        if not isinstance(dims_dict, dict):
            dims_dict = {}
            manifest_data["dims"] = dims_dict
        dims_dict[modality] = expected_dim

        counts_dict = manifest_data.get("counts")
        if not isinstance(counts_dict, dict):
            counts_dict = {}
            manifest_data["counts"] = counts_dict
        counts_dict[modality] = mgr.ntotal

        manifest_data["saved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save_manifest(manifest_file, manifest_data)

    assert mgr is not None
    return mgr
