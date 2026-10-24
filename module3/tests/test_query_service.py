"""Integration tests for QueryService validation, preflight checks, thread safety, and API contracts."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Tuple

import pytest

from retrieval_engine.config import Module3Config
from retrieval_engine.query_service import QueryService, SearchResponse


def test_input_validation_empty_and_whitespace(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Empty or whitespace-only queries must return clear error status, never crash."""
    service, _, _, _ = indexed_test_environment

    resp_empty = service.search(query="")
    assert resp_empty.status == "error"
    assert "query" in resp_empty.message.lower()

    resp_ws = service.search(query="   \t\n  ")
    assert resp_ws.status == "error"
    assert "query" in resp_ws.message.lower()


def test_input_validation_query_length_cap(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Queries exceeding the character cap must return clear error status."""
    service, cfg, _, _ = indexed_test_environment

    long_query = "word " * (cfg.max_query_chars // 4 + 10)
    assert len(long_query) > cfg.max_query_chars

    resp = service.search(query=long_query)
    assert resp.status == "error"
    assert "limit" in resp.message.lower() or "too long" in resp.message.lower()


def test_input_validation_top_k_boundaries(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """top_k must accept integers 3, 4, 5 and reject all other values and types."""
    service, _, _, _ = indexed_test_environment

    # Valid top_k values (3, 4, 5)
    for k in (3, 4, 5):
        resp = service.search(query="valid query", top_k=k)
        assert resp.status in ("ok", "no_match")

    # Invalid integer top_k (0, 1, 2, 6, 10, -1)
    for bad_k in (0, 1, 2, 6, 10, -1):
        resp = service.search(query="valid query", top_k=bad_k)
        assert resp.status == "error"
        assert "top_k" in resp.message.lower()

    # Invalid non-integer types (float, string, bool, None)
    for bad_type_k in (3.5, "5", True, False, None):
        resp = service.search(query="valid query", top_k=bad_type_k)  # type: ignore
        assert resp.status == "error"
        assert "top_k" in resp.message.lower()


def test_input_validation_unknown_video_id(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Unknown video_ids must return clear validation error naming the invalid ID."""
    service, _, _, _ = indexed_test_environment

    resp = service.search(query="valid query", top_k=5, video_ids=["vid_non_existent_12345"])
    assert resp.status == "error"
    assert "vid_non_existent_12345" in resp.message


def test_preflight_empty_index_returns_index_not_ready(tmp_path: Path) -> None:
    """When the vector index is empty or has 0 videos, return status='index_not_ready'."""
    from embedding_store.config import Module2Config
    from embedding_store.metadata_store import MetadataStore
    from embedding_store.retrieval import VideoIndex
    from embedding_store.vector_index import VectorIndexManager

    empty_m2 = tmp_path / "empty_m2"
    empty_vdb = empty_m2 / "vector_db"
    empty_vdb.mkdir(parents=True, exist_ok=True)

    db_path = empty_vdb / "metadata.db"
    store = MetadataStore(db_path)
    vis_mgr = VectorIndexManager(dim=768, modality="visual")
    txt_mgr = VectorIndexManager(dim=384, modality="text")

    m2_cfg = Module2Config(base_dir=empty_m2, vector_db_dir=empty_vdb)
    v_index = VideoIndex(
        config=m2_cfg,
        store=store,
        vis_mgr=vis_mgr,
        txt_mgr=txt_mgr,
    )

    m3_cfg = Module3Config(base_dir=tmp_path / "module3", vector_db_dir=empty_vdb)
    service = QueryService(config=m3_cfg, video_index=v_index, warmup=False)

    resp = service.search(query="find something")
    assert resp.status == "index_not_ready"
    assert "Run 'python main.py index --all' in module2 first" in resp.message


def test_search_response_json_serializability(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """SearchResponse.to_dict() must serialize directly with json.dumps and match schema 3.1.0 (Item 8)."""
    service, _, _, _ = indexed_test_environment

    resp = service.search(query="multimodal query", top_k=5)
    resp_dict = resp.to_dict()

    serialized = json.dumps(resp_dict, indent=2)
    deserialized = json.loads(serialized)

    assert deserialized["schema_version"] == "3.1.0"
    assert deserialized["status"] in ("ok", "no_match")
    assert isinstance(deserialized["timings_ms"], dict)
    assert isinstance(deserialized["debug"], dict)
    assert isinstance(deserialized["notices"], list)
    assert "query_encode_visual_ms" in deserialized["timings_ms"]


def test_input_validation_modality_weights(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Validate modality weights: must be finite, non-negative numbers and not both zero (Item 3)."""
    service, _, _, _ = indexed_test_environment

    # Valid custom weights
    r_valid = service.search(query="valid query", top_k=5, modality_weights={"visual": 0.8, "text": 0.2})
    assert r_valid.status in ("ok", "no_match")

    # One zero weight is valid as long as not both 0
    r_vis_only = service.search(query="valid query", top_k=5, modality_weights={"visual": 1.0, "text": 0.0})
    assert r_vis_only.status in ("ok", "no_match")

    # Both zero is rejected
    r_both_zero = service.search(query="valid query", top_k=5, modality_weights={"visual": 0.0, "text": 0.0})
    assert r_both_zero.status == "error"
    assert "cannot both be zero" in r_both_zero.message or "greater than 0" in r_both_zero.message

    # Negative weight is rejected
    r_neg = service.search(query="valid query", top_k=5, modality_weights={"visual": -0.5, "text": 0.5})
    assert r_neg.status == "error"
    assert "non-negative" in r_neg.message

    # Non-finite or bool weight is rejected
    r_inf = service.search(query="valid query", top_k=5, modality_weights={"visual": float("inf"), "text": 0.5})
    assert r_inf.status == "error"

    r_bool = service.search(query="valid query", top_k=5, modality_weights={"visual": True, "text": 0.5})  # type: ignore
    assert r_bool.status == "error"


def test_user_facing_error_sanitization(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Error messages returned to user must not contain repr(exc), tracebacks, or local paths (Item 8)."""
    service, _, _, _ = indexed_test_environment

    # Test bad query
    resp = service.search(query="")
    assert resp.status == "error"
    assert "traceback" not in resp.message.lower()
    assert "\\" not in resp.message
    assert "exception" not in resp.message.lower()


def test_query_cache_and_telemetry_reuse(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Repeated identical queries must hit LRU cache and set models/index reuse flags."""
    service, _, _, _ = indexed_test_environment

    # First query (cold cache)
    resp1 = service.search(query="multimodal indexing test", top_k=5)
    assert resp1.debug["cache_hit_visual"] is False
    assert resp1.debug["cache_hit_text"] is False

    # Second query (warm cache)
    resp2 = service.search(query="multimodal indexing test", top_k=5)
    assert resp2.debug["cache_hit_visual"] is True
    assert resp2.debug["cache_hit_text"] is True
    assert resp2.debug["models_reused"] is True
    assert resp2.debug["index_reused"] is True


def test_thread_safety_concurrent_searches(
    indexed_test_environment: Tuple[QueryService, Module3Config, Any, Any]
) -> None:
    """Concurrent searches executed across multiple threads must succeed without race conditions."""
    service, _, _, _ = indexed_test_environment

    errors: list[Exception] = []
    responses: list[SearchResponse] = []

    def worker(worker_id: int) -> None:
        try:
            for _ in range(3):
                q = f"query from worker {worker_id}"
                r = service.search(query=q, top_k=5)
                responses.append(r)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent execution errors: {errors}"
    assert len(responses) == 12

