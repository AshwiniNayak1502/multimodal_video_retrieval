"""Diagnostic script dumping raw candidates, scores, calibration, fusion, and UI objects."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root, module2 and module3 to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODULE2_DIR = PROJECT_ROOT / "module2"
if str(MODULE2_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE2_DIR))
MODULE3_DIR = PROJECT_ROOT / "module3"
if str(MODULE3_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE3_DIR))

from module3.retrieval_engine.config import Module3Config
from module3.retrieval_engine.query_service import QueryService
from app.services.search_service import SearchEngineWrapper
from app.db.app_db import AppDatabase

def run_diagnostics(query: str = "spring mvc", target_video_id: str = "vid_d9b1180af485c6cd"):
    print(f"======================================================================")
    print(f"DIAGNOSTIC SEARCH DUMP FOR: '{query}' ON {target_video_id}")
    print(f"======================================================================\n")

    cfg = Module3Config()
    qs = QueryService(config=cfg, warmup=False)
    index = qs.index

    # 1. FAISS Index Properties
    print("--- 1. FAISS Index Metric & Dimensions ---")
    print("Visual Index Metric:", index.vis_mgr.index.metric_type, "(0=InnerProduct, 1=L2)")
    print("Visual Index NTotal:", index.vis_mgr.index.ntotal, "Dim:", index.vis_mgr.dim)
    print("Text Index Metric:", index.txt_mgr.index.metric_type, "(0=InnerProduct, 1=L2)")
    print("Text Index NTotal:", index.txt_mgr.index.ntotal, "Dim:", index.txt_mgr.dim)

    # 2. Query Embeddings
    print("\n--- 2. Query Encoding ---")
    vis_query_vec = index.visual_embedder.embed_query_text(query)
    txt_query_vec = index.text_embedder.embed_query_text(query)
    print("Visual query norm:", float(np_norm := (vis_query_vec**2).sum()**0.5), "Shape:", vis_query_vec.shape)
    print("Text query norm:", float((txt_query_vec**2).sum()**0.5), "Shape:", txt_query_vec.shape)

    # 3. Raw FAISS Retrieval
    print("\n--- 3. Raw FAISS Retrieval (top 10 each modality) ---")
    vis_scores, vis_ids = index.vis_mgr.search(vis_query_vec, top_k=15)
    txt_scores, txt_ids = index.txt_mgr.search(txt_query_vec, top_k=15)

    print("\nTop 10 Visual FAISS Hits:")
    meta_vis = index.store.get_visual_metadata_by_ids([int(v) for v in vis_ids if int(v) != -1])
    for rank, (score, vid) in enumerate(zip(vis_scores, vis_ids), 1):
        meta = meta_vis.get(int(vid))
        v_video_id = meta.get("video_id") if meta else "NOT_FOUND"
        is_target = " [TARGET SAMPLE 6]" if v_video_id == target_video_id else ""
        print(f"  [{rank}] FAISS ID={vid}, Raw Cosine={score:.4f}, Video={v_video_id}, Time={meta.get('timestamp_seconds') if meta else None}s{is_target}")

    print("\nTop 10 Text FAISS Hits:")
    meta_txt = index.store.get_text_metadata_by_ids([int(t) for t in txt_ids if int(t) != -1])
    for rank, (score, tid) in enumerate(zip(txt_scores, txt_ids), 1):
        meta = meta_txt.get(int(tid))
        t_video_id = meta.get("video_id") if meta else "NOT_FOUND"
        is_target = " [TARGET SAMPLE 6]" if t_video_id == target_video_id else ""
        excerpt = meta.get("transcript_text", "")[:50] if meta else ""
        print(f"  [{rank}] FAISS ID={tid}, Raw Cosine={score:.4f}, Video={t_video_id}, Range={meta.get('start_timestamp') if meta else None}-{meta.get('end_timestamp') if meta else None}s, Text='{excerpt}'{is_target}")

    # 4. Filtered Retrieval on Target Video ID
    print(f"\n--- 4. QueryService Search for Modes on {target_video_id} ---")
    modes = [
        ("Combined", ("visual", "text")),
        ("Transcript only", ("text",)),
        ("Visual only", ("visual",)),
    ]

    for mode_name, modalities in modes:
        print(f"\n==================== MODE: {mode_name} ({modalities}) ====================")
        resp = qs.search(
            query=query,
            top_k=5,
            video_ids=[target_video_id],
            modalities=modalities,
        )
        print(f"Status: {resp.status}, Message: {resp.message}")
        print(f"Results Count: {len(resp.results)}")
        for idx, res in enumerate(resp.results):
            res_dict = res.to_dict()
            print(f"  Result [{idx}]:")
            print(f"    start_seconds: {res.start_seconds}, end_seconds: {res.end_seconds}")
            print(f"    start_formatted: {res.start_formatted}, end_formatted: {res.end_formatted}")
            print(f"    seek_seconds: {res.seek_seconds}")
            print(f"    final_score: {res.final_score}")
            print(f"    visual_score: {res.visual_score}, text_score: {res.text_score}")
            print(f"    visual_raw_cosine: {res.visual_raw_cosine}, text_raw_cosine: {res.text_raw_cosine}")
            print(f"    modalities: {res.modalities}")
            print(f"    transcript_text: {res.transcript_text}")
            print(f"    frame_id: {res.frame_id}")

    # 5. App UI Object simulation via SearchEngineWrapper
    print(f"\n--- 5. App UI Object from SearchEngineWrapper ---")
    db = AppDatabase(PROJECT_ROOT / "data" / "app.db")
    wrapper = SearchEngineWrapper()
    user_resp = wrapper.search_user_videos(
        user_id=1,
        query=query,
        top_k=5,
        modality_mode="Combined",
        filter_video_id=target_video_id,
        db=db,
    )
    print(f"Wrapper Status: {user_resp.status}, Total results: {len(user_resp.results)}")
    for idx, r in enumerate(user_resp.results):
        print(f"  UI Result [{idx}]:")
        print(f"    dict keys: {list(r.__dict__.keys()) if hasattr(r, '__dict__') else dir(r)}")
        # Check what fields search_view.py reads!
        print(f"    getattr start_time: {getattr(r, 'start_time', None)}")
        print(f"    getattr end_time: {getattr(r, 'end_time', None)}")
        print(f"    getattr score: {getattr(r, 'score', None)}")
        print(f"    getattr final_score: {getattr(r, 'final_score', None)}")
        print(f"    getattr start_seconds: {getattr(r, 'start_seconds', None)}")
        print(f"    getattr end_seconds: {getattr(r, 'end_seconds', None)}")

if __name__ == "__main__":
    run_diagnostics()
