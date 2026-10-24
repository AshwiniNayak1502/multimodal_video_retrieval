"""Database and Vector Index Integrity Verification Suite.

Performs comprehensive validation:
1. Every vector_id in FAISS resolves to an existing SQLite row.
2. Every vector_id in SQLite is indexed in FAISS.
3. Every visual row has a frame file reference and timestamp_seconds >= 0.
4. Every text row satisfies start >= 0 and end >= start.
5. Exact vector count equality between SQLite and FAISS indices.
6. Dimension consistency across manifest, SQLite, and FAISS indices.
7. No duplicate frame_id or segment_uid records.
8. Orphan-free: all vectors reference an existing video_id in the videos table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss

from .config import Module2Config
from .logging_config import get_logger
from .metadata_store import MetadataStore
from .vector_index import VectorIndexManager, load_manifest

logger = get_logger("validation")


def validate_integrity(config: Optional[Module2Config] = None) -> bool:
    """Execute complete validation suite and display PASS/FAIL report."""
    cfg = config or Module2Config()
    db_path = cfg.vector_db_dir / "metadata.db"
    vis_index_file = cfg.vector_db_dir / "visual.index"
    txt_index_file = cfg.vector_db_dir / "text.index"
    manifest_file = cfg.vector_db_dir / "index_manifest.json"

    print("==================================================")
    print("Executing Module 2 Vector Database Integrity Validation")
    print(f"Target DB: {db_path}")
    print("==================================================")

    if not db_path.is_file():
        print("[FAIL] SQLite database does not exist.")
        return False

    store = MetadataStore(db_path)
    cur = store.connection.cursor()

    checks: List[Tuple[str, bool, str]] = []

    # 1. Video and Vector Counts in SQLite
    cur.execute("SELECT COUNT(*) AS c FROM videos")
    video_count = cur.fetchone()["c"]

    cur.execute("SELECT vector_id FROM visual_vectors")
    sql_vis_ids = set(int(r["vector_id"]) for r in cur.fetchall())

    cur.execute("SELECT vector_id FROM text_vectors")
    sql_txt_ids = set(int(r["vector_id"]) for r in cur.fetchall())

    checks.append((
        "Database accessibility",
        True,
        f"{video_count} videos, {len(sql_vis_ids)} visual vectors, {len(sql_txt_ids)} text vectors",
    ))

    # 2. FAISS Indices Existence and Dimensions
    vis_index_exists = vis_index_file.is_file()
    txt_index_exists = txt_index_file.is_file()

    checks.append((
        "FAISS visual index file exists",
        vis_index_exists,
        str(vis_index_file) if vis_index_exists else "Missing",
    ))
    checks.append((
        "FAISS text index file exists",
        txt_index_exists,
        str(txt_index_file) if txt_index_exists else "Missing",
    ))

    if not vis_index_exists or not txt_index_exists:
        _print_report(checks)
        return False

    vis_index = faiss.read_index(str(vis_index_file))
    txt_index = faiss.read_index(str(txt_index_file))

    stored_vis_dim = int(store.get_meta("visual_dim") or 768)
    stored_txt_dim = int(store.get_meta("text_dim") or 384)

    checks.append((
        "Visual index dimension consistency",
        vis_index.d == stored_vis_dim,
        f"FAISS: {vis_index.d}, Meta: {stored_vis_dim}",
    ))
    checks.append((
        "Text index dimension consistency",
        txt_index.d == stored_txt_dim,
        f"FAISS: {txt_index.d}, Meta: {stored_txt_dim}",
    ))

    # 3. Vector count matching
    checks.append((
        "Visual vector count match (SQLite vs FAISS)",
        vis_index.ntotal == len(sql_vis_ids),
        f"FAISS: {vis_index.ntotal}, SQLite: {len(sql_vis_ids)}",
    ))
    checks.append((
        "Text vector count match (SQLite vs FAISS)",
        txt_index.ntotal == len(sql_txt_ids),
        f"FAISS: {txt_index.ntotal}, SQLite: {len(sql_txt_ids)}",
    ))

    # 4. Bidirectional ID Resolution
    faiss_vis_ids = set(int(x) for x in faiss.vector_to_array(vis_index.id_map)) if vis_index.ntotal > 0 else set()
    faiss_txt_ids = set(int(x) for x in faiss.vector_to_array(txt_index.id_map)) if txt_index.ntotal > 0 else set()

    vis_missing_in_sql = faiss_vis_ids - sql_vis_ids
    vis_missing_in_faiss = sql_vis_ids - faiss_vis_ids

    checks.append((
        "Visual vector IDs resolve bidirectionally",
        len(vis_missing_in_sql) == 0 and len(vis_missing_in_faiss) == 0,
        f"Missing in SQL: {len(vis_missing_in_sql)}, Missing in FAISS: {len(vis_missing_in_faiss)}",
    ))

    txt_missing_in_sql = faiss_txt_ids - sql_txt_ids
    txt_missing_in_faiss = sql_txt_ids - faiss_txt_ids

    checks.append((
        "Text vector IDs resolve bidirectionally",
        len(txt_missing_in_sql) == 0 and len(txt_missing_in_faiss) == 0,
        f"Missing in SQL: {len(txt_missing_in_sql)}, Missing in FAISS: {len(txt_missing_in_faiss)}",
    ))

    # 5. Visual records validation
    cur.execute("""
        SELECT COUNT(*) AS c FROM visual_vectors
        WHERE timestamp_seconds < 0 OR frame_path IS NULL OR frame_path = ''
    """)
    invalid_visual_rows = cur.fetchone()["c"]
    checks.append((
        "Visual rows validity (timestamp >= 0, frame_path present)",
        invalid_visual_rows == 0,
        f"{invalid_visual_rows} invalid rows",
    ))

    # 6. Text records validation
    cur.execute("""
        SELECT COUNT(*) AS c FROM text_vectors
        WHERE start_timestamp < 0 OR end_timestamp < start_timestamp
           OR transcript_text IS NULL OR transcript_text = ''
    """)
    invalid_text_rows = cur.fetchone()["c"]
    checks.append((
        "Text rows validity (start >= 0, end >= start, non-empty text)",
        invalid_text_rows == 0,
        f"{invalid_text_rows} invalid rows",
    ))

    # 7. Uniqueness constraints
    cur.execute("SELECT frame_id, COUNT(*) AS c FROM visual_vectors GROUP BY frame_id HAVING c > 1")
    dup_frames = cur.fetchall()
    checks.append((
        "Visual frame_id uniqueness",
        len(dup_frames) == 0,
        f"{len(dup_frames)} duplicates found",
    ))

    cur.execute("SELECT segment_uid, COUNT(*) AS c FROM text_vectors GROUP BY segment_uid HAVING c > 1")
    dup_segs = cur.fetchall()
    checks.append((
        "Text segment_uid uniqueness",
        len(dup_segs) == 0,
        f"{len(dup_segs)} duplicates found",
    ))

    # 8. Orphan records check
    cur.execute("""
        SELECT COUNT(*) AS c FROM visual_vectors v
        WHERE NOT EXISTS (SELECT 1 FROM videos vid WHERE vid.video_id = v.video_id)
    """)
    orphan_vis = cur.fetchone()["c"]

    cur.execute("""
        SELECT COUNT(*) AS c FROM text_vectors t
        WHERE NOT EXISTS (SELECT 1 FROM videos vid WHERE vid.video_id = t.video_id)
    """)
    orphan_txt = cur.fetchone()["c"]

    checks.append((
        "Orphan-free check (all vectors reference existing videos)",
        orphan_vis == 0 and orphan_txt == 0,
        f"{orphan_vis} orphan visual rows, {orphan_txt} orphan text rows",
    ))

    # 9. Index manifest check
    manifest_data = load_manifest(manifest_file)
    manifest_valid = bool(manifest_data) and "models" in manifest_data and "counts" in manifest_data
    checks.append((
        "Index manifest file validity",
        manifest_valid,
        f"Manifest counts: {manifest_data.get('counts', {}) if manifest_valid else 'Invalid'}",
    ))

    store.close()
    return _print_report(checks)


def _print_report(checks: List[Tuple[str, bool, str]]) -> bool:
    """Format and print verification checklist."""
    all_passed = True
    print("\nChecklist Results:")
    print("----------------------------------------------------------------------")
    for name, passed, details in checks:
        status = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f"{status:<7} | {name:<45} | {details}")
    print("----------------------------------------------------------------------")
    print(f"Overall Status: {'ALL CHECKS PASSED (Integrity OK)' if all_passed else 'INTEGRITY ISSUES DETECTED'}\n")
    return all_passed
