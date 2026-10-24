"""Incremental, duplicate-safe indexing orchestrator for Module 2.

Discovers completed Module 1 video outputs and performs incremental synchronization:
- Compares items against SQLite by content_hash, timestamps, and model version.
- Skips unchanged items (zero compute, zero model calls).
- Re-embeds modified items and prunes removed frames/segments.
- Purges older video entries if a different video_id points to the same video path.
- Handles corrupt images gracefully without aborting batches or videos.
- Atomically synchronizes in-memory FAISS indices and persists them to disk.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from .config import Module2Config
from .embedders import (
    MiniLMTextEmbedder,
    SigLIPVisualEmbedder,
    TextEmbedderProtocol,
    VisualEmbedderProtocol,
)
from .logging_config import get_logger
from .m1_reader import iter_videos, read_master_metadata
from .metadata_store import INDEX_SCHEMA_VERSION, MetadataStore
from .schema import IndexFailure, TextItem, VideoRecord, VisualItem
from .vector_index import (
    VectorIndexManager,
    load_manifest,
    load_or_rebuild_index,
    save_manifest,
)

logger = get_logger("indexer")


class IndexSummary:
    """Tracks cumulative counters across an indexing run."""

    def __init__(self) -> None:
        self.videos_processed: int = 0
        self.videos_skipped: int = 0
        self.videos_purged: int = 0
        self.visual_embedded: int = 0
        self.visual_skipped: int = 0
        self.visual_removed: int = 0
        self.visual_failed: int = 0
        self.text_embedded: int = 0
        self.text_skipped: int = 0
        self.text_removed: int = 0
        self.text_failed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "videos": {
                "processed": self.videos_processed,
                "skipped": self.videos_skipped,
                "purged": self.videos_purged,
            },
            "visual": {
                "embedded": self.visual_embedded,
                "skipped": self.visual_skipped,
                "removed": self.visual_removed,
                "failed": self.visual_failed,
            },
            "text": {
                "embedded": self.text_embedded,
                "skipped": self.text_skipped,
                "removed": self.text_removed,
                "failed": self.text_failed,
            },
        }


class VideoIndexer:
    """High-level indexer orchestrating metadata extraction, embedding, and storage."""

    def __init__(
        self,
        config: Module2Config,
        visual_embedder: Optional[VisualEmbedderProtocol] = None,
        text_embedder: Optional[TextEmbedderProtocol] = None,
        store: Optional[MetadataStore] = None,
    ) -> None:
        self.config = config
        self.config.ensure_directories()

        db_path = self.config.vector_db_dir / "metadata.db"
        self.store = store or MetadataStore(db_path)

        # Lazy embedders: instantiate default if not injected
        self._visual_embedder = visual_embedder
        self._text_embedder = text_embedder

        self.visual_index_file = self.config.vector_db_dir / "visual.index"
        self.text_index_file = self.config.vector_db_dir / "text.index"
        self.manifest_file = self.config.vector_db_dir / "index_manifest.json"

        self.vis_mgr: Optional[VectorIndexManager] = None
        self.txt_mgr: Optional[VectorIndexManager] = None

    @property
    def visual_embedder(self) -> VisualEmbedderProtocol:
        if self._visual_embedder is None:
            self._visual_embedder = SigLIPVisualEmbedder(
                model_name=self.config.visual_model_name,
                device=self.config.device,
                batch_size=self.config.visual_batch_size or 16,
                models_dir=self.config.models_dir,
                offline_mode=self.config.offline_mode,
            )
        return self._visual_embedder

    @property
    def text_embedder(self) -> TextEmbedderProtocol:
        if self._text_embedder is None:
            self._text_embedder = MiniLMTextEmbedder(
                model_name=self.config.text_model_name,
                device=self.config.device,
                batch_size=self.config.text_batch_size,
                models_dir=self.config.models_dir,
                offline_mode=self.config.offline_mode,
            )
        return self._text_embedder

    def init_indices(self) -> None:
        """Initialize metadata models and load/verify FAISS indices."""
        vis_dim = self.visual_embedder.dim
        txt_dim = self.text_embedder.dim

        # Validate database model compatibility
        self.store.check_or_set_models(
            visual_model=self.visual_embedder.model_name,
            visual_dim=vis_dim,
            text_model=self.text_embedder.model_name,
            text_dim=txt_dim,
        )

        # Load or rebuild FAISS indices
        self.vis_mgr = load_or_rebuild_index(
            modality="visual",
            index_file=self.visual_index_file,
            manifest_file=self.manifest_file,
            store=self.store,
            expected_dim=vis_dim,
            expected_model=self.visual_embedder.model_name,
        )

        self.txt_mgr = load_or_rebuild_index(
            modality="text",
            index_file=self.text_index_file,
            manifest_file=self.manifest_file,
            store=self.store,
            expected_dim=txt_dim,
            expected_model=self.text_embedder.model_name,
        )

    def save_indices(self) -> None:
        """Persist FAISS indices and update index_manifest.json."""
        if self.vis_mgr is not None:
            self.vis_mgr.save_index(self.visual_index_file)
        if self.txt_mgr is not None:
            self.txt_mgr.save_index(self.text_index_file)

        manifest_data = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "models": {
                "visual": self.visual_embedder.model_name,
                "text": self.text_embedder.model_name,
            },
            "dims": {
                "visual": self.visual_embedder.dim,
                "text": self.text_embedder.dim,
            },
            "counts": {
                "visual": self.vis_mgr.ntotal if self.vis_mgr else 0,
                "text": self.txt_mgr.ntotal if self.txt_mgr else 0,
            },
            "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        save_manifest(self.manifest_file, manifest_data)
        logger.info(
            "Saved vector indices and manifest: %d visual vectors, %d text vectors",
            manifest_data["counts"]["visual"],
            manifest_data["counts"]["text"],
        )

    def rebuild_all(self) -> None:
        """Wipe database and vector indices, preparing for complete clean re-indexing."""
        logger.info("Rebuilding vector storage: wiping metadata database and FAISS index files...")
        self.store.close()

        # Remove database files
        db_path = self.config.vector_db_dir / "metadata.db"
        for suffix in ("", "-wal", "-shm"):
            p = Path(f"{db_path}{suffix}")
            if p.is_file():
                try:
                    p.unlink()
                except Exception as exc:
                    logger.warning("Could not remove DB file %s: %s", p, exc)

        # Remove indices and manifest
        for p in (self.visual_index_file, self.text_index_file, self.manifest_file):
            if p.is_file():
                try:
                    p.unlink()
                except Exception as exc:
                    logger.warning("Could not remove index file %s: %s", p, exc)

        # Re-initialize
        self.store = MetadataStore(db_path)
        self.vis_mgr = None
        self.txt_mgr = None

    def prune_missing_videos(self) -> int:
        """Delete videos from the database whose Module 1 metadata JSON no longer exists."""
        pruned_count = 0
        db_videos = self.store.list_videos()
        for v in db_videos:
            m1_path = Path(v["m1_metadata_path"])
            if not m1_path.is_file():
                vid_id = v["video_id"]
                logger.info(
                    "Pruning video %s from database: Module 1 metadata JSON no longer exists at %s",
                    vid_id,
                    m1_path,
                )
                vis_ids, txt_ids = self.store.delete_video(vid_id)
                if self.vis_mgr and vis_ids:
                    self.vis_mgr.remove(vis_ids)
                if self.txt_mgr and txt_ids:
                    self.txt_mgr.remove(txt_ids)
                pruned_count += 1
        return pruned_count

    def index_video(
        self,
        video_record: VideoRecord,
        visual_items: List[VisualItem],
        text_items: List[TextItem],
        failures: List[IndexFailure],
        force_reindex: bool = False,
        visual_only: bool = False,
        text_only: bool = False,
        summary: Optional[IndexSummary] = None,
    ) -> None:
        """Incrementally index a single video.

        All SQLite operations for this video are committed in ONE atomic transaction.
        In-memory FAISS indices are updated immediately after commit.
        """
        vid_id = video_record.video_id
        logger.info("--- Processing video %s (%s) ---", vid_id, video_record.video_filename)

        # 1. Path uniqueness check: a path holds only one video
        existing_vid_for_path = self.store.find_video_by_path(video_record.video_path)
        if existing_vid_for_path and existing_vid_for_path != vid_id:
            logger.warning(
                "Video path '%s' previously mapped to video_id '%s'. Purging older video entry.",
                video_record.video_path,
                existing_vid_for_path,
            )
            v_ids, t_ids = self.store.delete_video(existing_vid_for_path)
            if self.vis_mgr and v_ids:
                self.vis_mgr.remove(v_ids)
            if self.txt_mgr and t_ids:
                self.txt_mgr.remove(t_ids)
            if summary:
                summary.videos_purged += 1

        existing_vis = self.store.get_existing_visual_items(vid_id)
        existing_txt = self.store.get_existing_text_items(vid_id)

        # Items to process
        new_vis_items: List[VisualItem] = []
        stale_vis_vector_ids: List[int] = []

        new_txt_items: List[TextItem] = []
        stale_txt_vector_ids: List[int] = []

        # 2. Reconcile Visual Items
        if not text_only:
            incoming_frame_ids: Set[str] = set()
            for v_item in visual_items:
                incoming_frame_ids.add(v_item.frame_id)
                old = existing_vis.get(v_item.frame_id)

                if old is None or force_reindex:
                    if old is not None:
                        stale_vis_vector_ids.append(old["vector_id"])
                    new_vis_items.append(v_item)
                else:
                    # Compare content hash, timestamp, and model
                    same_hash = old["content_hash"] == v_item.content_hash
                    same_ts = abs(float(old["timestamp_seconds"]) - v_item.timestamp_seconds) < 1e-4
                    same_model = old["embedding_model"] == self.visual_embedder.model_name

                    if same_hash and same_ts and same_model:
                        if summary:
                            summary.visual_skipped += 1
                    else:
                        stale_vis_vector_ids.append(old["vector_id"])
                        new_vis_items.append(v_item)

            # Detect stale frames no longer in Module 1 output
            for f_id, old_data in existing_vis.items():
                if f_id not in incoming_frame_ids:
                    stale_vis_vector_ids.append(old_data["vector_id"])

        # 3. Reconcile Text Items
        if not visual_only:
            incoming_seg_uids: Set[str] = set()
            for t_item in text_items:
                incoming_seg_uids.add(t_item.segment_uid)
                old = existing_txt.get(t_item.segment_uid)

                if old is None or force_reindex:
                    if old is not None:
                        stale_txt_vector_ids.append(old["vector_id"])
                    new_txt_items.append(t_item)
                else:
                    same_hash = old["content_hash"] == t_item.content_hash
                    same_start = abs(float(old["start_timestamp"]) - t_item.start_timestamp) < 1e-4
                    same_end = abs(float(old["end_timestamp"]) - t_item.end_timestamp) < 1e-4
                    same_model = old["embedding_model"] == self.text_embedder.model_name

                    if same_hash and same_start and same_end and same_model:
                        if summary:
                            summary.text_skipped += 1
                    else:
                        stale_txt_vector_ids.append(old["vector_id"])
                        new_txt_items.append(t_item)

            # Detect stale segments
            for seg_uid, old_data in existing_txt.items():
                if seg_uid not in incoming_seg_uids:
                    stale_txt_vector_ids.append(old_data["vector_id"])

        # 4. Generate Embeddings for New Items
        embedded_vis_data: List[Tuple[VisualItem, np.ndarray]] = []
        if new_vis_items:
            logger.info("Embedding %d visual frames for %s...", len(new_vis_items), vid_id)
            # Batch embedding
            vis_batch_size = self.config.visual_batch_size or 16
            for idx in range(0, len(new_vis_items), vis_batch_size):
                batch_items = new_vis_items[idx : idx + vis_batch_size]
                logger.info("video %s: frames %d/%d", vid_id, idx + len(batch_items), len(new_vis_items))

                # Open images, gracefully skipping corrupt ones
                valid_batch_items: List[VisualItem] = []
                loaded_images: List[Image.Image] = []

                for item in batch_items:
                    try:
                        with Image.open(item.frame_path) as img:
                            loaded_images.append(img.convert("RGB"))
                            valid_batch_items.append(item)
                    except Exception as img_exc:
                        msg = f"Failed to open image file {item.frame_path}: {img_exc}"
                        logger.warning(msg)
                        failures.append(
                            IndexFailure(
                                video_id=vid_id,
                                modality="visual",
                                item_key=item.frame_id,
                                reason=msg,
                                failed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            )
                        )
                        if summary:
                            summary.visual_failed += 1

                if loaded_images:
                    try:
                        batch_embeddings = self.visual_embedder.embed_images(loaded_images)
                        for item, emb in zip(valid_batch_items, batch_embeddings):
                            embedded_vis_data.append((item, emb))
                    except Exception as emb_exc:
                        logger.error(
                            "Batch embedding failed (%s); retrying item by item.", emb_exc
                        )
                        for item, img in zip(valid_batch_items, loaded_images):
                            try:
                                single_emb = self.visual_embedder.embed_images([img])[0]
                                embedded_vis_data.append((item, single_emb))
                            except Exception as single_exc:
                                msg = f"Single item embedding failed for {item.frame_id}: {single_exc}"
                                logger.error(msg)
                                failures.append(
                                    IndexFailure(
                                        video_id=vid_id,
                                        modality="visual",
                                        item_key=item.frame_id,
                                        reason=msg,
                                        failed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    )
                                )
                                if summary:
                                    summary.visual_failed += 1

        embedded_txt_data: List[Tuple[TextItem, np.ndarray]] = []
        if new_txt_items:
            logger.info("Embedding %d transcript segments for %s...", len(new_txt_items), vid_id)
            texts = [item.transcript_text for item in new_txt_items]
            try:
                txt_embeddings = self.text_embedder.embed_texts(texts)
                for item, emb in zip(new_txt_items, txt_embeddings):
                    embedded_txt_data.append((item, emb))
            except Exception as txt_exc:
                logger.error("Text embedding batch failed (%s); retrying segment by segment.", txt_exc)
                for item in new_txt_items:
                    try:
                        emb = self.text_embedder.embed_texts([item.transcript_text])[0]
                        embedded_txt_data.append((item, emb))
                    except Exception as s_exc:
                        msg = f"Text segment embedding failed for {item.segment_uid}: {s_exc}"
                        logger.error(msg)
                        failures.append(
                            IndexFailure(
                                video_id=vid_id,
                                modality="text",
                                item_key=item.segment_uid,
                                reason=msg,
                                failed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            )
                        )
                        if summary:
                            summary.text_failed += 1

        # 5. Atomic SQLite Commit
        new_vis_ids: List[int] = []
        new_vis_vecs: List[np.ndarray] = []

        new_txt_ids: List[int] = []
        new_txt_vecs: List[np.ndarray] = []

        with self.store.transaction():
            # Update video record
            self.store.upsert_video(video_record)

            # Purge stale visual rows
            for old_vid in stale_vis_vector_ids:
                self.store.delete_visual_vector(old_vid)

            # Purge stale text rows
            for old_tid in stale_txt_vector_ids:
                self.store.delete_text_vector(old_tid)

            # Insert new visual vectors
            for item, emb in embedded_vis_data:
                vector_id = self.store.insert_visual_vector(
                    item=item,
                    embedding=emb,
                    embedding_model=self.visual_embedder.model_name,
                )
                new_vis_ids.append(vector_id)
                new_vis_vecs.append(emb)
                # Successful item: clear prior failure records
                self.store.clear_failures(vid_id, modality="visual", item_key=item.frame_id)

            # Insert new text vectors
            for item, emb in embedded_txt_data:
                vector_id = self.store.insert_text_vector(
                    item=item,
                    embedding=emb,
                    embedding_model=self.text_embedder.model_name,
                )
                new_txt_ids.append(vector_id)
                new_txt_vecs.append(emb)
                self.store.clear_failures(vid_id, modality="text", item_key=item.segment_uid)

            # Record any new failures
            for fail in failures:
                self.store.record_failure(fail)

        # 6. Update in-memory FAISS indices
        if stale_vis_vector_ids and self.vis_mgr:
            self.vis_mgr.remove(stale_vis_vector_ids)
            if summary:
                summary.visual_removed += len(stale_vis_vector_ids)

        if new_vis_ids and self.vis_mgr:
            self.vis_mgr.add(
                vectors=np.vstack(new_vis_vecs),
                vector_ids=np.array(new_vis_ids, dtype=np.int64),
            )
            if summary:
                summary.visual_embedded += len(new_vis_ids)

        if stale_txt_vector_ids and self.txt_mgr:
            self.txt_mgr.remove(stale_txt_vector_ids)
            if summary:
                summary.text_removed += len(stale_txt_vector_ids)

        if new_txt_ids and self.txt_mgr:
            self.txt_mgr.add(
                vectors=np.vstack(new_txt_vecs),
                vector_ids=np.array(new_txt_ids, dtype=np.int64),
            )
            if summary:
                summary.text_embedded += len(new_txt_ids)

        if summary:
            summary.videos_processed += 1

        logger.info(
            "Finished indexing video %s (visual: +%d/-%d, text: +%d/-%d)",
            vid_id,
            len(new_vis_ids),
            len(stale_vis_vector_ids),
            len(new_txt_ids),
            len(stale_txt_vector_ids),
        )

    def run_indexing(
        self,
        video_id: Optional[str] = None,
        force_reindex: bool = False,
        rebuild: bool = False,
        prune_missing: bool = False,
        visual_only: bool = False,
        text_only: bool = False,
        save_each: bool = False,
    ) -> IndexSummary:
        """Execute complete indexing workflow according to specified flags."""
        summary = IndexSummary()

        if rebuild:
            self.rebuild_all()

        self.init_indices()

        if prune_missing:
            pruned = self.prune_missing_videos()
            logger.info("Pruned %d missing video(s) from store.", pruned)

        # Gather target videos
        if video_id:
            m1_json = self.config.module1_output_dir / "metadata" / f"{video_id}.json"
            if not m1_json.is_file():
                err = f"Module 1 metadata file not found for video_id '{video_id}': {m1_json}"
                logger.error(err)
                raise FileNotFoundError(err)

            parsed = read_master_metadata(m1_json, self.config)
            if parsed is None:
                logger.warning("Video %s could not be parsed or is incomplete; skipping.", video_id)
                summary.videos_skipped += 1
                return summary

            targets = [parsed]
        else:
            targets = list(iter_videos(self.config))

        total_targets = len(targets)
        logger.info("Indexing %d candidate video(s)...", total_targets)

        for idx, (v_rec, v_items, t_items, fails) in enumerate(targets, 1):
            logger.info("[Video %d/%d] %s", idx, total_targets, v_rec.video_id)
            self.index_video(
                video_record=v_rec,
                visual_items=v_items,
                text_items=t_items,
                failures=fails,
                force_reindex=force_reindex,
                visual_only=visual_only,
                text_only=text_only,
                summary=summary,
            )
            if save_each:
                self.save_indices()

        # Post-indexing integrity verification and self-healing (R4b)
        healed = self.verify_and_heal_integrity(
            targets=targets,
            summary=summary,
            visual_only=visual_only,
            text_only=text_only,
        )
        if healed:
            logger.info("Integrity self-healing completed; saving persistent vector indices.")

        # Final persistent save
        self.save_indices()

        logger.info("==================================================")
        logger.info("Indexing Run Completed. Summary:")
        logger.info("  Videos: Processed=%d, Skipped=%d, Purged=%d", summary.videos_processed, summary.videos_skipped, summary.videos_purged)
        logger.info("  Visual: Embedded=%d, Skipped=%d, Removed=%d, Failed=%d", summary.visual_embedded, summary.visual_skipped, summary.visual_removed, summary.visual_failed)
        logger.info("  Text:   Embedded=%d, Skipped=%d, Removed=%d, Failed=%d", summary.text_embedded, summary.text_skipped, summary.text_removed, summary.text_failed)
        logger.info("==================================================")

        return summary

    def verify_and_heal_integrity(
        self,
        targets: List[Tuple[VideoRecord, List[VisualItem], List[TextItem], List[IndexFailure]]],
        summary: Optional[IndexSummary] = None,
        visual_only: bool = False,
        text_only: bool = False,
    ) -> bool:
        """Post-indexing integrity check: verifies SQLite has expected count of vectors per video.

        If missing vectors are detected, logs a WARNING and self-heals by re-embedding them.
        Returns True if healing was performed.
        """
        healed_any = False
        for v_rec, v_items, t_items, fails in targets:
            vid_id = v_rec.video_id

            # Check visual vectors
            if not text_only and v_items:
                existing_vis = self.store.get_existing_visual_items(vid_id)
                if len(existing_vis) < len(v_items):
                    logger.warning(
                        "Post-index integrity warning for video %s: expected %d visual vectors, but found %d. Self-healing missing items...",
                        vid_id,
                        len(v_items),
                        len(existing_vis),
                    )
                    missing_vis = [item for item in v_items if item.frame_id not in existing_vis]
                    if missing_vis:
                        healed_vis_data: List[Tuple[VisualItem, np.ndarray]] = []
                        vis_batch_size = self.config.visual_batch_size or 16
                        for idx in range(0, len(missing_vis), vis_batch_size):
                            batch_items = missing_vis[idx : idx + vis_batch_size]
                            loaded_imgs: List[Image.Image] = []
                            valid_batch_items: List[VisualItem] = []
                            for item in batch_items:
                                try:
                                    with Image.open(item.frame_path) as img:
                                        loaded_imgs.append(img.convert("RGB"))
                                        valid_batch_items.append(item)
                                except Exception as exc:
                                    logger.warning("Integrity self-heal: cannot open %s: %s", item.frame_path, exc)
                            if loaded_imgs:
                                batch_embs = self.visual_embedder.embed_images(loaded_imgs)
                                for item, emb in zip(valid_batch_items, batch_embs):
                                    healed_vis_data.append((item, emb))

                        if healed_vis_data:
                            new_ids: List[int] = []
                            new_vecs: List[np.ndarray] = []
                            with self.store.transaction():
                                for item, emb in healed_vis_data:
                                    vec_id = self.store.insert_visual_vector(
                                        item=item,
                                        embedding=emb,
                                        embedding_model=self.visual_embedder.model_name,
                                    )
                                    new_ids.append(vec_id)
                                    new_vecs.append(emb)
                            if self.vis_mgr and new_ids:
                                self.vis_mgr.add(vectors=np.vstack(new_vecs), vector_ids=new_ids)
                            if summary:
                                summary.visual_embedded += len(new_ids)
                            healed_any = True
                            logger.info("Self-healed %d missing visual vectors for video %s.", len(new_ids), vid_id)

            # Check text vectors
            if not visual_only and t_items:
                existing_txt = self.store.get_existing_text_items(vid_id)
                if len(existing_txt) < len(t_items):
                    logger.warning(
                        "Post-index integrity warning for video %s: expected %d text vectors, but found %d. Self-healing missing items...",
                        vid_id,
                        len(t_items),
                        len(existing_txt),
                    )
                    missing_txt = [item for item in t_items if item.segment_uid not in existing_txt]
                    if missing_txt:
                        healed_txt_data: List[Tuple[TextItem, np.ndarray]] = []
                        texts = [item.transcript_text for item in missing_txt]
                        txt_embs = self.text_embedder.embed_texts(texts)
                        for item, emb in zip(missing_txt, txt_embs):
                            healed_txt_data.append((item, emb))

                        if healed_txt_data:
                            new_tids: List[int] = []
                            new_tvecs: List[np.ndarray] = []
                            with self.store.transaction():
                                for item, emb in healed_txt_data:
                                    vec_id = self.store.insert_text_vector(
                                        item=item,
                                        embedding=emb,
                                        embedding_model=self.text_embedder.model_name,
                                    )
                                    new_tids.append(vec_id)
                                    new_tvecs.append(emb)
                            if self.txt_mgr and new_tids:
                                self.txt_mgr.add(vectors=np.vstack(new_tvecs), vector_ids=new_tids)
                            if summary:
                                summary.text_embedded += len(new_tids)
                            healed_any = True
                            logger.info("Self-healed %d missing text vectors for video %s.", len(new_tids), vid_id)

        return healed_any
