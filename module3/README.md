# Module 3: Multimodal Semantic Query Processing & Segment Retrieval

Part of the **Fully Local, Offline-at-Runtime Multimodal Video Retrieval System**.

Module 3 accepts natural-language search queries, encodes them across isolated visual and textual embedding spaces using the exact neural encoders from Module 2, applies principled relevance gating, resolves timestamped candidate segments by video ID, dynamically fuses evidence across modalities, and produces ranked, timestamped video segments suitable for video playback seek operations.

---

## Architecture Overview

```text
Natural Language Query
        │
        ├──[ SigLIP2 Text Tower ]──────> Visual Space Query Vector (768-d)
        │                                        │
        │                                 FAISS visual.index
        │                                        │
        │                                 Visual Frame Candidates
        │                                        │
        │                                 Sigmoid Calibration & Gate (P >= 0.15)
        │                                        │
        ├──[ MiniLM Text Tower ]───────> Text Space Query Vector (384-d)
        │                                        │
        │                                 FAISS text.index
        │                                        │
        │                                 Speech Transcript Candidates
        │                                        │
        │                                 Monotonic Linear Gate (Cosine >= 0.22)
        │                                        │
        ▼                                        ▼
    Per-Video ID Segment Assembly & Temporal Linkage (Tolerance = frame_interval_seconds)
                                │
                        Score Fusion Engine
        (fusion_mode="weighted" | "noisy_or" + Agreement Bonus + Raw Cosine Tie-Break)
                                │
                    Top-K Ranked Results (K in 3..5)
                                │
                    SearchResponse (Schema 3.1.0)
```

---

## Key Invariants & Design Principles

1. **Strictly Local & Offline at Runtime**:
   - Zero cloud API calls (`HF_HUB_OFFLINE=1`).
   - Pure local execution against SQLite (`metadata.db`) and FAISS (`visual.index`, `text.index`).
2. **Modality Isolation & Independent Encoders**:
   - Visual space: `google/siglip2-base-patch16-224` text tower (768-d).
   - Text space: `sentence-transformers/all-MiniLM-L6-v2` (384-d).
   - Embedding spaces and raw cosine scores are never mixed directly.
3. **Principled Relevance Gating**:
   - Visual: SigLIP sigmoid probability calibration `P(relevant) = sigmoid(scale * cos + bias)` with gate threshold `P >= 0.15`.
   - Text: Monotonic linear calibration with cosine gate threshold `cos >= 0.22`.
   - Uncorrelated queries (e.g., "recipe for baking sourdough bread") are completely rejected at the gate, returning `status="no_match"`.
4. **Configurable Score Fusion**:
   - `fusion_mode="weighted"` (default): Computes `(wv * v + wt * t) / (wv + wt) + agreement_bonus` (capped at 1.0) for corroborated cross-modal segments. Single-modality segments preserve their own calibrated score (missing modality is never treated as 0.0).
   - `fusion_mode="noisy_or"`: Computes `1.0 - (1.0 - v) * (1.0 - t)` for probabilistic independence.
   - Raw cosine scores break ties between identical calibrated scores before RRF rank tie-breaking.
5. **Deterministic Timestamp Spans**:
   - Visual-only matches form approximate spans `[t - interval/2, t + interval/2]`.
   - Speech matches use exact WhisperX transcript intervals `[start_time, end_time]`.
   - Linked matches encompass the unified envelope `[min(start), max(end)]`.

---

## Configuration Reference

Loaded from explicit `--config <PATH>` or default `module3/config.json`:

| Parameter | Type | Default | Valid Range / Options | Description |
|---|---|---|---|---|
| `min_top_k` | int | `3` | `3` | Minimum allowable Top-K result count |
| `max_top_k` | int | `5` | `5` | Maximum allowable Top-K result count |
| `default_top_k` | int | `5` | `3..5` | Default Top-K results returned |
| `fusion_mode` | str | `"weighted"` | `"weighted"`, `"noisy_or"` | Multimodal score fusion strategy |
| `visual_weight` | float | `0.6` | `[0.0, inf)` | Weight for visual calibrated probability |
| `text_weight` | float | `0.4` | `[0.0, inf)` | Weight for speech transcript calibrated score |
| `agreement_bonus` | float | `0.05` | `[0.0, 1.0]` | Additive bonus for corroborated cross-modal segments |
| `link_tolerance_s` | float/null | `null` | `null` or `>= 0.0` | Cross-modal linking window tolerance (`null` = video frame interval) |
| `visual_min_prob` | float | `0.15` | `[0.0, 1.0]` | Visual sigmoid probability relevance gate |
| `text_min_cosine` | float | `0.22` | `[-1.0, 1.0]` | Transcript raw cosine relevance gate |
| `final_min_score` | float | `0.20` | `[0.0, 1.0]` | Final fused score threshold for result inclusion |
| `query_cache_size` | int | `128` | `>= 1` | LRU query vector cache capacity per modality |
| `visual_pool` | int | `50` | `>= 1` | Candidate retrieval pool size from FAISS visual index |
| `text_pool` | int | `50` | `>= 1` | Candidate retrieval pool size from FAISS text index |
| `rrf_k` | int | `60` | `>= 1` | Reciprocal Rank Fusion constant for tie-breaking |

*Note: `dedup_similarity_threshold` was removed as frame de-duplication is performed via temporal adjacency of gated hits.*

---

## Response Schema (`schema_version = "3.1.0"`)

All queries return a structured `SearchResponse` object with JSON-serializable output:

```json
{
  "schema_version": "3.1.0",
  "status": "ok",
  "message": "Found 3 matching segment(s).",
  "notices": [
    "Visual search unavailable for this query; transcript results only."
  ],
  "results": [
    {
      "segment_id": "seg_e9f58d28adfac07e_0000_0005",
      "video_id": "vid_e9f58d28adfac07e",
      "video_filename": "videoplayback.mp4",
      "video_path": "C:\\Users\\Lenovo\\Desktop\\Final Project\\module1\\output\\videos\\videoplayback.mp4",
      "video_file_exists": true,
      "start_seconds": 0.0,
      "end_seconds": 5.0,
      "start_formatted": "00:00:00",
      "end_formatted": "00:00:05",
      "duration_seconds": 5.0,
      "seek_seconds": 0.0,
      "span_type": "exact_speech",
      "resolution_seconds": 0.0,
      "modalities": ["visual", "text"],
      "final_score": 1.0,
      "visual_score": 0.9996,
      "text_score": 1.0,
      "visual_raw_cosine": 0.2096,
      "text_raw_cosine": 0.9767,
      "rrf_score": 0.0328,
      "reason": "Cross-modal match: visual (1.00) and transcript (1.00) corroborate this segment (weighted fused: 1.00, bonus=0.05).",
      "frame_id": "frame_0000",
      "matched_at_seconds": 0.0,
      "transcript_text": "Java 25 introduces scoped values as a preview feature..."
    }
  ],
  "timings_ms": {
    "query_encode_visual_ms": 0.01,
    "query_encode_text_ms": 0.00,
    "search_visual_ms": 28.50,
    "search_text_ms": 8.42,
    "calibrate_fuse_ms": 3.80,
    "segment_build_ms": 0.45,
    "total_ms": 41.20
  },
  "debug": {
    "cache_hit_visual": true,
    "cache_hit_text": true,
    "models_reused": true,
    "index_reused": true
  }
}
```

---

## Real-Model Benchmark & Threshold Evidence

### 1. Empirical Score Distributions (Measured via `tools/tune_thresholds.py`)

Run against real indexed library (`videoplayback.mp4`, `sample.mp4`, `silent.mp4`):

```text
VISUAL MODALITY (SigLIP2):
  Positives (N=3):
    Raw Cosine:       Min=0.1715, Mean=0.1855, Max=0.1974
    Calibrated Prob:  Min=0.9276, Mean=0.9703, Max=0.9958
  Negatives / False Alarms (N=4):
    Raw Cosine:       Min=-0.0253, Mean=0.0172, Max=0.0759
    Calibrated Prob:  Min=0.0000, Mean=0.0001, Max=0.0003

TEXT MODALITY (MiniLM-L6-v2):
  Positives (N=4):
    Raw Cosine:       Min=0.2955, Mean=0.6831, Max=0.9767
    Calibrated Score: Min=0.1592, Mean=0.7315, Max=1.0000
  Negatives / False Alarms (N=4):
    Raw Cosine:       Min=0.0515, Mean=0.1014, Max=0.1701
    Calibrated Score: Min=0.0000, Mean=0.0000, Max=0.0000

Empirical Separation:
  - Visual: Clear decision margin between negative max (0.0759 cos / 0.0003 prob) and positive min (0.1715 cos / 0.9276 prob). Current gate visual_min_prob=0.15 cleanly rejects all negatives while accepting all relevant hits.
  - Text: Clear decision margin between negative max (0.1701 cos / 0.0000 cal) and positive min (0.2955 cos / 0.1592 cal). Current gate text_min_cosine=0.22 cleanly rejects all negatives.
```

### 2. Measured Latency & Telemetry (Measured via `tools/benchmark_latency.py`)

| Stage | Cold Query (First Load) | Warm Query (LRU Cached, N=30) |
|---|---|---|
| Query Encode Visual (SigLIP2) | 28,205.05 ms | 0.01 ms |
| Query Encode Text (MiniLM) | 2,012.73 ms | 0.00 ms |
| FAISS Visual Search (`IndexFlatIP`) | 26.34 ms | 29.07 ms |
| FAISS Text Search (`IndexFlatIP`) | 8.68 ms | 8.62 ms |
| Relevance Gating & Calibration | 3.92 ms | 4.16 ms |
| Segment Assembly & Ranking | 1.22 ms | 0.47 ms |
| **Total End-to-End Latency** | **30,258.02 ms** | **42.38 ms** |

*Model and Index Re-use*: Models loaded exactly 1 time across the entire runtime session. Query cache hit rate: 83.3% (30/36).

### 3. Fusion Mode Ranking Comparison (`noisy_or` vs `weighted`)

From `module3/output/fusion_comparison_report.txt`:

| Query | Rank | Segment Window | Noisy-OR Score | Weighted Score | Modalities |
|---|---|---|---|---|---|
| "What are scoped values..." | 1 | `[00:00:00 - 00:00:05]` | 1.0000 | 1.0000 | Visual + Text |
| | 2 | `[00:00:08 - 00:00:14]` | 0.9996 | 0.9542 | Visual + Text |
| | 3 | `[00:00:20 - 00:00:26]` | 0.8845 | 0.6468 | Visual + Text |
| "a purple title card..." | 1 | `[00:00:00 - 00:00:05]` | 0.9640 | 0.6786 | Visual |
| "slide with Code Example" | 1 | `[00:00:19 - 00:00:25]` | 0.4340 | 0.4340 | Visual |

*Key Distinction*: In `weighted` mode, modality weights and agreement bonuses directly differentiate evidence quality, preventing saturation and allowing single-modality matches to retain their principled calibrated score without distortion.

---

## CLI Usage Guide

All commands support `--config <PATH>`:

```bash
# 1. System Health Check
python module3/main.py health --config module3/config.json
python module3/main.py health --json

# 2. List Indexed Videos
python module3/main.py videos --config module3/config.json

# 3. Multimodal Search
python module3/main.py search "What are scoped values in Java?" --top-k 5
python module3/main.py search "purple title card" --top-k 3 --json
python module3/main.py search "Code Example" --visual-weight 0.8 --text-weight 0.2 --debug

# 4. Interactive Search REPL
python module3/main.py interactive
```

---

## Test Suite Status

Comprehensive test suite verification:

| Test Suite | Total Tests | Passed | Failed | Skipped | Notes |
|---|---|---|---|---|---|
| `module1` | 25 | 25 | 0 | 0 | Video validation, audio/frame extraction, WhisperX |
| `module2` | 29 | 28 | 0 | 1 | `test_real_models_sample_indexing` skipped (fixture video absent) |
| `module3` | 61 | 61 | 0 | 0 | Full unit, integration, CLI, scenarios, and real-model suite |

---

## Known Limitations

1. **2-Second Frame Sampling Granularity**:
   Frames are sampled every 2.0 seconds by Module 1. Visual-only segment bounds are approximate spans centered at `t ± interval/2` (`±1.0s`), which can slightly precede or follow visual scene transitions.
2. **WhisperX `tiny` Transcript Resolution**:
   Speech transcripts generated by WhisperX `tiny` may misspell obscure technical terms or produce noisy timestamps during rapid overlapping dialogue.
3. **CPU Cold-Start Latency**:
   On standard CPU architectures, initial SigLIP2 and MiniLM model loading requires ~30.2 seconds. Subsequent warm queries execute in ~42.4 ms via LRU caching and in-memory indexes.
