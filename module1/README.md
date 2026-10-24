# Module 1: Video Ingestion, Representative Frame Extraction, and Audio Transcription

Part of the **Multimodal Semantic Indexing and Neural Retrieval for Local Video Libraries** system.

Module 1 is responsible for the offline ingestion and decomposition of local video files (`.mp4` and `.mkv`). It extracts representative, time-sampled visual frames, isolates audio tracks, and generates word-aligned speech transcripts using a local WhisperX model. All outputs are stored in structured, deterministic formats to be consumed by downstream modules (visual embedding, text embedding, vector storage, and neural retrieval).

---

## Architecture Overview

```
                      Local Video (.mp4 / .mkv)
                                 │
                                 ▼
                     Integrity & Stream Validation
                                 │
                                 ▼
                    Deterministic ID Generation
                          (e.g. vid_d3147dcb7284cc89)
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
                ▼                                 ▼
     Representative Frames                  Audio Track
        (Time Sampling)                  (Extracted ONCE)
                │                                 │
          OpenCV Decoding                   FFmpeg (WAV)
                │                                 │
     Timestamped Frame JPEGs                      ▼
  output/frames/<video_id>/               Local WhisperX ASR
                                                  │
                                          Silero VAD + Alignment
                                                  │
                                                  ▼
                                      Timestamped Transcript JSON
                                     output/transcripts/<video_id>.json
```

---

## 1. Python Environment Setup

Module 1 is tested and verified on **Python 3.10+** (tested on Python 3.13 on Windows 11).

### Create and Activate Virtual Environment

```powershell
# Navigate to the module directory
cd "C:\Users\Lenovo\Desktop\Final Project\module1"

# Create a virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Or on Command Prompt (CMD)
.venv\Scripts\activate.bat
```

---

## 2. Installing Dependencies

Install all core dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

### Dependencies Included:
- `opencv-python>=4.8.0`: Time-based video decoding and frame export.
- `whisperx>=3.1.0`: Faster-Whisper ASR with Silero VAD and Phoneme-level word alignment.
- `torch>=2.0.0` & `torchaudio>=2.0.0`: Local neural inference backend.
- `numpy>=1.24.0`: Array processing and numerical operations.
- `pytest>=7.0.0`: Comprehensive unit and integration test suite.

---

## 3. Installing FFmpeg on Windows & Verifying PATH

FFmpeg is strictly required for container introspection and single-pass audio extraction.

### Option A: Using Winget (Recommended)
```powershell
winget install Gyan.FFmpeg
```

### Option B: Manual Installation
1. Download `ffmpeg-release-essentials.zip` from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
2. Extract the archive to `C:\ffmpeg`.
3. Add `C:\ffmpeg\bin` to your Windows System `PATH` environment variable.
4. Open a new PowerShell terminal and verify:

```powershell
ffmpeg -version
ffprobe -version
```

Both commands should display version banners (e.g., `ffmpeg version 8.x ...`).

---

## 4. WhisperX Setup & Model Download

Module 1 runs **100% offline at runtime**. Model weights are downloaded automatically on the first run and cached in your user profile:

- **ASR Model Cache:** `~/.cache/huggingface/hub/models--Systran--faster-whisper-<model_name>`
- **VAD Cache:** `~/.cache/torch/hub/snakers4_silero-vad_master`
- **Wav2Vec2 Alignment Models:** `~/.cache/huggingface/hub/models--*wav2vec2*`

### Hardware Acceleration (CUDA vs. CPU)
- **Automatic Device Detection:** If an NVIDIA GPU with CUDA is detected, `cuda` with `float16` compute type is automatically used.
- **CPU Fallback:** On systems without CUDA, `cpu` with `int8` quantization is used, delivering fast, memory-efficient inference without GPU hardware.
- **Offline VAD:** Uses `vad_method="silero"`, which operates fully locally without requiring a Hugging Face user access token.

---

## 5. Input Directory Structure

Place raw video files (`.mp4` or `.mkv`) in:

```
module1/input/videos/
```

Nested subdirectories are automatically supported (e.g. `module1/input/videos/lectures/lecture_01.mp4`).

---

## 6. Running Module 1 CLI

The command-line interface is exposed via `main.py`:

### Batch Mode (All Discovered Videos)
Process all `.mp4` and `.mkv` files in `input/videos/`:

```powershell
python main.py --all
```

### Single Video Mode
Process a specific video file:

```powershell
python main.py --video "input/videos/sample_presentation.mp4"
```

### Force Reprocessing (`--force`)
By default, Module 1 checks if a video has already completed and skips it. Pass `--force` to reprocess and overwrite existing outputs:

```powershell
python main.py --video "input/videos/sample_presentation.mp4" --force
python main.py --all --force
```

### Customizing Sampling Interval
To sample frames every 3.5 seconds instead of the default 2.0 seconds:

```powershell
python main.py --all --interval 3.5
```

### Verbose Debug Logging
```powershell
python main.py --all -v
```

---

## 7. Storage Structure of Extracted Outputs

All outputs reside within `module1/output/`:

```
module1/
├── input/
│   └── videos/
│       ├── sample_presentation.mp4
│       └── sample_silent_clip.mkv
└── output/
    ├── frames/
    │   └── <video_id>/
    │       ├── frame_0000_0.000s.jpg
    │       ├── frame_0001_2.000s.jpg
    │       ├── frame_0002_4.000s.jpg
    │       └── frame_0003_6.000s.jpg
    ├── audio/
    │   └── <video_id>.wav
    ├── transcripts/
    │   └── <video_id>.json
    └── metadata/
        └── <video_id>.json
```

---

## 8. Timestamp Representation & Alignment Standards

Every timestamp is explicitly recorded as a floating-point number representing **true elapsed seconds** from the start of the source video:

1. **Frame Timestamps:**
   - Filename format: `frame_<sample_index:04d>_<timestamp_seconds:.3f}s.jpg`
   - Example: `frame_0002_4.000s.jpg` (sampled at exactly 4.000 seconds).
   - The timestamp is computed using hardware container positions (`CAP_PROP_POS_MSEC` / `frame_num / fps`) to ensure true temporal alignment.

2. **Transcript Timestamps:**
   - Segment timestamps: `start_time` and `end_time` in seconds.
   - Word timestamps: `start` and `end` in seconds with acoustic alignment score `score`.

---

## 9. Sample Transcript JSON Structure

Example of `output/transcripts/<video_id>.json`:

```json
{
  "video_id": "vid_d3147dcb7284cc89",
  "audio_path": "C:\\Users\\...\\module1\\output\\audio\\vid_d3147dcb7284cc89.wav",
  "language": "en",
  "has_word_alignment": true,
  "segment_count": 2,
  "word_count": 18,
  "segments": [
    {
      "segment_id": 0,
      "start_time": 0.17,
      "end_time": 2.856,
      "text": "Welcome to the Multimodal Video Retrieval Project.",
      "words": [
        {
          "word": "Welcome",
          "start": 0.17,
          "end": 0.531,
          "score": 0.845
        },
        {
          "word": "to",
          "start": 0.571,
          "end": 0.671,
          "score": 0.834
        },
        {
          "word": "the",
          "start": 0.691,
          "end": 0.771,
          "score": 0.828
        },
        {
          "word": "Multimodal",
          "start": 0.812,
          "end": 1.413,
          "score": 0.839
        },
        {
          "word": "Video",
          "start": 1.453,
          "end": 1.774,
          "score": 0.948
        },
        {
          "word": "Retrieval",
          "start": 1.834,
          "end": 2.315,
          "score": 0.781
        },
        {
          "word": "Project.",
          "start": 2.355,
          "end": 2.856,
          "score": 0.966
        }
      ]
    }
  ]
}
```

---

## 10. Sample Consolidated Metadata JSON Structure

Example of `output/metadata/<video_id>.json`:

```json
{
  "video_id": "vid_d3147dcb7284cc89",
  "source_video_path": "C:\\Users\\...\\sample_presentation.mp4",
  "file_name": "sample_presentation.mp4",
  "file_size_bytes": 149970,
  "processing_status": "completed",
  "error_message": null,
  "processed_at": "2026-10-08T14:42:08.749869+00:00",
  "pipeline_duration_seconds": 31.34,
  "video_metadata": {
    "video_id": "vid_d3147dcb7284cc89",
    "file_name": "sample_presentation.mp4",
    "file_size_bytes": 149970,
    "file_extension": ".mp4",
    "duration_seconds": 8.0,
    "width": 640,
    "height": 360,
    "fps": 25.0,
    "frame_count": 200,
    "video_codec": "h264",
    "has_audio": true,
    "audio_codec": "aac",
    "audio_channels": 1,
    "audio_sample_rate": 22050
  },
  "frame_extraction": {
    "frame_interval_seconds": 2.0,
    "extracted_frame_count": 4,
    "frames_dir": "C:\\Users\\...\\output\\frames\\vid_d3147dcb7284cc89",
    "frames": [
      {
        "video_id": "vid_d3147dcb7284cc89",
        "frame_index": 0,
        "source_frame_number": 0,
        "timestamp_seconds": 0.0,
        "file_path": "C:\\Users\\...\\output\\frames\\vid_d3147dcb7284cc89\\frame_0000_0.000s.jpg",
        "width": 640,
        "height": 360
      }
    ]
  },
  "audio_extraction": {
    "has_audio": true,
    "audio_path": "C:\\Users\\...\\output\\audio\\vid_d3147dcb7284cc89.wav",
    "sample_rate": 16000,
    "channels": 1,
    "file_size_bytes": 255684
  },
  "transcription": {
    "has_audio": true,
    "transcript_path": "C:\\Users\\...\\output\\transcripts\\vid_d3147dcb7284cc89.json",
    "language": "en",
    "segment_count": 2,
    "word_count": 18,
    "has_word_alignment": true
  }
}
```

---

## 11. Troubleshooting Common Errors

### 1. `ffmpeg` or `ffprobe` Not Found
- **Symptom:** `FileNotFoundError: [WinError 2] The system cannot find the file specified` or `ffprobe failed`.
- **Solution:** Ensure FFmpeg is installed and `C:\ffmpeg\bin` is added to your user/system PATH. Restart your PowerShell prompt. You can also override the paths in `Module1Config(ffmpeg_path=r"C:\path\to\ffmpeg.exe")`.

### 2. Video Has No Audio Track
- **Behavior:** Normal and expected. Module 1 checks whether an audio stream exists via `ffprobe`. If missing, it logs `Video has no audio track. Gracefully skipping audio extraction.` and finishes with status `completed_no_audio`. Frame extraction continues normally.

### 3. Non-Speech Audio (Music / Silence / Tones)
- **Behavior:** When an audio track contains only silence or music, Voice Activity Detection (Silero VAD) detects 0 speech segments. Module 1 handles this gracefully, producing a transcript JSON with `segment_count: 0` without crashing.

### 4. PyTorch / CUDA Configuration
- **Symptom:** `CUDA out of memory` or `Torch not compiled with CUDA enabled`.
- **Solution:** The pipeline automatically detects CUDA availability. If CUDA is not present, it automatically uses `cpu` with `int8` quantization. To explicitly force CPU mode, set `device="cpu"` and `compute_type="int8"` in `src/config.py`.

### 5. Corrupt or Unreadable Video File
- **Behavior:** Corrupt video files fail validation during Stage 1 (`validate_video`) with a descriptive log message and are marked with status `failed` in the batch summary. The batch run never crashes and moves on to the remaining videos.

---

## 12. Running Tests

To run the complete 25-test automated test suite:

```powershell
python -m pytest module1/tests -v
```
