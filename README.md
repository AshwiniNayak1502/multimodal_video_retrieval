# VideoContextEngine

**VideoContextEngine** is a fully local, privacy-preserving Multimodal Video Semantic Retrieval System designed and built for Windows 11. It indexes and searches video libraries using state-of-the-art multimodal deep learning models (SigLIP2, all-MiniLM-L6-v2, and WhisperX) without cloud dependencies, external inference APIs, or SaaS telemetry.

---

## 🏛️ System Architecture

```text
                               +--------------------------------------------+
                               |        Streamlit Web Frontend (app/)       |
                               |  - Modern Dark Navy & Purple Theme (#7C5CFF)|
                               |  - Argon2id Auth + SHA-256 Token Sessions  |
                               |  - Cross-User Scoping & Video Isolation    |
                               |  - Video Player with Precise Timestamp Cue |
                               |  - Real Key Moments Navigation Timeline    |
                               +---------------------+----------------------+
                                                     |
                         +---------------------------+---------------------------+
                         |                                                       |
                         v                                                       v
        +----------------------------------+                   +----------------------------------+
        |   App Database (data/app.db)     |                   | Ingestion Runner (Queue Worker)  |
        | - users (argon2id passwords)     |                   | - Single background queue thread |
        | - sessions (hashed tokens, 12h)  |                   | - Recovers crashed/stale jobs    |
        | - videos (refcounted ownership)  |                   | - Spawns Module 1 & 2 subprocs   |
        +----------------------------------+                   +----------------+-----------------+
                                                                                |
                         +------------------------------------------------------+
                         |
                         v
+---------------------------------------------------------------------------------------------------+
|                                   Core Local Pipeline Modules                                     |
+---------------------------------------------------------------------------------------------------+
|  Module 1: Preprocessing & Ingestion                                                              |
|  - Frame Extraction: OpenCV adaptive frame extraction saved to module1/output/frames/<video_id>/  |
|  - Audio Transcription: WhisperX with word-level timestamps & VAD segmentation                    |
|  - Output Metadata: Deterministic hashing, duration, fps, resolution, transcript JSON             |
+---------------------------------------------------------------------------------------------------+
|  Module 2: Multimodal Embedding & Storage                                                         |
|  - Visual Embedder: SigLIP2 (google/siglip2-base-patch16-224, 768-d L2 normalized)                |
|  - Textual Embedder: all-MiniLM-L6-v2 (384-d L2 normalized)                                       |
|  - Vector Index: FAISS IndexIDMap2 + IndexFlatIP with SQLite metadata store (metadata.db)          |
+---------------------------------------------------------------------------------------------------+
|  Module 3: Semantic Retrieval Engine                                                              |
|  - Modalities: "Combined", "Transcript only", "Visual only"                                       |
|  - Score Calibration: SigLIP native sigmoid + MiniLM cosine calibration                           |
|  - Strict Isolation: User-owned video_ids filtering in every search request                       |
|  - Dynamic Range Playback: FFmpeg faststart browser proxy generator for MKV / non-web containers    |
+---------------------------------------------------------------------------------------------------+
```

---

## 💻 Prerequisites & Setup (Windows 11)

### 1. Python 3.13 Installation
Ensure Python 3.13 (64-bit) is installed and available in your PowerShell PATH:
```powershell
python --version
# Expected: Python 3.13.x
```

### 2. FFmpeg & FFprobe Installation
FFmpeg is required for frame extraction, audio separation, and browser playback proxy generation:
1. Download Gyan.dev essentials build from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) or install via `winget`:
   ```powershell
   winget install Gyan.FFmpeg
   ```
2. Verify both utilities in PowerShell:
   ```powershell
   ffmpeg -version
   ffprobe -version
   ```

### 3. Virtual Environment Setup (PowerShell)
From the repository root directory:
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Environment Variables Configuration
Copy the sample environment configuration file:
```powershell
Copy-Item .env.example .env
```
All defaults in `.env.example` run offline with zero cloud secrets.

---

## 🚀 One-Time Model Warm-up & Cache Preparation

Before launching the app for the first time, warm up the model caches:
```powershell
# 1. Warm up SigLIP2 and MiniLM embedders (Module 2)
python module2/main.py download-models

# 2. Verify Module 3 query engine initialization
python module3/main.py --help
```

---

## 🎬 Running the Application

Launch the Streamlit frontend with a single command:
```powershell
streamlit run app/streamlit_app.py
```
The application will launch automatically in your default browser at `http://localhost:8501`.

### Key Features to Test in Browser:
1. **User Authentication:** Create a local account with email, password (>= 8 chars), and full name. Argon2id hashes the credentials in `data/app.db`.
2. **Video Upload:** Drag and drop an `.mp4` or `.mkv` video. The pipeline queues the file, extracts frames, generates word-level transcriptions via WhisperX, and creates FAISS vectors.
3. **Cross-User Content Reuse:** If User B uploads the identical video file that User A already uploaded, the engine recognizes the SHA-256 digest and grants instant library access without redundant reprocessing.
4. **Strict Isolation:** User B cannot see, search, or play videos uploaded by User A.
5. **Modalities Search:** Search via **Combined**, **Transcript only**, or **Visual only** modes.
6. **Key Moments & Instant Seeking:** Click on matched segments or Key Moments timeline chips to jump playback to the exact second.

---

## 🧪 Automated Test Suites

Run the complete test suite across all modules:

```powershell
# App frontend and services suite
python -m pytest tests/app -v

# Module 1 (Video Discovery, Frame Extraction & Transcription)
python -m pytest module1/tests -q

# Module 2 (Embedding Generation & Vector Storage)
python -m pytest module2/tests -q

# Module 3 (Retrieval & Fusion Engine)
python -m pytest module3/tests -q -m "not real_model"
```

Verify Python syntax compilation across the whole app:
```powershell
python -m compileall app
```

---

## 🛠️ Troubleshooting & FAQs

### 1. "No module named 'torchcodec'" UserWarning
This is an optional warning from `pyannote.audio` during WhisperX audio alignment. Built-in fallback audio decoders in `whisperx` / `ffmpeg` handle decoding transparently without affecting transcription accuracy.

### 2. "App shows 'No related video found'"
Ensure that:
- Your uploaded video has completed ingestion (check Video Library status badge).
- The search query is relevant to either the spoken audio or visual scene.
- If testing a silent video, select **"Visual only"** or **"Combined"** modality.

### 3. MKV Playback in Browser
Browsers cannot natively decode MKV containers with certain audio tracks. VideoContextEngine automatically creates a faststart H.264/AAC MP4 proxy under `data/playback/` using FFmpeg on the fly, ensuring smooth timeline seeking.

### 4. Resetting the Application Database
To reset all users and video metadata without modifying your original videos:
```powershell
Remove-Item -Force data/app.db
Remove-Item -Force -Recurse data/playback
```
The app will automatically regenerate the schema and empty tables on next launch.
