# VideoContextEngine Desktop Packaging Plan
**Target Architecture:** Windows 10/11 x64 Standalone Desktop Application  
**Runtime:** Python 3.13.0 x64, PyInstaller 6.x, pywebview (Edge WebView2), Inno Setup 6  
**Status:** Approved Architecture & Implementation Blueprint  

---

## 1. Executive Summary & Objectives

The goal is to convert **VideoContextEngine** into a self-contained, native Windows desktop application that runs in a standalone window (via `pywebview` backed by Microsoft Edge WebView2), requiring no terminal, no Python interpreter, and no pre-installed dependencies from the end user.

### Core Non-Negotiables:
1. **Local-First AI Preserved:** Embeddings, transcription, audio alignment, VAD, and FAISS vector retrieval run 100% locally on the user's machine. No third-party AI cloud inference APIs are introduced.
2. **Robust Standalone Window:** Replaces browser-based launch with a dedicated native window with custom icon, minimum dimensions (900x600), clean shutdown, and single-instance mutex.
3. **Multi-process Frozen Execution:** Subprocesses (`Module 1` frame extraction/transcription and `Module 2` indexing) must not attempt to execute `sys.executable "main.py"` directly when frozen, which would spawn infinite UI loops. Instead, they re-dispatch to the frozen executable using worker flags (`--run-module1`, `--run-module2`).
4. **Isolated User Data Directory:** Application databases, logs, models, uploads, and search indices reside in `%LOCALAPPDATA%\VideoContextEngine` rather than the read-only program directory.
5. **Secure Authentication:** Local Argon2id password hashing, DPAPI-protected auth secret generation on first run, structured SMTP error diagnosis, and desktop-native Google OAuth using system browser loopback redirect with PKCE (avoiding Google's block on embedded webviews).
6. **Zero Leaked Secrets:** Hardened repository hygiene, secret scanning, and template sanitization.

---

## 2. Dependency & Packaging Inventory

### Runtime Dependencies:
| Component | Pinned Version | Packaging Hazard / Mitigation |
| :--- | :--- | :--- |
| **Python** | 3.13.0 (win_amd64) | Verified compatible with PyInstaller 6.22+. |
| **Streamlit** | 1.66.1 | Static assets (`streamlit/static`), runtime config, tornado, protobuf hidden imports. |
| **PyTorch Suite** | torch 2.11.0, torchvision 0.26.0, torchaudio 2.11.0 | Heavy shared DLLs (`c10.dll`, `torch_cpu.dll`, OpenMP `libiomp5md.dll`). Collect via PyInstaller hooks. |
| **Transformers** | 5.1.0 | Dynamic module loading, model architectures (`siglip2`). |
| **Sentence-Transformers** | 5.4.1 | Model card configs, tokenizer files. |
| **Faster-Whisper / CTranslate2** | 1.2.1 / 4.8.2 | Bundled C++ inference engines and Intel MKL / cuBLAS DLLs. |
| **WhisperX** | 3.8.6 | PyAnnote VAD, torchaudio backends, alignment models. |
| **FAISS** | faiss-cpu 1.13.2 | C++ swig bindings, OpenMP thread pool. |
| **OpenCV** | 4.13.0.92 | Headless video decoding, ffmpeg DLLs. |
| **pywebview** | 6.2.1 | Needs `pythonnet` / `clr_loader` for Edge WebView2 interop. |
| **Keyring** | 25.7.0 | Windows Credential Manager / DPAPI backend for storing secrets safely. |
| **FFmpeg / FFprobe** | 8.1 essentials (Gyan.dev) | Bundled into `bin/` and added to runtime `PATH` dynamically. |

---

## 3. Subprocess Execution & Frozen Re-Dispatch Audit

### The Hazard:
In development:
- `IngestionRunner` launches `[sys.executable, "main.py", "--video", ...]` in `module1/`
- and `[sys.executable, "main.py", "index", ...]` in `module2/`
- `desktop_launcher.py` launches `[sys.executable, "desktop_launcher.py", "--streamlit-server"]`

When frozen into `VideoContextEngine.exe`, `sys.executable` points to `VideoContextEngine.exe`. Spawning `[sys.executable, "main.py", ...]` causes Windows to launch another copy of the desktop GUI, which creates a fork bomb of GUI instances.

### The Migration Architecture:
`desktop_launcher.py` acts as the root entry point:
```
VideoContextEngine.exe (CLI dispatcher)
│
├── (Default / No args) ──> GUI Launcher:
│     ├─ Enforce Single-Instance Mutex (Global\VideoContextEngine_Instance_Mutex)
│     ├─ Bind ephemeral dynamic port (port 0)
│     ├─ Spawn: VideoContextEngine.exe --streamlit-server <port> (in Job Object)
│     ├─ Wait for /_stcore/health (timeout 180s)
│     └─ Open pywebview window (Edge Chromium)
│
├── --streamlit-server <port>
│     └─ Execute streamlit.web.cli.main() directly in-process with --server.port=<port>
│
├── --run-module1 <args...>
│     └─ In-process dispatch to module1.main.main(args) with CREATE_NO_WINDOW
│
├── --run-module2 <args...>
│     └─ In-process dispatch to module2.main.main(args) with CREATE_NO_WINDOW
│
├── --diagnose-smtp
│     └─ Run standalone SMTP connectivity, TLS handshake, and auth diagnostics
│
└── --download-models
      └─ Trigger offline model pre-caching to %LOCALAPPDATA%\VideoContextEngine\models
```

---

## 4. Path Virtualization Architecture (`USER_DATA`)

In a standard Windows installation, the application resides in `C:\Program Files\` or `%LOCALAPPDATA%\Programs\VideoContextEngine\`. Writing data, SQLite databases, indices, or logs to the program directory fails due to write permissions or gets wiped during application updates.

### Virtualization Rule:
All mutable runtime state is redirected to:
`%LOCALAPPDATA%\VideoContextEngine` (`USER_DATA`)

Directory Structure:
```
%LOCALAPPDATA%\VideoContextEngine/
├── config/
│   ├── settings.json
│   └── secret.key (DPAPI protected auth secret)
├── data/
│   ├── app.db (SQLite: users, sessions, videos)
│   └── metadata.db (Module 2 FAISS metadata)
├── models/
│   ├── huggingface/ (HF_HOME)
│   ├── torch/ (TORCH_HOME)
│   └── whisper/
├── vectors/
│   ├── visual.index
│   ├── text.index
│   └── index_manifest.json
├── uploads/
│   └── u<user_id>/...
├── playback/
│   └── <video_id>.mp4
└── logs/
    ├── desktop.log
    ├── streamlit.log
    └── ingestion.log
```

`app/paths.py` and `app/settings.py` initialize these paths before any imports. Environment variables `HF_HOME`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, and `PATH` (adding bundled `bin\`) are set up at process initialization.

---

## 5. Model Inventory & First-Run Setup

### Embedded Models:
1. **Vision-Language Model:** `google/siglip2-base-patch16-224` (768-dimensional visual & text representations).
2. **Text Fusion Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional query representation).
3. **Speech Recognition:** WhisperX `tiny` ASR model with Faster-Whisper CTranslate2 backend.
4. **Phoneme Alignment:** PyAnnote VAD & Wav2Vec2 alignment models.

### First-Run Provisioning Strategy:
- When launched, the application checks if model weights exist in `USER_DATA\models`.
- If missing, a dedicated first-run Setup / Download screen or CLI mode downloads them with progress indicators.
- Once downloaded, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are asserted to prevent network delays or unexpected outbound calls.

---

## 6. Authentication & Google OAuth Redesign for Desktop

### Why the Web Approach Breaks:
1. `st.login()` redirects inside the Streamlit iframe / browser view.
2. In a desktop app, pywebview uses Microsoft Edge WebView2.
3. Google OAuth explicitly **blocks embedded user-agents / webviews** (disallowed useragent error `403 disallowed_useragent`) for security reasons.

### The Desktop-Native Solution:
1. **Loopback HTTP Server:** Application spins up a temporary single-request HTTP server on `http://127.0.0.1:<random_port>/oauth2callback`.
2. **System Default Browser:** Application constructs the Google OAuth authorization URL with PKCE (`code_challenge` S256, `state`, `nonce`) and opens it in the user's default browser (e.g., Chrome, Edge, Firefox) using `webbrowser.open()`.
3. **Loopback Capture:** The user signs in safely in their familiar browser. Google redirects to the local loopback server with `?code=...&state=...`.
4. **Token Exchange & Verification:** The loopback handler exchanges the authorization code for ID and Access tokens using PKCE `code_verifier`, verifies the Google ID token (validating audience, issuer, expiry), and extracts email, name, and `sub` (Google user ID).
5. **Account Linking:** Matches an existing user by email or creates a new verified user account.
6. **Streamlit Session Handoff:** The loopback server generates a cryptographic single-use, 60-second handoff token stored in memory, displays a "You can close this tab and return to VideoContextEngine" page, and redirects the pywebview window to `http://127.0.0.1:<port>/?desktop_auth=<handoff_token>`.
7. Streamlit verifies the handoff token, creates the local session, and logs the user in immediately.

---

## 7. Email Verification & SMTP Diagnosis

### Hardening:
- On first run, a cryptographically secure 64-character `VRS_AUTH_SECRET` is automatically generated if none is configured, protected via Windows DPAPI, and stored in `USER_DATA\config\secret.key`.
- Verification codes are 6 digits, valid for 10 minutes, single-use, and HMAC-SHA256 hashed.
- Resend cooldown (60s) and hourly limits are enforced in SQLite.

### Diagnosis Mode (`--diagnose-smtp`):
Provides clear categorization of SMTP failures:
- **Authentication Failed (535 5.7.8):** Clear guidance on generating a 16-character Google App Password with 2-Step Verification enabled.
- **Connection Refused / Timeout:** Port 587/465 block, firewall, or ISP blocking.
- **TLS Handshake Error:** Certificate validation or protocol mismatch.
- **Recipient Rejected (550/553):** Sender domain or authorization error.

---

## 8. Installer & Packaging Architecture

### PyInstaller (`VideoContextEngine.spec`):
- Mode: **onedir** (directory output for instant startup; onefile creates massive temp extraction overhead on multi-gigabyte PyTorch apps).
- Windows Subsystem: Windowed / GUI (`console=False`).
- Icon: `assets/icon.ico`.
- Data inclusion: `app/`, `module1/`, `module2/`, `module3/`, `bin/` (FFmpeg/FFprobe), static assets.

### Inno Setup 6 (`installer/VideoContextEngine.iss`):
- Privileges: `PrivilegesRequired=lowest` (installs cleanly into `{localappdata}\Programs\VideoContextEngine` without requiring administrator elevation).
- WebView2 Check: Detects if Edge WebView2 runtime is present; prompts or triggers evergreen bootstrapper if absent.
- Start Menu & Desktop Shortcuts with uninstall support.
- User Data Protection: Never wipes or touches `%LOCALAPPDATA%\VideoContextEngine` during updates or uninstallation.

---

## 9. Secret Audit Findings & Mitigation

### Audit Findings:
1. `.streamlit/secrets.toml.example` previously contained real client ID and secret values (`GOCSPX-[REDACTED-HISTORICAL-SECRET]`).
   - **Remediation:** Sanitized with descriptive placeholders (`CHANGE_ME_...`).
2. `.gitignore` was missing protection for `.env.*`, `smtp*.env`, `*.bak`, `*.backup`, `client_secret*.json`, and bundled `.exe` binaries.
   - **Remediation:** Hardened `.gitignore` and verified untracked status.
3. Pre-build automated scanner created in `build/build_windows.ps1` to abort builds if active API keys, app passwords, or Google client secrets are detected in committed files or staging directories.
