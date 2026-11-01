"""Runtime path virtualization and environment initialization for VideoContextEngine.

Manages distinction between read-only application code directory (sys._MEIPASS when frozen)
and persistent user data directory (%LOCALAPPDATA%\\VideoContextEngine when frozen, repo root in dev).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    """Check if running in a PyInstaller frozen bundle."""
    return getattr(sys, "frozen", False)


def get_app_dir() -> Path:
    """Return read-only application code directory."""
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass).resolve()
        return Path(sys.executable).parent.resolve()
    return Path(__file__).resolve().parent.parent


def get_user_data_dir() -> Path:
    """Return writable user data root.
    
    In a frozen production build, always resolves to %LOCALAPPDATA%\\VideoContextEngine
    (or custom VRS_USER_DATA_DIR if set) to ensure write permissions and survival of updates.
    In local dev mode, defaults to repo root unless VRS_DATA_DIR / VRS_USER_DATA_DIR is set.
    """
    env_user_data = os.getenv("VRS_USER_DATA_DIR")
    if env_user_data:
        return Path(env_user_data).resolve()

    if is_frozen():
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            base = Path(local_app_data).resolve() / "VideoContextEngine"
        else:
            base = Path.home() / ".videocontextengine"
        return base.resolve()

    # In local development mode:
    # If VRS_DATA_DIR is explicitly given, we respect it
    env_data_dir = os.getenv("VRS_DATA_DIR")
    if env_data_dir:
        p = Path(env_data_dir).resolve()
        # If VRS_DATA_DIR ends with 'data', its parent is user_data_dir
        if p.name.lower() == "data":
            return p.parent
        return p

    return get_app_dir()


APP_DIR = get_app_dir()
USER_DATA_DIR = get_user_data_dir()

# Writable directories under USER_DATA_DIR
DATA_DIR = USER_DATA_DIR / "data"
APP_DB_PATH = DATA_DIR / "app.db"
PLAYBACK_DIR = USER_DATA_DIR / "playback"
MODELS_DIR = USER_DATA_DIR / "models"
LOGS_DIR = USER_DATA_DIR / "logs"
CONFIG_DIR = USER_DATA_DIR / "config"

# Sub-module output and input roots
if is_frozen() or os.getenv("VRS_USER_DATA_DIR"):
    MODULE1_INPUT_DIR = USER_DATA_DIR / "module1" / "input" / "videos"
    MODULE1_OUTPUT_DIR = USER_DATA_DIR / "module1" / "output"
    MODULE2_OUTPUT_DIR = USER_DATA_DIR / "module2" / "output"
else:
    MODULE1_INPUT_DIR = APP_DIR / "module1" / "input" / "videos"
    MODULE1_OUTPUT_DIR = APP_DIR / "module1" / "output"
    MODULE2_OUTPUT_DIR = APP_DIR / "module2" / "output"


def init_runtime_environment() -> None:
    """Configure PATH for bundled binaries and set model cache environment variables."""
    # Ensure user data directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLAYBACK_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MODULE1_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODULE1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODULE2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Prepend bundled bin/ to PATH for ffmpeg and ffprobe
    bin_candidates = [
        APP_DIR / "bin",
        Path(sys.executable).parent / "bin" if is_frozen() else None,
    ]
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for b in bin_candidates:
        if b and b.is_dir():
            b_str = str(b.resolve())
            if b_str not in path_entries:
                path_entries.insert(0, b_str)
    os.environ["PATH"] = os.pathsep.join(path_entries)

    # 2. Configure model caches to point to persistent user data
    if not os.getenv("HF_HOME"):
        os.environ["HF_HOME"] = str((MODELS_DIR / "huggingface").resolve())
    if not os.getenv("TRANSFORMERS_CACHE"):
        os.environ["TRANSFORMERS_CACHE"] = str((MODELS_DIR / "huggingface" / "hub").resolve())
    if not os.getenv("TORCH_HOME"):
        os.environ["TORCH_HOME"] = str((MODELS_DIR / "torch").resolve())

    # 3. Export VRS_USER_DATA_DIR so any spawned child processes inherit it
    os.environ["VRS_USER_DATA_DIR"] = str(USER_DATA_DIR.resolve())


# Initialize immediately on import
init_runtime_environment()

