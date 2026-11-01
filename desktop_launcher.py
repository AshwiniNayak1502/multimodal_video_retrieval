"""VideoContextEngine Standalone Desktop Application Launcher.

Provides:
1. Multi-process frozen CLI dispatcher (--run-module1, --run-module2, --diagnose-smtp, --download-models, --streamlit-server).
2. Windows Single-Instance Named Mutex enforcement (Global\\VideoContextEngine_Instance_Mutex).
3. Dynamic ephemeral port allocation.
4. Microsoft Edge WebView2 runtime detection.
5. Windows Job Object management (kill-on-close process containment).
6. pywebview desktop window integration with custom icon and lifecycle handling.
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import multiprocessing
import os
import socket
import subprocess
import sys
import time
import winreg
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

# Call freeze_support immediately for PyInstaller multiprocessing
multiprocessing.freeze_support()

# Import path virtualization
from app.paths import (
    APP_DIR,
    CONFIG_DIR,
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    USER_DATA_DIR,
    init_runtime_environment,
    is_frozen,
)

# Initialize runtime environment
init_runtime_environment()

HOST = "127.0.0.1"
MUTEX_NAME = "Global\\VideoContextEngine_Instance_Mutex"
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
ERROR_ALREADY_EXISTS = 183


def setup_desktop_logging() -> logging.Logger:
    """Configure desktop launcher logging to both file and console."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "desktop.log"

    logger = logging.getLogger("desktop_launcher")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


logger = setup_desktop_logging()


def find_free_port() -> int:
    """Allocate an ephemeral available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def check_webview2_installed() -> bool:
    """Check Windows registry for Edge WebView2 Runtime installation."""
    if sys.platform != "win32":
        return True

    reg_paths = [
        r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-F38F-4475-B9A1-AA05C3C668EC}",
        r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-F38F-4475-B9A1-AA05C3C668EC}",
    ]
    roots = [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]

    for root in roots:
        for path in reg_paths:
            try:
                with winreg.OpenKey(root, path) as key:
                    pv, _ = winreg.QueryValueEx(key, "pv")
                    if pv and pv != "0.0.0.0":
                        return True
            except OSError:
                continue

    return False


def create_job_object() -> Optional[int]:
    """Create a Windows Job Object configured to terminate child processes on exit."""
    if sys.platform != "win32":
        return None

    try:
        kernel32 = ctypes.windll.kernel32
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        JobObjectExtendedLimitInformation = 9

        res = kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not res:
            logger.warning("Failed to set Job Object kill-on-close flag.")
            return None

        return job
    except Exception as exc:
        logger.warning("Could not initialize Windows Job Object: %s", exc)
        return None


def assign_process_to_job(job: int, process_handle: int) -> bool:
    """Assign child process to Job Object."""
    if not job or sys.platform != "win32":
        return False
    try:
        kernel32 = ctypes.windll.kernel32
        return bool(kernel32.AssignProcessToJobObject(job, process_handle))
    except Exception as exc:
        logger.warning("Error assigning process to Job Object: %s", exc)
        return False


def run_streamlit_server(port: int) -> None:
    """Bootstrap Streamlit server child process without opening external browser."""
    app_entry = APP_DIR / "app" / "streamlit_app.py"
    if not app_entry.is_file():
        raise FileNotFoundError(f"Streamlit application entry point not found at: {app_entry}")

    from streamlit.web.cli import main as streamlit_main

    sys.argv = [
        "streamlit",
        "run",
        str(app_entry),
        "--server.address",
        HOST,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
    ]
    logger.info("Starting Streamlit server on http://%s:%d...", HOST, port)
    streamlit_main()


def wait_for_server(url: str, process: subprocess.Popen, timeout: float = 180.0) -> None:
    """Poll server readiness endpoint /_stcore/health until 200 OK or timeout."""
    health_url = f"{url}/_stcore/health"
    deadline = time.time() + timeout
    logger.info("Waiting for Streamlit server readiness at %s...", health_url)

    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Streamlit server process exited prematurely with code {process.returncode}.")

        try:
            with urlopen(health_url, timeout=1.5) as response:
                if response.status == 200:
                    logger.info("Streamlit server is ready.")
                    return
        except Exception:
            time.sleep(0.5)

    raise TimeoutError(f"Streamlit server did not become ready within {timeout} seconds.")


def run_smtp_diagnosis() -> int:
    """CLI runner for SMTP connectivity and credentials diagnosis."""
    from app.services.email_service import diagnose_smtp_connection

    print("==================================================")
    print("VideoContextEngine: SMTP Connectivity Diagnosis")
    print("==================================================")
    res = diagnose_smtp_connection()

    print(f"Step Reached:   {res.step_reached}")
    print(f"Status:         {'SUCCESS' if res.success else 'FAILED'}")
    if res.error_category:
        print(f"Error Category: {res.error_category}")
    if res.error_message:
        print(f"Error Details:  {res.error_message}")
    if res.recommendation:
        print(f"\nRecommendation:\n{res.recommendation}")
    print("==================================================")
    return 0 if res.success else 1


def run_model_download() -> int:
    """CLI runner to pre-cache all required machine learning models to USER_DATA/models."""
    print("==================================================")
    print("VideoContextEngine: Model Pre-caching")
    print(f"Target Cache: {MODELS_DIR}")
    print("==================================================")

    # 1. Download Module 2 models (SigLIP2 and MiniLM)
    from module2.embedding_store.config import Module2Config
    from module2.embedding_store.embedders import download_models

    m2_cfg = Module2Config(models_dir=MODELS_DIR / "huggingface")
    download_models(m2_cfg)

    # 2. Download WhisperX model
    try:
        print("\n[3/3] Downloading Speech Recognition model (Whisper 'tiny')...")
        import faster_whisper

        faster_whisper.WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            download_root=str(MODELS_DIR / "whisper"),
        )
        print("Whisper speech model downloaded successfully.")
    except Exception as exc:
        print(f"Warning: Whisper download error: {exc}")

    print("\nModel caching complete.")
    return 0


def main_gui() -> int:
    """Main desktop GUI launcher."""
    logger.info("Launching VideoContextEngine Desktop...")

    # 1. Single Instance Mutex
    mutex = None
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            mutex = kernel32.CreateMutexW(None, False, MUTEX_NAME)
            last_err = kernel32.GetLastError()
            if last_err == ERROR_ALREADY_EXISTS:
                logger.warning("Another instance of VideoContextEngine is already running.")
                ctypes.windll.user32.MessageBoxW(
                    None,
                    "VideoContextEngine is already running.",
                    "VideoContextEngine",
                    0x40,  # MB_OK | MB_ICONINFORMATION
                )
                return 0
        except Exception as m_exc:
            logger.warning("Could not create single-instance mutex: %s", m_exc)

    # 2. Edge WebView2 check
    if not check_webview2_installed():
        logger.error("Microsoft Edge WebView2 Runtime is not installed.")
        res = 0
        if sys.platform == "win32":
            res = ctypes.windll.user32.MessageBoxW(
                None,
                "Microsoft Edge WebView2 Runtime is required to run VideoContextEngine.\n\n"
                "Would you like to open the official download page now?",
                "WebView2 Runtime Missing",
                0x01 | 0x30,  # MB_OKCANCEL | MB_ICONWARNING
            )
        if res == 1:  # IDOK
            import webbrowser
            webbrowser.open("https://go.microsoft.com/fwlink/p/?LinkId=2124703")
        return 1

    # 3. Allocate free port
    port = find_free_port()
    url = f"http://{HOST}:{port}"
    logger.info("Allocated port %d for Streamlit server.", port)

    # 4. Create Windows Job Object
    job = create_job_object()

    # 5. Spawn Streamlit child process
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW

    if is_frozen():
        server_cmd = [sys.executable, "--streamlit-server", str(port)]
    else:
        server_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--streamlit-server",
            str(port),
        ]

    logger.info("Spawning server process: %s", " ".join(server_cmd))
    server_process = subprocess.Popen(
        server_cmd,
        cwd=str(APP_DIR),
        creationflags=creationflags,
        env=os.environ.copy(),
    )

    if job:
        # On Windows, server_process._handle provides the process HANDLE
        handle = getattr(server_process, "_handle", None)
        if handle:
            assign_process_to_job(job, int(handle))

    # 6. Wait for server readiness and launch pywebview window
    try:
        wait_for_server(url, server_process, timeout=180.0)

        import webview

        icon_path = APP_DIR / "assets" / "icon.ico"
        icon_arg = str(icon_path.resolve()) if icon_path.is_file() else None

        window = webview.create_window(
            title="VideoContextEngine",
            url=url,
            width=1280,
            height=820,
            min_size=(900, 600),
            background_color="#0A0A1F",
        )
        logger.info("Starting pywebview window...")
        webview.start(icon=icon_arg, private_mode=False)

    except Exception as exc:
        logger.error("Error during application runtime: %s", exc, exc_info=True)
        if sys.platform == "win32":
            ctypes.windll.user32.MessageBoxW(
                None,
                f"Failed to start VideoContextEngine:\n\n{exc}",
                "VideoContextEngine Error",
                0x10,  # MB_OK | MB_ICONERROR
            )
        return 1
    finally:
        logger.info("Terminating server process...")
        if server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                server_process.kill()
        if mutex and sys.platform == "win32":
            try:
                ctypes.windll.kernel32.CloseHandle(mutex)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    # Check for CLI worker flags
    if "--run-module1" in sys.argv:
        idx = sys.argv.index("--run-module1")
        sys.argv = [sys.argv[0]] + sys.argv[idx + 1 :]
        m1_dir = str((APP_DIR / "module1").resolve())
        if m1_dir not in sys.path:
            sys.path.insert(0, m1_dir)
        from module1.main import main as m1_main

        sys.exit(m1_main())

    if "--run-module2" in sys.argv:
        idx = sys.argv.index("--run-module2")
        sys.argv = [sys.argv[0]] + sys.argv[idx + 1 :]
        m2_dir = str((APP_DIR / "module2").resolve())
        if m2_dir not in sys.path:
            sys.path.insert(0, m2_dir)
        from module2.main import main as m2_main

        sys.exit(m2_main())

    if "--diagnose-smtp" in sys.argv:
        sys.exit(run_smtp_diagnosis())

    if "--download-models" in sys.argv:
        sys.exit(run_model_download())

    if "--streamlit-server" in sys.argv:
        idx = sys.argv.index("--streamlit-server")
        port = int(sys.argv[idx + 1]) if len(sys.argv) > idx + 1 else 8501
        run_streamlit_server(port)
        sys.exit(0)

    # Default: launch standalone desktop GUI
    sys.exit(main_gui())
