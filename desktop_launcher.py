
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import webview

BASE_DIR = Path(__file__).resolve().parent
APP_PATH = BASE_DIR / "app" / "streamlit_app.py"
HOST = "127.0.0.1"
PORT = 8501
URL = f"http://{HOST}:{PORT}"


def port_is_open():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((HOST, PORT)) == 0


def run_streamlit_server():
    if not APP_PATH.exists():
        raise FileNotFoundError(f"Application not found: {APP_PATH}")

    from streamlit.web.cli import main as streamlit_main

    sys.argv = [
        "streamlit", "run", str(APP_PATH),
        "--server.address", HOST,
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false",
    ]
    streamlit_main()


def wait_for_server(process, timeout=180):
    deadline = time.time() + timeout

    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Server process exited with code {process.returncode}."
            )

        try:
            with urlopen(URL + "/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)

    raise TimeoutError("Streamlit did not become ready within 180 seconds.")


def main():
    if not APP_PATH.exists():
        raise FileNotFoundError(f"Application not found: {APP_PATH}")

    if port_is_open():
        raise RuntimeError(
            "Port 8501 is already in use. Close the existing app first."
        )

    if getattr(sys, "frozen", False):
        command = [sys.executable, "--streamlit-server"]
    else:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--streamlit-server",
        ]

    server = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        env=os.environ.copy(),
    )

    try:
        wait_for_server(server)
        webview.create_window(
            "VideoContextEngine", URL,
            width=1280, height=820, min_size=(900, 600),
        )
        webview.start()
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    if "--streamlit-server" in sys.argv[1:]:
        run_streamlit_server()
    else:
        main()
