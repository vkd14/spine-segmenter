"""Native macOS desktop wrapper for the Streamlit spine-segmentation app.

Architecture:
    1. Pick a free TCP port.
    2. Start `streamlit run app.py` headless on that port as a subprocess.
    3. Poll /_stcore/health until the server responds.
    4. Open a pywebview native window (uses macOS WKWebView under the hood)
       pointing at http://127.0.0.1:<port>/.
    5. When the window is closed, terminate the Streamlit process group.

Run via: ./launch_spine_seg_app.command   (or directly: .venv/bin/python desktop_app.py)
"""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import webview  # pywebview

HERE = Path(__file__).resolve().parent
APP_PY = HERE / "app.py"
STREAMLIT_BIN = HERE / ".venv" / "bin" / "streamlit"


def _find_free_port() -> int:
    """Bind to port 0 to let the kernel pick a free port, then close immediately."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout_s: float = 60.0) -> bool:
    deadline = time.time() + timeout_s
    last_err: str = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as r:
                if r.status == 200:
                    return True
        except urllib.error.URLError as e:
            last_err = str(e)
        except Exception as e:
            last_err = repr(e)
        time.sleep(0.4)
    print(f"[desktop_app] Streamlit didn't start within {timeout_s}s. Last error: {last_err}",
          file=sys.stderr)
    return False


def _start_streamlit(port: int) -> subprocess.Popen:
    if not STREAMLIT_BIN.exists():
        raise FileNotFoundError(
            f".venv missing — expected {STREAMLIT_BIN}. "
            "Create it with: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt"
        )
    cmd = [
        str(STREAMLIT_BIN), "run", str(APP_PY),
        "--server.headless=true",
        f"--server.port={port}",
        "--server.address=127.0.0.1",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
        "--client.toolbarMode=minimal",
        # Per-file upload limit (MB). Default is 200; spine MRIs can exceed that.
        "--server.maxUploadSize=600",
        # Websocket message cap (MB). Must be >= maxUploadSize or large uploads
        # silently disconnect at the message-size threshold.
        "--server.maxMessageSize=600",
    ]
    # Start in a new session so we can kill the whole process group later.
    proc = subprocess.Popen(
        cmd,
        cwd=str(HERE),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    return proc


def _kill(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=4)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


def main() -> int:
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}/"
    print(f"[desktop_app] Starting Streamlit on {url}")
    proc = _start_streamlit(port)
    atexit.register(_kill, proc)

    if not _wait_for_server(url + "_stcore/health"):
        _kill(proc)
        print("[desktop_app] Streamlit failed to come up. Exiting.", file=sys.stderr)
        return 1

    print(f"[desktop_app] Server ready. Opening native window…")
    webview.create_window(
        title="Lumbar L1-L5 Spine Segmentation",
        url=url,
        width=1500,
        height=950,
        min_size=(900, 600),
        background_color="#111111",
        confirm_close=False,
    )
    try:
        # Default GUI on macOS = WKWebView via pyobjc.
        webview.start(debug=False)
    finally:
        _kill(proc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
