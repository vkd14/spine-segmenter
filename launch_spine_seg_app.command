#!/bin/bash
# Double-click this file from Finder to launch the Lumbar Spine Segmentation
# app in a NATIVE macOS window (pywebview + WKWebView). No browser opens —
# you get a real Mac app window with title bar and Dock entry. Closing the
# window shuts down the embedded Streamlit server cleanly.

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ ! -d ".venv" ]; then
    echo "Error: .venv not found in $DIR"
    echo "Run: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    read -n 1 -s -r -p "Press any key to close…"
    exit 1
fi

echo "Launching Lumbar Spine Segmentation (native window)…"
exec "$DIR/.venv/bin/python" "$DIR/desktop_app.py"
