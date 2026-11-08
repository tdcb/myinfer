#!/bin/zsh
# Double-click launcher for the M5Engine web UI + OpenAI-compatible server (MLX backend).
# Usage: ./Launch_M5Engine_UI.command [port] [model]
cd "$(dirname "$0")"

PORT=${1:-8080}
MODEL=${2:-}

if [ ! -d ".venv" ]; then
  echo "[*] Creating Python environment (.venv) and installing MLX..."
  python3 -m venv .venv || exit 1
  ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q -r requirements.txt || exit 1
fi

echo "[*] M5Engine on http://127.0.0.1:${PORT}"
(sleep 2 && open "http://127.0.0.1:${PORT}") &
if [ -n "${MODEL}" ]; then
  ./.venv/bin/python -m m5engine serve --port "${PORT}" --model "${MODEL}"
else
  ./.venv/bin/python -m m5engine serve --port "${PORT}"
fi
