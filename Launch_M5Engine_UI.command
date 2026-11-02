#!/bin/zsh
# 1-Click macOS Finder Launcher for M5Engine Control Center & Web UI
cd "$(dirname "$0")"

PORT=${1:-8080}
MODEL=${2:-dense}

if [ ! -f "./m5engine" ]; then
  echo "[*] Building M5Engine for Apple M5 Pro..."
  make -j8
fi

echo "[*] Launching M5Engine Web UI & Control Plane on http://127.0.0.1:${PORT} ..."
(sleep 0.8 && open "http://127.0.0.1:${PORT}") &
./m5engine serve --model "${MODEL}" --port "${PORT}"
