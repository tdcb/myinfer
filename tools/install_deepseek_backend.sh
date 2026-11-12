#!/usr/bin/env bash
# Install the optional oMLX backend that serves DeepSeek-V4.1-Flash (Apache-2.0,
# https://github.com/jundot/omlx) into this checkout's virtual environment.
#
# Needs macOS on Apple Silicon and Python 3.11+ (mlx-lm >= 0.32 and oMLX require it).
# The pin matches m5engine/deepseek_v41.py; bump both together.
set -euo pipefail
cd "$(dirname "$0")/.."
PIN="${OMLX_PIN:-c74f1b18558842e6b10fcfa6e2e901bb382129d4}"

pick_python() {
  for c in python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
      echo "$c"; return
    fi
  done
  echo "Python 3.11 or newer is required (brew install python@3.12)" >&2; exit 1
}

if [ -x .venv/bin/python ] && ! .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "[*] .venv uses $(.venv/bin/python --version); recreating it with a newer Python"
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  PY="$(pick_python)"
  echo "[*] Creating .venv with $PY"
  "$PY" -m venv .venv
  ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q -r requirements.txt
fi
echo "[*] Installing oMLX @ ${PIN} (this pulls mlx 0.32.4, mlx-lm, transformers, mlx-vlm; several minutes)"
./.venv/bin/pip install -q "omlx @ git+https://github.com/jundot/omlx@${PIN}"
./.venv/bin/python - <<'PY'
from m5engine import deepseek_v41 as d
ok, detail = d.available()
print(("[ok] DeepSeek V4.1 backend ready: " if ok else "[!!] backend not importable: ") + detail)
raise SystemExit(0 if ok else 1)
PY
