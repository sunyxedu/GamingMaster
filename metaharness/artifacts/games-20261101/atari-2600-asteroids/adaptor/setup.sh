#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${1:-python3}"
mkdir -p "$ROOT/runtime/tmp" "$ROOT/runtime/pip-cache" "$ROOT/runtime/uv-cache"
export TMPDIR="$ROOT/runtime/tmp" PIP_CACHE_DIR="$ROOT/runtime/pip-cache" UV_CACHE_DIR="$ROOT/runtime/uv-cache"
if "$PYTHON" -I -m pip --version >/dev/null 2>&1; then
  "$PYTHON" -I -m pip install -r "$ROOT/requirements.txt"
else
  uv pip install --python "$PYTHON" -r "$ROOT/requirements.txt"
fi
"$PYTHON" -I - <<'PY'
import ale_py, gymnasium
from metaharness.games.base import ActionSpec
print('Dependencies and evaluator interface available:', ale_py.__version__, gymnasium.__version__)
PY
