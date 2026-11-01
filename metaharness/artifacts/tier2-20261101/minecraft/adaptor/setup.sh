#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"
# The evaluator venv is next to this adaptor; PYTHON_BIN remains an explicit override.
PYTHON_BIN=${PYTHON_BIN:-"$ROOT/../.venv/bin/python"}
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN=${PYTHON_BIN_FALLBACK:-python3}
fi
exec "$PYTHON_BIN" -I "$ROOT/setup.py"
