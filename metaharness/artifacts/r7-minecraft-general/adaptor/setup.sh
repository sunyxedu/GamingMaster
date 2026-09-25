#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec "${PYTHON_BIN:-<repo>/.venv/bin/python}" setup.py
