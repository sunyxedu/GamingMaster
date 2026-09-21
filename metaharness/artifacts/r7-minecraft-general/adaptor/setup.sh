#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec "${PYTHON_BIN:-/Users/hemlix/Documents/GamingMaster/.venv/bin/python}" setup.py
