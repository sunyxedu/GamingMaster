#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
ADAPTOR_PYTHON="${1:-/home/hemlix/GamingMaster-mh-tier1/runs/venvs/catan/bin/python}"
ADAPTOR_WHEELHOUSE="${CATAN_WHEELHOUSE:-runtime/wheelhouse}"
uv pip install --python "$ADAPTOR_PYTHON" --no-index --find-links "$ADAPTOR_WHEELHOUSE" --require-hashes -r requirements.lock
uv pip check --python "$ADAPTOR_PYTHON"
