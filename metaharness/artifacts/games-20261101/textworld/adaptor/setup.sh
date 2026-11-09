#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
evaluator_python="${1:-/home/hemlix/GamingMaster-mh-tier1/runs/venvs/textworld/bin/python}"
uv pip install --python "$evaluator_python" --no-index --find-links runtime/wheelhouse --require-hashes -r requirements.lock
uv pip check --python "$evaluator_python"
