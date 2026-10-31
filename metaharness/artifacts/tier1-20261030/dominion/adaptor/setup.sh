#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
DOMINION_PYTHON="${1:-/home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python}"
"$DOMINION_PYTHON" -m pip install --no-index --require-hashes --force-reinstall -r requirements.txt
"$DOMINION_PYTHON" -m pip check
