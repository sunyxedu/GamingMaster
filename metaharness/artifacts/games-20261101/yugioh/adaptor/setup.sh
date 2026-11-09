#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
YGO_PYTHON="${YGO_PYTHON:-/home/hemlix/GamingMaster-mh-tier1/runs/venvs/yugioh/bin/python}"
# Sources and assets are already staged. --download restores missing pinned inputs.
"$YGO_PYTHON" setup_assets.py "$@"
