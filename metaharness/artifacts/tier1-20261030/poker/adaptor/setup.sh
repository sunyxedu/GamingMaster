#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
POKER_PYTHON=/home/hemlix/GamingMaster-mh-tier1/runs/venvs/poker/bin/python
# The bundled pure-Python wheel makes setup offline as well as gameplay.
sha256sum --check wheelhouse.sha256
"$POKER_PYTHON" -m pip install --no-index --find-links=runtime/wheelhouse -r requirements.txt
"$POKER_PYTHON" -m pip check
"$POKER_PYTHON" -c 'import importlib.metadata as m; assert m.version("pokerkit") == "0.7.6"'
