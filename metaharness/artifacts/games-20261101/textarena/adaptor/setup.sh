#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
EVALUATOR_PY=/home/hemlix/GamingMaster-mh-tier1/runs/venvs/textarena/bin/python
"$EVALUATOR_PY" -m pip install --no-index --find-links runtime/wheelhouse --require-hashes -r requirements.txt
"$EVALUATOR_PY" -m pip check
