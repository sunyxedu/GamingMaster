#!/bin/sh
set -eu
cd "$(dirname "$0")"
if command -v python >/dev/null 2>&1; then
    task_python="$(command -v python)"
elif [ -x ../../../.venv/bin/python ]; then
    task_python="$(cd ../../../.venv/bin && pwd)/python"
else
    task_python="python3"
fi
"$task_python" -c 'import sys; assert sys.version_info >= (3, 11), "Use Python 3.11+"; import metaharness.games.base'
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$task_python" -r requirements.txt
else
    "$task_python" -m pip install -r requirements.txt
fi
"$task_python" -m unittest -v test_game
"$task_python" -c 'from metaharness.games.adaptor import validate_adaptor; r = validate_adaptor("."); print(r.to_markdown()); raise SystemExit(not r.ok)'
