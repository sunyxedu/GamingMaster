#!/bin/sh
set -eu
cd "$(dirname "$0")"
PYTHON=<repo>/.venv/bin/python
"$PYTHON" -m pip --version >/dev/null 2>&1 || "$PYTHON" -m ensurepip --upgrade
"$PYTHON" -m pip install -r requirements.txt
"$PYTHON" - <<'PY'
from game import make_game, MANIFEST
with_game = make_game()
try:
    for seed, (game_id, expected) in enumerate(MANIFEST):
        obs = with_game.reset(seed)
        assert obs['win_levels'] == expected
        print(game_id, obs['state'], f"{len(obs['frame'])}x{len(obs['frame'][0])}", obs['available_actions'])
finally:
    with_game.close()
PY
