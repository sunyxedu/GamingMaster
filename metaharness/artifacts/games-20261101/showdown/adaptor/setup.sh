#!/usr/bin/env bash
set -euo pipefail
adaptor_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
showdown_root="${SHOWDOWN_ROOT:-$HOME/gm-tools/pokemon-showdown}"
showdown_node="${SHOWDOWN_NODE:-node}"
foulplay_root="${FOUL_PLAY_ROOT:-$HOME/gm-tools/foul-play}"
evaluator_python=/home/hemlix/GamingMaster-mh-tier1/runs/venvs/showdown/bin/python
expected_revision=a5df8274e85b0889bf2a9b3422a08b39732374fc
foulplay_revision=8f541f80f91f960603673796559f796b10ccf0fb
test "$(git -C "$showdown_root" rev-parse HEAD)" = "$expected_revision"
SHOWDOWN_ROOT="$showdown_root" "$showdown_node" - <<'JS'
const [major, minor] = process.versions.node.split('.').map(Number);
if (major < 22 || (major === 22 && minor < 18)) throw Error('Node >=22.18 is required');
const {BattleStream, getPlayerStreams, Dex} = require(process.env.SHOWDOWN_ROOT);
if (!BattleStream || !getPlayerStreams || !Dex.formats.get('gen9randombattle').exists) {
  throw Error('The supplied Showdown checkout must be built');
}
console.log('Offline Showdown dependency check passed; Node ' + process.versions.node);
JS
if [ ! -d "$foulplay_root/.git" ]; then
  git clone https://github.com/pmariglia/foul-play.git "$foulplay_root"
  git -C "$foulplay_root" checkout "$foulplay_revision"
fi
test "$(git -C "$foulplay_root" rev-parse HEAD)" = "$foulplay_revision"
# poke-engine's default is Gen 9; explicitly build with Terastallization enabled.
# Rust/cargo is needed only at build time. uv works with the supplied pip-less venv.
uv pip install --python "$evaluator_python" -r "$adaptor_root/requirements.txt" \
  --config-settings 'build-args=--features poke-engine/terastallization --no-default-features'
mkdir -p "$adaptor_root/runtime" "$foulplay_root/fp/data/pkmn_sets_cache"
test "$(sha256sum "$adaptor_root/data/gen9randombattle.json" | cut -d ' ' -f 1)" = 6a4b6fc8391d334e252001d8a48ec383159a7ccfdcab23830ab6def00aab793c
"$evaluator_python" -c 'from metaharness.games.base import ActionSpec; from poke_engine import monte_carlo_tree_search; print("Evaluator Python and MCTS engine dependency checks passed")'
