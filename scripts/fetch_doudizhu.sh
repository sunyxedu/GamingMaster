#!/usr/bin/env bash
# Dou Dizhu: the imitation corpus end to end.  There is no public human data, so the
# records are DouZero self-play: scripts/generate_doudizhu.py downloads the public
# DouZero agents (ADP / WP checkpoints, Apache-2.0, pinned + SHA-256 checked) and plays
# them against each other under the gmrules standard rules (a documented heuristic
# bids: DouZero has no auction); `gmdata doudizhu convert --hq` replays and verifies
# every record with the simulator's rules and writes candidate-action gm-shards.
# Mac + Linux; needs cargo and the project's Python environment (uv).
#
#   scripts/fetch_doudizhu.sh [OUT_DIR] [GAMES]
#   OUT_DIR default data/doudizhu, GAMES default 1000000 (+ GAMES/50 validation games)
#   env: WORKERS (default: all cores), SEED (default 0), AGENTS (default adp,wp),
#        PYTHON (default "uv run python"; Slurm jobs: .venv/bin/python),
#        GM_REFERENCE_CACHE (download cache, default ~/.cache/gm-reference),
#        GMDATA (a prebuilt gmdata binary: skips the cargo build)
#
# Compute nodes without internet: run `$PYTHON scripts/generate_doudizhu.py --fetch_only`
# on a login node first (the downloads land in the cache).  Expected full scale (1M
# games): ~36M frames (3M per-seat samples), ~28 GB of shards, ~22 games/s per worker.
# Parts are written atomically and skipped when present: a rerun resumes.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-data/doudizhu}
GAMES=${2:-1000000}
SEED=${SEED:-0}
AGENTS=${AGENTS:-adp,wp}
WORKERS=${WORKERS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}
PYTHON=${PYTHON:-uv run python}

if [ -n "${GMDATA:-}" ]; then
  G=$GMDATA
else
  # a user-level rustup (the cluster's ~/.cargo) when cargo is not on PATH
  command -v cargo >/dev/null || export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
  (cd "$ROOT/data_processing" && cargo build --release -q -p gmdata)
  G="$ROOT/data_processing/target/release/gmdata"
fi
cd "$ROOT"

VAL=$((GAMES / 50 + 1))
# shellcheck disable=SC2086  # PYTHON may be a command with arguments
$PYTHON scripts/generate_doudizhu.py --out "$OUT/raw" --games "$GAMES" --seed "$SEED" \
    --agents "$AGENTS" --workers "$WORKERS"
# shellcheck disable=SC2086
$PYTHON scripts/generate_doudizhu.py --out "$OUT/raw_val" --games "$VAL" --seed $((SEED + 1)) \
    --agents "$AGENTS" --workers "$WORKERS"
# high-quality recipe: the strong DouZero agents' seats (all of them with the default mix)
"$G" doudizhu convert --input "$OUT/raw" --out "$OUT/train" --hq
"$G" doudizhu convert --input "$OUT/raw_val" --out "$OUT/val" --hq
echo "doudizhu shards: $OUT/train $OUT/val"
