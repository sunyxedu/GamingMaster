#!/usr/bin/env bash
# Kuhn poker (the toy reference game): data end to end — synthetic hand logs
# ("download"), high-quality conversion to candidate-action gm-shards, a val split.
# The template for scripts/fetch_<game>.sh of a gmrules game: fetch raw records,
# `gmdata <game> convert` (replays and verifies every record with the simulator's
# rules), print the stats.  Mac + Linux; needs cargo (data_processing/README.md).
#
#   scripts/fetch_kuhn_poker.sh [OUT_DIR] [HANDS] [PLAYERS]
#   OUT_DIR default data/kuhn_poker, HANDS default 1000000, PLAYERS default 2
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-data/kuhn_poker}
HANDS=${2:-1000000}
PLAYERS=${3:-2}
SEED=${SEED:-0}

# a user-level rustup (the cluster's ~/.cargo) when cargo is not on PATH
command -v cargo >/dev/null || export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
cd "$ROOT/data_processing"
cargo build --release -q -p gmdata
G="$ROOT/data_processing/target/release/gmdata"
cd "$ROOT"

mkdir -p "$OUT/raw"
"$G" kuhn-poker generate --out "$OUT/raw/train.log" --hands "$HANDS" --players "$PLAYERS" --seed "$SEED"
"$G" kuhn-poker generate --out "$OUT/raw_val/val.log" --hands $((HANDS / 50 + 1)) \
    --players "$PLAYERS" --seed $((SEED + 1))
# high-quality recipe: learn from the scripted expert's seats only
"$G" kuhn-poker convert --input "$OUT/raw" --out "$OUT/train" --players "$PLAYERS" --hq
"$G" kuhn-poker convert --input "$OUT/raw_val" --out "$OUT/val" --players "$PLAYERS" --hq
echo "kuhn_poker shards: $OUT/train $OUT/val"
