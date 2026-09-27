#!/usr/bin/env bash
# Xiangqi data end to end: download the Chinese Chess Practical Dataset (CCPD, CC BY 4.0,
# GitHub archive at a pinned commit), convert it to from-to gm-shards with
# `gmdata xiangqi convert` — every move of every record is resolved against the legal
# moves and the game is replayed and verified by the gmrules rules the simulator uses —
# as disjoint train / val splits, and print the stats.  Mac + Linux; needs cargo
# (data_processing/README.md).  The whole corpus is small: this IS the full-scale run
# (~1 min, ~0.7 GB of shards).
#
#   scripts/fetch_xiangqi.sh [OUT_DIR] [RAW_DIR]
#     OUT_DIR  default data/xiangqi   (OUT_DIR/train, OUT_DIR/val)
#     RAW_DIR  default ~/gm-data/xiangqi (the 60 MB archive + ATTRIBUTION.txt)
#   HQ=1 (default): master games (CCPD 大師對局), >= 20 plies, known result; HQ=0: every
#   game from the initial position.  FORCE=1 replaces existing OUT_DIR/train|val
#   (the shard writer appends to a non-empty directory, so they are refused otherwise).
#
# Expected at CCPD commit 368a47a9 (HQ=1): train 26,980 games / 2,230,893 frames
# (621 MB), val 567 games / 46,371 frames (13 MB); HQ=0: 27,548 / 578 games.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-data/xiangqi}
RAW=${2:-$HOME/gm-data/xiangqi}
HQ=${HQ:-1}
FORCE=${FORCE:-0}
COMMIT=${CCPD_COMMIT:-368a47a947773dd8692c026e286dd19b6277b993}

for d in "$OUT/train" "$OUT/val"; do
  if [ -d "$d" ] && [ -n "$(ls -A "$d")" ]; then
    if [ "$FORCE" = 1 ]; then
      rm -rf "$d"
    else
      echo "$d exists and is not empty (FORCE=1 replaces it)" >&2
      exit 1
    fi
  fi
done

# a user-level rustup (the cluster's ~/.cargo) when cargo is not on PATH
command -v cargo >/dev/null || export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
cd "$ROOT/data_processing"
cargo build --release -q -p gmdata
G="$ROOT/data_processing/target/release/gmdata"
cd "$ROOT"

ZIP="$RAW/ccpd-${COMMIT:0:8}.zip"
if [ ! -s "$ZIP" ]; then
  "$G" xiangqi fetch --out "$RAW" --commit "$COMMIT"
fi
FLAGS=()
if [ "$HQ" = 1 ]; then
  FLAGS+=(--hq)
fi
"$G" xiangqi convert --input "$ZIP" --out "$OUT/train" --split train ${FLAGS[@]+"${FLAGS[@]}"}
"$G" xiangqi convert --input "$ZIP" --out "$OUT/val" --split val ${FLAGS[@]+"${FLAGS[@]}"}
cp "$RAW/ATTRIBUTION.txt" "$OUT/ATTRIBUTION.txt" 2>/dev/null || true
echo "xiangqi shards: $OUT/train $OUT/val"
