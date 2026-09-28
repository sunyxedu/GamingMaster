#!/usr/bin/env bash
# Versus-Tetris IL data end to end: Cold Clear self-play on the gmrules tetris game
# (data_processing/rs/gmtetris, its own cargo workspace) in batches of BATCH games, then
# `gmdata rules convert --game tetris` (every record replayed and verified) into
# OUT/train and OUT/val. Idempotent / resumable: a batch whose .jsonl.gz exists is kept
# (written only once complete), the conversion is redone from the batches. Mac + Linux;
# needs cargo.
#
#   scripts/generate_tetris.sh OUT_DIR GAMES [WORKERS]
#   env: NODES=400000     Cold Clear search nodes per move (both seats; the strength
#                         measurements are in docs: 50k beats 5k 83 %, 200k ~ 50k 53 %,
#                         800k beats 200k 60 %)
#        BATCH=500  VAL_GAMES=GAMES/50  SEED=1  NICE=19  CONVERT_ONLY=1
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:?OUT_DIR}
GAMES=${2:?GAMES}
WORKERS=${3:-4}
NODES=${NODES:-400000}
BATCH=${BATCH:-500}
VAL_GAMES=${VAL_GAMES:-$(( GAMES / 50 > 0 ? GAMES / 50 : 1 ))}
SEED=${SEED:-1}
NICE=${NICE:-19}
mkdir -p "$OUT/raw" "$OUT/raw_val" "$OUT/logs"

command -v cargo >/dev/null || export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
(cd "$ROOT/data_processing/rs/gmtetris" && cargo build --release -q)
(cd "$ROOT/data_processing" && cargo build --release -q -p gmdata)
T="$ROOT/data_processing/rs/gmtetris/target/release/gmtetris"
G="$ROOT/data_processing/target/release/gmdata"

run_batches() {  # dir games seed0
  local dir=$1 total=$2 s0=$3 b=0 n
  while (( b * BATCH < total )); do
    n=$(( total - b * BATCH < BATCH ? total - b * BATCH : BATCH ))
    f="$dir/batch-$(printf %05d $b).jsonl.gz"
    if [ ! -s "$f" ]; then
      nice -n "$NICE" "$T" selfplay --games "$n" --nodes "$NODES" --workers "$WORKERS" \
          --seed $(( s0 + b )) --out "$f.part.jsonl.gz" > "$OUT/logs/$(basename "$dir")-$b.json"
      mv "$f.part.jsonl.gz" "$f"
      echo "$(date +%T) $f: $(cat "$OUT/logs/$(basename "$dir")-$b.json")"
    fi
    b=$(( b + 1 ))
  done
}

if [ "${CONVERT_ONLY:-0}" != 1 ]; then
  run_batches "$OUT/raw_val" "$VAL_GAMES" $(( SEED * 1000000 + 900000 ))
  run_batches "$OUT/raw" "$GAMES" $(( SEED * 1000000 ))
fi
for split in train val; do
  src=$OUT/raw; [ $split = val ] && src=$OUT/raw_val
  find "$src" -name '*.part.jsonl.gz' -delete      # an interrupted batch
  rm -rf "${OUT:?}/$split"
  "$G" rules convert --game tetris --input "$src" --out "$OUT/$split" > "$OUT/logs/convert-$split.json"
  echo "$split: $(head -c 300 "$OUT/logs/convert-$split.json")"
done
