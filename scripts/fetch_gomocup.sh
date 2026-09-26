#!/bin/bash
# Gomocup tournament records (AI vs AI, 2010 onwards) → freestyle 20×20 gm-shards.
#
#   OUT=~/gm-data/gomoku bash scripts/fetch_gomocup.sh [FIRST_YEAR] [LAST_YEAR]
#
# Downloads each year's results archive from gomocup.org into $OUT/raw/<year>,
# links the freestyle leagues (Freestyle*, Fastgame, Unlimited; 20×20, five or
# more wins) into $OUT/sel20 and converts them with `gmdata gomoku convert`
# (other board sizes are dropped by the converter as wrong_board_size).
# 2010-2026 gives ~289k games / 12.5M frames (~2 GB of shards).
set -euo pipefail
OUT=${OUT:-$HOME/gm-data/gomoku}
FIRST=${1:-2010}
LAST=${2:-$(date +%Y)}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GMDATA=${GMDATA:-$ROOT/data_processing/target/release/gmdata}
mkdir -p "$OUT/raw" "$OUT/sel20"
for y in $(seq "$FIRST" "$LAST"); do
    zip="$OUT/raw/gomocup$y.zip"
    if [ ! -s "$zip" ]; then
        code=$(curl -sL -m 600 -o "$zip" -w "%{http_code}" \
            "https://gomocup.org/static/tournaments/$y/results/gomocup${y}results.zip")
        if [ "$code" != 200 ]; then rm -f "$zip"; echo "$y: HTTP $code, skipped"; continue; fi
    fi
    mkdir -p "$OUT/raw/$y" && unzip -qo "$zip" -d "$OUT/raw/$y"
    for d in "$OUT/raw/$y"/*/; do
        b=$(basename "$d")
        case $b in Freestyle*|Fastgame*|Unlimited*) ln -sfn "$d" "$OUT/sel20/${y}_$b" ;; esac
    done
done
"$GMDATA" gomoku convert --input "$OUT/sel20" --out "$OUT/shards20" --board-size 20 --rule freestyle
