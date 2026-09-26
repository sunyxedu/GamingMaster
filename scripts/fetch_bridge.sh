#!/usr/bin/env bash
# Contract bridge: data end to end — download the OpenSpiel WBridge5 dataset (GCS;
# ~1.16M deals bid in SAYC and played by WBridge5; dealer N, nobody vulnerable) and
# Ben's WBC play data (GitHub release "training-data"; ~1.9M boards played by bridge
# robots, every dealer / vulnerability), then convert them to candidate-action
# gm-shards with `gmdata bridge convert`, which replays and verifies every record with
# the gmrules bridge rules (the simulator's own code; recorded contracts / results
# are checked too).  Optional third source "bba": Ben's BBA bidding sets (EPBot
# auctions with each deal's result, no play; the SAYC sets by default).  Mac + Linux;
# needs cargo (data_processing/README.md).  BBO vugraph archives are NOT fetched
# (pending a decision; the LIN reader is ready).
#
#   scripts/fetch_bridge.sh [OUT_DIR]        OUT_DIR default data/bridge
#   SOURCES="openspiel ben"                  sources (default; add "bba" for the
#                                            bidding sets)
#   BBA_SETS="BEN-sayc,Robo-Sayc,..."        bba sets (default: the 4 SAYC sets; all
#                                            10: see `gmdata bridge fetch --help`)
#   MAX_RECORDS=N                            cap every conversion (smoke runs)
#   KEEP_RAW=0                               delete the downloads after converting
#
# Output (one shard family: same meta and tensors, so one --data root per split):
#   OUT/train/{openspiel,ben,bba}  OUT/val/{openspiel,ben,bba}  OUT/test/openspiel
#   → train_multi_game --data bridge=OUT/train --val bridge=OUT/val
# Ben and BBA have no official split: deals go to val by hash (1 in 100; both rooms
# of a deal stay together).  Sizes (~810 bytes per decision frame): openspiel train
# 1,158,441 deals, 72.7M frames, ~59 GB (valid / test 10k deals, ~0.5 GB each); ben
# 1,883,853 boards kept of 1,887,139, 117.9M frames, ~95 GB; bba ~1.15M auctions,
# ~12.3M frames, ~10 GB per set.  Downloads: 0.38 GB + 0.15 GB (+ 0.08 GB per bba
# set).  Re-runs skip the conversions already finished (OUT/<split>/<source>/.done).
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-data/bridge}
SOURCES=${SOURCES:-"openspiel ben"}
KEEP_RAW=${KEEP_RAW:-1}
MAX=()
if [ -n "${MAX_RECORDS:-}" ]; then
    MAX=(--max-records "$MAX_RECORDS")
fi

# a user-level rustup (the cluster's ~/.cargo) when cargo is not on PATH
command -v cargo >/dev/null || export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
(cd "$ROOT/data_processing" && cargo build --release -q -p gmdata)
G="$ROOT/data_processing/target/release/gmdata"

# convert <dest> <gmdata bridge convert args...>: skipped when finished before; an
# interrupted conversion is redone from scratch (the writer appends to a directory)
convert() {
    local dest=$1
    shift
    if [ -f "$dest/.done" ]; then
        echo "have $dest"
        return
    fi
    rm -rf "$dest"
    local stats
    stats=$("$G" bridge convert --out "$dest" "$@" ${MAX[@]+"${MAX[@]}"})
    echo "$stats"
    printf '%s\n' "$stats" > "$dest/.done"
}

for src in $SOURCES; do
    case "$src" in
        openspiel)
            "$G" bridge fetch --source openspiel --out "$OUT/raw/openspiel"
            convert "$OUT/train/openspiel" --format openspiel --input "$OUT/raw/openspiel/train.txt"
            convert "$OUT/val/openspiel" --format openspiel --input "$OUT/raw/openspiel/valid.txt"
            convert "$OUT/test/openspiel" --format openspiel --input "$OUT/raw/openspiel/test.txt"
            ;;
        ben)
            "$G" bridge fetch --source ben --out "$OUT/raw/ben"
            convert "$OUT/train/ben" --format jack --input "$OUT/raw/ben/play_jackformat.zip" --split train
            convert "$OUT/val/ben" --format jack --input "$OUT/raw/ben/play_jackformat.zip" --split val
            ;;
        bba)
            "$G" bridge fetch --source bba --out "$OUT/raw/bba" ${BBA_SETS:+--sets "$BBA_SETS"}
            convert "$OUT/train/bba" --format pbn --input "$OUT/raw/bba" --split train
            convert "$OUT/val/bba" --format pbn --input "$OUT/raw/bba" --split val
            ;;
        *)
            echo "unknown source '$src' (openspiel | ben | bba)" >&2
            exit 2
            ;;
    esac
    if [ "$KEEP_RAW" = "0" ]; then
        rm -rf "${OUT:?}/raw/$src"
    fi
done
echo "bridge shards: $OUT/train $OUT/val (+ $OUT/test/openspiel)"
