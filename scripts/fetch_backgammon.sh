#!/usr/bin/env bash
# Backgammon: data end to end — gnubg 2-ply self-play matches (`gmdata backgammon
# generate`: our rules engine runs every match with seeded dice, gnubg decides both
# seats, and gnubg's full legal move list is checked against ours at every checker
# play), then conversion to candidate-action gm-shards (`gmdata backgammon convert`:
# every record replayed and verified by the simulator's rules; gnubg's equities become
# soft targets, its match-winning chances engine values), plus a val split played
# with other dice.  Mac (arm64) + Linux (x86-64, glibc >= 2.28: the gnubg wheels);
# needs cargo and uv — gnubg (GPL-3.0-or-later) runs as an external process in a
# throw-away uv environment (`uv run --no-project --with gnubg==1.1.0a16`, fetched
# once), never installed into the project.
#
#   scripts/fetch_backgammon.sh [OUT_DIR] [MATCHES] [WORKERS]
#   OUT_DIR default data/backgammon, MATCHES default 100000, WORKERS default = CPUs
#   env: SEED (1), PLIES (2), MATCH_LENGTH (7), VAL_EVERY (50: one val match per 50),
#        FIRST (0: global index of the first match), RAW_ONLY=1 / CONVERT_ONLY=1,
#        GMDATA (a prebuilt gmdata binary: skips the cargo build)
#
# Full scale (cluster, one job per node; ~13 s per 7-point match per worker, one core
# and ~80 MB each): 100k matches ≈ 36M frames ≈ 80 GB of shards + ~1 GB of gz
# transcripts, ~1.4 h on 256 cores. Several nodes: the same SEED, disjoint ranges,
# convert once at the end —
#   FIRST=0     RAW_ONLY=1 scripts/fetch_backgammon.sh OUT 50000 128   # node 1
#   FIRST=50000 RAW_ONLY=1 scripts/fetch_backgammon.sh OUT 50000 128   # node 2
#   CONVERT_ONLY=1 scripts/fetch_backgammon.sh OUT
# A gnubg / rules disagreement is logged to OUT/raw*/discrepancies-*.ndjson (the match
# is skipped; the run goes on) and reported at the end — it must stay 0.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-data/backgammon}
MATCHES=${2:-100000}
WORKERS=${3:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}
SEED=${SEED:-1}
PLIES=${PLIES:-2}
MATCH_LENGTH=${MATCH_LENGTH:-7}
VAL_EVERY=${VAL_EVERY:-50}
FIRST=${FIRST:-0}

# user-level uv install (the cluster's ~/.local) when not on PATH
command -v uv >/dev/null || export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "uv not found (https://docs.astral.sh/uv/)" >&2; exit 1; }
if [ -n "${GMDATA:-}" ]; then
  G=$GMDATA
else
  # a user-level rustup (the cluster's ~/.cargo) when cargo is not on PATH
  command -v cargo >/dev/null || export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
  (cd "$ROOT/data_processing" && cargo build --release -q -p gmdata)
  G="$ROOT/data_processing/target/release/gmdata"
fi
cd "$ROOT"

VAL=$(( MATCHES / VAL_EVERY ))
[ "$VAL" -ge 1 ] || VAL=1
VAL_FIRST=$(( FIRST / VAL_EVERY ))

if [ "${CONVERT_ONLY:-0}" != 1 ]; then
    mkdir -p "$OUT/raw" "$OUT/raw_val"
    "$G" backgammon generate --out "$OUT/raw" --matches "$MATCHES" --first-match "$FIRST" \
        --workers "$WORKERS" --plies "$PLIES" --match-length "$MATCH_LENGTH" --seed "$SEED" \
        --keep-going > "$OUT/raw/generate-s$SEED-m$FIRST.json"
    "$G" backgammon generate --out "$OUT/raw_val" --matches "$VAL" --first-match "$VAL_FIRST" \
        --workers "$WORKERS" --plies "$PLIES" --match-length "$MATCH_LENGTH" \
        --seed $(( SEED + 1000 )) --keep-going > "$OUT/raw_val/generate-s$(( SEED + 1000 ))-m$VAL_FIRST.json"
    cat "$OUT/raw/generate-s$SEED-m$FIRST.json" "$OUT/raw_val/generate-s$(( SEED + 1000 ))-m$VAL_FIRST.json"
fi
if [ "${RAW_ONLY:-0}" != 1 ]; then
    for split in train val; do
        raw="$OUT/raw"
        [ "$split" = val ] && raw="$OUT/raw_val"
        rm -rf "$OUT/$split"
        "$G" backgammon convert --input "$raw" --out "$OUT/$split" --match-length "$MATCH_LENGTH" \
            --soft-k 8 --engine-values | tee "$OUT/convert-$split.json"
    done
    echo "backgammon shards: $OUT/train $OUT/val"
fi
if ls "$OUT"/raw*/discrepancies-*.ndjson >/dev/null 2>&1; then
    echo "WARNING: gnubg / gmrules disagreements (rules bugs to fix):" >&2
    wc -l "$OUT"/raw*/discrepancies-*.ndjson >&2
fi
