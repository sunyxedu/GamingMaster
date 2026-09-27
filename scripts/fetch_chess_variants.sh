#!/usr/bin/env bash
# Lichess chess variants end to end: the monthly rated dumps of database.lichess.org
# (CC0) of chess960, crazyhouse, atomic, antichess, three-check, king of the hill, horde
# and racing kings -> verified gm-shards of the ONE gmrules game `chess_variants`
# (`gmdata chess-variants convert`: every SAN resolved by shakmaty, every record replayed
# by the rules the simulator uses, every rules end checked against the PGN result),
# month by month and variant by variant: download one dump, convert it, delete it
# (KEEP_RAW=1 keeps it).  Mac + Linux; needs cargo (data_processing/README.md) or a
# prebuilt binary in GMDATA.  Idempotent / resumable: a (split, variant, month) whose
# stats JSON exists in $OUT/logs is skipped; an interrupted one is redone from scratch
# (its shard directory is cleared first: the shard writer appends to a non-empty
# directory); a complete download is reused.
#
#   scripts/fetch_chess_variants.sh [OUT_DIR]   (default data/chess_variants under the repo;
#                                                a relative OUT_DIR is taken from the
#                                                current directory)
#   env: VARIANTS="chess960 crazyhouse atomic antichess threecheck kingofthehill horde racingkings"
#        MONTHS="2026-08"   train months, YYYY-MM (`gmdata chess-variants list` shows the dumps)
#        VAL_MONTHS=""      validation months (whole months held out; keep them out of MONTHS)
#        HQ=1               convert --hq: rated, normal termination, >= 60 s, >= 4 plies and
#                           the top 10 % of each dump by the lower Elo
#        CONVERT_FLAGS=""   extra convert flags, e.g. "--min-clock-s 5"
#        MAX_BYTES=<N>      a sample: the first N compressed bytes of each dump (.head.pgn)
#        KEEP_RAW=0         1 keeps each downloaded dump (dumps given with SKIP_FETCH=1 are kept)
#        RAW=$OUT/raw   GMDATA=<prebuilt gmdata binary>
#        SKIP_FETCH=1       convert the dumps already under RAW (no network)
#
# Output: $OUT/<train|val>/<variant>/<month>/chess_variants_* (every variant and month
# shares one shard meta: --data chess_variants=$OUT/train --val chess_variants=$OUT/val);
# the conversion stats (records / converted / illegal / returns_mismatch / adjudicated /
# frames ...) in $OUT/logs/<split>_<variant>_<month>.json.
# Sizes (2026-08): 1.85 M rated games in 430 MB of dumps (chess960 116 MB, the largest);
# HQ=1 keeps 6.6-8.5 % of the games (the 2026-08 samples), ~0.14 M games / ~7.5 M
# frames / ~2 GB of shards per month (~290 bytes per frame); illegal and
# returns_mismatch are 0 on the samples.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
abspath() { mkdir -p "$1" && (cd -- "$1" && pwd); }
OUT=$(abspath "${1:-$ROOT/data/chess_variants}")
VARIANTS=${VARIANTS:-"chess960 crazyhouse atomic antichess threecheck kingofthehill horde racingkings"}
MONTHS=${MONTHS:-2026-08}
VAL_MONTHS=${VAL_MONTHS:-}
HQ=${HQ:-1}
KEEP_RAW=${KEEP_RAW:-0}
MAX_BYTES=${MAX_BYTES:-}
RAW=$(abspath "${RAW:-$OUT/raw}")
read -r -a EXTRA <<< "${CONVERT_FLAGS:-}"

if [ -n "${GMDATA:-}" ]; then
  G=$GMDATA
else
  # a user-level rustup (the cluster's ~/.cargo) when cargo is not on PATH
  command -v cargo >/dev/null || export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
  (cd "$ROOT/data_processing" && cargo build --release -q -p gmdata)
  G="$ROOT/data_processing/target/release/gmdata"
fi

# the dump directory names of database.lichess.org
lichess_dir() {
  case "$1" in
    threecheck) echo threeCheck ;;
    kingofthehill) echo kingOfTheHill ;;
    racingkings) echo racingKings ;;
    *) echo "$1" ;;
  esac
}

FLAGS=()
if [ "$HQ" = 1 ]; then FLAGS+=(--hq); fi
FETCH=()
SUFFIX=pgn.zst
if [ -n "$MAX_BYTES" ]; then
  FETCH=(--max-bytes "$MAX_BYTES")
  SUFFIX=head.pgn
fi

mkdir -p "$OUT/logs"
convert_split() {
  local split=$1 months=$2 m v log dump
  for m in $months; do
    for v in $VARIANTS; do
      log="$OUT/logs/${split}_${v}_$m.json"
      if [ -s "$log" ]; then
        echo "have $split $v $m"
        continue
      fi
      dump="$RAW/lichess_db_$(lichess_dir "$v")_rated_$m.$SUFFIX"
      if [ "${SKIP_FETCH:-0}" != 1 ] && [ ! -s "$dump" ]; then
        "$G" chess-variants fetch --variant "$v" --month "$m" --out "$RAW" ${FETCH[@]+"${FETCH[@]}"} \
            >> "$OUT/logs/fetch.jsonl"
      fi
      if [ ! -s "$dump" ]; then
        echo "no dump $dump" >&2
        exit 1
      fi
      rm -rf "${OUT:?}/$split/$v/$m"
      "$G" chess-variants convert --input "$dump" --variant "$v" --out "$OUT/$split/$v/$m" \
          ${FLAGS[@]+"${FLAGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} > "$log.part"
      mv "$log.part" "$log"
      echo "$split $v $m: $(head -c 300 "$log")"
      if [ "$KEEP_RAW" != 1 ] && [ "${SKIP_FETCH:-0}" != 1 ]; then rm -f "$dump"; fi
    done
  done
}

convert_split train "$MONTHS"
if [ -n "$VAL_MONTHS" ]; then convert_split val "$VAL_MONTHS"; fi
echo "chess_variants shards: $OUT/train${VAL_MONTHS:+ $OUT/val}"
