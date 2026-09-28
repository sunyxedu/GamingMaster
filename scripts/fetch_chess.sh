#!/usr/bin/env bash
# Lichess standard chess end to end: the monthly rated dumps of database.lichess.org (CC0,
# ~29 GB of .pgn.zst and ~95 M games a month in 2026) -> high-quality gm-shards, month by
# month: download one dump (curl, resumable), convert it (`gmdata chess convert --hq`: rated,
# normal termination, >= 20 plies, no BOT accounts, both players at the measured per-speed
# Elo — blitz 2400, rapid 2200, classical 2000, no bullet; decision log 96), delete it
# (KEEP_RAW=1 keeps it).  The converter skips the movetext of every game its tags reject, so
# a month converts in about the time zstd needs to decompress it.  Mac + Linux; needs cargo
# (data_processing/README.md) or a prebuilt binary in GMDATA.  Idempotent / resumable: a
# month whose stats JSON exists in $OUT/logs is skipped; an interrupted conversion is redone
# from scratch (its directory is cleared first); a complete download is reused, a partial
# one resumed.
#
#   scripts/fetch_chess.sh [OUT_DIR]   (default data/chess under the repo)
#   env: MONTHS="2026-08"     months, YYYY-MM
#        VAL_EVERY=20         1 game in N (by its Lichess URL) into <month>/val
#        CONVERT_FLAGS=""     extra convert flags, e.g. "--speed-elo blitz=2500,rapid=2300"
#        KEEP_RAW=0           1 keeps each downloaded dump
#        RAW=$OUT/raw   GMDATA=<prebuilt gmdata binary>
#        SKIP_FETCH=1         convert the dumps already under RAW (no network)
#
# Output: $OUT/<month>/train/chess_* and $OUT/<month>/val/chess_* (one shard meta for all
# months); the conversion stats (seen / kept / kept_val / kept_by_speed / tag_rejected ...)
# in $OUT/logs/<month>.json.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
abspath() { mkdir -p "$1" && (cd -- "$1" && pwd); }
OUT=$(abspath "${1:-$ROOT/data/chess}")
MONTHS=${MONTHS:-2026-08}
VAL_EVERY=${VAL_EVERY:-20}
KEEP_RAW=${KEEP_RAW:-0}
RAW=$(abspath "${RAW:-$OUT/raw}")
read -r -a EXTRA <<< "${CONVERT_FLAGS:-}"

if [ -n "${GMDATA:-}" ]; then
  G=$GMDATA
else
  command -v cargo >/dev/null || export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
  (cd "$ROOT/data_processing" && cargo build --release -q -p gmdata)
  G="$ROOT/data_processing/target/release/gmdata"
fi

mkdir -p "$OUT/logs"
for m in $MONTHS; do
  log="$OUT/logs/$m.json"
  if [ -s "$log" ]; then
    echo "have $m"
    continue
  fi
  name="lichess_db_standard_rated_$m.pgn.zst"
  dump="$RAW/$name"
  if [ "${SKIP_FETCH:-0}" != 1 ] && [ ! -s "$dump" ]; then
    # resumable (-C -), retried; the complete file is renamed into place
    curl -fL --retry 20 --retry-delay 30 --retry-all-errors -C - -o "$dump.part" \
        "https://database.lichess.org/standard/$name"
    mv "$dump.part" "$dump"
  fi
  if [ ! -s "$dump" ]; then
    echo "no dump $dump" >&2
    exit 1
  fi
  rm -rf "${OUT:?}/$m"
  t0=$(date +%s)
  "$G" chess convert --input "$dump" --out "$OUT/$m" --hq --val-every "$VAL_EVERY" \
      ${EXTRA[@]+"${EXTRA[@]}"} > "$log.part"
  mv "$log.part" "$log"
  echo "$m ($(( $(date +%s) - t0 )) s): $(head -c 400 "$log")"
  if [ "$KEEP_RAW" != 1 ] && [ "${SKIP_FETCH:-0}" != 1 ]; then rm -f "$dump"; fi
done
echo "chess shards: $OUT/<month>/train and $OUT/<month>/val"
