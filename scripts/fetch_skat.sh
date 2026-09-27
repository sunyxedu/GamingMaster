#!/usr/bin/env bash
# Skat: the International Skat Server's game archive end to end — download
# (iss-games-07-2024.sgf.bz2: 545 MiB, 9.1M games 2007-2024, humans and programs;
# plus the ISS ratings page), verify every record (every move replayed by the
# gmrules skat rules, every recorded result reproduced), per-player corpus
# statistics, high-quality conversion to candidate-action gm-shards and a 1% val
# split (by ISS game id).  Mac + Linux; needs cargo (data_processing/README.md).
# Details: data_processing/rs/gmdata/src/skat.rs.
#
#   scripts/fetch_skat.sh [OUT_DIR]          OUT_DIR default data/skat
#   HQ=0 scripts/fetch_skat.sh ...           every seat (no quality filter)
#   MAX_RECORDS=100000 scripts/fetch_skat.sh a quick partial run
#   CONVERT_FLAGS="--min-money 10" ...       extra `gmdata skat convert` flags
#   GMDATA=<prebuilt gmdata binary>          skips the cargo build
#
# Output: OUT/raw (archive + ratings page), OUT/verify.json + OUT/rejected.sgf (the
# parity report), OUT/players.json, OUT/train, OUT/val (gm-shards, game "skat"),
# OUT/convert_{train,val}.json. Expected sizes: see the module docs of
# data_processing/rs/gmdata/src/skat.rs ("Full scale"); re-runs skip the download.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-data/skat}
HQ=${HQ:-1}
MAX=()
if [ -n "${MAX_RECORDS:-}" ]; then MAX=(--max-records "$MAX_RECORDS"); fi
read -r -a EXTRA <<< "${CONVERT_FLAGS:-}"

if [ -n "${GMDATA:-}" ]; then
  G=$GMDATA
else
  # a user-level rustup (the cluster's ~/.cargo) when cargo is not on PATH
  command -v cargo >/dev/null || export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
  (cd "$ROOT/data_processing" && cargo build --release -q -p gmdata)
  G="$ROOT/data_processing/target/release/gmdata"
fi
cd "$ROOT"

mkdir -p "$OUT/raw"
"$G" skat fetch --out "$OUT/raw"
"$G" skat verify --input "$OUT/raw" --dump-rejected "$OUT/rejected.sgf" ${MAX[@]+"${MAX[@]}"} \
    | tee "$OUT/verify.json"
"$G" skat stats --input "$OUT/raw" --out "$OUT/players.json" ${MAX[@]+"${MAX[@]}"}

FILTER=()
if [ "$HQ" = 1 ]; then
    # high-quality recipe: the seats of players with >= 2000 games and >= 5 money points
    # per game in the corpus, or an ISS Elo >= 1050 (+-60) on the ratings page
    FILTER=(--hq --players "$OUT/players.json")
    if [ -s "$OUT/raw/iss-ratings.html" ]; then FILTER+=(--ratings "$OUT/raw/iss-ratings.html"); fi
fi
for split in train val; do
    "$G" skat convert --input "$OUT/raw" --out "$OUT/$split" --split "$split" \
        ${FILTER[@]+"${FILTER[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} ${MAX[@]+"${MAX[@]}"} \
        | tee "$OUT/convert_$split.json"
done
echo "skat shards: $OUT/train $OUT/val"
