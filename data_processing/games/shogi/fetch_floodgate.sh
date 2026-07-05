#!/usr/bin/env bash
# Fetch floodgate (wdoor) shogi game records for the shogi IL pipeline.
#
# The floodgate server at wdoor.c.u-tokyo.ac.jp publishes yearly 7z
# archives of ALL games played that year; each archive expands into
# one-game-per-file CSA records (nested by date), which is exactly what
# data_processing.games.shogi.convert consumes.
#
# Requirements:
#   - wget (or swap in `curl -LO`)
#   - 7z to extract (`brew install p7zip` / `apt install p7zip-full`)
#
# Usage:
#   ./fetch_floodgate.sh [YEAR ...]     # default: 2023
#
# Then convert:
#   uv run python -m data_processing.games.shogi.convert \
#       --input floodgate_csa/2023 --output_dir shards/shogi \
#       --shard_size 256 --min_moves 30

set -euo pipefail

YEARS=("${@:-2023}")
BASE_URL="http://wdoor.c.u-tokyo.ac.jp/shogi/x"
OUT_DIR="floodgate_csa"

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

for YEAR in "${YEARS[@]}"; do
    ARCHIVE="wdoor${YEAR}.7z"
    echo "== fetching ${ARCHIVE} =="
    # Yearly archive of every floodgate game (per-game CSA files inside).
    wget -c "${BASE_URL}/${ARCHIVE}"
    echo "== extracting ${ARCHIVE} -> ${OUT_DIR}/${YEAR} =="
    7z x -y -o"${YEAR}" "${ARCHIVE}"
done

echo "done. convert with:"
echo "  uv run python -m data_processing.games.shogi.convert --input ${OUT_DIR}/<year> --output_dir shards/shogi"
