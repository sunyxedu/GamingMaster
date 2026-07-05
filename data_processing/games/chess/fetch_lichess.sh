#!/usr/bin/env bash
# Download a monthly Lichess standard-rated PGN dump.
#
#   ./fetch_lichess.sh 2024-01 [out_dir]
#
# Sizes: recent months are ~30 GB compressed (~100M games); early years
# are far smaller (2013-01 is ~15 MB, ~120k games — good for smoke runs).
# Full list + checksums: https://database.lichess.org/
#
# The converter streams .pgn.zst directly (via the `zstandard` package),
# so there is NO need to decompress. If you want the raw .pgn anyway:
#   zstd -d <file>.pgn.zst        # needs `brew install zstd` / apt zstd
# (expect ~7x expansion on disk).

set -euo pipefail

YM="${1:?usage: fetch_lichess.sh YYYY-MM [out_dir]}"
OUT_DIR="${2:-data/chess/raw}"
FILE="lichess_db_standard_rated_${YM}.pgn.zst"
URL="https://database.lichess.org/standard/${FILE}"

mkdir -p "$OUT_DIR"
echo "Fetching ${URL} -> ${OUT_DIR}/${FILE}"
curl -L --fail --continue-at - -o "${OUT_DIR}/${FILE}" "$URL"
echo "Done: $(du -h "${OUT_DIR}/${FILE}" | cut -f1)"
echo "Next: uv run python -m data_processing.games.chess.convert \\"
echo "        --input ${OUT_DIR}/${FILE} --output_dir data/chess/train \\"
echo "        --min_elo 1600 --min_plies 10 --skip_bullet"
