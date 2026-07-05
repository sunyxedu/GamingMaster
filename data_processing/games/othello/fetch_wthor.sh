#!/usr/bin/env bash
# Fetch the WTHOR game database (Fédération Française d'Othello).
#
# The FFO publishes ~120k professional games (1977–present) as yearly
# zipped .wtb files ("base zippée par années") at
#   https://www.ffothello.org/wthor/base_zip/WTH_<YEAR>.ZIP
# Each zip contains one WTH_<YEAR>.wtb game file (16-byte header +
# 68-byte records; see wthor.py for the verified byte layout).  Player
# and tournament name tables (WTHOR.JOU / WTHOR.TRN) live in the same
# directory and are optional — the converter only stores numeric ids.
#
# Usage:  ./fetch_wthor.sh [dest_dir] [first_year] [last_year]
# Then:   uv run python -m data_processing.games.othello.convert \
#             --input <dest_dir> --output_dir data/othello/train
set -euo pipefail

DEST="${1:-data/othello/wthor}"
FIRST_YEAR="${2:-1977}"
LAST_YEAR="${3:-$(date +%Y)}"
BASE_URL="https://www.ffothello.org/wthor/base_zip"

mkdir -p "$DEST"
cd "$DEST"

for year in $(seq "$FIRST_YEAR" "$LAST_YEAR"); do
    zip="WTH_${year}.ZIP"
    if [ -f "WTH_${year}.wtb" ]; then
        echo "have WTH_${year}.wtb — skipping"
        continue
    fi
    echo "fetching ${BASE_URL}/${zip}"
    # -f: some years may not exist (no tournaments archived) — skip quietly.
    if curl -fsSL -o "$zip" "${BASE_URL}/${zip}"; then
        unzip -o -q "$zip"
        rm -f "$zip"
    else
        echo "  ${zip} not available — skipped"
    fi
done

echo "done: $(ls -1 ./*.wtb 2>/dev/null | wc -l | tr -d ' ') .wtb files in $DEST"
