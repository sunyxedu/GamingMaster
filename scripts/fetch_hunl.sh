#!/usr/bin/env bash
# Heads-up no-limit hold'em: the ACPC 2009-2017 heads-up no-limit hand histories
# (PHH format; "A Dataset of Poker Hand Histories", Zenodo record 10796885, CC-BY-4.0)
# -> verified candidate-layout gm-shards, end to end:
#   1. `gmdata hunl fetch`: HTTP range requests into the 20.3 GB Zenodo zip for ONLY
#      the 58 380 raw heads-up no-limit logs (4.85 GB, stored as .phhs.gz; resumable);
#   2. per competition year, `gmdata hunl convert` into a train and a val split
#      (hash-based, VAL_PER_MILLE): every hand replayed and verified by the gmrules
#      hunl rules against its recorded payoffs, duplicates dropped (the 2016 logs are
#      archived twice), the 2009 1/2/400 game rescaled x50 to 50/100/20000, and with
#      HQ=1 only the seats of the year's winning bots (bb/100 >= MIN_BB100 over all of
#      the year's hands).
# Mac + Linux; needs cargo (data_processing/README.md) or a prebuilt binary in
# GMDATA. Idempotent: a year / split whose stats JSON exists in $OUT/logs is skipped
# (an interrupted one is redone from scratch).
#
#   scripts/fetch_hunl.sh [OUT_DIR]      (default data/hunl under the repo; a relative
#                                         OUT_DIR is taken from the current directory)
#   env: YEARS="2009 2010 2011 2012 2013 2014 2016 2017"   HQ=1   MIN_BB100=0
#        KEEP_PER_MILLE=1000 (train: a uniform hash-based subsample, same on every run)
#        MAX_HANDS=<cap per year, train, first hands in file order>   VAL_PER_MILLE=5
#        RAW=$OUT/raw   GMDATA=<prebuilt gmdata binary>
#        FETCH_ARGS="--max-files-per-year 3"   (a small sample run)
#        SKIP_FETCH=1   (convert the logs already under RAW, e.g. rsynced; no network)
#
# Output: $OUT/train/<year>/hunl_*, $OUT/val/<year>/hunl_* (train with
# --data hunl=$OUT/train --val hunl=$OUT/val); conversion stats (records / converted /
# illegal / returns_mismatch / duplicates / frames ...) in $OUT/logs/*.json.
# Expected (all years): 175 M hands, 160.6 M unique (58 380 logs x 3 000 hands minus the
# duplicate 2016 archive), 5.56 decision frames per hand, 1.67 KB per frame on disk:
#   HQ=1 (default)        ~420 M frames, ~0.7 TB    (train; val 0.5 % of it)
#   HQ=0                  ~890 M frames, ~1.5 TB
#   KEEP_PER_MILLE=100    a tenth of either: HQ=1 ~16 M hands, ~42 M frames, ~70 GB
# For joint IL pretraining KEEP_PER_MILLE=100 is plenty and keeps GameShardDataset's
# in-memory per-sample index (~120 B per sample and process: ~20 GB for the full
# HQ corpus's ~160 M samples) small; the raw logs stay in RAW for a larger re-run.
# About 45 000 hands / s per conversion pass on 12 cores (HQ adds a parsing pass), so
# a few hours in all; the fetch is network-bound.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
abspath() { mkdir -p "$1" && (cd -- "$1" && pwd); }
OUT=$(abspath "${1:-$ROOT/data/hunl}")
YEARS=${YEARS:-"2009 2010 2011 2012 2013 2014 2016 2017"}
HQ=${HQ:-1}
MIN_BB100=${MIN_BB100:-0}
KEEP_PER_MILLE=${KEEP_PER_MILLE:-1000}
MAX_HANDS=${MAX_HANDS:-}
VAL_PER_MILLE=${VAL_PER_MILLE:-5}
RAW=$(abspath "${RAW:-$OUT/raw}")
FETCH_ARGS=${FETCH_ARGS:-}

if [ -n "${GMDATA:-}" ]; then
  G=$GMDATA
else
  # a user-level rustup (the cluster's ~/.cargo) when cargo is not on PATH
  command -v cargo >/dev/null || export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
  (cd "$ROOT/data_processing" && cargo build --release -q -p gmdata)
  G="$ROOT/data_processing/target/release/gmdata"
fi

mkdir -p "$OUT/logs"
if [ "${SKIP_FETCH:-0}" = 1 ]; then
  echo "SKIP_FETCH=1: converting the logs already under $RAW"
else
  # shellcheck disable=SC2086
  "$G" hunl fetch --out "$RAW" --years "$(echo $YEARS | tr ' ' ',')" $FETCH_ARGS > "$OUT/logs/fetch.json"
  cat "$OUT/logs/fetch.json"
fi

# (bash 3.2 safe: no empty-array expansion under set -u)
EXTRA=""
if [ "$HQ" = 1 ]; then EXTRA="--hq --min-bb100 $MIN_BB100"; fi
TRAIN_ONLY="--keep-per-mille $KEEP_PER_MILLE"
if [ -n "$MAX_HANDS" ]; then TRAIN_ONLY="$TRAIN_ONLY --max-records $MAX_HANDS"; fi

for y in $YEARS; do
  for split in train val; do
    log="$OUT/logs/convert_${split}_$y.json"
    if [ -s "$log" ]; then
      echo "have $split $y"
      continue
    fi
    rm -rf "${OUT:?}/$split/$y"
    limit=""
    if [ "$split" = train ]; then limit=$TRAIN_ONLY; fi
    # shellcheck disable=SC2086
    "$G" hunl convert --input "$RAW" --years "$y" --rescale --split "$split" \
        --val-per-mille "$VAL_PER_MILLE" $EXTRA $limit --out "$OUT/$split/$y" > "$log.part"
    mv "$log.part" "$log"
    echo "$split $y: $(head -c 400 "$log")"
  done
done
echo "hunl shards: $OUT/train $OUT/val"
