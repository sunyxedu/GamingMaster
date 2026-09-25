#!/bin/bash
#SBATCH --job-name=gm-multi
#SBATCH --partition=batch
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#
# One multi-game run (training.train_multi_game or training.eval_multi_game).
# Pass everything through environment-variable PREFIXES (never
# `sbatch --export=VAR=...`, which gets jobs killed on hephaestus):
#
#   RUN=core-rd-moe ARGS="--data chess=... --steps 20000 --override moe_experts=8" \
#       sbatch --requeue scripts/slurm/multi/multi_game_job.sh
#
#   MODULE=training.eval_multi_game RUN=eval-x ARGS="--ckpt ... --val ..." sbatch ...
#
# generals joins like any game: --data generals=DIR takes a gio record
# directory (gmdata generals ladder|pipeline output, e.g. .../training) or,
# much faster to load and ~10x smaller, its gm-shards
#   data_processing/target/release/gmdata generals shard --input DIR --out SHARDS
# Its sequences are ~10x longer and 627 tokens per frame, and they are cut to
# their FIRST frames: bound them with --game_max_len generals=512
# --game_val_max_len generals=512 --game_batch generals=2.
#
# Paths live under $GM_HOME (default ~/gm-core) — not /scratch.
set -euo pipefail
GM_HOME=${GM_HOME:-$HOME/gm-core}
RUN=${RUN:?set RUN}
MODULE=${MODULE:-training.train_multi_game}
OUT=${OUT:-$HOME/gm-runs/multi/$RUN}
mkdir -p "$OUT"
cd "$GM_HOME"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
export PYTHONPATH="$GM_HOME${PYTHONPATH:+:$PYTHONPATH}"
echo "[$(date)] $RUN on $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
EXTRA=""
if [ "$MODULE" = "training.train_multi_game" ]; then EXTRA="--save_dir $OUT"; fi
# shellcheck disable=SC2086
"$GM_HOME/.venv/bin/python" -u -m "$MODULE" $ARGS $EXTRA 2>&1 | tee "$OUT/log.txt"
echo "[$(date)] done"
