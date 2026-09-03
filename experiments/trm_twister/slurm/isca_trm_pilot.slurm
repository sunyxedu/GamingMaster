#!/bin/bash
# SLURM job array: stock-TWISTER TRM pilot: 2 games x {baseline, trm}, seed 0.
# Everything stock except dynamics_core on the trm arm; both arms share the
# modernized (gymnasium+ale-py) env layer, so the comparison is internal.
# Tasks: i/2 -> game, i%2 -> arm.
#
#   sbatch -A <account> experiments/isca_trm_pilot.slurm

#SBATCH --export=ALL
#SBATCH -D .
#SBATCH -p gpu
#SBATCH -A REPLACE_WITH_ACCOUNT_CODE   # overridden by -A on the sbatch line
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --array=0-3%4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=H.Ahmad@exeter.ac.uk
#SBATCH -o logs/twpilot_%A_%a.out
#SBATCH -e logs/twpilot_%A_%a.err

set -u

GAMES=(ms_pacman breakout)
ARMS=(transformer trm)

i=$SLURM_ARRAY_TASK_ID
GAME=${GAMES[$((i/2))]}
ARM=${ARMS[$((i%2))]}
SEED=0

# Reuse the echelon conda env (same framework lineage, torch 2.11 cu128,
# gymnasium+ale-py). Call its python directly and fail loudly.
PY="$HOME/echelon-env/bin/python"
if [ ! -x "$PY" ]; then
    echo "ERROR: $PY missing: rebuild with echelon/experiments/isca_setup.sh" >&2
    exit 1
fi
cd "$HOME/twister-trm"

export env_name="atari100k-${GAME}" run_name="twister_${ARM}"
if [ "$ARM" = "trm" ]; then
    export override_config='{"dynamics_core":"trm"}'
else
    export override_config='{}'
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
export TMPDIR="$HOME/tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR"
export PYTHONUNBUFFERED=1

echo "=== array task $i | game=$GAME arm=$ARM seed=$SEED | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
# wandb run name comes from callback_path: callbacks/twister_<arm>/atari100k-<game>
"$PY" main.py --wandb --seed "$SEED" --keep_last_k 2 --log_figure_period_epoch 9999
