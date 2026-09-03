#!/bin/bash
# SLURM job array: stock-TWISTER TRM pilot, ms_pacman seeds 1-2 x {baseline, trm}.
# Completes the 3-seed set with isca_trm_pilot.slurm (seed 0).
# Tasks: arm = i%2, seed = 1 + i/2.
#
#   sbatch -A <account> experiments/isca_trm_seeds.slurm

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
#SBATCH -o logs/twseeds_%A_%a.out
#SBATCH -e logs/twseeds_%A_%a.err

set -u

ARMS=(transformer trm)

i=$SLURM_ARRAY_TASK_ID
GAME=ms_pacman
ARM=${ARMS[$((i%2))]}
SEED=$((1 + i/2))

PY="$HOME/echelon-env/bin/python"
if [ ! -x "$PY" ]; then
    echo "ERROR: $PY missing: rebuild with echelon/experiments/isca_setup.sh" >&2
    exit 1
fi
cd "$HOME/twister-trm"

export env_name="atari100k-${GAME}" run_name="twister_${ARM}_s${SEED}"
if [ "$ARM" = "trm" ]; then
    export override_config='{"dynamics_core":"trm"}'
else
    export override_config='{}'
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
export TMPDIR="$HOME/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR"
export PYTHONUNBUFFERED=1

echo "=== array task $i | game=$GAME arm=$ARM seed=$SEED | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
"$PY" main.py --wandb --seed "$SEED" --keep_last_k 2 --log_figure_period_epoch 9999
