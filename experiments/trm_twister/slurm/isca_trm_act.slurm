#!/bin/bash
# SLURM job array: full TRM recipe (deep supervision, ACT halting, EMA) on the
# attention loop with H=2, L=3. 2 games x seeds 0-1. 4 tasks.
#
#   sbatch -A <account> experiments/isca_trm_act.slurm

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
#SBATCH -o logs/twact_%A_%a.out
#SBATCH -e logs/twact_%A_%a.err

set -u

GAMES=(ms_pacman breakout ms_pacman breakout)
SEEDS=(0 0 1 1)

i=$SLURM_ARRAY_TASK_ID
GAME=${GAMES[$i]}
ARM=act8
SEED=${SEEDS[$i]}

PY="$HOME/echelon-env/bin/python"
if [ ! -x "$PY" ]; then
    echo "ERROR: $PY missing: rebuild with echelon/experiments/isca_setup.sh" >&2
    exit 1
fi
cd "$HOME/twister-trm"

export override_config='{"dynamics_core":"trm","trm_loop":"attention","trm_H_cycles":2,"trm_L_cycles":3,"trm_halt_max_steps":3,"ema":true}'
export env_name="atari100k-${GAME}" run_name="twister_${ARM}_s${SEED}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
export TMPDIR="$HOME/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR"
export PYTHONUNBUFFERED=1

echo "=== array task $i | game=$GAME arm=$ARM seed=$SEED | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
"$PY" main.py --wandb --seed "$SEED" --keep_last_k 2 --log_figure_period_epoch 9999
