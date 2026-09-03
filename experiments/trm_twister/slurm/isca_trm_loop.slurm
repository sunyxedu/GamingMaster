#!/bin/bash
# SLURM job array: attention-in-loop TRM core, 2 games x 2 depths x seeds 0-1,
# plus the missing stock baseline (Breakout seed 1). 9 tasks.
#
#   sbatch -A <account> experiments/isca_trm_loop.slurm

#SBATCH --export=ALL
#SBATCH -D .
#SBATCH -p gpu
#SBATCH -A REPLACE_WITH_ACCOUNT_CODE   # overridden by -A on the sbatch line
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --array=0-8%6
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=H.Ahmad@exeter.ac.uk
#SBATCH -o logs/twloop_%A_%a.out
#SBATCH -e logs/twloop_%A_%a.err

set -u

GAMES=(ms_pacman ms_pacman breakout breakout ms_pacman ms_pacman breakout breakout breakout)
ARMS=(attn15 attn8 attn15 attn8 attn15 attn8 attn15 attn8 transformer)
SEEDS=(0 0 0 0 1 1 1 1 1)

i=$SLURM_ARRAY_TASK_ID
GAME=${GAMES[$i]}
ARM=${ARMS[$i]}
SEED=${SEEDS[$i]}

PY="$HOME/echelon-env/bin/python"
if [ ! -x "$PY" ]; then
    echo "ERROR: $PY missing: rebuild with echelon/experiments/isca_setup.sh" >&2
    exit 1
fi
cd "$HOME/twister-trm"

case "$ARM" in
    attn15)      export override_config='{"dynamics_core":"trm","trm_loop":"attention"}' ;;
    attn8)       export override_config='{"dynamics_core":"trm","trm_loop":"attention","trm_H_cycles":2,"trm_L_cycles":3}' ;;
    transformer) export override_config='{}' ;;
esac
export env_name="atari100k-${GAME}" run_name="twister_${ARM}_s${SEED}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
export TMPDIR="$HOME/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR"
export PYTHONUNBUFFERED=1

echo "=== array task $i | game=$GAME arm=$ARM seed=$SEED | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
"$PY" main.py --wandb --seed "$SEED" --keep_last_k 2 --log_figure_period_epoch 9999
