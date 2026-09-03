#!/bin/bash
# Resume one run from its last checkpoint (--load_last restores weights,
# optimizer, step, grad scaler, replay buffer, and the epoch counter).
# Used as a dependency of runs that may hit the 24 hour limit.
#
#   sbatch -A <account> --export=ALL,GAME=breakout,ARM=attn15,SEED=0 \
#          --dependency=afterany:<jobid>_<task> experiments/isca_trm_resume.slurm

#SBATCH --export=ALL
#SBATCH -D .
#SBATCH -p gpu
#SBATCH -A REPLACE_WITH_ACCOUNT_CODE   # overridden by -A on the sbatch line
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=H.Ahmad@exeter.ac.uk
#SBATCH -o logs/twresume_%j.out
#SBATCH -e logs/twresume_%j.err

set -u
: "${GAME:?}" "${ARM:?}" "${SEED:?}"

PY="$HOME/echelon-env/bin/python"
if [ ! -x "$PY" ]; then
    echo "ERROR: $PY missing: rebuild with echelon/experiments/isca_setup.sh" >&2
    exit 1
fi
cd "$HOME/twister-trm"

case "$ARM" in
    attn15)      export override_config='{"dynamics_core":"trm","trm_loop":"attention"}' ;;
    attn8)       export override_config='{"dynamics_core":"trm","trm_loop":"attention","trm_H_cycles":2,"trm_L_cycles":3}' ;;
    act8)        export override_config='{"dynamics_core":"trm","trm_loop":"attention","trm_H_cycles":2,"trm_L_cycles":3,"trm_halt_max_steps":3,"ema":true}' ;;
    trm)         export override_config='{"dynamics_core":"trm"}' ;;
    transformer) export override_config='{}' ;;
esac
export env_name="atari100k-${GAME}" run_name="twister_${ARM}_s${SEED}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
export TMPDIR="$HOME/tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR"
export PYTHONUNBUFFERED=1

echo "=== resume | game=$GAME arm=$ARM seed=$SEED | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
"$PY" main.py --wandb --seed "$SEED" --keep_last_k 2 --log_figure_period_epoch 9999 --load_last
