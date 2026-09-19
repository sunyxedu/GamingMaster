#!/bin/bash
#SBATCH --job-name=gm-il-p2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:8
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out

# Phase 2 IL — fine-tune the phase-1 floor on ALL human players (everything
# except the Human.exe bot bulk), to pull the policy toward human play
# (memory/dynamics were learned in phase 1). The *-val splits are held out
# for evaluation and deliberately NOT trained on.
# --finetune_from loads phase-1 WEIGHTS + architecture but starts a fresh run
# (epoch 1, new optimizer/schedule). Point CKPT at the best phase-1 checkpoint:
#
#   CKPT=/scratch/$USER/gaming-master/checkpoints/il_phase1/checkpoint_epoch40.pt \
#       sbatch scripts/slurm/il/train_il_phase2_job.sh
#
# Resuming an INTERRUPTED phase-2 run (any GPU count — ACCUM keeps the
# effective batch at TARGET_EFF; the checkpoint carries optimizer + LR
# schedule; CKPT is not needed):
#
#   RESUME=/scratch/$USER/gaming-master/checkpoints/il_phase2/checkpoint_epochN.pt \
#       sbatch scripts/slurm/il/train_il_phase2_job.sh
#
# Small high-end set → smaller effective batch (more optimizer steps).

export PYTHONWARNINGS="ignore::UserWarning"
export TORCH_CPP_LOG_LEVEL=ERROR
# Slurm executes a spool copy: start from the explicit checkout or submit directory.
_gm_start="${GAMINGMASTER_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname -- "${BASH_SOURCE[0]}")}}"
_gm_root=$(cd -- "$_gm_start" && pwd) || exit 1
while [[ ! -f "$_gm_root/scripts/slurm/_env.sh" ]]; do
    if [[ "$_gm_root" == / ]]; then
        echo "Cannot find GamingMaster; set GAMINGMASTER_ROOT to the checkout root." >&2
        exit 1
    fi
    _gm_root=$(dirname -- "$_gm_root")
done
source "$_gm_root/scripts/slurm/_env.sh" || exit 1
unset _gm_start _gm_root
export PYTHONUNBUFFERED=1
# Recurrent-mode memory (measured, T=1000): B=6 OOMs a 141GB card; B=3
# peaks at 134.5/139.8 GiB — no headroom for DDP buffers + variable
# batch shapes. B=2 (~90 GiB peak) is the safe production setting.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Node-local tmp for multiprocessing scratch: on NFS, deleting worker temp
# dirs hits ".nfs* Device or resource busy" tracebacks at every teardown.
export TMPDIR=/tmp

# Fresh start (--finetune_from phase-1 CKPT) and resume (--resume phase-2
# checkpoint) are mutually exclusive in training/train_il.py: a resumed run must keep
# its own optimizer/schedule, not re-seed from phase 1. CKPT is only needed
# for a fresh start.
if [ -n "${RESUME:-}" ]; then
    INIT_FLAGS=(--resume "$RESUME")
else
    : "${CKPT:?set CKPT=/path/to/phase1 checkpoint .pt}"
    INIT_FLAGS=(--finetune_from "$CKPT")
fi

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MICRO=${MICRO:-2}
TARGET_EFF=${TARGET_EFF:-48}
WORLD=$(( SLURM_NNODES * GPUS_PER_NODE ))
ACCUM=$(( TARGET_EFF / (MICRO * WORLD) ))
[ "$ACCUM" -lt 1 ] && ACCUM=1
echo "GPUs ${WORLD} | micro ${MICRO} x accum ${ACCUM} = effective $(( MICRO * WORLD * ACCUM ))"

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
TRAIN_ROOT=/scratch/$USER/gaming-master/data-v2/training
DATA_DIRS=(
    "$TRAIN_ROOT/Average Joe"
    "$TRAIN_ROOT/Hermione Granger"
    "$TRAIN_ROOT/L_7d_gae90_30k_ema"
    "$TRAIN_ROOT/Sinyth"
    "$TRAIN_ROOT/YiShengYiShi"
    "$TRAIN_ROOT/asdvlaksudhiouwenf"
    "$TRAIN_ROOT/bike9800"
    "$TRAIN_ROOT/cute_x1aoyu_ovo"
    "$TRAIN_ROOT/d3st1ny"
)

srun apptainer exec --nv /scratch/$USER/pytorch.sif \
    torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=$HEAD_NODE:29500 \
        --module training.train_il \
        --data_dir "${DATA_DIRS[@]}" \
        "${INIT_FLAGS[@]}" \
        --output_dir /scratch/$USER/gaming-master/checkpoints/il_phase2 \
        --world_model_weight 0.2 \
        --tf_start 0.2 --tf_end 0.0 \
        --batch_size $MICRO --grad_accum $ACCUM \
        --epochs 30 \
        --lr 3e-4 --warmup_steps 200 \
        --weight_decay 1e-4 \
        --num_workers 16 \
        --save_interval 1
