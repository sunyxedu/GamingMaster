#!/bin/bash
#SBATCH --job-name=gm-il-scratch
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:8
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-%j.out

# From-scratch IL: recurrent neural memory + world-model objective.
#
# The effective batch (~TARGET_EFF full games / optimizer step) is held
# constant by auto-scaling grad_accum from however many GPUs the job got —
# fewer GPUs only means slower wall-clock, never different training.
#
# Submit-time variants (no editing needed):
#   sbatch scripts/slurm/il/train_il_scratch_job.sh                    # 1 node x 8 H200
#   sbatch --nodes=2 scripts/slurm/il/train_il_scratch_job.sh          # 2 nodes x 8
#   sbatch --nodes=3 scripts/slurm/il/train_il_scratch_job.sh          # gpu[1-3] when drained
#   # Harvest the 4-full-card halves of gpu5/gpu6 with a uniform 4/node job:
#   GPUS_PER_NODE=4 sbatch --export=ALL --nodes=4 \
#       --gres=gpu:nvidia_h200_nvl:4 scripts/slurm/il/train_il_scratch_job.sh
#
# NOTE: MIG slices (gres names like nvidia_h200_nvl_3g.71gb) cannot join
# multi-GPU NCCL jobs — use them for single-GPU probes/smoke tests only.

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
export TMPDIR=/tmp   # live logs (unbuffered stdout)

# Uncomment if cross-node NCCL misbehaves (worked around IB issues before):
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=0

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MICRO=${MICRO:-2}                 # per-GPU micro-batch (probe before raising)
TARGET_EFF=${TARGET_EFF:-384}
WARMUP=${WARMUP:-100}   # ~5% of total optimizer steps     # target effective batch (full games/step)
WORLD=$(( SLURM_NNODES * GPUS_PER_NODE ))
ACCUM=$(( TARGET_EFF / (MICRO * WORLD) ))
[ "$ACCUM" -lt 1 ] && ACCUM=1
echo "GPUs: ${WORLD} (${SLURM_NNODES} nodes x ${GPUS_PER_NODE}) | micro ${MICRO} x accum ${ACCUM} = effective $(( MICRO * WORLD * ACCUM ))"

# Optional resume: RESUME=/path/to/checkpoint.pt sbatch ...
RESUME_FLAG=""
[ -n "${RESUME:-}" ] && RESUME_FLAG="--resume $RESUME"

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

# Training data: one dir per collected player (names with spaces are quoted).
TRAIN_ROOT=/scratch/$USER/gaming-master/data-v2/training
DATA_DIRS=(
    "$TRAIN_ROOT/bike9800"
    "$TRAIN_ROOT/Average Joe"
    "$TRAIN_ROOT/d3st1ny"
    "$TRAIN_ROOT/L_7d_gae90_30k_ema"
    "$TRAIN_ROOT/cute_x1aoyu_ovo"
)

srun apptainer exec --nv /scratch/$USER/pytorch.sif \
    torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=$HEAD_NODE:29500 \
        --module training.train_il \
        --data_dir "${DATA_DIRS[@]}" \
        $RESUME_FLAG \
        --output_dir /scratch/$USER/gaming-master/checkpoints/il_recurrent_v1 \
        --temporal_mixer recurrent \
        --world_model_weight 0.2 \
        --tf_start 0.4 --tf_end 0.0 \
        --batch_size $MICRO --grad_accum $ACCUM \
        --epochs 150 \
        --lr 1e-3 --warmup_steps $WARMUP \
        --weight_decay 1e-4 \
        --num_workers 16 \
        --save_interval 1 --save_steps 500
