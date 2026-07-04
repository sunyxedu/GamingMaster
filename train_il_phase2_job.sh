#!/bin/bash
#SBATCH --job-name=gm-il-p2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:8
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out

# Phase 2 IL — fine-tune the phase-1 floor on the HIGH-END players only, to
# pull the policy toward strong play (memory/dynamics were learned in phase 1).
# --finetune_from loads phase-1 WEIGHTS + architecture but starts a fresh run
# (epoch 1, new optimizer/schedule). Point CKPT at the best phase-1 checkpoint:
#
#   CKPT=/scratch/$USER/gaming-master/checkpoints/il_phase1/checkpoint_epoch40.pt \
#       sbatch train_il_phase2_job.sh
#
# Small high-end set → smaller effective batch (more optimizer steps).

export PYTHONWARNINGS="ignore::UserWarning"
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONPATH=~/GamingMaster
export PYTHONUNBUFFERED=1
# Recurrent-mode memory (measured, T=1000): B=6 OOMs a 141GB card; B=3
# peaks at 134.5/139.8 GiB — no headroom for DDP buffers + variable
# batch shapes. B=2 (~90 GiB peak) is the safe production setting.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${CKPT:?set CKPT=/path/to/phase1 checkpoint .pt}"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MICRO=${MICRO:-2}
TARGET_EFF=${TARGET_EFF:-48}
WORLD=$(( SLURM_NNODES * GPUS_PER_NODE ))
ACCUM=$(( TARGET_EFF / (MICRO * WORLD) ))
[ "$ACCUM" -lt 1 ] && ACCUM=1
echo "GPUs ${WORLD} | micro ${MICRO} x accum ${ACCUM} = effective $(( MICRO * WORLD * ACCUM ))"

# Optional resume: RESUME=/path/to/checkpoint.pt sbatch ... (any GPU count —
# ACCUM keeps the effective batch at TARGET_EFF, so the trajectory continues
# identically; the checkpoint carries optimizer + LR-schedule position)
RESUME_FLAG=""
[ -n "${RESUME:-}" ] && RESUME_FLAG="--resume $RESUME"

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
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
        ~/GamingMaster/train_il.py \
        --data_dir "${DATA_DIRS[@]}" \
        $RESUME_FLAG \
        --output_dir /scratch/$USER/gaming-master/checkpoints/il_phase2 \
        --finetune_from "$CKPT" \
        --world_model_weight 0.2 \
        --tf_start 0.2 --tf_end 0.0 \
        --batch_size $MICRO --grad_accum $ACCUM \
        --epochs 30 \
        --lr 3e-4 --warmup_steps 200 \
        --weight_decay 1e-4 \
        --num_workers 16 \
        --save_interval 1
