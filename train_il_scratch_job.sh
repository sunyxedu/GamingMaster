#!/bin/bash
#SBATCH --job-name=gm-il-scratch
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:8
#SBATCH --exclusive
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-%j.out

# From-scratch IL with the recurrent neural memory + world-model objective.
#
# Effective batch = batch_size x 32 GPUs x grad_accum = 6 x 32 x 2 = 384
# full-game sequences per optimizer step. batch_size 6 is the conservative
# start (estimated peak ~11-14 GB/sequence in bf16 at T=1000); probe first:
#
#   srun --gres=gpu:nvidia_h200_nvl:1 --time=00:30:00 --pty \
#     apptainer exec --nv /scratch/$USER/pytorch.sif \
#     python ~/GamingMaster/scripts/check_training_memory.py model \
#       --temporal-mixer recurrent --predict-next-obs \
#       --batch-size 8 --seq-len 1000
#
# and raise batch_size / lower grad_accum accordingly.

export PYTHONWARNINGS="ignore::UserWarning"
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONPATH=~/GamingMaster

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun apptainer exec --nv /scratch/$USER/pytorch.sif \
    torchrun --nproc_per_node=8 --nnodes=$SLURM_NNODES \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=$HEAD_NODE:29500 \
        ~/GamingMaster/train_il.py \
        --data_dir /scratch/$USER/gaming-master/data-v2/training/Human.exe \
        --output_dir /scratch/$USER/gaming-master/checkpoints/il_recurrent_v1 \
        --temporal_mixer recurrent \
        --world_model_weight 0.2 \
        --tf_start 0.4 --tf_end 0.0 \
        --batch_size 6 --grad_accum 2 \
        --epochs 150 \
        --lr 1e-3 --warmup_steps 500 \
        --weight_decay 1e-4 \
        --num_workers 16 \
        --save_interval 1 --save_steps 500
