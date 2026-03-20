#!/bin/bash
#SBATCH --job-name=gm-il-finetune
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:8
#SBATCH --exclusive
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-%j.out

export PYTHONWARNINGS="ignore::UserWarning"
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONPATH=~/GamingMaster

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun apptainer exec --nv /scratch/$USER/pytorch.sif \
    torchrun --nproc_per_node=8 --nnodes=4 \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=$HEAD_NODE:29500 \
        ~/GamingMaster/train_il.py \
        --data_dir /scratch/$USER/gaming-master/data-1k/training/Human.exe \
        --output_dir /scratch/$USER/gaming-master/checkpoints/finetune \
        --batch_size 2 \
        --epochs 76 \
        --num_workers 16 \
        --lr 5e-5 \
        --warmup_steps 100 \
        --weight_decay 3e-4 \
        --resume /scratch/$USER/gaming-master/checkpoints/checkpoint_epoch36.pt \
        --reset_scheduler \
        --tf_start 0.15 \
        --tf_end 0.05
