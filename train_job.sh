#!/bin/bash
#SBATCH --job-name=gm-il-train
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200_nvl:8
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

PYTHONPATH=~/GamingMaster apptainer exec --nv /scratch/$USER/pytorch.sif \
    torchrun --nproc_per_node=8 ~/GamingMaster/train_il.py \
        --data_dir /scratch/$USER/gaming-master/data/training/Human.exe \
        --output_dir /scratch/$USER/gaming-master/checkpoints \
        --batch_size 4 \
        --d_model 256 \
        --epochs 150 \
        --num_workers 16 \
        --log_interval 1
