#!/bin/bash
#SBATCH --job-name=gm-rl-train
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:8
#SBATCH --exclusive
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-rl-%j.out

export PYTHONWARNINGS="ignore::UserWarning"
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONPATH=~/GamingMaster

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun apptainer exec --nv /scratch/$USER/pytorch.sif \
    torchrun --nproc_per_node=8 --nnodes=4 \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=$HEAD_NODE:29500 \
        ~/GamingMaster/train_rl.py \
        --il_checkpoint /scratch/$USER/gaming-master/checkpoints/finetune/checkpoint_epoch67.pt \
        --output_dir /scratch/$USER/gaming-master/checkpoints/rl \
        --rollout_steps 1000 \
        --ppo_epochs 4 \
        --clip_range 0.2 \
        --gamma 0.99 \
        --gae_lambda 0.95 \
        --entropy_weight 0.03 \
        --kl_weight 0.01 \
        --il_weight 0.5 \
        --il_decay 0.9999 \
        --lr 1e-4 \
        --grad_clip 1.0 \
        --freeze_encoder_iters 1000 \
        --pool_size 10 \
        --pool_update_interval 100 \
        --iterations 10000 \
        --save_interval 100 \
        --log_interval 10
