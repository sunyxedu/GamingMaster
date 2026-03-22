#!/bin/bash
#SBATCH --job-name=gm-il-finetune
#SBATCH --partition=batch
#SBATCH --nodes=3
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
    torchrun --nproc_per_node=8 --nnodes=3 \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=$HEAD_NODE:29500 \
        ~/GamingMaster/train_il.py \
        --data_dir /scratch/$USER/gaming-master/data/training/asdvlaksudhiouwenf \
                   /scratch/$USER/gaming-master/data/training/qianglituomao \
                   /scratch/$USER/gaming-master/data/training/ejfvhnksdjkfassdfv \
                   /scratch/$USER/gaming-master/data/training/ToastedMelted \
                   /scratch/$USER/gaming-master/data/training/Crescent_ \
        --output_dir /scratch/$USER/gaming-master/checkpoints/rl_finetune \
        --batch_size 2 \
        --epochs 30 \
        --num_workers 16 \
        --lr 8e-5 \
        --warmup_steps 80 \
        --weight_decay 1e-3 \
        --resume /scratch/$USER/gaming-master/checkpoints/rl/rl_checkpoint_iter3300.pt \
        --reset_scheduler \
        --tf_start 0.4 \
        --tf_end 0.0
