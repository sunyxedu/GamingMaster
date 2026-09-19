#!/bin/bash
#SBATCH --job-name=gm-il-multi
#SBATCH --partition=batch
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:8
#SBATCH --exclusive
#SBATCH --nodelist=gpu1,gpu2,gpu3
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))
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

# NCCL debugging - see what's happening with cross-node communication
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Try TCP sockets if InfiniBand/RoCE is misconfigured
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

srun apptainer exec --nv /scratch/$USER/pytorch.sif \
    torchrun \
        --nproc_per_node=8 \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --module training.train_il \
        --data_dir /scratch/$USER/gaming-master/data/training/Human.exe \
        --output_dir /scratch/$USER/gaming-master/checkpoints \
        --batch_size 4 \
        --d_model 256 \
        --epochs 150 \
        --num_workers 16 \
        --log_interval 1
