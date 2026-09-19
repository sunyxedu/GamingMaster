#!/bin/bash
#SBATCH --job-name=gm-rl-train
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:8
#SBATCH --exclusive
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-rl-%j.out

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

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun apptainer exec --nv /scratch/$USER/pytorch.sif \
    torchrun --nproc_per_node=8 --nnodes=2 \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=$HEAD_NODE:29500 \
        --module training.train_rl \
        --il_checkpoint /scratch/$USER/gaming-master/checkpoints/rl_finetune/checkpoint_epoch27.pt \
        --output_dir /scratch/$USER/gaming-master/checkpoints/rl2 \
        --replay_dir /scratch/$USER/gaming-master/data-1k/replays \
        --filtered_file /scratch/$USER/gaming-master/data-1k/filtered/Human.exe.json \
        --episodes_per_gpu 4 \
        --rollout_steps 1000 \
        --ppo_epochs 2 \
        --clip_range 0.15 \
        --gamma 0.999 \
        --gae_lambda 0.95 \
        --entropy_weight 0.02 \
        --entropy_weight_decay 0.9998 \
        --entropy_weight_min 0.008 \
        --kl_weight 0.05 \
        --il_weight 0.5 \
        --il_decay 0.995 \
        --lr 6e-5 \
        --lr_decay 0.9999 \
        --grad_clip 1.0 \
        --freeze_encoder_iters 200 \
        --pool_size 15 \
        --pool_update_interval 50 \
        --pool_update_winrate 0.55 \
        --pool_update_window 10 \
        --resume /scratch/$USER/gaming-master/checkpoints/rl2/rl_checkpoint_iter900.pt \
        --iterations 10000 \
        --save_interval 25 \
        --save_replay_interval 10 \
        --log_interval 1 \
        --il_data_dir /scratch/$USER/gaming-master/data/training/asdvlaksudhiouwenf \
        --invalidity_weight 0.5 \
        --city_reward_decay 0.9995 \
        --dense_reward_decay 0.9997