#!/bin/bash
#SBATCH --job-name=gm-rl-p3
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-rl-%j.out

# Phase 3 RL — PPO self-play from the phase-2 IL policy. Each rank is a
# symmetric actor+learner: collect EPISODES self-play games vs a pool
# opponent, then a synchronized PPO + IL-auxiliary update.
#
# Starting point is the phase-2 VAL-LOSS optimum (epoch 5, val loss 2.33 —
# best-calibrated policy; later epochs memorize the training replays and
# their overconfident logits make poor PPO/KL anchors). The same checkpoint
# seeds the frozen KL reference and the opponent-pool baseline.
#
#   sbatch scripts/slurm/rl/train_rl_phase3_job.sh
#   RESUME=/scratch/$USER/gaming-master/checkpoints/rl_v2/rl_checkpoint_iterN.pt \
#       sbatch scripts/slurm/rl/train_rl_phase3_job.sh

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
# Recurrent-mode PPO re-runs full-episode BPTT (T up to 1000) like IL
# training; same allocator + node-local tmp settings apply.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/tmp

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
EPISODES=${EPISODES:-6}
CKPT=${CKPT:-/scratch/$USER/gaming-master/checkpoints/il_phase2/checkpoint_epoch5.pt}

if [ -n "${RESUME:-}" ]; then
    INIT_FLAGS=(--resume "$RESUME")
else
    INIT_FLAGS=(--il_checkpoint "$CKPT")
fi

# Real replay maps for the env (fall back to random maps if v2 raw replays
# aren't staged on scratch yet).
REPLAY_DIR=${REPLAY_DIR:-/scratch/$USER/gaming-master/data-v2/replays}
FILTERED_FILE=${FILTERED_FILE:-/scratch/$USER/gaming-master/data-v2/filtered/Human.exe.json}
MAP_FLAGS=()
if [ -d "$REPLAY_DIR" ] && [ -f "$FILTERED_FILE" ]; then
    MAP_FLAGS=(--replay_dir "$REPLAY_DIR" --filtered_file "$FILTERED_FILE")
else
    echo "[warn] map pool sources missing ($REPLAY_DIR / $FILTERED_FILE) — using random maps"
fi

# IL auxiliary loss on the same nine human players phase 2 fine-tuned on.
TRAIN_ROOT=/scratch/$USER/gaming-master/data-v2/training
IL_DIRS=(
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

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun apptainer exec --nv /scratch/$USER/pytorch.sif \
    torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=$HEAD_NODE:29500 \
        --module training.train_rl \
        "${INIT_FLAGS[@]}" \
        "${MAP_FLAGS[@]}" \
        --output_dir /scratch/$USER/gaming-master/checkpoints/rl_v2 \
        --il_data_dir "${IL_DIRS[@]}" \
        --episodes_per_gpu $EPISODES \
        --rollout_steps 1000 \
        --act_steps 1 \
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
        --invalidity_weight 0.5 \
        --city_reward_decay 0.9995 \
        --dense_reward_decay 0.9997 \
        --lr 6e-5 \
        --lr_decay 0.9999 \
        --grad_clip 1.0 \
        --freeze_encoder_iters 200 \
        --pool_size 15 \
        --pool_update_interval 50 \
        --pool_update_winrate 0.55 \
        --pool_update_window 10 \
        --iterations 10000 \
        --save_interval 25 \
        --save_replay_interval 10 \
        --log_interval 1
