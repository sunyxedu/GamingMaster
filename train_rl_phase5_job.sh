#!/bin/bash
#SBATCH --job-name=gm-rl-p5
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:8
#SBATCH --time=5-00:00:00
#SBATCH --no-requeue
#SBATCH --output=slurm-rl-%j.out

# Phase 5 RL — the AverageJoe-informed regime change. Keeps phase 4.1's
# reward design (damage-proportional city payout, capture-aux head,
# material-scored truncation) and adds, in one coherent phase:
#   - Hard legal masking of sources AND directions in rollout + PPO replay
#     (--rl_legal_mask): no reward/entropy budget spent on legality, KL
#     anchor compares masked (trained) distributions.
#   - Undiscounted horizon fit: gamma 1.0, lambda 0.9, 2048-step episodes
#     — long economic games reach the real ±1 terminal.
#   - Learner seat randomized per episode (--randomize_seat): the P0
#     move-priority advantage no longer trains or inflates statistics.
#   - Mirror self-play (--mirror_prob 0.5): both seats of model-vs-model
#     games train (~+50% samples); fresh pool opponent drawn per episode.
#   - Chunked optimizer steps + target-KL epoch guard (--accum_episodes,
#     --target_kl).
#   - EMA weights (--ema_decay) for eval, pool snapshots and deployment
#     (rl_ema_iterN.pt artifacts; use them for the ladder bot and as
#     refs for scripts/elo_ladder.py).
#   - Input features (--derive_army_channels, --turn_globals): linear
#     ownership-split army channels + production-cycle globals, enabled
#     via exactly function-preserving zero-pad migration (optimizer state
#     is intentionally dropped at this boundary; flags become sticky in
#     the checkpoint config).
#   - Eval: paired, seat-swapped, greedy vs the frozen IL anchor.
# 8 GPUs x 4 episodes keeps 32 games/iteration (mirror games add ~50%
# more trajectories on top) at ~2x phase-4 wall-clock speed.
#
# Weights, pool and iteration counter resume from the newest phase-5
# checkpoint if one exists, else newest phase-4, else phase-3 — a plain
# resubmission continues phase 5 instead of rolling back. Outputs land in
# a fresh directory so regimes stay separable. The KL anchor stays pinned
# to the IL policy (--kl_ref).
#
#   sbatch train_rl_phase5_job.sh              # first launch AND crash recovery
#   RESUME=/path/to/specific.pt sbatch train_rl_phase5_job.sh   # explicit pick

export PYTHONWARNINGS="ignore::UserWarning"
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONPATH=~/GamingMaster
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/tmp

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
# 4 episodes x 8 GPUs = 32 pool games/iteration — matches phase 3/4 batch
# statistics for the pool-gate winrate; mirror episodes add extra
# trajectories without touching the gate.
EPISODES=${EPISODES:-4}
IL_ANCHOR=/scratch/$USER/gaming-master/checkpoints/il_phase2/checkpoint_epoch5.pt
P5_DIR=/scratch/$USER/gaming-master/checkpoints/rl_phase5
P4_DIR=/scratch/$USER/gaming-master/checkpoints/rl_phase4
P3_DIR=/scratch/$USER/gaming-master/checkpoints/rl_v2

# Highest-iteration checkpoint (sort -V: by iteration number, not mtime);
# the [0-9].pt suffix keeps the standalone rl_ema_iterN.pt artifacts and
# anything unexpected out of the resume path.
if [ -z "${RESUME:-}" ]; then
    for DIR in "$P5_DIR" "$P4_DIR" "$P3_DIR"; do
        RESUME=$(ls "$DIR"/rl_checkpoint_iter*[0-9].pt 2>/dev/null | sort -V | tail -1)
        [ -n "$RESUME" ] && break
    done
fi
if [ ! -f "$RESUME" ]; then
    echo "ERROR: no resume checkpoint found (RESUME='$RESUME')" >&2
    exit 1
fi
if [ ! -f "$IL_ANCHOR" ]; then
    echo "ERROR: KL anchor checkpoint missing: $IL_ANCHOR" >&2
    exit 1
fi
echo "Resuming from: $RESUME"

REPLAY_DIR=${REPLAY_DIR:-/scratch/$USER/gaming-master/data-v2/replays}
FILTERED_FILE=${FILTERED_FILE:-/scratch/$USER/gaming-master/data-v2/filtered/Human.exe.json}
MAP_FLAGS=()
if [ -d "$REPLAY_DIR" ] && [ -f "$FILTERED_FILE" ]; then
    MAP_FLAGS=(--replay_dir "$REPLAY_DIR" --filtered_file "$FILTERED_FILE")
else
    echo "[warn] map pool sources missing ($REPLAY_DIR / $FILTERED_FILE) — using random maps"
fi

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
        ~/GamingMaster/train_rl.py \
        --resume "$RESUME" \
        --kl_ref "$IL_ANCHOR" \
        --reset_reward_decay \
        "${MAP_FLAGS[@]}" \
        --output_dir "$P5_DIR" \
        --il_data_dir "${IL_DIRS[@]}" \
        --episodes_per_gpu $EPISODES \
        --rollout_steps 2048 \
        --act_steps 1 \
        --ppo_epochs 2 \
        --accum_episodes 4 \
        --target_kl 0.015 \
        --clip_range 0.15 \
        --gamma 1.0 \
        --gae_lambda 0.9 \
        --entropy_weight 0.02 \
        --entropy_weight_decay 0.9998 \
        --entropy_weight_min 0.008 \
        --kl_weight 0.05 \
        --il_weight 0.5 \
        --il_decay 0.995 \
        --rl_legal_mask 1 \
        --randomize_seat 1 \
        --mirror_prob 0.5 \
        --derive_army_channels 1 \
        --turn_globals 1 \
        --ema_decay 0.998 \
        --capture_aux_weight 0.05 \
        --city_budget 0.8 \
        --eval_interval 50 \
        --eval_games 4 \
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
