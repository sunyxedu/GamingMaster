#!/bin/bash
#SBATCH --job-name=gm-rl-p4
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --time=5-00:00:00
#SBATCH --no-requeue
#SBATCH --output=slurm-rl-%j.out

# Phase 4 RL — continue PPO self-play from the phase-3 policy with city
# economy taught through three channels instead of one big capture bonus
# (the flat +0.25..0.33 of commit ed0c866 aggregated to 2-3× the terminal
# reward on city-dense maps and caused reward-hacking boom-bust cycles):
#   1. Damage-proportional neutral-city payout, budget-capped per map
#      (--city_budget): every whittle hit pays partially, a full capture
#      sums to exactly the per-city unit. Dense signal, unfarmable.
#   2. Auxiliary capture-feasibility head (--capture_aux_weight): exact
#      "would this move capture" labels from the env force the encoder to
#      represent army-count comparisons the log-compressed input hides.
#   3. Truncated games score ±0.35 by city-adjusted material instead of 0,
#      so long-game economy pays through the terminal channel.
# Winrate vs the frozen IL anchor is evaluated every --eval_interval iters
# as a shaping-independent progress signal.
#
# Weights, optimizer, opponent pool and iteration counter resume from the
# newest phase-4 checkpoint if one exists, else the newest phase-3
# checkpoint (rl_v2) — so a plain resubmission continues phase 4 instead
# of silently rolling back to phase 3 and overwriting phase-4 outputs.
# Outputs land in a fresh directory so the two reward regimes stay
# separable. The KL anchor is pinned back to the IL policy (--kl_ref): a
# plain resume would re-anchor onto the resumed rush-style policy and pull
# AGAINST the city behavior this phase is meant to induce.
# --reset_reward_decay restarts the shaping decay clocks at this phase
# boundary: without it the raised capture bonus would enter pre-annealed
# to 0.9995^300 and be half gone ~1100 iterations in.
#
#   sbatch train_rl_phase4_job.sh              # first launch AND crash recovery
#   RESUME=/path/to/specific.pt sbatch train_rl_phase4_job.sh   # explicit pick

export PYTHONWARNINGS="ignore::UserWarning"
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONPATH=~/GamingMaster
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/tmp

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
# 8 episodes x 4 GPUs = 32 games/iteration — matches phase 3's effective
# batch so PPO gradient noise and pool-gate winrate stats stay comparable.
EPISODES=${EPISODES:-8}
IL_ANCHOR=/scratch/$USER/gaming-master/checkpoints/il_phase2/checkpoint_epoch5.pt
P4_DIR=/scratch/$USER/gaming-master/checkpoints/rl_phase4
P3_DIR=/scratch/$USER/gaming-master/checkpoints/rl_v2

# Highest-iteration phase-4 checkpoint if any (sort -V: by iteration number,
# not mtime), else newest phase-3 one for the first launch.
if [ -z "${RESUME:-}" ]; then
    RESUME=$(ls "$P4_DIR"/rl_checkpoint_iter*.pt 2>/dev/null | sort -V | tail -1)
    [ -n "$RESUME" ] || RESUME=$(ls "$P3_DIR"/rl_checkpoint_iter*.pt 2>/dev/null | sort -V | tail -1)
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
        --output_dir "$P4_DIR" \
        --il_data_dir "${IL_DIRS[@]}" \
        --episodes_per_gpu $EPISODES \
        --rollout_steps 2048 \
        --act_steps 1 \
        --ppo_epochs 2 \
        --clip_range 0.15 \
        --gamma 1.0 \
        --gae_lambda 0.9 \
        --entropy_weight 0.02 \
        --entropy_weight_decay 0.9998 \
        --entropy_weight_min 0.008 \
        --kl_weight 0.05 \
        --il_weight 0.5 \
        --il_decay 0.995 \
        --invalidity_weight 0.5 \
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
