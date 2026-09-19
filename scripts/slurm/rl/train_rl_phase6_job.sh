#!/bin/bash
#SBATCH --job-name=gm-rl-p6
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --time=5-00:00:00
#SBATCH --no-requeue
#SBATCH --output=slurm-rl-p6-%j.out

# Phase 6 RL — the AverageJoe regime, adapted. Seeds from the phase-5 peak
# EMA (iter950: eval-vs-anchor 0.81, distilled from the last era with
# demonstrated self-play progress, BEFORE the 950-1010 collapse and the
# no-city equilibrium that followed) and changes the regime in one move:
#
#   - SPARSE REWARD (--sparse_reward): pure ±1 terminal + the ±0.35
#     material-scored soft terminal on truncation. City/dense shaping OFF.
#   - PURE MIRROR self-play (--mirror_prob 1.0): both seats are the live
#     model, every game trains both trajectories. No opponent pool in
#     training -> the pool-gate losing-streak feedback that drove the
#     phase-5 collapse (winrate 0.01 at iters ~985-1010) cannot exist.
#     The pool/Elo ladder remain EVAL tools (scripts/elo_ladder.py).
#   - TOP-25% |ADV| FILTER (--adv_filter_frac 0.25): AverageJoe's
#     signature — the policy trains on each episode's decisive samples;
#     the critic trains on all. Also mutes noise-dominated updates.
#   - HL-GAUSS CRITIC (--hl_gauss_value): 128 bins on [-1.5, 1.5],
#     sigma 0.06, CE against Gaussian-smeared targets. Value head is
#     REINITIALISED at this boundary (scalar -> distributional); early
#     iterations are effectively REINFORCE while it converges.
#   - ANCHOR DECAY (--kl_weight 0.08 --kl_weight_pow 0.2
#     --kl_weight_min 0.005): magnet-style schedule on OUR anchor (the
#     IL policy — better than a hand-written magnet: we have human data).
#     Strong while the fresh critic finds its feet, fading as self-play
#     takes over.
#   - IL FLOOR (--il_weight_min 0.02): permanent human-data pull. The
#     opening-quality gap (land@100 37 vs anchor 45.6) appeared exactly
#     when il_weight decayed to ~0; the floor is the fix we committed to.
#   - ENTROPY CEILING (--entropy_ceiling 2.3): no entropy bonus while
#     mean policy entropy is already above 2.3 — breaks the despair-
#     exploration spiral (phase-5 rode it to ent 3.0).
#   - SPARRING PARTNER (--sparring_prob 0.15 vs the frozen ema950 seed):
#     pure mirror cannot punish style drift — rush vs rush is symmetric,
#     so the over-aggressive short-game meta observed by iter ~1020
#     (ep_len ~300, city capture ~0.1/game, owner's verdict "too
#     aggressive") would never be corrected from within. 15% of episodes
#     against the frozen economic seed re-introduce the counter-style;
#     h2h at iter 1000 was 17-15, so these games carry real gradient in
#     both directions. No pool/gate machinery involved. il_weight_min
#     raised 0.02 -> 0.05 for the same reason (persistent human-style
#     pull).
#
# Deferred (next engineering pass, orthogonal to this regime): cross-
# episode batched rollout for 4-8x throughput.
#
# Weights: NO automatic resume chain into phase-5 checkpoints. First
# launch MUST come from the phase-5 peak EMA artifact (explicit below);
# crash recovery resumes the newest phase-6 checkpoint automatically.
# The KL anchor stays pinned to the IL policy.
#
#   sbatch scripts/slurm/rl/train_rl_phase6_job.sh                # first launch AND crash recovery
#   RESUME=/path/to/specific.pt sbatch scripts/slurm/rl/train_rl_phase6_job.sh   # explicit pick
#   EPISODES=4 sbatch --gres=gpu:nvidia_h200_nvl:2 --time=4:00:00 \
#       scripts/slurm/rl/train_rl_phase6_job.sh                   # 2-GPU smoke test
#
# LR note: train_rl restores lr = --lr * lr_decay^950 at this resume
# (iteration counter continues from 950), so --lr 6e-5 lands at an
# effective ~5.5e-5 for the first phase-6 iteration.

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/tmp

GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}
# EPISODES x GPUs = 32 games/iteration by default; every episode is a
# mirror game, so trajectory count is ~2x that at any width.
EPISODES=${EPISODES:-$(( 32 / GPUS_PER_NODE ))}
IL_ANCHOR=/scratch/$USER/gaming-master/checkpoints/il_phase2/checkpoint_epoch5.pt
P6_DIR=/scratch/$USER/gaming-master/checkpoints/rl_phase6
P5_EMA_SEED=/scratch/$USER/gaming-master/checkpoints/rl_phase5/rl_ema_iter950.pt

# Crash recovery: newest phase-6 checkpoint if one exists, else the
# phase-5 peak EMA seed. Never the raw phase-5 checkpoints — the raw
# iter950 weights are the mid-collapse policy; the EMA artifact is the
# clean distillation (same iteration, different tensors).
if [ -z "${RESUME:-}" ]; then
    RESUME=$(ls "$P6_DIR"/rl_checkpoint_iter*[0-9].pt 2>/dev/null | sort -V | tail -1)
    [ -n "$RESUME" ] || RESUME=$P5_EMA_SEED
fi
if [ ! -f "$RESUME" ]; then
    echo "ERROR: no resume checkpoint found (RESUME='$RESUME')" >&2
    exit 1
fi
if [ ! -f "$IL_ANCHOR" ]; then
    echo "ERROR: KL anchor checkpoint missing: $IL_ANCHOR" >&2
    exit 1
fi
if [ ! -f "$P5_EMA_SEED" ]; then
    echo "ERROR: sparring seed checkpoint missing: $P5_EMA_SEED" >&2
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
RDZV_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))

srun apptainer exec --nv /scratch/$USER/pytorch.sif \
    torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=$HEAD_NODE:$RDZV_PORT \
        --module training.train_rl \
        --resume "$RESUME" \
        --kl_ref "$IL_ANCHOR" \
        --reset_reward_decay \
        "${MAP_FLAGS[@]}" \
        --output_dir "$P6_DIR" \
        --il_data_dir "${IL_DIRS[@]}" \
        --episodes_per_gpu $EPISODES \
        --rollout_steps 1536 \
        --act_steps 1 \
        --ppo_epochs 2 \
        --accum_episodes 4 \
        --target_kl 0.015 \
        --clip_range 0.15 \
        --gamma 1.0 \
        --gae_lambda 0.9 \
        --sparse_reward 1 \
        --mirror_prob 1.0 \
        --sparring_prob 0.15 \
        --sparring_ckpt "$P5_EMA_SEED" \
        --randomize_seat 1 \
        --rl_legal_mask 1 \
        --adv_filter_frac 0.25 \
        --hl_gauss_value 1 \
        --entropy_weight 0.01 \
        --entropy_weight_decay 0.9998 \
        --entropy_weight_min 0.003 \
        --entropy_ceiling 2.3 \
        --kl_weight 0.08 \
        --kl_weight_pow 0.2 \
        --kl_weight_min 0.005 \
        --il_weight 0.3 \
        --il_decay 0.997 \
        --il_weight_min 0.05 \
        --derive_army_channels 1 \
        --turn_globals 1 \
        --ema_decay 0.998 \
        --capture_aux_weight 0.05 \
        --eval_interval 50 \
        --eval_games 4 \
        --lr 6e-5 \
        --lr_decay 0.9999 \
        --grad_clip 1.0 \
        --freeze_encoder_iters 0 \
        --iterations 10000 \
        --save_interval 25 \
        --save_replay_interval 10 \
        --log_interval 1
