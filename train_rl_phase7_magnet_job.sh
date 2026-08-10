#!/bin/bash
#SBATCH --job-name=gm-rl-p7m
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:6
#SBATCH --time=5-00:00:00
#SBATCH --requeue
#SBATCH --mem=256G
#SBATCH --output=slurm-rl-p7m-%j.out

# Phase 7 / run-4 "magnet" — adopt AverageJoe's anchoring, drop ours.
#
# WHY. Runs 1-3b all died the same two-step death: the IL (human-BC)
# anchor decays away on schedule, and the moment il_w drops below ~0.15
# the mirror echo chamber takes over — as entropy explosion (phase-6,
# 39746: ent 2.3->3.5 / 3.04), or, once the tightened gate suppressed
# entropy (40595: penalty 0.02, ceiling 1.8 DID hold ent < 2.3), as
# anchor-divergence instead: kl_reg 0.34 -> 1.2 at il_w~0.14, then
# ep_len 889 -> 1454 in four iterations. Patching the symptom moved the
# failure, never removed it: the anchor is both crutch and time bomb.
#
# THE FIX (what the #1-ladder reference implementation actually does):
# AverageJoe (github.com/strakam/AverageJoe) trains pure-mirror PPO with
# NO human anchor and NO entropy gate. Its single regulariser is
# ent_coef * KL(policy, expander-magnet): a fixed heuristic prior that
# scores expansion/capture moves (neutral 2, capturable enemy 3,
# capturable city 5, top-5 stacks +2, pass ~5%). The KL term clamps
# entropy from BOTH sides (collapse and flatten both raise it), so no
# reactive controller is needed; the coefficient never decays to zero
# (power-law floor), so the regulariser cannot silently vanish.
#
# RUN-4 REGIME (model architecture untouched):
#   1. --magnet 1: KL anchor = expander heuristic via the existing
#      reference-KL rails (reverse KL(magnet||pi), all four heads).
#      --kl_weight 0.01 --kl_weight_pow 0: CONSTANT weight — matches
#      AverageJoe's schedule value at our training age (0.05/it^0.2 ~=
#      0.010 at it~1600) and honours its no-vanish floor property.
#   2. NO human anchor: --il_weight 0, no --kl_ref, no --il_data_dir
#      (BC survives as the network initialisation, nothing more).
#   3. Entropy: small CONSTANT bonus 0.003 (their -H(pi) term inside the
#      magnet KL, decomposed), gate OFF (--entropy_ceiling 0). The
#      magnet KL is the anti-collapse/anti-flatten mechanism now.
#   4. PPO per the reference: --clip_range 0.2 --target_kl 0.02
#      --ppo_epochs 1 (single pass over the top-25% |adv| samples).
#   5. DELIBERATE DEVIATION — truncation shaping stays
#      (--trunc_penalty 0.3 --trunc_material 0.15): AverageJoe's zero
#      reward at truncation makes stalling strictly better than losing,
#      and OUR env at OUR batch size has twice demonstrated that stall
#      attractor (phase-6; 40595 iters 1708-1710). Their immunity is
#      scale (262k steps/iter), not the loss. Run-1 validated this fix.
#
# Unchanged: sparse +-1, pure mirror, seat randomisation, hard legal
# mask, top-25% |adv| filter, HL-Gauss critic, gamma 1.0, lambda 0.9,
# EMA 0.998, capture aux 0.05, ema1300 fixed exam every 50 iters.
#
# Weights: first launch seeds from run-3's measured best point
# (rl_ema_iter1600.pt, eval 0.62 vs ema1300); crash recovery resumes the
# newest run-4 checkpoint automatically.
#
#   sbatch train_rl_phase7_magnet_job.sh          # launch AND recovery
#   RESUME=/path/to.pt sbatch train_rl_phase7_magnet_job.sh
#   P7M_DIR=/scratch/.../smoke ITERATIONS=1602 EPISODES=2 \
#       EVAL_INTERVAL=1 EVAL_GAMES=1 \
#       sbatch --gres=gpu:nvidia_h200_nvl:2 --time=2:00:00 \
#       train_rl_phase7_magnet_job.sh             # two-iteration smoke
#   EPISODES=12 sbatch train_rl_phase7_magnet_job.sh  # production: 72/iter

export PYTHONWARNINGS="ignore::UserWarning"
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONPATH=~/GamingMaster
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/tmp

GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-6}}
# Production: EPISODES=12 x 6 GPUs = 72 games/iteration (phase-6 scale).
EPISODES=${EPISODES:-12}
P7M_DIR=${P7M_DIR:-/scratch/$USER/gaming-master/checkpoints/rl_phase7b/run4_magnet}
SEED_CKPT=/scratch/$USER/gaming-master/checkpoints/rl_phase7b/run3_prod/rl_ema_iter1600.pt
P6_EMA_SEED=/scratch/$USER/gaming-master/checkpoints/rl_phase6/rl_ema_iter1300.pt
EVAL_CKPT=${EVAL_CKPT:-$P6_EMA_SEED}
ITERATIONS=${ITERATIONS:-10000}
SAVE_INTERVAL=${SAVE_INTERVAL:-25}
SAVE_REPLAY_INTERVAL=${SAVE_REPLAY_INTERVAL:-10}
EVAL_INTERVAL=${EVAL_INTERVAL:-50}
EVAL_GAMES=${EVAL_GAMES:-4}
MAGNET_WEIGHT=${MAGNET_WEIGHT:-0.01}
ENTROPY_WEIGHT=${ENTROPY_WEIGHT:-0.003}
# Vectorized rollout: all EPISODES envs step in lockstep through one
# batched forward (mirror-only). This is the throughput lever — EPISODES
# can go to 64+/GPU without a proportional wall-clock cost.
VEC_ROLLOUT=${VEC_ROLLOUT:-0}
# With VEC_ROLLOUT: parallel env stepping over this many persistent
# worker processes per rank (0 = in-process serial stepping).
ENV_WORKERS=${ENV_WORKERS:-0}
# Episodes accumulated per optimizer step. The default 4 was tuned for
# 12 episodes/GPU; at 64 episodes/GPU it yields ~32 optimizer steps per
# iteration whose cumulative drift blows past target_kl (measured
# approx_kl 0.28-0.66 in run5) and whose small chunks chase
# truncation-penalty batches. AverageJoe accumulates 128 episodes per
# step; 32 restores the healthy-era step count at vectorized scale.
ACCUM_EPISODES=${ACCUM_EPISODES:-4}
# TRM pondering depth, FIXED for rollout/update/eval alike (PPO must
# replay the exact rollout policy; adaptive q_halt gating would break
# the ratio). >1 spends proportionally more forward compute — switch
# only after a depth probe (scripts/depth_probe.py) shows it pays.
ACT_STEPS=${ACT_STEPS:-1}
# Opening anchor (off by default): set OPENING_ANCHOR to the IL
# checkpoint to teach the first OPENING_TURNS turns the human opening —
# mirror self-play cannot punish shared opening habits and the magnet
# is turn-agnostic, so the opening needs its own teacher.
OPENING_ANCHOR=${OPENING_ANCHOR:-}
OPENING_TURNS=${OPENING_TURNS:-100}
OPENING_WEIGHT=${OPENING_WEIGHT:-0.05}
OPEN_FLAGS=()
if [ -n "$OPENING_ANCHOR" ]; then
    OPEN_FLAGS=(--opening_anchor "$OPENING_ANCHOR"
                --opening_anchor_turns "$OPENING_TURNS"
                --opening_anchor_weight "$OPENING_WEIGHT")
fi

# Crash recovery: newest run-4 checkpoint if one exists, else the seed.
if [ -z "${RESUME:-}" ]; then
    RESUME=$(ls "$P7M_DIR"/rl_checkpoint_iter*[0-9].pt 2>/dev/null | sort -V | tail -1)
    [ -n "$RESUME" ] || RESUME=$SEED_CKPT
fi
if [ ! -f "$RESUME" ]; then
    echo "ERROR: no resume checkpoint found (RESUME='$RESUME')" >&2
    exit 1
fi
if [ ! -f "$EVAL_CKPT" ]; then
    echo "ERROR: fixed eval checkpoint missing: $EVAL_CKPT" >&2
    exit 1
fi
echo "Resuming from:  $RESUME"
echo "Anchor:         expander magnet (weight $MAGNET_WEIGHT, constant)"
echo "Fixed exam vs:  $EVAL_CKPT"

REPLAY_DIR=${REPLAY_DIR:-/scratch/$USER/gaming-master/data-v2/replays}
FILTERED_FILE=${FILTERED_FILE:-/scratch/$USER/gaming-master/data-v2/filtered/Human.exe.json}
MAP_FLAGS=()
if [ -d "$REPLAY_DIR" ] && [ -f "$FILTERED_FILE" ]; then
    MAP_FLAGS=(--replay_dir "$REPLAY_DIR" --filtered_file "$FILTERED_FILE")
else
    echo "[warn] map pool sources missing ($REPLAY_DIR / $FILTERED_FILE) — using random maps"
fi

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
RDZV_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))

srun apptainer exec --nv /scratch/$USER/pytorch.sif \
    torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=$HEAD_NODE:$RDZV_PORT \
        ~/GamingMaster/train_rl.py \
        --resume "$RESUME" \
        --reset_reward_decay \
        "${MAP_FLAGS[@]}" \
        --output_dir "$P7M_DIR" \
        --episodes_per_gpu $EPISODES \
        --vec_rollout "$VEC_ROLLOUT" \
        --env_workers "$ENV_WORKERS" \
        --rollout_steps 1536 \
        --act_steps "$ACT_STEPS" \
        --ppo_epochs 1 \
        --accum_episodes "$ACCUM_EPISODES" \
        --target_kl 0.02 \
        --clip_range 0.2 \
        --gamma 1.0 \
        --gae_lambda 0.9 \
        --sparse_reward 1 \
        --trunc_penalty 0.3 \
        --trunc_material 0.15 \
        --mirror_prob 1.0 \
        --sparring_prob 0.0 \
        --randomize_seat 1 \
        --rl_legal_mask 1 \
        --adv_filter_frac 0.25 \
        --hl_gauss_value 1 \
        --magnet 1 \
        "${OPEN_FLAGS[@]}" \
        --kl_weight "$MAGNET_WEIGHT" \
        --kl_weight_pow 0 \
        --kl_weight_min "$MAGNET_WEIGHT" \
        --il_weight 0 \
        --il_decay 1.0 \
        --il_weight_min 0 \
        --entropy_weight "$ENTROPY_WEIGHT" \
        --entropy_weight_decay 1.0 \
        --entropy_weight_min "$ENTROPY_WEIGHT" \
        --entropy_ceiling 0 \
        --entropy_penalty 0 \
        --derive_army_channels 1 \
        --turn_globals 1 \
        --ema_decay 0.998 \
        --capture_aux_weight 0.05 \
        --eval_ckpt "$EVAL_CKPT" \
        --eval_interval "$EVAL_INTERVAL" \
        --eval_games "$EVAL_GAMES" \
        --lr 6e-5 \
        --lr_decay 0.9999 \
        --grad_clip 1.0 \
        --freeze_encoder_iters 0 \
        --iterations "$ITERATIONS" \
        --save_interval "$SAVE_INTERVAL" \
        --save_replay_interval "$SAVE_REPLAY_INTERVAL" \
        --log_interval 1
