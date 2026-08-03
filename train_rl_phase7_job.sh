#!/bin/bash
#SBATCH --job-name=gm-rl-p7c
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --time=5-00:00:00
#SBATCH --no-requeue
#SBATCH --output=slurm-rl-p7c-%j.out

# Phase 7c / run-3 — recovery after the run-2b frozen-opponent spiral.
#
# Run-2b exposed three coupled mistakes:
#   1. "rollback 1450" resumed rl_checkpoint_iter1450.pt (RAW learner), but
#      the measured 53% h2h health point was rl_ema_iter1450.pt.  Run-3 seeds
#      the learner itself from that exact EMA artifact, with a fresh optimizer
#      and fresh EMA shadow.
#   2. Even ema950 sparring fell 48 -> 41 -> 31 -> 27%, while entropy and
#      episode length rose together.  Frozen opponents are therefore removed
#      from the PPO distribution; anti-drift opponents are read-only exams.
#   3. The negative entropy weight was filtered to the top-25% |adv| samples,
#      leaving 75% of the trajectory outside the safety controller.  Negative
#      entropy pressure now covers all valid timesteps and is doubled to 0.01.
#
# The fixed exam remains ema1300, paired/greedy on every eval interval.
#
# WHY A NEW PHASE — the phase-6 collapse (iters ~1355-1462, job 34175):
# pure-mirror economic play drifted into a degenerate equilibrium where
# BOTH seats stall to the turn limit and farm the ±0.35 material soft
# terminal instead of ever winning. Observed signature:
#   ep_len 400 -> 1400 (cap 1536), entropy 2.3 -> 3.5 with the zeroing
#   gate firing EVERY iteration (it stops adding fuel but cannot push
#   down), sparring w/l vs ema950 -> rows of 0/X, city bash 5-7/game at
#   fail ~0.7, eval-vs-anchor 0.83(1350) -> 0.65(1400).
# Mechanism: a truncated lead scored nearly as well as a win, the anchor
# had decayed away (il_w 0.065, klw 0.024), and 15% sparring was too weak
# to counter the 85% mirror echo chamber.
#
# THE RUN-3 REGIME:
#   1. TRUNCATION PENALTY (--trunc_penalty 0.3 --trunc_material 0.15):
#      hitting the clock now costs both players 0.3, and the material
#      term shrinks to ±0.15. Best truncation outcome for the leader is
#      ~-0.15 vs +1.0 for winning -> leads must be converted.
#   2. GLOBAL ACTIVE ENTROPY SUPPRESSION (--entropy_ceiling 1.9
#      --entropy_penalty 0.01): above the ceiling the negative entropy
#      weight covers every valid timestep, not only the advantage-filtered
#      quarter. The positive exploration bonus remains filtered.
#   3. ANCHOR (--il_weight 0.3 --il_weight_min 0.05): phase-6's proven
#      schedule. RUN-1 LESSON: the floor was doubled to 0.1 here and the
#      strong pull Goodharted eval-vs-anchor to 0.83 while real strength
#      fell (h2h vs seed 34% at iter 1475) — closer-to-anchor beats the
#      anchor more, proving nothing. KL-anchor clock resets automatically
#      (new output dir => klw restarts at 0.08 and decays afresh).
#   4. PURE MIRROR TRAINING (--sparring_prob 0): the truncation and entropy
#      fixes now guard the two known mirror failure modes. Frozen opponents
#      are excluded from gradients and ema1300 is retained as the paired,
#      greedy fixed exam (--eval_ckpt).
#
# RUN-2 (2026-07-23): run-1 (iters 1326-1500, archived in
# rl_phase7_run1/) validated the anti-stalling fixes — ep_len 470-580,
# entropy gated correctly, zero clock-farming — but net strength fell
# vs the seed. Run-2b then proved that a weaker frozen opponent still
# re-created the same punishment-dominated spiral. Run-3 keeps the validated
# anti-stalling pieces and removes the common frozen-training cause.
#
# Unchanged from phase 6: sparse ±1, pure mirror otherwise, top-25% |adv|
# filter, HL-Gauss critic, gamma 1.0, lambda 0.9, clip 0.15,
# target_kl 0.015, EMA 0.998.
#
# Weights: first launch seeds from the phase-7 EMA1450 deployment artifact
# (the measured 53% h2h point); crash recovery resumes the newest run-3
# checkpoint automatically. KL anchor stays pinned to the IL policy.
#
#   sbatch train_rl_phase7_job.sh                # first launch AND crash recovery
#   RESUME=/path/to/specific.pt sbatch train_rl_phase7_job.sh   # explicit pick
#   P7_DIR=/scratch/.../smoke ITERATIONS=1452 EPISODES=2 \
#       EVAL_INTERVAL=1 EVAL_GAMES=1 \
#       sbatch --gres=gpu:nvidia_h200_nvl:2 --time=2:00:00 \
#       train_rl_phase7_job.sh                   # two-iteration smoke test
#   EPISODES=12 sbatch train_rl_phase7_job.sh    # production: 48 games/iter
#
# LR note: iteration numbering continues from 1450, so --lr 6e-5 lands at
# an effective ~5.2e-5 for the first run-3 iteration.

export PYTHONWARNINGS="ignore::UserWarning"
export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONPATH=~/GamingMaster
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/tmp

GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}
# EPISODES x GPUs = 32 games/iteration by default; production phase-6 ran
# EPISODES=12 on 6 GPUs (72/iter).
EPISODES=${EPISODES:-$(( 32 / GPUS_PER_NODE ))}
IL_ANCHOR=/scratch/$USER/gaming-master/checkpoints/il_phase2/checkpoint_epoch5.pt
# New output directory isolates run-3 from both the damaged run-2b weights
# and the original rl_phase7 BeeGFS metadata failure.
P7_DIR=${P7_DIR:-/scratch/$USER/gaming-master/checkpoints/rl_phase7b/run3_prod}
P7_EMA_SEED=/scratch/$USER/gaming-master/checkpoints/rl_phase7b/rl_ema_iter1450.pt
P6_EMA_SEED=/scratch/$USER/gaming-master/checkpoints/rl_phase6/rl_ema_iter1300.pt
EVAL_CKPT=${EVAL_CKPT:-$P6_EMA_SEED}
ITERATIONS=${ITERATIONS:-10000}
SAVE_INTERVAL=${SAVE_INTERVAL:-25}
SAVE_REPLAY_INTERVAL=${SAVE_REPLAY_INTERVAL:-10}
EVAL_INTERVAL=${EVAL_INTERVAL:-50}
EVAL_GAMES=${EVAL_GAMES:-4}
ENTROPY_CEILING=${ENTROPY_CEILING:-1.9}
ENTROPY_PENALTY=${ENTROPY_PENALTY:-0.01}

# Crash recovery: newest run-3 checkpoint if one exists, else the exact
# deployment artifact whose 53% h2h result defined the healthy rollback.
if [ -z "${RESUME:-}" ]; then
    RESUME=$(ls "$P7_DIR"/rl_checkpoint_iter*[0-9].pt 2>/dev/null | sort -V | tail -1)
    [ -n "$RESUME" ] || RESUME=$P7_EMA_SEED
fi
if [ ! -f "$RESUME" ]; then
    echo "ERROR: no resume checkpoint found (RESUME='$RESUME')" >&2
    exit 1
fi
if [ ! -f "$IL_ANCHOR" ]; then
    echo "ERROR: KL anchor checkpoint missing: $IL_ANCHOR" >&2
    exit 1
fi
if [ ! -f "$EVAL_CKPT" ]; then
    echo "ERROR: fixed eval checkpoint missing: $EVAL_CKPT" >&2
    exit 1
fi
echo "Resuming from: $RESUME"
echo "Training opponents: 100% live mirror"
echo "Fixed exam vs:      $EVAL_CKPT"

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
        ~/GamingMaster/train_rl.py \
        --resume "$RESUME" \
        --kl_ref "$IL_ANCHOR" \
        --reset_reward_decay \
        "${MAP_FLAGS[@]}" \
        --output_dir "$P7_DIR" \
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
        --trunc_penalty 0.3 \
        --trunc_material 0.15 \
        --mirror_prob 1.0 \
        --sparring_prob 0.0 \
        --randomize_seat 1 \
        --rl_legal_mask 1 \
        --adv_filter_frac 0.25 \
        --hl_gauss_value 1 \
        --entropy_weight 0.005 \
        --entropy_weight_decay 0.9998 \
        --entropy_weight_min 0.003 \
        --entropy_ceiling "$ENTROPY_CEILING" \
        --entropy_penalty "$ENTROPY_PENALTY" \
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
