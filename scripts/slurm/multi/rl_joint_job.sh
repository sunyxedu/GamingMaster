#!/bin/bash
#SBATCH --job-name=gm-rl-joint
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#
# Stage-2 joint mixed-game RL (training/train_rl_joint.py) on NGPU GPUs of ONE
# node:  torchrun --standalone --nproc_per_node=$NGPU -m training.train_rl_joint
# Everything is passed through environment-variable PREFIXES, like
# multi_game_ddp_job.sh (never `sbatch --export=VAR=...`, which gets jobs killed
# on hephaestus).  #SBATCH lines cannot read the environment, so:
#
#   # 4 GPUs (the #SBATCH default)
#   RUN=rlj-go1 INIT=$HOME/gm-runs/multi/core-rdmoe-4g/multi_final.pt \
#   ARGS="--rl generals --rl othello --data othello=... --data chess=... \
#         --data generals=... --replay_max_len generals=512" \
#       sbatch --requeue scripts/slurm/multi/rl_joint_job.sh
#
#   # any NGPU (default 4): run it with bash and it submits ITSELF with a matching
#   # --gres / --cpus-per-task (CPUS_PER_GPU, default 8 per GPU) / --mem (MEM_PER_GPU,
#   # default 96, GiB per GPU); GPU_TYPE (default nvidia_h200_nvl), SBATCH_EXTRA
#   # (extra sbatch flags)
#   NGPU=2 RUN=rlj-2g INIT=... ARGS="..." bash scripts/slurm/multi/rl_joint_job.sh
#
#   # NGPU MIG slices of one GPU (e.g. 2 x 1g.18gb): a MIG GPU_TYPE; per slice the
#   # defaults become 4 CPUs and 24 GiB of RAM (times the slice's compute units)
#   GPU_TYPE=nvidia_h200_nvl_1g.18gb NGPU=2 RUN=rlj-mig2 INIT=... ARGS="..." \
#       bash scripts/slurm/multi/rl_joint_job.sh
#
# MIG slices: each rank pins itself to its own slice (a CUDA process can use one
# MIG instance) and --dist_backend auto (passed below; DIST_BACKEND overrides)
# picks gloo (NCCL refuses two ranks on one physical GPU).  A 1g.18gb slice has
# ~16 GiB usable for the policy, the frozen reference, EMA and optimizer state:
# keep --rows_per_forward / --replay_batch_size small, and 4 CPUs per slice run
# the rollouts, so keep --envs modest.
#
# INIT is the Stage-1 train_multi_game checkpoint (its EMA weights = the initial
# policy and the frozen KL reference); it is passed again on a resume.
# Per-rank rollouts: --envs GAME=N lockstep envs per wave and rank, the
# learner-token budget --tokens_per_iter is global (split over the ranks).
# generals: bound its learner forwards with --rows_per_forward generals=1..2
# (episodes run up to --env_kw generals:max_turns=N frames, default 2048).
#
# Checkpoints / resume: rank 0 writes $OUT/rl_latest.pt every CKPT_MINUTES
# (default 30) minutes and at the end; the snapshot pool lives in
# $OUT/snapshots (evicted snapshot files are deleted only after the next
# rl_latest.pt is written).  If rl_latest.pt exists when the job starts (requeue
# after a node failure, or the same RUN resubmitted after the 24 h wall), the run
# continues from it (--resume, exact on the same NGPU; metrics.jsonl is cut back
# to the resumed iteration; a different INIT is refused); RESUME=0 starts from
# scratch instead.
#
# Paths live under $GM_HOME (default ~/gm-core) — not /scratch.
set -euo pipefail
GPU_TYPE=${GPU_TYPE:-nvidia_h200_nvl}
MIG_RE='_([0-9]+)g\.[0-9]+gb'   # MIG gres types, e.g. nvidia_h200_nvl_3g.71gb

if [ -z "${SLURM_JOB_ID:-}" ]; then
    # submission mode: request exactly NGPU GPUs (env vars travel with sbatch's
    # default environment propagation — no --export)
    : "${RUN:?set RUN}"
    : "${INIT:?set INIT (the Stage-1 train_multi_game checkpoint)}"
    NGPU=${NGPU:-4}
    if [[ "$GPU_TYPE" =~ $MIG_RE ]]; then
        # a MIG slice: resources scale with its compute units (the node's 128
        # CPUs / 1 TB are shared by 4 full GPUs and 28 1g slices)
        UNITS=${BASH_REMATCH[1]}
        CPUS_PER_GPU=${CPUS_PER_GPU:-$((4 * UNITS))}
        MEM_PER_GPU=${MEM_PER_GPU:-$((24 * UNITS))}
    fi
    self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    sinfo -p "${PARTITION:-batch}" -N -h \
        -O "NodeHost:20,Gres:40,GresUsed:50,StateCompact:10" 2>/dev/null || true
    export NGPU
    # shellcheck disable=SC2086
    exec sbatch --requeue --partition="${PARTITION:-batch}" --gres="gpu:$GPU_TYPE:$NGPU" \
        --cpus-per-task=$((${CPUS_PER_GPU:-8} * NGPU)) --mem="$((${MEM_PER_GPU:-96} * NGPU))G" \
        ${SBATCH_EXTRA:-} "$self"
fi

GM_HOME=${GM_HOME:-$HOME/gm-core}
RUN=${RUN:?set RUN}
INIT=${INIT:?set INIT (the Stage-1 train_multi_game checkpoint)}
OUT=${OUT:-$HOME/gm-runs/rl_joint/$RUN}
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    VISIBLE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    VISIBLE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
fi
NGPU=${NGPU:-${SLURM_GPUS_ON_NODE:-$VISIBLE}}
if [ "$NGPU" -lt 1 ] || [ "$NGPU" -gt "$VISIBLE" ]; then
    echo "NGPU=$NGPU but $VISIBLE GPU(s) visible in this allocation" >&2
    exit 1
fi
if [ "$NGPU" -gt 1 ] && [[ "${CUDA_VISIBLE_DEVICES:-}" == *MIG-* ]]; then
    # every rank pins its own slice (training.distributed.pin_rank_device)
    echo "MIG slices: $NGPU ranks, one slice each, gloo (--dist_backend ${DIST_BACKEND:-auto})"
fi
if [ ! -f "$INIT" ]; then
    echo "INIT=$INIT does not exist" >&2
    exit 1
fi
mkdir -p "$OUT"
RESUME_FROM=""
if [ "${RESUME:-1}" != 0 ] && [ -f "$OUT/rl_latest.pt" ]; then
    RESUME_FROM="$OUT/rl_latest.pt"
fi
cd "$GM_HOME"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
export PYTHONPATH="$GM_HOME${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
# node-local scratch for torchrun / DataLoader workers (NFS temp dirs trip
# ".nfs* Device or resource busy" at teardown)
export TMPDIR=/tmp
echo "[$(date)] $RUN on $(hostname) ngpu=$NGPU gpu=${CUDA_VISIBLE_DEVICES:-?} commit=$(git rev-parse --short HEAD) init=$INIT resume=${RESUME_FROM:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi -L 2>/dev/null | grep MIG || true
# ARGS may override --ckpt_minutes (the last occurrence wins); log.txt is
# appended to, so a resumed run keeps the earlier part
# shellcheck disable=SC2086
"$GM_HOME/.venv/bin/torchrun" --standalone --nproc_per_node="$NGPU" \
    -m training.train_rl_joint --init "$INIT" --ckpt_minutes "${CKPT_MINUTES:-30}" \
    --dist_backend "${DIST_BACKEND:-auto}" \
    --snapshot_dir "$OUT/snapshots" ${ARGS:-} \
    ${RESUME_FROM:+--resume "$RESUME_FROM"} --save_dir "$OUT" 2>&1 | tee -a "$OUT/log.txt"
echo "[$(date)] done"
