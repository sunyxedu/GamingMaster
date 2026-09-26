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
#   # --gres / --cpus-per-task (8 per GPU) / --mem (MEM_PER_GPU, default 96, GiB per
#   # GPU); GPU_TYPE (default nvidia_h200_nvl), SBATCH_EXTRA (extra sbatch flags)
#   NGPU=2 RUN=rlj-2g INIT=... ARGS="..." bash scripts/slurm/multi/rl_joint_job.sh
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
# $OUT/snapshots.  If rl_latest.pt exists when the job starts (requeue after a
# node failure, or the same RUN resubmitted after the 24 h wall), the run
# continues from it (--resume, exact on the same NGPU); RESUME=0 starts from
# scratch instead.
#
# Paths live under $GM_HOME (default ~/gm-core) — not /scratch.
set -euo pipefail
GPU_TYPE=${GPU_TYPE:-nvidia_h200_nvl}
MIG_RE='_[0-9]+g\.[0-9]+gb'     # MIG gres types, e.g. nvidia_h200_nvl_3g.71gb

if [ -z "${SLURM_JOB_ID:-}" ]; then
    # submission mode: request exactly NGPU GPUs (env vars travel with sbatch's
    # default environment propagation — no --export)
    : "${RUN:?set RUN}"
    : "${INIT:?set INIT (the Stage-1 train_multi_game checkpoint)}"
    NGPU=${NGPU:-4}
    if [ "$NGPU" -gt 1 ] && [[ "$GPU_TYPE" =~ $MIG_RE ]]; then
        echo "GPU_TYPE=$GPU_TYPE is a MIG slice: a CUDA process sees one MIG instance," \
             "so DDP needs full GPUs (GPU_TYPE=nvidia_h200_nvl) or NGPU=1" >&2
        exit 1
    fi
    self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    sinfo -p "${PARTITION:-batch}" -N -h \
        -O "NodeHost:20,Gres:40,GresUsed:50,StateCompact:10" 2>/dev/null || true
    export NGPU
    # shellcheck disable=SC2086
    exec sbatch --requeue --partition="${PARTITION:-batch}" --gres="gpu:$GPU_TYPE:$NGPU" \
        --cpus-per-task=$((8 * NGPU)) --mem="$((${MEM_PER_GPU:-96} * NGPU))G" \
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
if [ "$NGPU" -gt 1 ] && { [[ "${CUDA_VISIBLE_DEVICES:-}" == *MIG-* ]] ||
        scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -Eq "gpu:[A-Za-z0-9_]*$MIG_RE"; }
then
    echo "MIG slices in this allocation: DDP needs NGPU full GPUs" >&2
    exit 1
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
# ARGS may override --ckpt_minutes (the last occurrence wins); log.txt is
# appended to, so a resumed run keeps the earlier part
# shellcheck disable=SC2086
"$GM_HOME/.venv/bin/torchrun" --standalone --nproc_per_node="$NGPU" \
    -m training.train_rl_joint --init "$INIT" --ckpt_minutes "${CKPT_MINUTES:-30}" \
    --snapshot_dir "$OUT/snapshots" ${ARGS:-} \
    ${RESUME_FROM:+--resume "$RESUME_FROM"} --save_dir "$OUT" 2>&1 | tee -a "$OUT/log.txt"
echo "[$(date)] done"
