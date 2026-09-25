#!/bin/bash
#SBATCH --job-name=gm-multi-ddp
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#
# One data-parallel multi-game training run on NGPU GPUs of ONE node:
#   torchrun --standalone --nproc_per_node=$NGPU -m training.train_multi_game
# Everything is passed through environment-variable PREFIXES, like
# multi_game_job.sh (never `sbatch --export=VAR=...`, which gets jobs killed on
# hephaestus).  #SBATCH lines cannot read the environment, so:
#
#   # 4 GPUs (the #SBATCH default)
#   RUN=core-rdmoe-4g ARGS="--data chess=... --steps 20000 --override moe_experts=8" \
#       sbatch --requeue scripts/slurm/multi/multi_game_ddp_job.sh
#
#   # any NGPU (default 4): run it with bash and it submits ITSELF with a matching
#   # --gres / --cpus-per-task (8 per GPU) / --mem (MEM_PER_GPU, default 96, GiB per
#   # GPU); GPU_TYPE (default nvidia_h200_nvl), SBATCH_EXTRA (extra sbatch flags)
#   NGPU=2 RUN=core-rdmoe-2g ARGS="..." bash scripts/slurm/multi/multi_game_ddp_job.sh
#
# Check there is room first — NGPU free FULL GPUs on a single node (MIG slices
# such as nvidia_h200_nvl_3g.71gb cannot run DDP: a process sees one instance):
#   sinfo -p batch -N -h -O "NodeHost:20,Gres:40,GresUsed:50,StateCompact:10"
#
# generals: --data generals=DIR (gio records or `gmdata generals shard` output;
# see multi_game_job.sh) with --game_max_len / --game_val_max_len /
# --game_batch generals=... to bound its long sequences.
#
# Objective — choose the global batch explicitly:
#   --global_games_per_step K   K micro-batches per optimizer step in total
#                               (split over the ranks): the single-GPU
#                               --games_per_step K objective, same --lr/--steps
#   --games_per_step K          K PER RANK (default: number of games), i.e.
#                               NGPU*K per step: the global batch grows NGPU-fold
#                               at the same --steps (optimizer steps) and --lr
#
# Checkpoints / resume: rank 0 writes $OUT/multi_latest.pt every CKPT_EVERY
# (default 500) optimizer steps.  If it exists when the job starts (requeue
# after a node failure, or the same RUN resubmitted after the 24 h wall), the
# run continues from it (--resume); RESUME=0 starts from scratch instead.
# Size --steps to the wall anyway: a resume restarts the data order.
#
# Paths live under $GM_HOME (default ~/gm-core) — not /scratch.
set -euo pipefail
GPU_TYPE=${GPU_TYPE:-nvidia_h200_nvl}
MIG_RE='_[0-9]+g\.[0-9]+gb'     # MIG gres types, e.g. nvidia_h200_nvl_3g.71gb

if [ -z "${SLURM_JOB_ID:-}" ]; then
    # submission mode: request exactly NGPU GPUs (env vars travel with sbatch's
    # default environment propagation — no --export)
    : "${RUN:?set RUN}"
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
OUT=${OUT:-$HOME/gm-runs/multi/$RUN}
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
mkdir -p "$OUT"
RESUME_FROM=""
if [ "${RESUME:-1}" != 0 ] && [ -f "$OUT/multi_latest.pt" ]; then
    RESUME_FROM="$OUT/multi_latest.pt"
fi
cd "$GM_HOME"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
export PYTHONPATH="$GM_HOME${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
# node-local scratch for torchrun / DataLoader workers (NFS temp dirs trip
# ".nfs* Device or resource busy" at teardown)
export TMPDIR=/tmp
echo "[$(date)] $RUN on $(hostname) ngpu=$NGPU gpu=${CUDA_VISIBLE_DEVICES:-?} commit=$(git rev-parse --short HEAD) resume=${RESUME_FROM:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# ARGS may override --ckpt_every (the last occurrence wins); log.txt is appended
# to, so a resumed run keeps the earlier part
# shellcheck disable=SC2086
"$GM_HOME/.venv/bin/torchrun" --standalone --nproc_per_node="$NGPU" \
    -m training.train_multi_game --ckpt_every "${CKPT_EVERY:-500}" ${ARGS:-} \
    ${RESUME_FROM:+--resume "$RESUME_FROM"} --save_dir "$OUT" 2>&1 | tee -a "$OUT/log.txt"
echo "[$(date)] done"
