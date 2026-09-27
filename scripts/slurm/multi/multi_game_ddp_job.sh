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
#   # --gres / --cpus-per-task (CPUS_PER_GPU, default 8 per GPU) / --mem (MEM_PER_GPU,
#   # default 96, GiB per GPU); GPU_TYPE (default nvidia_h200_nvl), SBATCH_EXTRA
#   # (extra sbatch flags)
#   NGPU=2 RUN=core-rdmoe-2g ARGS="..." bash scripts/slurm/multi/multi_game_ddp_job.sh
#
#   # NGPU MIG slices of one GPU (e.g. 2 x 1g.18gb): the same command with a MIG
#   # GPU_TYPE; per slice the defaults become 4 CPUs and 24 GiB of RAM (times the
#   # slice's compute units, e.g. 3g.71gb: 12 CPUs / 72 GiB)
#   GPU_TYPE=nvidia_h200_nvl_1g.18gb NGPU=2 RUN=core-mig2 ARGS="..." \
#       bash scripts/slurm/multi/multi_game_ddp_job.sh
#
# Check there is room first — NGPU free GPUs (or slices of one type) on a single node:
#   sinfo -p batch -N -h -O "NodeHost:20,Gres:40,GresUsed:50,StateCompact:10"
#
# MIG slices: a CUDA process can use ONE MIG instance and NCCL refuses two ranks
# on the same physical GPU, so plain torchrun is enough — each rank narrows
# CUDA_VISIBLE_DEVICES to its own slice before touching CUDA and runs on cuda:0,
# and --dist_backend auto (passed below; DIST_BACKEND overrides) picks gloo, which
# all-reduces CUDA tensors through pinned host memory (~3.4 GB/s between two
# slices: ~0.25-0.5 s per step for a 208M-parameter model's gradient).  A 1g.18gb
# slice has ~16 GiB usable: bound the heavy games' batches (e.g. --game_batch
# generals=1 --game_max_len generals=256, --game_batch <game>=2 for long
# sequences) and prefer --bf16; the full-GPU default --batch_size 8 may not fit.
#
# generals: --data generals=DIR (`gmdata generals shard` output, or gio
# records for small corpora; see multi_game_job.sh) with --game_max_len /
# --game_batch generals=... to bound its long sequences; teacher forcing
# defaults to 0 for generals (--game_tf generals=0.4:0 anneals), and
# --game_loss_weight generals=0.4 evens out its larger loss scale.
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
# New games join a running RUN: resubmit it with more --data entries in ARGS
# (any order; the checkpoint's games must all stay) and they start fresh
# around the trained core (train_multi_game.py docstring, "Resuming").  Pass
# --global_games_per_step (or --games_per_step) explicitly: its default, the
# number of games, would grow the batch with every game added.
#
# Sharded optimizer state (ZeRO-1, training/zero.py): SHARD_OPTIMIZER=1 (or
# --shard_optimizer in ARGS) makes every rank keep AdamATan2's two moments and
# the EMA shadows of its 1/NGPU partition of the parameters only — 8 + 12/NGPU
# instead of 20 bytes per parameter; weights and gradients stay full.  Training
# is bit-identical to the unsharded run and the checkpoint format is the same,
# so a run may switch the flag or NGPU across a resume.  SHARD_OPTIMIZER=1 is
# ignored with NGPU=1 (the trainer refuses the flag there).  Cost: one extra
# bucketed broadcast of the updated weights per step (about half the traffic of
# the gradient all-reduce).  Per-rank numbers and expected peaks for a config:
#   uv run python scripts/zero1_memory.py --experts 32 64 --world 12
# e.g. the 64-expert core (~364M parameters) on 12 x 1g.18gb: ~11.4 GiB peak per
# rank sharded vs ~15.1 GiB unsharded (extrapolated from the measured 11.9 GiB
# of the 32-expert run):
#   GPU_TYPE=nvidia_h200_nvl_1g.18gb NGPU=12 SHARD_OPTIMIZER=1 RUN=core-moe64 \
#       ARGS="--override moe_experts=64 ..." bash scripts/slurm/multi/multi_game_ddp_job.sh
#
# Paths live under $GM_HOME (default ~/gm-core) — not /scratch.
set -euo pipefail
GPU_TYPE=${GPU_TYPE:-nvidia_h200_nvl}
MIG_RE='_([0-9]+)g\.[0-9]+gb'   # MIG gres types, e.g. nvidia_h200_nvl_3g.71gb

if [ -z "${SLURM_JOB_ID:-}" ]; then
    # submission mode: request exactly NGPU GPUs (env vars travel with sbatch's
    # default environment propagation — no --export)
    : "${RUN:?set RUN}"
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
if [ "$NGPU" -gt 1 ] && [[ "${CUDA_VISIBLE_DEVICES:-}" == *MIG-* ]]; then
    # every rank pins its own slice (training.distributed.pin_rank_device)
    echo "MIG slices: $NGPU ranks, one slice each, gloo (--dist_backend ${DIST_BACKEND:-auto})"
fi
SHARD_FLAG=""
if [ "${SHARD_OPTIMIZER:-0}" != 0 ]; then
    if [ "$NGPU" -gt 1 ]; then
        SHARD_FLAG="--shard_optimizer"
        echo "ZeRO-1: optimizer state + EMA sharded over $NGPU ranks (--shard_optimizer)"
    else
        echo "SHARD_OPTIMIZER=1 ignored: one rank, nothing to shard over"
    fi
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
# variable-length batches (whole-record training) fragment the caching allocator's
# segments: expandable segments reuse them (override by exporting another value)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
echo "[$(date)] $RUN on $(hostname) ngpu=$NGPU gpu=${CUDA_VISIBLE_DEVICES:-?} commit=$(git rev-parse --short HEAD) resume=${RESUME_FROM:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi -L 2>/dev/null | grep MIG || true
# ARGS may override --ckpt_every (the last occurrence wins); log.txt is appended
# to, so a resumed run keeps the earlier part
# shellcheck disable=SC2086
"$GM_HOME/.venv/bin/torchrun" --standalone --nproc_per_node="$NGPU" \
    -m training.train_multi_game --ckpt_every "${CKPT_EVERY:-500}" \
    --dist_backend "${DIST_BACKEND:-auto}" ${SHARD_FLAG} ${ARGS:-} \
    ${RESUME_FROM:+--resume "$RESUME_FROM"} --save_dir "$OUT" 2>&1 | tee -a "$OUT/log.txt"
echo "[$(date)] done"
