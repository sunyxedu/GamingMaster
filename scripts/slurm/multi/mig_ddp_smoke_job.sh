#!/bin/bash
#SBATCH --job-name=gm-mig-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl_1g.18gb:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#
# Smoke test of data parallelism over MIG slices (training/distributed.py) on
# NGPU (default 2) slices of one GPU, in three steps:
#   1. the DDP equivalence tests, whose CUDA variants now run gloo on CUDA
#      tensors with every rank pinned to its own slice (tests/test_mig_ddp.py,
#      test_grad_sync, test_multi_game_ddp, test_rl_joint_dist,
#      test_rl_joint_trainer);
#   2. scripts/bench_grad_sync.py: gradient sync time per bucket size for a
#      208M-parameter model (gloo between two slices);
#   3. a short real `torchrun -m training.train_multi_game` run on synthetic
#      othello + chess shards with --dist_backend auto (the multi_game_ddp_job.sh
#      command line); the log must show one gloo rank per slice.
#
#   sbatch scripts/slurm/multi/mig_ddp_smoke_job.sh          # 2 x 1g.18gb
#   GPU_TYPE=nvidia_h200_nvl_1g.18gb NGPU=3 bash scripts/slurm/multi/mig_ddp_smoke_job.sh
#
# (bash mode submits itself with --gres gpu:$GPU_TYPE:$NGPU, 4 CPUs and 24 GiB
# of RAM per slice; environment-variable prefixes only, never sbatch --export.)
# GM_HOME (default ~/gm-core) must be a checkout of the code under test with
# its .venv; outputs go to $OUT (default ~/gm-runs/mig-smoke/<job id>).
set -euo pipefail
GPU_TYPE=${GPU_TYPE:-nvidia_h200_nvl_1g.18gb}

if [ -z "${SLURM_JOB_ID:-}" ]; then
    NGPU=${NGPU:-2}
    self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    export NGPU
    # shellcheck disable=SC2086
    exec sbatch --partition="${PARTITION:-batch}" --gres="gpu:$GPU_TYPE:$NGPU" \
        --cpus-per-task=$((4 * NGPU)) --mem="$((24 * NGPU))G" ${SBATCH_EXTRA:-} "$self"
fi

GM_HOME=${GM_HOME:-$HOME/gm-core}
OUT=${OUT:-$HOME/gm-runs/mig-smoke/$SLURM_JOB_ID}
VISIBLE=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c . || true)
NGPU=${NGPU:-$VISIBLE}
mkdir -p "$OUT"
cd "$GM_HOME"
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
export PYTHONPATH="$GM_HOME${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} TMPDIR=/tmp
PY="$GM_HOME/.venv/bin/python"
echo "[$(date)] mig smoke on $(hostname) ngpu=$NGPU commit=$(git rev-parse --short HEAD 2>/dev/null)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
status=0

echo "=== 1/3 DDP equivalence tests (gloo on CUDA tensors, one slice per rank)"
"$PY" -m pytest -q -rs tests/test_mig_ddp.py tests/test_grad_sync.py \
    tests/test_multi_game_ddp.py tests/test_rl_joint_dist.py tests/test_rl_joint_trainer.py \
    2>&1 | tee "$OUT/pytest.txt" || status=1

echo "=== 2/3 gradient-sync benchmark (208M parameters)"
"$GM_HOME/.venv/bin/torchrun" --standalone --nproc_per_node="$NGPU" \
    scripts/bench_grad_sync.py --params 208e6 --bucket_mb 0 5 25 100 400 \
    2>&1 | tee "$OUT/bench.txt" || status=1

echo "=== 3/3 train_multi_game under torchrun on synthetic shards"
"$PY" - "$OUT/data" <<'EOF'
import os, sys
from data_processing.games.dataset import save_shard
from tests.test_core_moe import _board_samples
for g in ("othello", "chess"):
    os.makedirs(f"{sys.argv[1]}/{g}", exist_ok=True)
    save_shard(f"{sys.argv[1]}/{g}/a.pt", g, _board_samples(g, 64, 32))
EOF
"$GM_HOME/.venv/bin/torchrun" --standalone --nproc_per_node="$NGPU" \
    -m training.train_multi_game --dist_backend "${DIST_BACKEND:-auto}" \
    --data othello="$OUT/data/othello" --data chess="$OUT/data/chess" \
    --val othello="$OUT/data/othello" --steps 30 --batch_size 4 --warmup_steps 5 \
    --log_every 10 --eval_every 30 --ckpt_every 30 --save_dir "$OUT/ck" \
    2>&1 | tee "$OUT/train.txt" || status=1
ranks=$(grep -c "^\[dist\] rank .*backend=gloo device=cuda:0" "$OUT/train.txt" || true)
if [ "$ranks" != "$NGPU" ] || [ ! -f "$OUT/ck/multi_final.pt" ]; then
    echo "train_multi_game: $ranks/$NGPU gloo ranks on cuda:0, final checkpoint" \
         "$( [ -f "$OUT/ck/multi_final.pt" ] && echo present || echo MISSING )" >&2
    status=1
fi
echo "[$(date)] done, status=$status (logs in $OUT)"
exit "$status"
