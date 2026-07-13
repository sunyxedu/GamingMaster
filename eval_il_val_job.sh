#!/bin/bash
#SBATCH --job-name=gm-il-val
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

# Evaluate the LAST few phase-2 IL epoch checkpoints on the held-out *-val
# splits (mid-epoch _step saves are ignored). Checkpoints are sharded
# round-robin across however many GPUs the job was granted (auto-detected),
# one eval_il.py process per GPU; a merged, loss-sorted summary is printed
# at the end.
#
#   sbatch eval_il_val_job.sh                                # last 5, 4 GPUs
#   LAST=10 sbatch --gres=gpu:nvidia_h200_nvl:2 eval_il_val_job.sh
#   EPOCHS="10 15 20 23" sbatch eval_il_val_job.sh           # explicit epochs
#
# Per-GPU CSVs are APPENDED across runs, so the merged table at the end
# accumulates every epoch evaluated so far — rerunning with new EPOCHS
# extends the same sorted summary.
#
# Override the checkpoint dir or output:
#   CKPT_DIR=... OUT_DIR=... sbatch eval_il_val_job.sh

export PYTHONWARNINGS="ignore::UserWarning"
export PYTHONPATH=~/GamingMaster
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp

CKPT_DIR=${CKPT_DIR:-/scratch/$USER/gaming-master/checkpoints/il_phase2}
OUT_DIR=${OUT_DIR:-$CKPT_DIR/val_eval}
# One shard per granted GPU — follows whatever --gres the job was submitted with
GPUS=${GPUS:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}}
LAST=${LAST:-5}
mkdir -p "$OUT_DIR"

TRAIN_ROOT=/scratch/$USER/gaming-master/data-v2/training
# Same nine players phase 2 trained on — their held-out val splits.
VAL_DIRS=(
    "$TRAIN_ROOT/Average Joe-val"
    "$TRAIN_ROOT/Hermione Granger-val"
    "$TRAIN_ROOT/L_7d_gae90_30k_ema-val"
    "$TRAIN_ROOT/Sinyth-val"
    "$TRAIN_ROOT/YiShengYiShi-val"
    "$TRAIN_ROOT/asdvlaksudhiouwenf-val"
    "$TRAIN_ROOT/bike9800-val"
    "$TRAIN_ROOT/cute_x1aoyu_ovo-val"
    "$TRAIN_ROOT/d3st1ny-val"
)

# Epoch checkpoints only (skip the mid-epoch _step500 saves), numeric
# order. EPOCHS="10 15 20" picks those exact epochs; otherwise the last $LAST.
if [ -n "${EPOCHS:-}" ]; then
    CKPTS=()
    for e in $EPOCHS; do
        f="$CKPT_DIR/checkpoint_epoch$e.pt"
        [ -f "$f" ] && CKPTS+=("$f") || echo "[warn] missing $f — skipped"
    done
else
    mapfile -t CKPTS < <(ls "$CKPT_DIR"/checkpoint_epoch*.pt | grep -v _step | sort -V)
    CKPTS=("${CKPTS[@]: -$LAST}")
fi
echo "Evaluating ${#CKPTS[@]} checkpoints (${CKPTS[0]##*/} .. ${CKPTS[-1]##*/}) on ${GPUS} GPUs"

for i in $(seq 0 $((GPUS - 1))); do
    # Round-robin shard: GPU i takes checkpoints i, i+GPUS, i+2*GPUS, ...
    SHARD=()
    for j in $(seq $i $GPUS $(( ${#CKPTS[@]} - 1 ))); do
        SHARD+=("${CKPTS[$j]}")
    done
    [ ${#SHARD[@]} -eq 0 ] && continue
    CUDA_VISIBLE_DEVICES=$i apptainer exec --nv /scratch/$USER/pytorch.sif \
        python3 ~/GamingMaster/eval_il.py \
            --checkpoints "${SHARD[@]}" \
            --data_dir "${VAL_DIRS[@]}" \
            --world_model_weight 0.2 \
            --batch_size 8 --num_workers 8 \
            --csv "$OUT_DIR/val_gpu$i.csv" \
            > "$OUT_DIR/val_gpu$i.log" 2>&1 &
done
wait

echo
echo "=== Merged results (sorted by val loss) ==="
{ head -n1 "$OUT_DIR"/val_gpu0.csv; tail -q -n +2 "$OUT_DIR"/val_gpu*.csv | sort -t, -k2 -g; } \
    | tee "$OUT_DIR/val_all_sorted.csv" | column -s, -t
