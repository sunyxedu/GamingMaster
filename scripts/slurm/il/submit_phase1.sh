#!/bin/bash
# Submit phase-1 IL with any nodes x gpus-per-node shape. The effective batch
# stays at TARGET_EFF via auto-computed grad accumulation, so every shape
# trains the SAME trajectory — only wall-clock differs.
#
# Usage:
#   scripts/slurm/il/submit_phase1.sh 3 4        # 3 nodes x 4 GPUs = 12 (fits busy nodes)
#   scripts/slurm/il/submit_phase1.sh 1 8        # classic single node
#   scripts/slurm/il/submit_phase1.sh 2 6        # 12 GPUs as 2x6
#   RESUME=/path/ckpt.pt scripts/slurm/il/submit_phase1.sh 3 4    # continue a run
#
# Prefer shapes where 2 x nodes x gpus divides 384 exactly:
#   total GPUs 4, 6, 8, 12, 16, 24, 32 are all exact.
set -euo pipefail

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

NODES=${1:?usage: scripts/slurm/il/submit_phase1.sh <nodes> <gpus_per_node>}
GPN=${2:?usage: scripts/slurm/il/submit_phase1.sh <nodes> <gpus_per_node>}
WORLD=$(( NODES * GPN ))
MICRO=${MICRO:-2}
TARGET_EFF=${TARGET_EFF:-384}
ACCUM=$(( TARGET_EFF / (MICRO * WORLD) ))
[ "$ACCUM" -lt 1 ] && ACCUM=1
EFF=$(( MICRO * WORLD * ACCUM ))
echo "shape: ${NODES} nodes x ${GPN} GPUs = ${WORLD} | accum ${ACCUM} -> effective ${EFF}"
[ "$EFF" -ne "$TARGET_EFF" ] && echo "WARN: effective ${EFF} != target ${TARGET_EFF} (inexact division)"

jid=$(GPUS_PER_NODE=$GPN sbatch --parsable --export=ALL --chdir="$GAMINGMASTER_ROOT" \
    --nodes="$NODES" --gres=gpu:nvidia_h200_nvl:"$GPN" \
    "$GAMINGMASTER_ROOT/scripts/slurm/il/train_il_phase1_job.sh")
echo "submitted job $jid  ->  tail -f slurm-${jid}.out"
