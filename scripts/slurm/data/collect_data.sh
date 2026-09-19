#!/bin/bash
# Submit one data-collection job per player (one cpu node each; Slurm
# queues the overflow). Usage:
#   scripts/slurm/data/collect_data.sh                 # default roster below
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

PLAYERS=(
    "bike9800"
    "Average Joe"
    "d3st1ny"
    "L_7d_gae90_30k_ema"
    "cute_x1aoyu_ovo"
)
for P in "${PLAYERS[@]}"; do
    jid=$(sbatch --parsable --export=ALL --chdir="$GAMINGMASTER_ROOT" \
        -J "gm-data-${P// /_}" "$GAMINGMASTER_ROOT/scripts/slurm/data/prepare_data_job.sh" "$P")
    echo "submitted $jid  ->  $P"
done

cat <<'NEXT'

Monitor:   squeue -u $USER
           grep -h "trainable" slurm-*.out | tail
Finished?  grep -h "RESULT" slurm-*.out     # merge+verify run automatically;
                                            # expect "all files valid" per job
NEXT
