#!/bin/bash
# Submit one data-collection job per player (one cpu node each; Slurm
# queues the overflow). Usage:
#   ./collect_data.sh                 # default roster below
#   ./collect_data.sh 2000           # override --target via TARGET env
set -euo pipefail

PLAYERS=(
    "bike9800"
    "Average Joe"
    "d3st1ny"
    "L_7d_gae90_30k_ema"
    "cute_x1aoyu_ovo"
)
OUT=/scratch/$USER/gaming-master/data-v2

for P in "${PLAYERS[@]}"; do
    jid=$(sbatch --parsable -J "gm-data-${P// /_}" prepare_data_job.sh "$P")
    echo "submitted $jid  ->  $P"
done

cat <<'NEXT'

Monitor:   squeue -u $USER
           grep -h "trainable" slurm-*.out | tail
All done?  grep -h "done:" slurm-*.out

Then merge + validate each player:
  for P in "bike9800" "Average Joe" "d3st1ny" "L_7d_gae90_30k_ema" "cute_x1aoyu_ovo"; do
    .venv/bin/python -m scripts.prepare_data -p "$P" -o /scratch/$USER/gaming-master/data-v2 --merge
  done
  srun -p cpu --cpus-per-task=8 --time=2:00:00 bash -lc '
    cd ~/GamingMaster
    for P in "bike9800" "Average Joe" "d3st1ny" "L_7d_gae90_30k_ema" "cute_x1aoyu_ovo"; do
      .venv/bin/python scripts/validate_dataset.py \
          "/scratch/$USER/gaming-master/data-v2/training/$P"
    done'
NEXT
