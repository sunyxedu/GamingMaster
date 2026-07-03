#!/bin/bash
#SBATCH --job-name=gm-data-v2
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out

# Unified data pipeline: fetch → filter → download → simulate → convert,
# one command, resumable via per-player manifests. --target counts games
# that actually enter the training set.
#
# The same command works everywhere: if this compute node has no internet,
# run it on the login node first (numpy-free there: games park in the
# "simulated" state) and this job finishes the conversions.
#
# Datasets generated before the 2026-07 simulator fixes are polluted —
# always write to a fresh data-v2 directory.

# The pipeline overlaps network and CPU (background downloader feeding an
# always-full process pool), so --workers = all cores saturates one node.
#
# To saturate ALL FOUR cpu nodes on a single player, submit as a job array —
# each task owns a disjoint shard of the player's history (no coordination):
#
#   sbatch --array=0-3 prepare_data_job.sh
#   # after all tasks finish, merge the shard lists once:
#   .venv/bin/python -m scripts.prepare_data -p Human.exe \
#       -o /scratch/$USER/gaming-master/data-v2 --merge
#
# Multiple players? Pass the name as the script argument (space-safe):
#   sbatch prepare_data_job.sh "Average Joe"
# or submit the whole roster with collect_data.sh.
#
# No srun (single-task job; step creation times out on the cpu partition)
# and no apptainer (the container lacks the lzstring package — the repo
# venv on shared home has everything and cpu nodes run its wheels fine).
export PYTHONPATH=~/GamingMaster
cd ~/GamingMaster

# Shard spec from the array task id (defaults to 0/1 = unsharded)
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    SHARD="$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"
else
    SHARD="0/1"
fi

.venv/bin/python -m scripts.prepare_data \
        --players "${1:-${PLAYER:-Human.exe}}" \
        --shard "$SHARD" \
        --output /scratch/$USER/gaming-master/data-v2 \
        --target 10000 \
        --min-turns 80 \
        --min-opponent-stars 40 \
        --workers 20
