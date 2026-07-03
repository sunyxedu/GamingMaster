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

# The pipeline overlaps network and CPU (background downloader feeding a
# always-full process pool), so --workers = all cores saturates the node.
#
# Multiple players? One node each fills cpu[1-4]:
#   for P in Human.exe qianglituomao Crescent_; do
#     sbatch --export=ALL,PLAYER=$P prepare_data_job.sh
#   done
# (set --players "${PLAYER:-Human.exe}" below to use the env override)
#
# No srun (single-task job; step creation times out on the cpu partition)
# and no apptainer (the container lacks the lzstring package — the repo
# venv on shared home has everything and cpu nodes run its wheels fine).
export PYTHONPATH=~/GamingMaster
cd ~/GamingMaster

.venv/bin/python -m scripts.prepare_data \
        --players "${PLAYER:-Human.exe}" \
        --output /scratch/$USER/gaming-master/data-v2 \
        --target 3000 \
        --min-turns 80 \
        --min-opponent-stars 40 \
        --workers 20
