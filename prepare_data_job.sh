#!/bin/bash
#SBATCH --job-name=gm-data-v2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --exclude=gpu[1-6]        # cpu[1-4] are idle — keep GPU nodes free

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

export PYTHONPATH=~/GamingMaster

srun apptainer exec /scratch/$USER/pytorch.sif \
    python ~/GamingMaster/scripts/prepare_data.py \
        --players Human.exe \
        --output /scratch/$USER/gaming-master/data-v2 \
        --target 3000 \
        --min-turns 80 \
        --min-opponent-stars 40 \
        --workers 64
