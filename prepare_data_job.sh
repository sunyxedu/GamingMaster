#!/bin/bash
#SBATCH --job-name=gm-data-v2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --exclude=gpu[1-6]        # cpu[1-4] are idle — keep GPU nodes free

# Regenerate IL training data with the FIXED simulator.
# Datasets generated before the 2026-07 simulator fixes (mountain moves,
# dropped-move labels, afk winners, oversize/feature/modifier rejection)
# are polluted and must not be used — write to a fresh data-v2 directory.
#
# The 'download' stage needs internet: if compute nodes are offline, run
#   uv run python -m scripts.prepare_data --players ... --stages download ...
# on the login node first; this job then only filters/simulates/converts.

export PYTHONPATH=~/GamingMaster

srun apptainer exec /scratch/$USER/pytorch.sif \
    python ~/GamingMaster/scripts/prepare_data.py \
        --players Human.exe \
        --output /scratch/$USER/gaming-master/data-v2 \
        --stages filter simulate convert \
        --min-turns 80 \
        --min-opponent-stars 40
