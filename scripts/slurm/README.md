# Slurm jobs

These are cluster profiles for the neural-agent/data workflows. They retain
their training parameters, resource requests and dataset/checkpoint defaults.
They assume Slurm, the named cluster partitions/GPU resources, shared
`/scratch/$USER/gaming-master/` storage and, for GPU jobs, the Apptainer image
`/scratch/$USER/pytorch.sif`. Adjust those profiles for a different cluster.
Data jobs use the checkout's `.venv/bin/python`, created with `uv sync`.

| Directory | Jobs and helpers |
|---|---|
| `data/` | `collect_data.sh` submits the player roster; `prepare_data_job.sh` runs ingestion |
| `il/` | Phase 1/2, scratch and earlier IL profiles; `submit_phase1.sh` selects node/GPU shape |
| `rl/` | PPO profiles, including phases 3, 5, 6 and 7 and the magnet variant |
| `eval/` | `eval_il_val_job.sh` evaluates IL checkpoints |

From the checkout root:

```bash
bash scripts/slurm/data/collect_data.sh
sbatch scripts/slurm/data/prepare_data_job.sh "Average Joe"
bash scripts/slurm/il/submit_phase1.sh 3 4
sbatch scripts/slurm/il/train_il_scratch_job.sh
sbatch scripts/slurm/rl/train_rl_phase7_job.sh
sbatch scripts/slurm/eval/eval_il_val_job.sh
```

Read each job's header for its inputs and environment overrides. Submission
helpers can also be invoked by absolute path from another working directory.
They pass absolute job paths and `--chdir` to `sbatch` and export the checkout
location to the job.

## Checkout discovery

Slurm copies a submitted script to a spool directory, so a job cannot locate
the repository solely relative to its own filename. Jobs start from
`GAMINGMASTER_ROOT`, then `SLURM_SUBMIT_DIR`, then the script's directory, using
the first nonempty value. They walk upward to find `scripts/slurm/_env.sh` and
fail if it cannot be found. The shared environment changes to the checkout
root and prepends it to `PYTHONPATH`, preserving existing entries.

When directly submitting a job from outside the checkout, set its location:

```bash
export GAMINGMASTER_ROOT=/path/to/GamingMaster
sbatch --chdir="$GAMINGMASTER_ROOT" \
    "$GAMINGMASTER_ROOT/scripts/slurm/il/train_il_phase1_job.sh"
```

The checkout, datasets and container must be visible on every allocated node.
Python launches use `python -m training.<module>` or
`torchrun ... --module training.<module>`. No job assumes `~/GamingMaster`.
Default `slurm-*.out` logs are ignored by Git; use `sbatch --output=...` to
choose another existing log directory.
