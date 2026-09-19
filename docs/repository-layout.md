# Repository conventions

The root holds project metadata, the overview README and the package directories.
Runnable training code, cluster jobs and detailed documentation have dedicated
homes. Local credentials, datasets, checkpoints and run logs are ignored.

## Why this layout

The reference projects use different package layouts but separate responsibilities:

| Reference | Relevant organization |
|---|---|
| [Transformers](https://github.com/huggingface/transformers) | `src/transformers/` for the library; separate `examples/`, `scripts/`, `tests/`, `docs/` |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | A root-level `sklearn/` package; separate `examples/`, `benchmarks/`, `build_tools/`, `maint_tools/`, `doc/` |
| [PyTorch](https://github.com/pytorch/pytorch) | A root-level `torch/` package; separate `tools/`, `scripts/`, `benchmarks/`, `test/`, `docs/` |

GamingMaster keeps its existing package names: changing them would unnecessarily
affect imports and serialized model references. The `training` package makes
training entrypoints importable and distributable. A `src/` migration is not
required to separate these responsibilities.

## Where new files belong

| Content | Location |
|---|---|
| Astra/Jev backends, generated-harness search and game contracts | `metaharness/` |
| Shared neural network, losses and per-game observation/action adapters | `model/` |
| Neural-agent self-play environments | `env/` |
| Replay ingestion, conversion and dataset code | `data_processing/` |
| IL, RL and checkpoint evaluation entrypoints | `training/` |
| Data, visualization and analysis commands | `scripts/` |
| Slurm submissions and job profiles | `scripts/slurm/{data,il,rl,eval}/` |
| Tests and small fixtures | `tests/` |
| Guides, architecture and reports | `docs/` |
| Published harness programs, traces and measurements | `metaharness/artifacts/` |
| Local outputs | Ignored `runs/`, `data/`, `checkpoints/`, `viz_*/` |
| Local credentials/settings | Ignored `.env`, `.claude/`; variable names in `.env.example` |

Keep experiment archives intact; new experiments get new archives. Keep game
data download helpers next to their converter in `data_processing/games/`.
Add new commands to the relevant package instead of adding executable files at
the root. Add an `examples/` directory when there are standalone user examples
to put in it; do not create empty directories just to match another project.

## Training command migration

Run these commands from the checkout after `uv sync`. Pass the same arguments
as before; training and checkpoint formats are unchanged.

| Previous entrypoint | Current command |
|---|---|
| `python train_il.py` | `uv run python -m training.train_il` |
| `python train_rl.py` | `uv run python -m training.train_rl` |
| `python train_il_game.py` | `uv run python -m training.train_il_game` |
| `python train_rl_game.py` | `uv run python -m training.train_rl_game` |
| `python eval_il.py` | `uv run python -m training.eval_il` |

Distributed launches use `torchrun --nproc_per_node=8 --module training.train_il`
(or `training.train_rl`). Python imports use `from training.train_rl import ...`.
The wheel includes both `training` and its `env` dependency; installed modules
can also run from outside the checkout using that environment's Python.

All former root shell scripts are under [scripts/slurm/](../scripts/slurm/README.md).
Use the new paths in personal submission commands; there are no forwarding
scripts at the root.

## Checks for entrypoint changes

Run `uv run pytest tests/ -q`. The suite checks training CLI imports and exercises
Slurm submission/path resolution with stub commands, including paths containing
spaces and Slurm's copied job scripts. These checks do not submit jobs or prove
cluster resource availability.

If package contents change, also run `uv build --wheel` and check the installed
entrypoints from outside the checkout. Update command examples and imports when
moving a file, and keep the root README focused on the overview and quick start.
