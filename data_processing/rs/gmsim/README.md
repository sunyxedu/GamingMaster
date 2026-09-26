# gmsim — gmrules simulators for Python

`gmsim` exposes every game registered in `rs/gmrules` (`registry::GAMES`) to Python:
batched simulators (`gmsim.Sim`, the RL environments behind `env/vec_gmsim.py`),
verified replay of trajectories (`gmsim.replay`), single states for step-by-step play
(`gmsim.State`, differential tests against reference implementations) and the
invariance / conformance battery (`gmsim.check_game`, `gmsim.perturb_hidden`). The
rules and encoders are gmrules', the same code the `gmdata` converters run.

Layout: a PyO3 crate (`src/lib.rs` → extension module `gmsim._native`, abi3 for
CPython >= 3.10) plus the pure-Python package `python/gmsim/` that views the
extension's buffers as numpy arrays. Built by maturin (`pyproject.toml`).

## Build

The root `pyproject.toml` depends on `gmsim` through a uv path source, so

```sh
uv sync            # builds and installs gmsim (Rust release build, ~20 s first time)
```

builds it on macOS (arm64 / x86-64) and Linux (x86-64 / aarch64). Requirements:
`cargo` on `PATH` (stable Rust, https://rustup.rs) — maturin itself comes from PyPI as
a build dependency. On the cluster (user-level rustup in `~/.cargo`):

```sh
source ~/.cargo/env          # or: export PATH="$HOME/.cargo/bin:$PATH"
uv sync
```

Slurm jobs run `.venv/bin/python` / `torchrun` directly and never rebuild; run
`uv sync` on the login node after pulling Rust changes.

Rebuilds are automatic: `[tool.uv] cache-keys` lists this crate's and gmrules'
sources, so `uv sync` / `uv run` rebuild the wheel whenever they change (a new game
in gmrules is picked up by the next `uv run`). Force one with
`uv sync --reinstall-package gmsim`.

`cargo build` / `cargo test` in `data_processing/` skip this crate (it is not in the
workspace `default-members`: a PyO3 extension only links through maturin); its logic
is tested in gmrules (`cargo test --release -p gmrules`) and from Python
(`uv run pytest tests/test_candidate_*.py tests/test_gmrules_*.py`).

## API (see `python/gmsim/__init__.py`)

Frames are named tensors in three groups declared by the game's spec (`obs`,
`legal`, `target`; each a list of `{"name", "shape", "dtype", "bits"}`), plus the
`action_keys` (int64 targets = the policy's action). Candidate-action games:
obs `entities` / `entity_mask` / (`entity_type`) / `globals` / `actions` /
`action_mask`, legal `action`, action key `action_idx`.

```python
import gmsim
gmsim.games()                                   # ['kuhn_poker', 'goofspiel', 'pig', 'breakthrough', ...]
spec = gmsim.spec_of("kuhn_poker", {"players": 3})   # schema, obs/legal/target tensors, action_keys, meta, layout
sim = gmsim.Sim("kuhn_poker", {"players": 3}, n_envs=64, threads=4)
sim.reset(range(64), seeds)                     # per-episode seeds
o = sim.observe()                               # {"obs": {name: [R, ...]}, "legal": {...}, active, step_idx, ...}
info = sim.step({"action_idx": idx})            # or an int array [R, K]; rewards [N, S], done, applied, acted
sim.random_actions(seed)                        # {action key: [R]}: uniform legal actions (any schema)
sim.trajectory(0)                               # int64 [n, 2] (seat, id), seat -1 = chance
r = gmsim.replay("kuhn_poker", cfg, events)     # r["obs"] / r["legal"] / r["target"]: [D, ...] per decision
st = gmsim.State("breakthrough"); st.frame(0); st.target(0, a); st.decode(0, keys)
gmsim.perturb_hidden(game, cfg, events, seat, seed)
gmsim.check_game(game, cfg, n_games=200)        # raises ValueError on a violation
```

`Sim.reset` / `Sim.step` / `replay` / `check_game` release the GIL; `threads`: 1 =
the calling thread (default), 0 = rayon's global pool, n > 1 = n threads (results
never depend on it).
