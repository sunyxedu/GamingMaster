# Sokoban interfaces

Sokoban has no network service or universal official gameplay API. The useful interfaces are local simulators, level files, and (only for a graphical commercial copy) screen/input automation. The recommendation below is designed for the Linux headless evaluator and Python 3.12.3 at `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/sokoban/bin/python`.

## Options

| Interface | Maturity and exposure | Latency | Cost and legal/ToS risk |
|---|---|---|---|
| **Standalone Python ASCII-grid engine over pinned Boxoban (chosen)** | Small deterministic engine; parse XSB text, expose JSON observation and four-action transition. Full control, reproducible seeds, push/walk counters, and no rendering dependency. It must be implemented by the adaptor phase. | Microseconds to milliseconds per action; model latency dominates. | Python standard library only and free. Boxoban is generally distributed under Apache-2.0, but verify the pinned repository's [LICENSE](https://github.com/google-deepmind/boxoban-levels/blob/master/LICENSE) and retain notices. No service account or anti-cheat issue. |
| `gym-sokoban` legacy Gym package | Mature reference implementation with discrete actions, RGB/state-array rendering, rewards, and done flags; inspect [README](https://github.com/mpSchrader/gym-sokoban) and [sokoban_env.py](https://github.com/mpSchrader/gym-sokoban/blob/master/gym_sokoban/envs/sokoban_env.py). It may pull NumPy, Gym, and pygame and its old Gym API can conflict with Python 3.12. | Fast locally, but import/render overhead and dependency installation add setup risk. | Open-source project; check its license and transitive licenses. No external ToS. Do not rely on its reward numbers without pinning a commit. |
| Jumanji Sokoban | Modern functional/JAX environment family ([repository](https://github.com/instadeepai/jumanji), [docs](https://instadeepai.github.io/jumanji/)); useful for batched training and typed state, but requires JAX and its version-specific API. | Fast after JAX warm-up; heavy install and possible CPU memory cost. | Open source; verify licenses and Python 3.12 wheels. No external service. |
| Pycolab custom game | Headless ASCII-grid engine from DeepMind ([repository](https://github.com/google-deepmind/pycolab)). It supplies scheduling/rendering primitives, but Sokoban rules and deadlock logic still must be written. | Low runtime latency. | Open source; old dependencies and project maintenance make it a weaker current choice. |
| Griddly declarative environment | C++/Python grid-game engine ([repository](https://github.com/Bam4d/Griddly)); can expose observations/actions and headless execution if its Sokoban example and wheels are available. | Low once initialized; native install can be substantial. | Open source; verify current package and license. No game-server ToS. |
| JSoko or another desktop solver | Java GUI application with a strong solver and level editor ([SourceForge](https://sourceforge.net/projects/jsoko/)). Can be driven only through its file/solver interfaces or GUI automation. | Solver can be fast, but process and GUI automation add seconds and brittle synchronization. | Free software, but check distribution license. GUI is unsuitable for the server and using its solver may make the benchmark trivial or violate an intended agent restriction. |
| Screen capture plus input injection | `xdotool`, XTest, Wayland tools, or OCR against a third-party graphical game. Observes pixels and controls keyboard events only. | Tens to hundreds of ms per frame, plus OCR errors. | Requires GUI and a copy of the game; possible game EULA/anti-cheat violation. No value on the headless machine. |
| Emulator hooks | Relevant only if running a console/old binary ROM. Memory addresses and input APIs are version-specific; no canonical Sokoban hook. | Potentially low, but reverse engineering and synchronization dominate. | ROM and emulator licensing/ToS risk; not appropriate for this free offline benchmark. |

## Chosen adaptor contract

Choose the standalone engine. It avoids unavailable wheels and GUI, gives Astra a lossless textual state, and makes seed-to-level mapping and scoring explicit. It is also easier to audit than a shaped-reward package. The adaptor phase should implement the following small boundary:

```text
reset(seed: int) -> observation JSON
step(action: "up"|"down"|"left"|"right") -> observation JSON
```

`reset` must map a seed deterministically to a level in the pinned corpus (for example, `index = stable_hash(seed) % len(levels)`; never use process-randomized Python `hash`). `step` applies the transition rules in [brief.md](brief.md), increments `actions`, and returns `done` when all goals are occupied or the action cap is reached. Include `level_id`, `seed`, `rows`, `cols`, `player`, `boxes`, `goals`, `reachable` (or a reachable mask), `last_action`, `last_event` (`walk`, `push`, `blocked`, or `solved`), `pushes`, `walk_steps`, `actions`, `boxes_on_goals`, `done`, and `valid_actions` in JSON. A canonical ASCII board in the same object is useful for prompt readability.

The engine should normalize CRLF, preserve spaces inside rows, reject malformed levels, and compute `valid_actions` from the current state. Keep level order and a SHA-256 manifest in the run metadata. Do not silently download, generate, or substitute levels during evaluation.

## Installation and environment

No third-party package is required. Use the evaluator interpreter explicitly:

```text
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/sokoban/bin/python
```

The user/staging process must provide:

* `SOKOBAN_LEVELSET` (required): absolute path to the pinned Boxoban or separately licensed classic XSB corpus copied onto the offline machine. Do not put a credential or secret in this variable.
* `SOKOBAN_MAX_ACTIONS` (optional): integer action cap supplied by the benchmark harness; record the value in each run.

Setup steps are: (1) obtain Boxoban from [google-deepmind/boxoban-levels](https://github.com/google-deepmind/boxoban-levels) on a connected staging machine, verify the license, revision, and SHA-256; (2) copy only the selected text split and notices into the offline workspace; (3) parse every level once and fail fast on mismatched box/goal counts; (4) run deterministic transition and reset smoke tests; and (5) expose the JSON boundary above to the generated harness. There are no server addresses, account credentials, API keys, or network calls.

## Why not install a package anyway?

`gym-sokoban` and Jumanji are useful references and optional validation backends. They are poor defaults here because the venv currently has only pip, network resolution is unavailable, and old Gym/pygame/JAX dependency stacks can be difficult to reproduce. If a connected staging machine installs one, pin exact versions and compare its transitions against the standalone engine on a small hand-checked level set before trusting it.
