# MiniGrid / BabyAI interfaces

Evidence status, 2026-10-30: the specified evaluator Python was inspected and is CPython 3.12.3; `minigrid`, `gymnasium`, `numpy`, and `pygame` are absent. Requests to the official docs, GitHub raw source, and PyPI failed DNS resolution. URLs below identify authoritative verification targets; they were not retrieved here. Version-specific details must be checked against a locally supplied distribution before implementation. No installation or environment rollout was possible in this session. This is a concrete handoff specification with explicit verification gates, not a claim of a tested integration.

The recommended adaptor is the official Python environment API. MiniGrid is a local, deterministic benchmark suite, so it has no server, account, matchmaking service, or game credentials. The authoritative project is the Farama Foundation repository and its documentation ([project](https://github.com/Farama-Foundation/Minigrid), [basic usage](https://minigrid.farama.org/content/basic_usage/), [environment catalog](https://minigrid.farama.org/environments/minigrid/)). Pin and record the package version because environment registration and private attributes can change.

## Interface comparison

| Interface | What it exposes and latency | Maturity/cost | Legal/operational risk | Assessment |
|---|---|---|---|---|
| **Gymnasium API (`gym.make("MiniGrid-...-v0")`)** | Synchronous `reset(seed)` and `step(action)`; structured dict observation (`image`, `direction`, `mission`) and scalar reward/termination. A call advances exactly one game step; model thinking does not advance the world. | Official, maintained, free; CPU-only for normal use. | No online service or anti-cheat issue. Must follow package license and pin versions. | **Choose this.** It is the benchmark interface and gives exact actions and state without vision or input injection. |
| Direct MiniGrid internals (`env.unwrapped`, grid/world objects) | Adds exact grid cells, agent position/direction, carried object, mission and step counter; useful for debugging and a text-state serializer. | Official source but semi-private, version-sensitive; free and negligible latency. | No external risk; coupling to internals can break after upgrades. | Use only inside an optional debug/diagnostic path; the playing adaptor should rely on the public observation. |
| Official wrappers | `FullyObsWrapper` gives a full symbolic grid; `ViewSizeWrapper` changes egocentric window; `RGBImgPartialObsWrapper`/`RGBImgObsWrapper` produce pixels; `ImgObsWrapper`, `FlatObsWrapper`, and `OneHotPartialObsWrapper` alter representation. See [wrapper API](https://minigrid.farama.org/api/wrappers/) and [source](https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/wrappers.py). | Maintained with MiniGrid; free; wrapper overhead is tiny. | Wrapper changes task observability and can invalidate a benchmark's intended partial-observation setting. | Use only when the evaluator explicitly permits it. A textual serializer around the default dict is the safest standard. |
| RL/bot libraries (Stable-Baselines3, CleanRL, RLlib) | These consume a Gymnasium env and issue the same discrete actions; vectorized environments can batch episodes but add no new state access. | Mature general RL tooling, free/open source, but extra dependencies and policy inference latency. | A pretrained policy may be trained on a different ID/version and silently exploit full observations. | Optional offline baseline only; not needed for the Astra harness. |
| Gymnasium registry/inspection | `gym.envs.registry`, `gym.spec(id)`, `env.observation_space`, `env.action_space`, and `env.unwrapped.max_steps` reveal IDs and budgets before play. | Stable Gymnasium tooling; free. | Registry contents vary by installed release. | Required setup audit; never hard-code all IDs or budgets without checking the installed package. |
| Pixel rendering plus screen/input injection (`pygame`, `pyautogui`, `xdotool`/XTest) | `env.render()` can return an RGB array with `render_mode="rgb_array"`; `render_mode="human"` opens a Pygame window. OS screenshot and keyboard tools could read/control a human UI. | Rendering is supported, but screen scraping is slower and GUI-dependent; rendering is optional and pyautogui/xdotool add setup. | No online account, but brittle coordinate/key mapping, X display requirements, and headless failures; screen capture is an unnecessary data path. | Fallback only when a different evaluator exposes pixels instead of Gymnasium. Do not choose on this Linux headless host. |
| BabyAI standalone/legacy bot APIs | Historical BabyAI repositories and imitation/RL agents can provide language parsers or policies, but the maintained tasks are in MiniGrid. A policy library still needs the same local Gymnasium environment. | Research code quality and APIs vary; free but often unmaintained. | Dependency drift and task-version mismatch; no service benefit. | Do not depend on it. Use the installed MiniGrid BabyAI environment IDs and verifier. |
| Emulator/mod/plugin/bot API | There is no separate game executable, network protocol, mod loader, or emulator: MiniGrid is the simulator itself. | Not applicable. | Any unofficial fork can silently change rewards or task semantics. | Not applicable. |

## Installation and setup

The evaluator Python is `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/minigrid/bin/python` (CPython 3.12.3). Install into that interpreter, preferably from an already downloaded wheelhouse because this host is offline:

```bash
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/minigrid/bin/python -m pip install minigrid
# If a wheelhouse is supplied for offline use:
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/minigrid/bin/python -m pip install --no-index --find-links /path/to/wheelhouse minigrid
```

Modern `minigrid` distributions normally depend on Gymnasium, NumPy, and Pygame (Pygame may be imported even without rendering). Install declared dependencies rather than skipping Pygame for text-only play; let the supplied wheel's metadata settle exact versions. Farama-Notifications, cloudpickle, and typing-extensions are common transitive Gymnasium dependencies. No PyTorch, CUDA, browser, ROM, or separate `babyai`/legacy `gym-minigrid` package is needed. Python 3.12 compatibility must be demonstrated by resolving Linux x86_64 CPython 3.12-compatible dependency wheels and importing them in the specified venv; it was not demonstrated here. Verify imports and registration in the same interpreter:

```python
import gymnasium as gym
import minigrid                         # registers MiniGrid/BabyAI IDs
env = gym.make("MiniGrid-Empty-5x5-v0")
obs, info = env.reset(seed=0)
print(env.spec.id, env.action_space, env.observation_space)
```

No environment variables, API keys, server addresses, accounts, or secrets are required. `env` configuration belongs in the harness (selected environment ID, seed, optional wrapper), not in credentials. No SDL variable should be necessary when no window is created; `SDL_VIDEODRIVER=dummy` is an optional rendering workaround, not a game credential. The current tool sandbox also cannot write to the evaluator venv, which is outside the study writable root; installation must occur in a phase with write access to that venv. The adaptor should:

1. Import `minigrid` before `gym.make`, enumerate/validate the evaluator's registered IDs, and record `minigrid`/Gymnasium versions.
2. Construct the evaluator-selected ID with its normal kwargs and no rendering; inspect `action_space.n`, `observation_space`, `env.spec`, and the effective `max_steps` (including an outer `TimeLimit`).
3. Call `obs, info = env.reset(seed=seed)` for every episode. Seed the action space only if the harness samples actions. Do not reuse an old `obs` after reset.
4. Serialize each observation to Astra as JSON/text: mission string; direction (`0` east, `1` south, `2` west, `3` north in current MiniGrid); image dimensions; and each visible cell's object/color/state names decoded from `minigrid.core.constants`. Preserve unknown/occluded cells as unknown. Keep action count, carried object, reward, terminated, truncated, and a compact action history in harness state. Derive inventory from the public observation as described below; expose only the allowed view, never hidden grid contents or verifier targets.
5. Ask Astra for one legal action name or integer, validate it against the installed `Actions` enum, call exactly one `env.step(action)`, and return the resulting five-tuple. The standard action order is `left=0`, `right=1`, `forward=2`, `pickup=3`, `drop=4`, `toggle=5`, `done=6`; query the enum at runtime because a specialized environment may restrict actions.
6. Stop on either `terminated` or `truncated`, then reset. Determine success from the environment's documented success signal (prefer `info["success"]` when supplied; otherwise the task's positive terminal reward/verifier), and log raw reward plus terminal flags rather than guessing from `terminated` alone.

## Observation and control details

The default observation is a Gymnasium dictionary. `image` is a small `uint8` symbolic tensor with one triplet per visible cell (object type, color, and object state); `direction` is the agent orientation; `mission` is the natural-language instruction. The usual view is a 7x7 egocentric window, but `view_size`, wrappers, and individual tasks can differ. It is not an RGB screenshot. The public observation does not expose the complete map; exploration and memory are required. The internal grid is available for diagnostics only and would remove intended partial observability if passed to the model.

The following is the standard base encoding, to confirm against the installed [constants](https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/core/constants.py), [grid encoding](https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/core/grid.py), and [gen_obs implementation](https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/minigrid_env.py):

- Tensor indexing is `image[x, y, channel]`, not conventional image row/column indexing. For a textual row `y`, iterate `x` across that row. With odd view size V, the agent is at `(V//2, V-1)`, looking toward decreasing view y; the front cell is `(V//2, V-2)`.
- Object indices normally are unseen=0, empty=1, wall=2, floor=3, door=4, key=5, ball=6, box=7, goal=8, lava=9, agent=10. Colors normally are red=0, green=1, blue=2, purple=3, yellow=4, grey=5. Door state normally is open=0, closed=1, locked=2. Decode from constants, and only interpret the third channel as door state for doors. A zero channel on a key does not mean an open key.
- In the base partial view, the agent's cell represents the carried object, or empty if not carrying; it is not a normal floor observation and need not contain an `agent` code. Extract inventory there and mark the agent separately in the text. Full-observation wrappers have a different agent representation. Test this before promising public-observation inventory support for every wrapper.
- Relative cell offset is `forward = V-1-y`, `right = x-V//2`. In a map with y increasing south, facing vectors for east/south/west/north are `(1,0),(0,1),(-1,0),(0,-1)`; the right vector is the next vector clockwise. Integrate only successful movements, accounting for doors, obstacles and carried objects. Global `agent_pos` is not a default observation; start a relative map at `(0,0)` rather than disclosing hidden absolute coordinates.
- Some levels allow seeing through walls, and wrappers change this. Preserve exactly the observation mask supplied by the env rather than inventing occlusion rules. Cache prior observed cells with age; an unseen cell this turn does not erase a previously seen object.

Suggested text contract: environment ID, mission, allowed action names, steps used/limit, direction, inventory, view coordinates/convention, a grid of decoded cells with `unknown`, and last action/reward/terminal flags. JSON must contain ordinary numbers/lists/strings, not NumPy objects. Distinguish an absent optional field from a known empty value. Log `info` for diagnostics, but whitelist fields passed to models so debug metadata cannot reveal hidden targets.

Actions are discrete and adjacent-cell based: turn left/right, move forward, pick up, drop, toggle (door/box/key interaction), and `done`. A blocked forward, invalid pickup/drop, or toggle still consumes a step in the normal environment. Interaction requires facing the adjacent object. A matching key unlocks a locked door; carrying capacity is one object. Specialized tasks can override action effects, so the adaptor must use the actual action space and task verifier.

The API is synchronous: wall-clock time spent in Astra does not consume an environment step. This makes the game tier 1. The evaluator may impose an external process timeout, but MiniGrid itself has no decision deadline. Jev can answer small recognition questions from the JSON state, but it should not run an asynchronous action loop.

## Scoring extraction

For standard sparse MiniGrid tasks, a successful terminal reward is commonly `1 - 0.9 * step_count / max_steps`; timeout, lava death, and other failures are zero. This is implemented in [the core environment](https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/minigrid_env.py), but task-specific variants must be checked. The harness should emit `success`, `steps_used`, `max_steps`, `remaining_steps`, `terminated`, `truncated`, raw reward/return, and environment ID/seed. Use the benchmark's success-and-remaining-steps score as the primary metric, and keep raw reward as a diagnostic. Never infer success solely from `terminated` because lava/death and time limits also terminate episodes.

## Offline/reproducibility checks

Before running held-out seeds, run a smoke test over every evaluator-selected ID: reset with two seeds, print observation keys/shapes, step a legal no-op-like turn, and verify termination handling. Save the exact package versions and ID list. If `pip install` cannot reach an index, provide wheels for `minigrid`, `gymnasium`, `numpy`, and their transitive dependencies in a local wheelhouse; no credentials are needed.

On a connected Linux x86_64 Python 3.12 machine, resolve and download a pinned `minigrid` release with all dependencies using `pip download --only-binary=:all: --dest wheelhouse minigrid==<validated-version>`, transfer the directory, then install using the offline command above and run `pip check`. Verify wheel tags and the host's manylinux/glibc compatibility; do not copy wheels for another Python or OS. Record version pins and wheel hashes. For headless smoke testing use `render_mode=None`, not `human`; a display backend must not be required merely to step a text-only episode.

## Scenario selection and admission gates

If the evaluator already supplies an ID, use it. Otherwise fix a versioned, ordered allowlist and a documented seed mapping. Derive two independent deterministic values from the external seed using a named stable algorithm (e.g. SHA-256 over `minigrid-suite-v1:id:<seed>` and `minigrid-suite-v1:layout:<seed>`); use one to choose the ID by explicit weights and the other as Gymnasium reset seed. Do not use Python's randomized `hash()`, a process-global RNG, or discovery order from the registry. Log external seed, actual reset seed, ID, kwargs, and version. Sample reset seeds separately within each family when reporting per-family success; a uniform random ID mix can conceal a hard family's failures.

An initial candidate allowlist to validate in the installed registry is `MiniGrid-Empty-5x5-v0`, `MiniGrid-Empty-Random-6x6-v0`, `MiniGrid-DoorKey-5x5-v0`, `MiniGrid-DoorKey-6x6-v0`, `MiniGrid-MultiRoom-N2-S4-v0`, `BabyAI-GoToRedBallGrey-v0`, `BabyAI-GoToObj-v0`, `BabyAI-PickupLoc-v0`, `BabyAI-Open-v0`, and `BabyAI-PutNextLocal-v0`. These are candidate IDs, not a claim that they were registered locally. Empty fixed-start variants are useful smoke tests but may not vary meaningfully with seed; random-start, object, room, and mission variants should dominate held-out evaluation. No opponents or accounts exist.

For each ID admit it only after confirming success extraction, max steps, mission semantics, observation decoding, and seeded reproducibility. Specialized GoToObject/GoToDoor tasks may require `done`; do not confuse them with BabyAI GoTo verification. DynamicObstacles can expose a reduced action space and moves hazards on each step; Memory has cue-dependent branch success. PutNear/Fetch/other overrides and any reward-shaping wrapper need separate success logic. Exclude unaudited variants rather than guessing from a generic goal tile. Do not add ActionBonus, PositionBonus, NoDeath, observation-expanding, or stochastic-action wrappers to the scored benchmark unless requested.

For audited sparse-success IDs, positive terminal reward can be the success signal if the implementation confirms no positive failure/shaping reward. Do not assume `info['success']` exists; base environments often return an empty dict. A time-limit cutoff is a failure unless a successful terminal event also occurred on that last step. A time-limit wrapper may impose a different budget from the core reward denominator: choose and record one effective evaluation budget and score from it, while retaining raw return separately. `decision_latency_s=-1.0` in study.json is a documented sentinel for no intrinsic deadline, not an executable timeout; evaluator wall-clock limits remain separate.

Required adaptor checks: same ID/seed gives the same initial mission/grid; view x/y and facing transform agree; picking up/dropping updates public inventory; opening/unlocking changes door state; `done` behavior is audited; blocked actions spend a step; success, hazard failure, timeout and last-step success are distinguished. Check dependency imports without a display before launching a suite. These are integration checks for the next phase; no game-playing code has been written here.

## Further source targets and licensing

- Gymnasium reset/step API and seeding: https://gymnasium.farama.org/api/env/ and https://gymnasium.farama.org/introduction/basic_usage/
- MiniGrid package metadata / Python requirements: https://pypi.org/project/minigrid/ and https://github.com/Farama-Foundation/Minigrid/blob/master/pyproject.toml
- Registration and complete ID list: https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/__init__.py
- BabyAI catalog and verifier: https://minigrid.farama.org/environments/babyai/ and https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/envs/babyai/core/verifier.py
- Historical BabyAI agents: https://github.com/mila-iqia/babyai ; general RL integration: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
- Manual keyboard tool: https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/manual_control.py
- MiniGrid is distributed under the MIT license; retain required notices for copied code: https://github.com/Farama-Foundation/Minigrid/blob/master/LICENSE . Check wheel dependency licenses if redistributing a bundle. No per-game fee, remote-service ToS, anti-cheat bypass, or external account is involved.

Latency estimates in the table are qualitative, not measurements: small symbolic CPU steps should be much faster than an Astra call; large rendering/vector/RL stacks add overhead but no game-time urgency.
