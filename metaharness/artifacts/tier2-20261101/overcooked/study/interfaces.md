# Programmatic interfaces for Overcooked

Research date: 2026-10-30. The user specifically requests the open-source **HumanCompatibleAI/overcooked_ai** environment, headless Linux, no account, symbolic state, and a scripted partner. That is the recommended interface. The commercial game and GUI approaches are documented as alternatives for completeness, but should not be used for this run.

## Decision

### Chosen: official Overcooked-AI Python MDP/environment, plus an adaptor-owned 6-Hz ticking worker

Use `OvercookedGridworld` for exact deterministic transitions and `OvercookedEnv` for episode bookkeeping. Construct a concrete layout with `OvercookedGridworld.from_layout_name(layout_name, **overrides)`, wrap it with `OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)`, and use `env.state` as the canonical symbolic observation. Call `env.step((action0, action1))` with native action objects. Use `env.reset()` or `mdp.get_random_start_state_fn(...)()` for reset variants. Serialize/deserialize through `OvercookedState.to_dict()` / `OvercookedState.from_dict()`; use `env.lossless_state_encoding_mdp(state)` only for a tensor policy, not as the canonical model prompt.

The next adaptor controls one index and supplies the second action from the scripted partner. The cleanest interface is to make the adaptor an `Agent` with `action(state) -> (action, info)` and run the partner as a second `Agent` in `AgentPair`, or call the pair's actions directly. A strong baseline is `GreedyHumanModel(env.mlam)`; its implementation returns a native action and `{"action_probs": ...}`. It is intentionally limited: its own docstring says it is intended for the 3-onion order and may not work on `forced_coordination`. For mixed recipes or forced handoffs, write a deterministic partner around `MediumLevelActionManager` / `MotionPlanner` or use a fixed plan, rather than pretending the stock partner is general.

This beats every other option here because it is local, open-source/MIT, account-free, deterministic, symbolic, CPU-capable, and exposes the exact transition state and reward. It avoids screenshots, OCR, GUI focus, network variance, commercial game state ambiguity, anti-cheat, and input-injection fragility. It is also the reference implementation used by the benchmark paper. The one important addition is ours: a worker thread/process that calls one joint transition every 1/6 second while Astra/Jev reason off-thread. Without that scheduler the native API is synchronous and should be treated as tier 1 in execution, despite the study's requested tier-2 model architecture.

**Important compatibility finding:** upstream `pyproject.toml` currently declares `requires-python = ">=3.10,<3.11"` and `uv.lock` pins Python 3.10. The evaluator is Python 3.12.3. Do not use the default dependency-resolving install blindly. Install the GitHub source with an explicit compatibility override (`pip install --ignore-requires-python ...` or build/install a local wheel without enforcing metadata), then install only the runtime dependencies. In a smoke test here, the source package imported and stepped under Python 3.12.3 after installing `numpy<2`, `scipy`, `gymnasium`, `pygame`, `opencv-python-headless`, `dill`, `tqdm`, `ipywidgets`, and current setuptools. If a future source revision breaks 3.12, vendor/pin the inspected revision or use a separate Python 3.10 runtime; do not silently fall back to a stale PyPI wheel.

## Exact chosen API

### Imports and construction

```python
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld, OvercookedState,
)
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import (
    Agent, AgentPair, GreedyHumanModel,
)

mdp = OvercookedGridworld.from_layout_name(
    "cramped_room", old_dynamics=False
)
env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)
partner = GreedyHumanModel(env.mlam)
```

`OvercookedGridworld.from_layout_name` reads packaged `.layout` files. `from_grid` supports generated grids and configuration overrides. Use `env.mdp` and `env.state` after reset. `OvercookedEnv` defaults to an effectively infinite horizon (`MAX_HORIZON=1e10`), so pass 400 explicitly. `DEFAULT_ENV_PARAMS` is `{"horizon": 400}`.

### Step and return contract

The core call is:

```python
next_state, sparse_reward, done, env_info = env.step((controlled_action, partner_action))
```

The actions are native objects from `Action.ALL_ACTIONS`:

```python
Action.ALL_ACTIONS == [
    Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST,
    Action.STAY, Action.INTERACT,
]
```

Do not send Gym integer indices to `OvercookedEnv.step`; translate with `Action.INDEX_TO_ACTION`. The `Overcooked` Gymnasium wrapper accepts a two-integer tuple, but it randomizes the primary-agent index and is less direct for a custom symbolic harness.

`env_info` includes:

- `sparse_r_by_agent`: per-chef delivery reward for this transition;
- `shaped_r_by_agent`: shaping reward, separate from score;
- `agent_infos`: per-agent action metadata;
- `episode` on the terminal transition, including `ep_sparse_r`, `ep_shaped_r`, `ep_length`, and `ep_game_stats`;
- optional `phi_s` / `phi_s_prime` only when potential display is requested.

`env.state` / `next_state` is an `OvercookedState` with:

```text
players: tuple[PlayerState]
  position: (x,y), orientation: one of the four direction tuples,
  held_object: None or ObjectState/SoupState
objects: dict[(x,y), ObjectState or SoupState]  # excludes held objects
bonus_orders: sorted Recipe list
all_orders: sorted Recipe list
timestep: int
```

For JSON-safe prompts, call `state.to_dict()`. It contains `players`, an object list, `bonus_orders`, `all_orders`, and `timestep`; each player includes `position`, `orientation`, and a serialized held object. `SoupState` includes ingredients, cooking tick, cooking/ready/idle flags, and cook time. Include `terrain_mtx` and `layout_name` in the adaptor's observation because `state.to_dict()` alone does not identify walls, dispensers, pots, or serving locations.

A compact but lossless observation record should include:

```text
{
  episode_id, layout_name, terrain_mtx, horizon,
  state: state.to_dict(),
  sparse_score, last_sparse_reward, last_events,
  sparse_r_by_agent, shaped_r_by_agent,
  partner_action_last, controlled_agent_index,
  deadline_tick, wall_clock_monotonic
}
```

Do not use `str(state)` as a machine protocol. `env.mdp.state_string(state)` is useful for human debugging only.

### Partner integration

Official agent contract:

```python
class Agent:
    def action(self, state):
        return action, action_info_dict
```

For a pair, `AgentPair.joint_action(state)` returns two `(action, info)` tuples; unpack actions before passing them to `env.step`. `AgentPair` sets agent indices 0 and 1. Call `pair.reset()` between episodes. If the partner is stateful, do not share the same object for both agents unless `allow_duplicate_agents=True` and you understand the index reset behavior.

`GreedyHumanModel` uses a precomputed `MediumLevelActionManager` and `MotionPlanner`, chooses a high-level goal, and executes only the first primitive action. Set its index before direct calls. Its default `auto_unstuck=True` uses previous-state history. If requesting batches, `actions(states, agent_indices)` resets history per item; do not use that method when continuity is important.

For a custom scripted partner, prefer this hierarchy:

1. Reuse `MotionPlanner.get_plan(start_pos_and_or, motion_goal)` for shortest legal paths.
2. Use MLAM helpers such as `pickup_onion_actions`, `pickup_tomato_actions`, `pickup_dish_actions`, `put_onion_in_pot_actions`, `put_tomato_in_pot_actions`, `start_cooking_actions`, `pickup_soup_with_dish_actions`, and `deliver_soup_actions`.
3. Return only the first action in the plan per tick; replan after every interaction, collision, changed pot readiness, or partner handoff.
4. For `forced_coordination`, explicitly reserve roles and counters; do not use the stock greedy assertion that only accepts a single three-onion order.

## Tier-2 adaptor scheduler (required design)

The chosen Python environment is a synchronous transition oracle. The adaptor must supply the world clock:

1. At `reset(seed, scenario)`, choose the layout and start state, create a fresh episode ID, reset both agents, precompute `mlam`/motion planning, zero score, and publish tick 0.
2. Start a dedicated high-priority worker at `tick_period = 1/6` seconds. It owns calls to `env.step` and the authoritative state. It must never call Astra or Jev synchronously.
3. Each tick, consume the newest valid controlled-agent primitive. If absent or stale, use `Action.STAY`. Obtain the scripted partner's action from its continuously running worker/cache; if absent, use its safe continuation/STAY. Form one joint action and call `env.step` exactly once.
4. Publish a snapshot after the transition. Include the state tick and monotonically increasing episode ID so delayed model results are discarded.
5. Keep a short action queue (at most one or two ticks) rather than allowing stale queues to execute. A directional action is a primitive, not a plan commitment.
6. Stop at `done` / tick 400, report score, then join workers and clear queues.
7. For deterministic replay, run a synchronous mode with an explicit `advance_one_tick` function; never compare its wall-clock latency with tier-2 mode.

The official demo is informative but not the chosen control surface. Its server uses Socket.IO, queues client actions, advances at 6 FPS, and emits `state_pong`; NPC actions are computed in a background consumer. It is useful as evidence for cadence and as a possible network interface, but its client/session/lobby protocol and wall-clock game timer add unnecessary fragility for this offline task.

## Other practical interfaces

| Interface | Maturity / latency | Exposes | Legal/ToS risk | Cost and verdict |
|---|---|---|---|---|
| **Official `OvercookedGridworld` + `OvercookedEnv` (chosen)** | Research reference implementation; deterministic local transition, typically sub-millisecond to a few milliseconds after planner cache; model latency is external | Full symbolic state, exact terrain, actions, sparse/shaped rewards, event logs, lossless/feature encodings | MIT source license; no commercial account or online service | Free, CPU/local. Best fit. Python 3.12 metadata mismatch is the main risk. |
| **Official demo server (`overcooked_demo`, Flask-SocketIO)** | Maintained with repository; 6-Hz default game loop, network/event-loop latency and action queueing | JSON state with terrain, serialized `OvercookedState`, score, time left; action event names `UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`, `SPACE`; trajectory data | Local use is low risk under MIT; deploying public service, bundled game assets, or collecting player data needs review | Free, but brings Flask/SocketIO/Docker/session complexity. Use only if an adaptor specifically needs WebSocket control. |
| **Official scripted/planning agents (`GreedyHumanModel`, MLAM, MotionPlanner)** | Mature benchmark baseline; planner creation cached to disk; each primitive call fast after initialization | Partner action plus action probabilities; full state required | MIT source; no service risk | Free. Recommended partner for standard three-onion layouts; not universally capable. |
| **CoGrid Overcooked implementation** | Separate current multi-agent framework; NumPy/JAX backends can be fast; API maturity/documentation differs | Gymnasium/PettingZoo-style observations, cardinal action space, layouts and extended variants | Its own license/dependencies; no commercial service risk | Free, but not the requested reference implementation and its 7-action semantics differ. Keep as fallback/validation only. |
| **DI-engine wrapper** | Training/evaluation wrapper around Overcooked-AI; fixed 400-step episodes; framework overhead | Six discrete actions, tensor features (e.g. `(5,4,26)` per agent), shaped/sparse reward option | Open-source framework terms; no commercial risk | Free but brings DI-engine and training dependencies; unnecessary for a symbolic Astra/Jev harness. |
| **OC2-Modding (BepInEx/.NET)** | Community PC Overcooked! 2 framework; version/DLL fragile; game-frame latency | Mod-dependent internals/QoL/randomizer hooks, not the Overcooked-AI JSON MDP | Requires owned commercial PC game; mod/online play may conflict with game publisher/platform policy and multiplayer ToS; never use online without review | Free mod, but requires game, .NET/BepInEx, and a display-capable game. Not usable on this no-account offline server. |
| **Steam/Proton executable + screen capture and input injection** | GUI/render loop; capture and input often tens of ms to frames; brittle focus/window/display setup | Pixels/OCR and key events only unless reverse-engineered; no guaranteed symbolic score/state | Commercial game automation can violate platform/game terms, especially online/competitive use; anti-cheat risk | Game/license, X/Wayland virtual display, capture/injection tools, substantial engineering. Explicitly reject here. |
| **Emulator hooks / memory inspection / remote-control protocol** | No official Overcooked-AI hook; high maintenance and version-specific; latency uncertain | Potentially pixels, memory, or controller events, but undocumented and unsafe across builds | Highest reverse-engineering, anti-cheat, EULA/ToS, and distribution risk | No practical benefit for the requested symbolic environment. Reject. |

Sources for alternatives: [CoGrid Overcooked docs](https://cogrid.readthedocs.io/en/latest/environments/overcooked/), [DI-engine docs](https://di-engine-docs.readthedocs.io/en/latest/13_envs/overcooked.html), [OC2-Modding](https://github.com/toasterparty/oc2-modding), [official demo source](https://github.com/HumanCompatibleAI/overcooked_ai/tree/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_demo).

## Installation for the next phase

### Required software

Use the evaluator interpreter exactly:

```text
/home/hemlix/GamingMaster-mh-tier2/runs/overcooked/.venv/bin/python
```

Required source and runtime packages:

```text
git
pip
numpy<2
scipy
gymnasium
dill
tqdm
setuptools>=77.0.3
opencv-python-headless  # replaces GUI opencv for headless use
pygame                  # import/runtime dependency; use SDL_VIDEODRIVER=dummy
ipywidgets              # package dependency; not needed by the tick loop
```

Install current upstream source while overriding its overly narrow Python metadata, then runtime packages. A safer adaptor phase should pin the inspected commit and record it:

```bash
VENV=/home/hemlix/GamingMaster-mh-tier2/runs/overcooked/.venv
$VENV/bin/pip install --ignore-requires-python \
  "git+https://github.com/HumanCompatibleAI/overcooked_ai.git@739950a079cdaed5a44fcc662efc40244c205d06"
$VENV/bin/pip install \
  dill gymnasium ipywidgets 'numpy<2' opencv-python-headless pygame scipy \
  'setuptools>=77.0.3' tqdm
```

Do not install the optional `harl` extra: it adds obsolete/heavy Ray/TensorFlow/W&B components and is not needed for a scripted partner or symbolic state. Do not run `pygame` display rendering; set `SDL_VIDEODRIVER=dummy` if an import path initializes it.

### Credentials and environment variables

No credentials, account, server, API key, or network endpoint are required for the chosen local MDP. The chosen interface's `env` list is therefore empty. Optional process settings are not credentials:

- `SDL_VIDEODRIVER=dummy`: set to avoid display requirements if pygame is imported.
- `PYTHONHASHSEED=<integer>`: optional reproducibility setting; it does not replace NumPy seeding.
- `OVERCOOKED_AI_COMMIT=<commit>`: optional adaptor-owned pin for logging only; do not expect upstream code to read it.

If the next phase chooses the demo server instead, its optional `CONF_PATH` points to a local JSON configuration; no account is needed, but the server is not selected here.

### Verification checklist

Run from a clean adaptor directory, not from an untrusted downloaded source directory:

1. Import `Action`, `OvercookedGridworld`, `OvercookedState`, and `OvercookedEnv` with the evaluator Python.
2. Construct `cramped_room`, reset, apply four STAY joint actions, and assert timestep reaches 4 and score remains 0.
3. Serialize `env.state.to_dict()` with standard JSON and reconstruct with `OvercookedState.from_dict`; assert equality.
4. Construct `GreedyHumanModel(env.mlam)`, call `action(state)` for both agent indices, and assert returned values are in `Action.ALL_ACTIONS`.
5. Run the upstream core tests only if their optional GUI/import dependencies are satisfied; the package's own source tests are authoritative for dynamics.
6. Test the adaptor scheduler with a fake slow model: the tick count must continue to advance and stale model replies must be dropped.

## Sources

- Official README/install/code structure: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/README.md
- Package metadata and Python constraint: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/pyproject.toml
- Environment API: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/mdp/overcooked_env.py
- MDP/state/rules/rewards: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/mdp/overcooked_mdp.py
- Actions: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/mdp/actions.py
- Agents: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/agents/agent.py
- Benchmark evaluator: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/agents/benchmarking.py
- Layout files: https://github.com/HumanCompatibleAI/overcooked_ai/tree/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_ai_py/data/layouts
- Original benchmark paper: https://arxiv.org/abs/1910.05789
- Official demo clock/server: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_demo/server/app.py
- Official demo game/action queue: https://github.com/HumanCompatibleAI/overcooked_ai/blob/739950a079cdaed5a44fcc662efc40244c205d06/src/overcooked_demo/server/game.py
