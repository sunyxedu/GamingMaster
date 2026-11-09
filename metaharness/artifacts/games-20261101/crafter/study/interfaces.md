# Crafter interfaces

## Recommended: official Python environment (API / official env)

Install the upstream package in the evaluator interpreter: `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/crafter/bin/python -m pip install crafter` (pin the version that is available to the offline wheel cache; the current PyPI line is commonly `crafter==1.8.x`). The source of truth is [github.com/danijar/crafter](https://github.com/danijar/crafter), with API details in [`crafter/env.py`](https://github.com/danijar/crafter/blob/main/crafter/env.py) and [Read the Docs](https://crafter.readthedocs.io/en/latest/).

Typical adaptor shape:

```python
import crafter
env = crafter.Env()                 # optionally configure length/area/size
obs = env.reset(seed=seed)
obs, reward, terminated, truncated, info = env.step(action_id)
```

Use `env.observation_space` and `env.action_space`, not assumptions from a different Gym version. Exposed semantic fields are the inventory and achievements mappings (plus reward/discount); image is RGB. For the requested textual map, add a serializer that reads the local world/player state from the installed release (`env._world`, `env._player`, and nearby object/entity arrays are the usual private attributes) and emits a radius grid with coordinates, block/entity names, facing, health, food, drink, energy, inventory and achievements. Treat those attributes as a version-checked adapter and fail loudly if absent. The public fallback is to pass the RGB image through a local tile classifier and expose the resulting grid as text; no GUI is needed. Keep a raw image option for Astra debugging.

Control is one integer action per call: movement, `do`, `sleep`, and four placement actions. Calling `step` is synchronous and has negligible API latency. There are no credentials, server addresses, API keys, or ToS restrictions for local benchmark use. Cost is free software; package dependencies (NumPy, Pillow, Pygame/Gym or Gymnasium according to release) must be present in the offline wheel cache. On headless Linux set `SDL_VIDEODRIVER=dummy` if Pygame attempts to initialise a display.

Setup checklist:

1. Install `crafter` and its pinned dependencies into the evaluator venv, with wheels available offline.
2. Run a smoke test that prints `env.action_space`, observation keys/shapes, and one seeded reset.
3. Discover the installed action-name mapping and semantic/private world fields; record the package version in adaptor logs.
4. Implement `reset(seed)` and a JSON serializer containing local map, player vitals, inventory and achievements.
5. Implement `step(action)` with tuple-shape compatibility for old/new Gym APIs and close the environment in `finally`.
6. Run headless (`SDL_VIDEODRIVER=dummy`) and retain optional `crafter.Recorder` output for diagnosis.

This beats every alternative because it is the benchmark's own deterministic simulator, exposes exact actions and achievement accounting, runs offline without a GUI or account, and advances only when the adaptor calls `step`.

## Other practical routes

| Route | Maturity / latency / exposure | Cost and risk |
|---|---|---|
| Direct private-engine inspection (`env._world`, player/entity objects) | Mature in the upstream implementation but version-sensitive; zero-copy and low latency; gives exact tiles, entities, vitals and coordinates. | Free and local. Private API can break on upgrade; pin and smoke-test. No legal/ToS issue for this open-source benchmark. |
| RGB observation plus local semantic decoder | Public API and release-stable; one-step latency; gives only pixels, so inventory/achievement fields should still be taken from the observation. | Free, headless. Decoder engineering error and ambiguity around hidden tiles; no external service. |
| `crafter.Recorder` / JSON episode logs | Mature wrapper for recording frames and metrics, not a controller; writes after each step and adds I/O latency. | Free; disk usage only. Useful for replay/evaluation, not as the primary action interface. |
| Screen capture plus keyboard/mouse injection (SDL/Pygame, Xvfb) | Possible only by launching a human renderer; high latency and brittle focus/display dependencies; pixels but no semantic state. | Free but poor fit for a server without GUI; no benefit over the API. |
| Generic Gym/Gymnasium wrappers or RL bot libraries | Wrappers can normalize reset/step and vectorize environments, but do not expose extra game semantics; latency is low. | Free; dependency/version mismatch risk. They are optional around the official env, not a separate game interface. |

No network endpoint, mod/plugin protocol, emulator hook, or account credential is required. Do not put credential values in configuration; for this local route there are no required environment variables. `SDL_VIDEODRIVER` is an optional runtime variable, not a secret.

