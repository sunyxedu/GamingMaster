# Programmatic interfaces for Flappy Bird

## Decision and recommendation

**Chosen interface: the maintained `flappy-bird-gymnasium` package, feature-vector mode, with no renderer.** It is a local Gymnasium environment with a synchronous `reset(seed)` / `step(action)` interface, deterministic seeded pipe generation, direct symbolic observations, and no display, account, server, or credentials. It is the only option below that directly matches the offline headless constraint and lets Jev answer from a compact JSON state instead of inferring pixels. It also avoids the latency, fragility, and legal ambiguity of driving a third-party web/mobile copy.

Package source and license: [markub3327/flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium), [PyPI metadata](https://pypi.org/pypi/flappy-bird-gymnasium/json), [MIT license](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/LICENSE). The repository describes itself as based on the older [Talendar/flappy-bird-gym](https://github.com/Talendar/flappy-bird-gym) and [FlapPyBird](https://github.com/sourabhv/FlapPyBird); it is a faithful clone/environment, not an official API for the withdrawn commercial app.

## Option A — Gymnasium symbolic environment (chosen)

### What to install

Use the evaluator interpreter, not system Python:

```bash
/home/hemlix/GamingMaster-mh-tier2/runs/flappy-bird/.venv/bin/python -m pip install flappy-bird-gymnasium
```

PyPI 0.4.0 declares Python `>=3.9` and dependencies `gymnasium`, `numpy`, `pygame`, and `matplotlib`; Python 3.12.3 is within the declared range. The package's own setup metadata is visible in [`setup.py`](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/setup.py). For reproducible evaluation, pin the resolved package versions in the adaptor environment after testing, rather than silently using a newer release.

### Exact setup and adaptor contract

1. Install the package into the evaluator venv.
2. Import `flappy_bird_gymnasium` before calling `gymnasium.make`, so the package registers `FlappyBird-v0`.
3. Create `env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False, normalize_obs=False, audio_on=False)`. `render_mode=None` avoids display initialization; feature mode gives symbolic values and is easier for a language-model harness than 180 rays.
4. Call `obs, info = env.reset(seed=int(seed))`. Do not use the old Gym 4-tuple API.
5. Convert the NumPy array into a JSON object before sending it to Jev. Preserve the field order below and include the previous action, frame number, cumulative score, reward, `terminated`, and `truncated` in the wrapper's JSON (those wrapper fields are not supplied by the package and must be bookkeeping).
6. Ask Jev for a typed binary choice (`FLAP` or `IDLE`) from the current JSON. Validate the answer locally and map it to 1 or 0; never allow model text to become an arbitrary action.
7. Call `obs, reward, terminated, truncated, info = env.step(action)`. End the episode on either flag, call `env.close()`, and report `info["score"]`.
8. During Astra-driven calibration, use short batches of steps and inspect score/termination; during live control, do not block the environment waiting for Astra.

The official package usage example is in the [README](https://github.com/markub3327/flappy-bird-gymnasium#usage), and the return contract is defined by [Gymnasium's Env API](https://gymnasium.farama.org/api/env/).

### Observation schema

With `use_lidar=False, normalize_obs=False`, the 12-vector is:

```text
[
  last_pipe_x, last_pipe_top_y, last_pipe_bottom_y,
  next_pipe_x, next_pipe_top_y, next_pipe_bottom_y,
  next_next_pipe_x, next_next_pipe_top_y, next_next_pipe_bottom_y,
  bird_y, bird_vertical_velocity, bird_rotation
]
```

The source calls the first three pipe records “last”, “next”, and “next next”; they are the three pipe pairs sorted by x, with off-screen-ahead pipes represented using screen bounds. The top/bottom values are the lower edge of the upper pipe and upper edge of the lower pipe, i.e. the open gap boundaries. With `normalize_obs=True`, divide x by screen width, y by screen height, velocity by `PLAYER_MAX_VEL_Y` (10), and rotation by 90. Prefer unnormalized values in the wrapper because they make exact geometric diagnostics and logs easier; normalize only if the prompt-size budget requires it.

The source of truth is [`_get_observation_features`](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/flappy_bird_env.py#L470-L526). The action is `0 = do nothing`, `1 = flap`; the `Actions` enum and action space are in [`flappy_bird_env.py`](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/flappy_bird_env.py).

### Timing and latency

A `step` advances exactly one simulation frame. The package advertises 30 render FPS, and pipes move -4 px/frame in its default 288x512 world. Rendering is not required for stepping and omitting it makes simulation faster than real time. A Jev call should be budgeted at roughly 0.5 s in the supplied architecture, which is too slow for a strict one-call-per-frame live loop; the adaptor should either (a) run Jev in a low-latency local/parallel call and keep a previously selected action between calls, or (b) ask Jev for a short typed action schedule/hold duration and execute it while polling safety state. The safer generated-harness design is one Jev decision every 1–3 frames near a pipe, with a local code-level emergency flap rule if the model call misses its deadline.

The environment itself is deterministic after `reset(seed)`, but it does not wait for wall-clock model computation. This is why the game remains tier 2 even though its API is synchronous.

### Cost, maturity, and risk

- **Maturity:** 0.4.0, MIT, 116 GitHub stars and active repository metadata as of the study date; it has a conventional Gymnasium API and tests, but it is a community clone rather than a Farama-maintained official environment.
- **Latency:** local function call / NumPy/Pygame simulation; no network latency and no rendering required. A single frame is cheap enough to batch, but model inference remains the likely bottleneck.
- **Exposure:** exact numerical state, reward, score, termination, optional RGB array, optional LIDAR. The feature vector is the preferred form.
- **Legal/ToS:** package MIT license covers the code. The Flappy Bird name and original artwork are not thereby relicensed; retain the package's attribution and avoid distributing proprietary original assets. For this offline evaluator there is no third-party account or service ToS.
- **Cost:** free/open source; only local pip and package dependencies.

## Option B — legacy `flappy-bird-gym`

Install with `pip install flappy-bird-gym`. Its README documents `FlappyBird-v0` (two numerical values: horizontal distance to next pipe and vertical difference to its hole) and `FlappyBird-rgb-v0` (RGB arrays), with old Gym `reset()` and 4-value `step()` conventions. Source: [README](https://github.com/Talendar/flappy-bird-gym#readme), [setup.py](https://github.com/Talendar/flappy-bird-gym/blob/main/setup.py).

- **Maturity:** influential and MIT-licensed, but its pinned requirements (`gym ~= 0.18.0`, `numpy ~= 1.19.5`, `pygame ~= 2.0.1`) are old and awkward on Python 3.12.
- **Latency/exposure:** local symbolic or pixel observations; no network. The two-feature state is compact but omits vertical velocity, so it is less useful for a reactive controller than the chosen 12-vector.
- **Legal/cost:** same clone/asset caveat; free. No credentials.
- **Verdict:** useful fallback if the newer package cannot install, but do not choose it first on this evaluator because of old dependencies and API compatibility.

## Option C — clone the game source and add a symbolic wrapper

[FlapPyBird](https://github.com/sourabhv/FlapPyBird) is a well-known Python/Pygame MIT-licensed clone. It supports Python 3, `make init`, Space/Up controls, a `DEBUG=True` mode, and an optional browser build; it has no stable external agent API. A newer small [headless-testing clone](https://github.com/br8nfroze/flappy-bird-clone) advertises randomized pipes, collision/scoring tests, and `python -m unittest discover -s tests -v`, but has no license in its repository metadata and is a one-commit learning project.

- **Maturity:** FlapPyBird is mature as example code, but its loop is designed around Pygame events/rendering; the small clone is immature despite its tests.
- **Latency/exposure:** after forking, direct Python state can be fastest and most complete, but the adaptor must maintain a patch and expose/reset/seed the internals itself.
- **Legal/cost:** FlapPyBird code is MIT; asset/name rights remain a separate concern. The small clone has no visible license, so do not copy it into a deliverable without permission.
- **Verdict:** sensible fallback when exact code modification is acceptable; worse than Gymnasium because it adds maintenance and does not provide a standard seeded API out of the box.

## Option D — MPC / optimization controller

[FlapPyBird-MPC](https://github.com/philzook58/FlapPyBird-MPC) is an MIT-licensed research fork of FlapPyBird. It models ballistic motion and discrete flap booleans as a mixed-integer linear program, replans in a receding horizon, and uses CVXPY/Gurobi. The authors explicitly say the long-lookahead configuration does not run in real time and uses a coarser future timestep trick; see its [README](https://github.com/philzook58/FlapPyBird-MPC#readme).

- **Maturity:** a useful reference controller, not a maintained general environment.
- **Latency:** solver calls and Gurobi setup are incompatible with a ~0.5 s Jev control budget unless heavily simplified; Gurobi may require a license.
- **Exposure:** direct game state and predicted trajectories, if integrated.
- **Legal/cost:** code MIT; CVXPY free, Gurobi licensing/cost may apply. No account is needed for the chosen environment, but this option may need a solver license/environment variable.
- **Verdict:** put the model-predictive idea in Astra/code strategy notes, not in the baseline adaptor.

## Option E — browser copy with Playwright

For a browser-hosted copy such as [flappybird.io](https://flappybird.io/how-to-play), Playwright can launch Chromium, locate the game canvas, take screenshots, and inject `page.keyboard.press("Space")` or `page.keyboard.down/up`. Official APIs: [Keyboard](https://playwright.dev/python/docs/api/class-keyboard), [Page](https://playwright.dev/python/docs/api/class-page).

- **Maturity:** Playwright is mature; the particular website/game is outside the adaptor's control and can change.
- **Latency/exposure:** keyboard events are low overhead, but screenshot/canvas inference and browser scheduling add variable latency. Unless the page exposes a debug object (not assumed), observations are pixels and need CV; browser canvas state is not a public API.
- **Legal/ToS:** automated play, scraping, and leaderboard submission may violate a site's rules; do not submit scores or bypass anti-cheat. The site's clone artwork/name has its own rights. Network access contradicts the offline constraint.
- **Cost/credentials:** Playwright Python plus a browser binary; no account for a casual copy, but a site could require one. No credentials should be put in study files.
- **Verdict:** not viable for this run: network/display dependence, fragile pixel coordinates, and possible ToS issues.

## Option F — screen capture + desktop input injection

A generic bot can capture a visible game with [MSS](https://python-mss.readthedocs.io/stable/usage.html) or [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/), use OpenCV/Pillow to detect bird and pipes, and inject Space/click. MSS supports Linux X11 backends and uses `DISPLAY`; PyAutoGUI sends keyboard events to the focused window and on Linux requires X11/Python-Xlib plus screenshot tooling such as `scrot` (see its [installation docs](https://pyautogui.readthedocs.io/en/latest/install.html)).

- **Maturity:** the libraries are mature; each game's CV detector and window geometry are bespoke.
- **Latency/exposure:** MSS's XShm backend is fast, but capture, image processing, focus, and injected input create more jitter than symbolic stepping. It exposes screenshots only; bird/pipe coordinates must be inferred and calibrated.
- **Legal/ToS:** automation may violate a hosted game's ToS or anti-cheat rules; screen/input automation is appropriate only for a local copy or authorized test. Do not use it to evade anti-cheat.
- **Cost:** open source, but Linux packages (`scrot`, X11 libraries, possibly OpenCV) and an X display are required. No account for a local copy.
- **Headless caveat:** a virtual display can be provided by [Xvfb](https://xorg.freedesktop.org/archive/X11R7.5/doc/man/man1/Xvfb.1.html), an X server for machines with no display hardware/input devices, but this is unnecessary for the chosen symbolic environment and still requires launching the game in that display.
- **Verdict:** last-resort compatibility interface only.

## Option G — Android emulator / ADB

If an authorized Android APK is available, [Android Debug Bridge](https://developer.android.com/tools/adb) can connect to an emulator/device, stream screenshots with `adb exec-out screencap -p`, and send shell input/touch/key events. `ANDROID_SERIAL` selects a device. The app itself remains pixel-only unless instrumented.

- **Maturity:** ADB is mature and robust for testing; the original Flappy Bird APK is not a supported downloadable dependency and obtaining/distributing it may be legally problematic.
- **Latency/exposure:** screenshot subprocess and device transport add substantial jitter; state is pixels, not JSON.
- **Legal/ToS/cost:** requires an APK, Android SDK/emulator and substantial disk/CPU; app licensing and automation/leaderboard rules apply. No account is assumed, but do not hard-code device credentials or serials.
- **Verdict:** inappropriate for offline headless evaluation unless the user supplies an authorized APK and emulator image.

## Environment variables and credentials

For the chosen Gymnasium interface, **no credentials and no environment variables are required**. Do not invent or write values. The only setup is local pip installation and the adaptor's Python import.

If a future adaptor deliberately selects an alternate interface, it must request rather than hard-code:

| Variable | Meaning | Required for chosen interface? |
|---|---|---|
| `DISPLAY` | X11 display for GUI/MSS/PyAutoGUI/Xvfb | No |
| `MSS_DISPLAY` | wrapper-specific display override, e.g. `:99` | No |
| `ANDROID_SERIAL` | ADB device/emulator serial | No |
| `PLAYWRIGHT_BROWSERS_PATH` | optional browser installation location | No |
| `GAME_URL` | authorized browser game URL | No |
| `GAME_WINDOW` | wrapper-specific capture rectangle | No |

Never place API keys, passwords, account cookies, APK signing keys, or real device credentials in this directory.

## Operational failure checks

The adaptor phase should verify, using the evaluator Python:

```bash
/home/hemlix/GamingMaster-mh-tier2/runs/flappy-bird/.venv/bin/python -m pip install flappy-bird-gymnasium
/home/hemlix/GamingMaster-mh-tier2/runs/flappy-bird/.venv/bin/python -c 'import flappy_bird_gymnasium, gymnasium; e=gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False, normalize_obs=False); o,i=e.reset(seed=0); print(o.shape, e.action_space, i); print(e.step(0)[0].shape); e.close()'
```

If that fails because a transitive dependency has no Python 3.12 wheel, the adaptor should first try the package's current compatible dependency versions. Only then fall back to a local symbolic implementation based on the MIT source; do not silently switch to an online GUI or old Gym package.
