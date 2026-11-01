# Quaver interfaces and adaptor choice

## Recommendation for this run

**Choose a local `.qua` chart simulator with a JSON observation/action protocol.** It is the only option that works with all stated constraints (offline, headless Linux, no GPU/audio device, no account) and still tests timing, lanes, long-note state, judgement, combo, accuracy, score, and held-out chart variation. It uses the same open chart representation and the official Standard\* windows/weights, rather than pretending that a screenshot bot can meet an 18 ms window.

This is not a claim that a simulator is the same as the competitive client. It does not reproduce audio/UI/Steam/network latency, server validation, visual scroll, or every current modifier/timing-group edge case. It must label results `local_sim` and keep a fidelity version in its output. Use generated charts or user-supplied/openly licensed chart files where possible: community map music/assets can have separate copyright/licensing terms. Use the native client backend only when the user supplies a Steam installation/account and a display/input path.

## Chosen interface: local simulator contract

### Inputs and observations

The adaptor reads a native YAML `.qua` file, validates the fields below, and runs a virtual song clock. It should expose JSON (one object per observation) with at least:

```json
{
  "interface": "quaver-local-sim-v1",
  "clock_ms": 1932,
  "mode": "Keys4",
  "lanes": 4,
  "judgement_window": "Standard*",
  "visible_notes": [
    {"id": 12, "lane": 2, "start_ms": 2000, "end_ms": 0, "state": "pending"},
    {"id": 13, "lane": 4, "start_ms": 2020, "end_ms": 2600, "state": "pending"}
  ],
  "held_lanes": [],
  "recent": [{"id": 11, "part": "press", "judgement": "Marvelous", "error_ms": 1}],
  "judgements": {"Marvelous": 11, "Perfect": 1, "Great": 0, "Good": 0, "Okay": 0, "Miss": 0},
  "score": 12345,
  "accuracy_percent": 99.85,
  "combo": 12,
  "max_combo": 12,
  "health": 100.0,
  "terminal": false
}
```

`visible_notes` must be a fair visual lookahead, not the entire hidden chart. A sensible default is objects whose target start/end is between `clock_ms - 164` and `clock_ms + 2000` (or a configurable screen/lookahead duration selected by the scenario). Include target timestamps because the real game visually encodes time via scroll; do not require Astra/Jev to infer milliseconds from pixels in the simulator. Include only notes already visible to a hypothetical client; a separate offline chart-planning mode may expose the full chart to code for evaluation of scheduling.

The input action is a JSON array of timestamped events:

```json
[
  {"type":"press", "lane":2, "time_ms":1999},
  {"type":"press", "lane":4, "time_ms":2021},
  {"type":"release", "lane":4, "time_ms":2598}
]
```

Actions are lane events, not physical key names. A press chooses the earliest pending tap/LN in that lane; a release chooses the earliest LN currently held in that lane. Process equal-time events together, in stable input order, and reject or report duplicate impossible state changes. `step` may accept a batch and advance the virtual clock; `reset(seed)` must fully clear pending/held objects, score, health, combo, and histories. `finish` advances past all miss/release deadlines and returns the final result.

### Rules to implement

1. Parse `Mode`, `TimingPoints`, `HitObjects`, and optional `QuaVersion`; normalize the official case-sensitive fields. Require lane 1..4 or 1..7 and `EndTime == 0` for taps or `EndTime > StartTime` for LNs. Sort by start time/lane. `HitObjects`/`TimingPoints` and `StartTime`/`Lane`/`EndTime` are documented by the [official `Qua` model](https://github.com/Quaver/Quaver.API/blob/master/Quaver.API/Maps/Qua.cs) and [official API2 schema](https://pkg.go.dev/github.com/Quaver/api2/qua).
2. Use Standard\* press windows `18/43/76/106/127/164 ms` for Marvelous/Perfect/Great/Good/Okay/Miss. Classify by the smallest absolute-error threshold; an expired unhit object is a Miss (and an expired unhit LN contributes its required second Miss as well). The values and accuracy weights are in the [official gameplay wiki](https://quavergame.com/docs/gameplay#judgement-timing-windows) and [official processor](https://github.com/Quaver/Quaver.API/blob/master/Quaver.API/Maps/Processors/Scoring/ScoreProcessorKeys.cs#L80-L145).
3. Count a tap as one judgement and an LN as two. Score the press at `StartTime` and release at `EndTime`. Official release windows are multiplied by 1.5; an LN release too early is a miss/does not produce a normal release judgement, and leaving the key down past the release deadline is Good by default (`LNMissJudgement=Good`). Preserve an explicit scenario setting for custom `LNMissJudgement`, because the official client allows custom judgement windows. See [official release scorer](https://github.com/Quaver/Quaver.API/blob/master/Quaver.API/Maps/Processors/Scoring/ScoreProcessorKeys.cs#L179-L237) and [official missed-release manager](https://github.com/Quaver/Quaver/blob/develop/Quaver.Shared/Screens/Gameplay/Rulesets/Keys/HitObjects/HitObjectManagerKeys.cs#L663-L715).
4. Accuracy is the weighted mean: Marvelous 100, Perfect 98.25, Great 65, Good 25, Okay -100, Miss -50, with a nonnegative final clamp as in the official processor. Combo breaks on Miss in ranked/default mode; Good LN auto-release does not break it. Report all judgement counts and errors, not only a rounded grade.
5. Implement the official normalized score/multiplier if score is evaluated; otherwise report accuracy, miss count, full-combo flag, max combo, and score separately. Never report a perfect score just because the simulator accepted an action batch.
6. Require nondecreasing action timestamps relative to the virtual clock (or return a structured late-action error); process equal-time events as one batch. Start timing at `0` and use virtual time. No wall-clock sleep is needed in local evaluation. A real-time mode, if added, should use a monotonic clock and schedule events early enough to account for injection overhead; never ask a foundation model to sleep for individual notes.

### Software and setup

For the chosen backend, install only into the evaluator's venv:

```bash
/home/hemlix/GamingMaster-mh-tier2/runs/quaver/.venv/bin/python -m pip install 'PyYAML>=6.0,<7'
```

Python 3.12.3 is the required interpreter. The adaptor phase should:

1. Verify `sys.executable` is the supplied venv and import `yaml`.
2. Load only charts from `QUAVER_CHART_ROOT`; reject paths outside that root.
3. Parse/validate `.qua`; reject unsupported mines, timing groups, or chart versions explicitly (or implement them before enabling those seeds).
4. Choose a deterministic chart from `reset(seed)` and record seed, mode, difficulty, chart hash, ruleset version, and `local_sim` fidelity.
5. Return the JSON observations/actions above through the generated harness, with no network, Steam, display, audio, or credential dependency.
6. Add unit fixtures that test exact boundary errors (±18, ±43, ±76, ±106, ±127, ±164), simultaneous chords, LN start/release, early release, no release, and misses.

Environment variables/credentials for this backend:

- `QUAVER_CHART_ROOT`: directory containing allowed `.qua` charts (required; no secret).
- `QUAVER_RULESET`: ruleset identifier, normally `standard-star-v1` (required; the adaptor must reject unknown values).
- No account, server address, API key, Steam ticket, or network access is required.

### Variable/credential inventory for alternatives

These are names only; no credential value belongs in this study directory. They are not needed by the chosen simulator:

| fallback | variable or secret | when required |
|---|---|---|
| real Steam client | `QUAVER_STEAM_LIBRARY` (path), `DISPLAY`, optionally `XDG_RUNTIME_DIR`, plus a Steam account logged into the client | only for a local client launch; the account must not be used for automated online/ranked play |
| real-client key bridge | `QUAVER_LANE_KEYS` (non-secret lane-to-key mapping), `/dev/uinput` access or X11 permissions | only for offline input injection |
| API v2 | `QUAVER_API_BASE_URL` (default `https://api.quavergame.com/v2`), `QUAVER_JWT` in an `Authorization: Bearer ...` header, or `QUAVER_CLIENT_SECRET` in `X-Client-Secret` for protected downloads | only for authenticated API routes; obtain credentials through the official service, never invent/store values here |
| real-time Z protocol | `QUAVER_Z_SERVER` (server WebSocket address), plus the official client/account authentication material | only for multiplayer/spectator experiments; no live key-control API and not usable offline |
| source/tool build | `DOTNET_ROOT`/`PATH` pointing to a compatible .NET SDK and local `QUAVER_TOOLS_PATH` if the API server is being developed | only for optional C# tooling; not part of the Python adaptor |

## Other practical interfaces

### 1. Official open-source client + Linux screen capture and keyboard injection

- **What it exposes:** The real client renders the playfield and result UI; observation is a screenshot (or OCR/template recognition) plus any externally visible result screen. Controls are lane key-down/key-up events. The open-source client has the actual input manager and configurable key layouts; [the official input manager](https://github.com/Quaver/Quaver/blob/develop/Quaver.Shared/Screens/Gameplay/Rulesets/Input/InputManagerKeys.cs) shows per-lane press/release handling, and [the config](https://github.com/Quaver/Quaver/blob/develop/Quaver.Shared/Config/ConfigManager.cs#L876-L890) shows the defaults.
- **Tools:** Steam Quaver client (Steam app 980610), X11 `Xvfb` for a virtual display, `xdotool`/XTest or a `/dev/uinput` virtual keyboard for input, and `mss`/ImageMagick for capture. `ydotool`/uinput can work without X but needs the daemon/device permission; X11 tools need `DISPLAY` and a focused window. Software-rendered Xvfb/llvmpipe may be needed, but Quaver's graphics/audio path is not guaranteed headless.
- **Latency:** Capture/OCR and UI scheduling are frame-rate and OS dependent (roughly one to many frames, often tens of ms), which is marginal against ±18 ms. uinput/XTest is lower-latency than OCR but still not deterministic; the agent must pre-schedule from chart data if it can.
- **Maturity:** The game client is official and Linux-supported; the automation layer is a fragile integration with no stable bot API. Real-client state is much less exposed than simulator JSON.
- **Cost/requirements:** Quaver is free on Steam, but Steam must be installed/running and an account/client installation is needed. Xvfb and input permissions are extra Linux setup. The user explicitly has no account, GPU, display, or audio device, so this is not usable in this run.
- **Legal/ToS risk:** High for online/ranked play. The official [rules](https://quavergame.com/docs/rules) call using third-party tools, editing the client, or unintended exploits cheating, require the official Steam client when connecting to the server, and state cheating/client editing can cause permanent bans. Keep any screen/input experiment offline and do not submit scores.

### 2. Wine/Proton/emulator compatibility hooks

- **Applicability:** Quaver has an official native Linux/Steam target, so there is no separate mobile/console emulator with a documented game-state API. Running the Windows build through Wine/Proton can provide a compatibility layer, but it still presents a graphical client; Wine/X11 window capture and virtual keyboard injection are the same screen/input technique as above.
- **Maturity/latency:** Proton/Wine compatibility may make an otherwise unavailable build launch, but it adds another translation/scheduling layer and has no Quaver-specific observation or control interface. It is worse than the native client plus simulator for an 18 ms target and is not a headless guarantee.
- **Cost/legal:** Steam/Quaver, a compatible Wine/Proton prefix, X server/virtual display, and input permissions are required. Emulator/process hooks or DLL injection are high-risk third-party client modification under the [official rules](https://quavergame.com/docs/rules); use neither for online/ranked play. No emulator hook is selected.

### 3. Official `.qua`/`.qr` library and `Quaver.Tools` (offline parser/replay tooling)

- **What it exposes:** The official [Quaver.API](https://github.com/Quaver/Quaver.API) library parses/writes maps and replays, converts formats, and calculates difficulty. `Quaver.Tools` is a .NET CLI around that library. `.qr` replays expose recorded frames/hit statistics and can be written for playback; they are an analysis/replay artifact, not a live action API.
- **Latency:** Excellent for pre-run parsing and offline scoring; no live controls or live observation. A C# subprocess boundary would be unnecessary overhead for every note.
- **Maturity:** Official, source-available, used by the project; library targets `netstandard2.1`, while the tools/tests use .NET 6 in the repository. It is more faithful than reimplementing format semantics but does not remove the need for a Python harness protocol.
- **Cost:** Free/Mozilla Public License 2.0 code; install a compatible .NET SDK/runtime and package dependencies. It does not solve the no-account/headless client problem.
- **Legal/ToS:** Offline chart/replay analysis is low-risk; generating/injecting bot replays into online play is not. Source: [Quaver.API README](https://github.com/Quaver/Quaver.API#whats-included), [replay model](https://github.com/Quaver/Quaver.API/tree/master/Quaver.API/Replays), [official tools](https://github.com/Quaver/Quaver.API/tree/master/Quaver.Tools).

### 4. Official web API v2

- **What it exposes:** Read-only web data and downloads: map/mapset metadata, public leaderboards, users, scores/replays, multiplayer history, and map/chart downloads. The official source registers routes such as `GET /v2/map/:id`, `GET /v2/mapset/search`, `GET /v2/download/map/:id`, `GET /v2/download/replay/:id`, and score endpoints in [server route registration](https://github.com/Quaver/api-v2/blob/master/cmd/api/server.go#L120-L270). It does not send key presses, start a local song, or expose live gameplay state.
- **Auth/cost/latency:** Public metadata paths may be unauthenticated; map/replay/mapset downloads and personal data require in-game/JWT authentication or a client secret. Production rate limiting is 100 requests/minute per IP in the official server source, with a few authenticated bypass routes. The service is remote and network-dependent; charts could be pre-fetched but the user has no account/network requirement.
- **Maturity:** Official v2 source exists, but its README says it is developed for internal network use and provides no support for external use. The wiki index is [here](https://wiki.quavergame.com/docs/api-v2).
- **Legal/ToS:** Reading public data is ordinary API use; automated downloads must respect authentication/rate limits and site rules. It remains useless as a control channel and does not authorize bot play.

### 5. Official real-time multiplayer protocol (`Z` / `Quaver.Server.Client`)

- **What it exposes:** The official `Z` service is the packet-based real-time login, session, multiplayer, spectator, and chat backend. Its public source includes game/lobby state, player status, judgement-summary packets, and spectator replay-frame packets; it does not expose the local lane key-down/up stream or a supported bot action API. Source: [Z README](https://github.com/Quaver/Z#readme), [packet directory](https://github.com/Quaver/Z/tree/master/packets), and the [main client's submodule declaration](https://github.com/Quaver/Quaver/blob/develop/.gitmodules).
- **Controls/latency:** A protocol client can join a multiplayer room, report/receive judgements, send lobby actions, and consume spectator frames, but cannot make the local Quaver ruleset press a key. Network timing is unsuitable for the ±18 ms local hit path, and spoofing judgement packets is not legitimate play.
- **Maturity:** Official infrastructure source, but the README explicitly says it is for internal use with no support/proper documentation. It requires MariaDB/MySQL, Redis, Steam API/Publisher credentials to run a server and is not an end-user bot library.
- **Cost/legal:** Requires an account/server access for the real service, network connectivity, and server-side dependencies if self-hosted. Joining the official service with automation or falsified judgements risks the same permanent cheating penalty in the [official rules](https://quavergame.com/docs/rules). It is unusable under the no-account/offline constraint.

### 6. Quaver editor Lua plugin API

- **What it exposes:** Editor plugins can inspect/edit map objects, timing points, scroll velocities, selection, and song time; they can place/remove objects and use undo/redo. The community [plugin guide](https://github.com/IceDynamix/QuaverPluginGuide/blob/master/quaver_plugin_guide.md) documents a MoonSharp Lua sandbox and the `state`, `map`, and `actions` objects. This is an editor integration, not a gameplay observer/input API.
- **Latency/maturity:** Runs callbacks/rendering every editor frame; adequate for chart tooling, not a millisecond gameplay control path. Community guide and examples exist, but the API is not a stable ranked-play bot interface.
- **Cost/legal:** Free and local; no account needed to edit local charts. The sandbox restricts file/network/module access, and the plugin is not present in ranked gameplay. Editing the official client or using a plugin to automate online play would trigger the [official anti-cheating rules](https://quavergame.com/docs/rules).
- **Useful role:** Generate/inspect deterministic seed charts before the simulator, or make a visual chart-analysis tool; do not select it for the agent action loop.

### 7. Community DLL/in-process autoplay projects

- **What it exposes:** [lanylow/quaver-autoplay](https://github.com/lanylow/quaver-autoplay) is a small C++ Windows-only internal autoplayer. Its README says to configure Windows virtual-key codes, compile, and inject into `Quaver.exe`; it claims all key modes but has no supported Linux build or stable external state protocol.
- **Latency/maturity:** In-process timing can be precise, but it is an old/minimal proof of concept with manual injection, no release/API, and unknown compatibility. It is not a viable headless Linux adaptor.
- **Cost/legal:** Requires Windows/Quaver process injection and carries high detection/ban risk; the project itself warns against using it while connected to Quaver servers. Never install it for this run or use it online. Its offline-only warning does not override Quaver's rules.

### 8. Custom client fork or direct source integration

- **What it exposes:** Building the [official open-source client](https://github.com/Quaver/Quaver) gives direct access to `GameplayRuleset`, `InputManagerKeys`, score processor, replay capture, and rendered state. A custom test-only build could expose a JSON/pipe bridge with exact internal state and actions.
- **Latency/maturity:** Potentially best latency/fidelity, because the bridge sits inside the game loop; high engineering/maintenance cost and no existing supported bridge. The official README requires the .NET 8 SDK and says Steam must be running for development. It is not practical under no account/audio/display and is explicitly unsafe for server connections if modified.
- **Cost/legal:** Free source but requires .NET 8, submodules/assets, build time, and likely graphics/audio runtime. MPL code obligations and separate asset licensing apply. The [official rules](https://quavergame.com/docs/rules) prohibit edited clients for server login; restrict a fork to local offline testing.

## Real-client fallback setup (not chosen)

Only use this if the user later supplies an account and accepts offline-only testing:

1. Install Steam and Quaver (app 980610), plus a compatible Linux graphics/audio stack; do not attempt server login with a modified client.
2. Install `xvfb`, `xdotool`/XTest, and `mss`; set `DISPLAY=:99` and launch `Xvfb :99 -screen 0 1280x720x24` with software rendering. Verify the client can load a local `.qua` without audio/GPU.
3. Configure lane keybinds in-game and keep a lane-to-key map outside secrets. Focus the game window; inject only key-down/up events from a monotonic scheduler.
4. Capture screenshots at a known cadence, crop the playfield, and detect lane/receptor/note positions. Calibrate audio/visual offset against a known chart; report the measured offset and timing jitter.
5. Run offline only and compare result-screen counts against the local simulator. Stop if capture/injection jitter exceeds 18 ms; do not use an online account or submit a result.

The simulator remains the chosen interface because it is deterministic, installable in the supplied Python 3.12.3 venv, and exposes exact JSON state without credentials.
