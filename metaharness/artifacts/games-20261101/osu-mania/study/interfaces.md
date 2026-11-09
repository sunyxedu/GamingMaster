# osu!mania programmatic interfaces

## Recommendation

Use a **local open-chart simulator** driven by native mania `.osu` files, implemented with the evaluator’s Python 3.12.3 standard library. It is the only option here that is headless, offline, deterministic, account-free, and naturally produces structured observations and timestamped actions. It also lets the benchmark vary key counts, OD, density, pattern families, and hold placement by seed. The simulator should implement the stable native ScoreV1 profile specified in [`brief.md`](brief.md), not silently claim to be current lazer.

The real osu! client is useful as a reference or offline spot-check, but it is a poor adaptor target on this server: there is no display, GPU, audio device, account, or documented headless control API. Live gameplay automation is also prohibited by the [osu! rules](https://osu.ppy.sh/wiki/en/Rules), so screen/input or bot routes must not be used against the online service.

## Interface comparison

| Route | Maturity | Latency and observation | Control | Legal/ToS risk | Cost and credentials |
|---|---|---|---|---|---|
| **Local `.osu` parser + simulator (chosen)** | The file format and stable formulas are official and well documented; the simulator itself must be tested by the adaptor phase | Deterministic local JSON; event processing can be sub-millisecond and does not depend on wall-clock scheduling | Timestamped lane down/up events; reset/advance; no OS input | Low when used only with local/generated charts; no online score or account | Free; evaluator Python only; no credentials |
| **osu! API v2** | Mature official web API | Network request latency and rate limits; metadata, beatmap, score/profile/leaderboard data, not the live playfield | No gameplay input, pause, or real-time state endpoint | Official API use is lower risk if terms/rate limits are followed, but it cannot play | Free registration/OAuth; client ID/secret or token for protected routes; see [API docs](https://osu.ppy.sh/docs/index.html) |
| **Official `.osu`/`.osr` file and replay pipeline** | Stable formats; replay tooling is widely used | Local file parsing is fast; `.osr` is post-hoc replay data, not a live observation | Can create or inspect replay frames, but a replay does not control a running client by itself | Low for offline analysis; submitting automated play online conflicts with fair play | Free; no credential for local files. `osrparse` and [osu-parsers](https://github.com/kionell/osu-parsers) are optional libraries |
| **Built-in Auto/replay playback** | Useful official-client validation path | Produces a rendered play/result, but not an agent-visible structured state | Auto/replay drives the client itself; it offers no decision-action hook for Astra/Jev | Do not submit automated results online; safe only as an offline chart/simulator spot-check | osu! client and local chart/replay; still needs a display-capable client |
| **Embed osu!lazer ruleset/source or its tests** | Very high fidelity to current open-source lazer code, but APIs are internal and change with master | Local deterministic ruleset calls can be fast once embedded; building the full client is slow/heavy | Possible only after writing a custom integration around ruleset objects/events; no documented public gameplay-agent API | Code/framework is MIT, but branding/resources have separate terms; live automation remains prohibited | .NET SDK and a large source checkout; no account for tests. [Repository](https://github.com/ppy/osu), [lazer gameplay differences](https://osu.ppy.sh/wiki/en/Client/Release_stream/Lazer/Gameplay_differences_in_osu!%28lazer%29) |
| **Real client + screen capture + key injection** | Mature desktop automation primitives, but brittle for a rhythm game | Screenshot/vision and OS input add frame, capture, compositor, and injection jitter—bad against 16 ms MAX windows | X11/Wayland virtual keyboard or `uinput`; physical lane bindings | Highest anti-cheat/ToS risk; osu! rules prohibit software that plays for a user. Use only for an explicitly isolated local experiment, never online | Client, virtual display, capture, input-injection tools, and likely audio/graphics dependencies; no account is acceptable only if the client supports offline local play |
| **Tosu/gosumemory-style memory telemetry** | Tosu is actively maintained; memory offsets are version-sensitive | Local HTTP/WebSocket telemetry can be low-latency and structured, but only after a running supported client | Primarily read-only; does not provide safe authoritative input control | Process-memory readers are a gray/high-risk choice around anti-cheat and online play; offline only | Tosu is [LGPL-3.0](https://github.com/tosuapp/tosu); platform/client compatibility must be checked; no API credential |
| **Community parsers/judges** | Varies: `osu-parsers` is active MIT TypeScript; `mania-judge` is WIP; `mania-rl-bot` is useful reference but has no reuse-friendly license | File/local simulator latency is good; JSON fields depend on the package | Usually produces chart/replay data, not OS input | Offline use is low risk; bot operation against online osu! is prohibited | Node/npm for TypeScript packages, or Python dependencies for a reference simulator; no account. See [mania-judge](https://github.com/zzzzv/mania-judge), [osu-parsers](https://github.com/kionell/osu-parsers), [mania-rl-bot](https://github.com/YameenMunir/mania-rl-bot) |
| **Desktop/mobile emulator + ADB/touch injection** | General emulator tooling is mature, but there is no official osu!mania agent environment | Screenshot and touch timing are less predictable than direct lane events; GPU/display/audio burden | ADB key/touch events, if the client and emulator map them | Same automation/ToS risk as a bot; not suitable for this headless server | Emulator, virtual GPU/display, client image; no benefit over the simulator |

### Important distinctions

* The official API is a data service, not a game server API. It cannot stream the current judgement line or send key presses. Do not put OAuth values in this study directory.
* A replay parser observes historical input frames after a play. It is useful for validation, but it is not an agent control interface. Built-in Auto/replay playback has the same limitation.
* Tosu/gosumemory can observe a client but cannot solve the missing input path, and offsets can break after a client update. The original [gosumemory](https://github.com/l3lackShark/gosumemory) and [Tosu](https://github.com/tosuapp/tosu) expose telemetry rather than a supported action API.
* `osu!lazer` source is an excellent reference for a second ruleset, but the README documents `dotnet run --project osu.Desktop`, not a supported headless agent protocol. The source is MIT for code/framework; resources and trademarks are separate.
* Community bots such as [osu-pilot](https://github.com/Wakype/osu-pilot) demonstrate direct `.osu` parsing, but their purpose is automated play. They are not an appropriate online integration or dependency for this benchmark.

## Selected adaptor contract

This is the contract the next phase can implement without a GUI. It is intentionally an event-driven simulator rather than a real-time sleep loop.

### `reset(seed)`

1. Derive all scenario choices from a stable hash of the supplied seed, not Python’s version-dependent random stream.
2. Select or generate one valid native mania `.osu` chart. The chart must declare `Mode: 3`, an integer key count, `CircleSize`, `OverallDifficulty`, one or more timing points, and valid tap/hold objects.
3. Start the song clock at 0 ms, clear key state, score, combo, health placeholder, and event queue.
4. Return an initial JSON state with `seed`, `ruleset: "stable-scorev1-native-mania"`, `key_count`, `od`, `current_time_ms`, `lookahead_ms`, chart duration, and only the notes in the visible/lookahead horizon. Do not reveal the entire held-out chart unless the benchmark explicitly wants an offline planning task.

### `observe()`

Return JSON, not a screenshot. At minimum include:

```json
{
  "time_ms": 1200.0,
  "key_count": 4,
  "od": 7.0,
  "lookahead_ms": 2000,
  "lanes": [
    {"lane": 0, "key_down": false,
     "upcoming": [{"id": 17, "start_ms": 1325, "end_ms": null, "kind": "tap"}]}
  ],
  "active_holds": [],
  "recent_results": [],
  "score": 0,
  "accuracy": 1.0,
  "combo": 0,
  "max_combo": 0,
  "remaining_objects": 50
}
```

Each upcoming object should expose a stable ID, lane, start/end times, kind, and optionally pattern labels generated from the visible chart. `recent_results` should include result, signed head/tail error, and object ID. The state should expose judgement windows and score counts either once in metadata or on every observation. Do not ask Jev to recompute them.

### `act(events)` and clock advancement

Accept a batch of future lane transitions:

```json
{"events": [
  {"time_ms": 1325.0, "lane": 0, "down": true},
  {"time_ms": 1350.0, "lane": 0, "down": false}
]}
```

`act` should validate lane bounds, monotonic event timestamps, and one logical key state per lane. It may accept future events up to a configurable planning horizon; events submitted after their judgement deadline are late and must be scored as such, not silently moved. `advance(to_time_ms)` processes transitions and object deadlines in timestamp order and returns the new JSON state. No `sleep`, display, audio, or OS key code is needed.

At a tap head, a down transition in the lane is the candidate hit. At a hold, the lane must be down near the head and remain down until a release near the tail; a premature release records a hold break and caps the stable result at MEH. Simultaneous objects require simultaneous transitions in their respective lanes. The low-level adaptor may translate a model’s selected target time into events, but it must not invent perfect events for objects the model did not select.

### Software and environment

Required:

* `/home/hemlix/GamingMaster-mh-tier2/runs/osu-mania/.venv/bin/python` (Python 3.12.3), using only the standard library for the selected path.
* A native `.osu` chart corpus is optional; generated charts avoid accounts and downloads.

Optional environment variables, with no credential values:

* `OSU_MANIA_CHART_ROOT`: directory of local `.osu` files to use instead of generated charts; omit for seed-generated scenarios.
* `OSU_MANIA_RULESET`: explicit profile name, default `stable-scorev1-native-mania`; reject unknown profiles rather than silently using lazer rules.
* `OSU_MANIA_LOOKAHEAD_MS`: visible planning horizon, default 2000 ms; keep this fixed within an evaluation split.

No server address, API key, OAuth token, osu! account, display, GPU, or audio device is required. If a future adaptor deliberately chooses the official API instead, keep `OSU_CLIENT_ID`, `OSU_CLIENT_SECRET`, and `OSU_ACCESS_TOKEN` in its process environment only; never write values here.

## Validation checklist

* Compare parser lane assignment against `floor(x*K/512)` and clamp it.
* Test each tap at exactly 0, just inside, exactly on, and just outside every stable window.
* Test early press (no effect), late MEH, misses, chords, repeated same-lane jacks, and a hold with early release.
* Cross-check ScoreV1 counts and score against the formula in the official [score page](https://osu.ppy.sh/wiki/en/Gameplay/Score/ScoreV1/osu!mania); keep MAX/300 accuracy weights distinct from their 320/300 score values.
* Keep a separate test profile if lazer compatibility is later added; do not combine stable one-judgement holds with lazer two-judgement holds.
