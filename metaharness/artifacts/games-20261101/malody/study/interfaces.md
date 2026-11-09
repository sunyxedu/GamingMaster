# Malody interfaces and adaptor plan

## Recommendation in one line

Use a **local deterministic Key-mode emulator** driven by native `.mc` JSON and timestamped actions. It is the only option that fits the supplied offline/headless/no-account/no-display/no-GPU/no-audio constraints and still lets the harness vary charts by seed. Treat the emulator's numerical score as an RM-compatible proxy unless the real Key client is later used to calibrate it.

## Interface inventory

| Route | What it exposes / controls | Maturity and latency | Cost and legal/ToS risk | Verdict |
|---|---|---|---|---|
| **Local `.mc` parser + Key simulator (chosen)** | Structured chart JSON becomes beats, BPM segments, note/hold events, judgments, score, combo and accuracy. Adaptor sends timestamped `press`/`release` actions or a batch. | No network or render loop; deterministic and effectively sub-millisecond in Python. Chart format and parser behavior are supported by multiple open-source references, but full Key scoring is not officially specified publicly. | No account, service, or paid software. The next phase writes its own simulator; do not copy code without checking each dependency's license. Fidelity risk is explicit: RM table is a proxy for Key. | **Choose.** Works on this server and is reproducible for held-out seeds. |
| **Official Malody V Steam/desktop client + Wine/Proton** | Real gameplay, chart loader, skin, audio and result screen. Automation would inject keyboard events and observe screenshots/OCR or a plugin. | Steam client is mature; input timing can be 1--20+ ms with a real event device but Wine/Xvfb/render/audio scheduling adds jitter. Only documented desktop requirement is Windows 7 SP1+ 64-bit, 2 GB RAM, DX10, 500 MB; the listing also says a Mugzone third-party account is required ([Steam](https://store.steampowered.com/app/1512940/Malody_V/)). | Paid Steam product/account and likely network/login; violates the no-account/offline constraints. Running a Windows binary under Wine is unvalidated and may fail without GPU/audio. Game automation may violate service rules; do not use on online rankings. | Useful only for later calibration on an authorized local account. Not the evaluator interface. |
| **Screen capture + keyboard injection** (`Xvfb`, `xdotool`, `ydotool`, `uinput`, evdev, OCR) | Screenshot pixels, optionally audio waveform; sends key-down/up events. It works with whatever client can run in a virtual display but has no semantic chart state. | Capture/OCR is tens to hundreds of ms and has frame/clock jitter; direct uinput is faster, but timing still depends on the client. | Open-source OS tools are free; root/uinput permissions and X/Wayland setup are hazards. Same client account/ToS risk as above, and visual automation can look like botting. | A fallback only; it cannot meet 25--40 ms accuracy reliably or expose deterministic state. |
| **Android/iOS client in an emulator** (ADB/touch or virtual keyboard) | Real client and rendered play; ADB can inject key/touch events and take screenshots. | Emulator/display startup and input transport add large, variable latency; audio/GPU/device emulation are unavailable or expensive on this server. | Client/account/network constraints remain; emulator license and game automation/ToS must be checked. | Not practical for this headless Linux evaluator. |
| **Malody V overlay/plugin/reverse-engineering hook** | Potentially internal chart, timing, input and result objects rather than pixels. The [MalodyV Rework Overlay](https://github.com/MCPsDE/malodyv-rework-overlay) is an unofficial BepInEx-style overlay project; the [hit-window study](https://github.com/Hxjjxg/malody-hitwindow) reverse-engineers desktop/mobile judge functions. | Potentially low-latency internal events, but version/ABI dependent and not a stable public API. The hit-window project targets Windows x86 `malody.exe` and Android arm64 `libmalodycpp.so`, not a Linux client. | DLL injection/reverse engineering may breach client terms, trigger anti-cheat, or break after updates. No Linux/headless path is documented. | Valuable for offline rule calibration by an authorized researcher; not the production adaptor. |
| **Replay inspection/rendering** ([Malody V replay renderer](https://github.com/jk21lcl/malody-v-replay-renderer)) | Reads `.mrv` replay plus `.mcz/.osz` chart package, pairs notes/inputs/judgments, and exports a local MP4; has a browser UI and `--no-browser`. | Good offline forensic/visualization utility; it is post-hoc, not a live action API. Its documented limitations include No Mod/Const/Flip only and special long-note pairing behavior. | MIT project; no account needed for local files. FFmpeg/browser are extra dependencies and video generation is slower than real time. | Use as an optional validation tool, not as control. |
| **Chart conversion/parsing libraries** ([`malody2osu`](https://github.com/Jakads/malody2osu), [`malody-convert`](https://github.com/kangalio/malody-convert), [rconv format docs](https://prefixaut.github.io/rconv/rconv/malody.html)) | Parse native `.mc`, read `mode_ext.column`, BPM/time points, notes and `endbeat`; converters can produce osu!mania but do not play Malody. | Mature enough as reference implementations; pure file processing is immediate. They do not define all modern client scoring or hold behavior. | Check repository licenses before vendoring. No account/network required for local charts. | Use the field conventions as references; implement the minimal parser in the next phase so the harness owns its behavior. |
| **Malody Store API / local chart server** ([official Store API docs](https://github.com/Mugzone/Malody_Store_API), [MorePracticeMalodyServer](https://github.com/RhythmCodec/MorePracticeMalodyServer)) | Search/download/upload chart packs and skins. The official API requires `uid`, a server-generated `key`, and an `api` version; it does not control live play. MorePractice is a local alternative chart server with SQLite. | HTTP latency is irrelevant after a chart is cached; server projects are less mature than a client and not a gameplay environment. | Official service requires account credentials and network. Upload/crawling can implicate ToS. Local server is free but GPL/other dependency obligations apply. | Optional chart provisioning only. Keep evaluator fixtures local and seeded instead. |
| **Malody Data Studio REST API** ([repository](https://github.com/ChuanYuanNotBoat/Malody_Data_Studio)) | Local REST queries for player/chart/ranking data and a crawler; it is read-only data access, not input or gameplay control. | Python/FastAPI/SQLite is easy to run, but crawling latency/network availability varies. | Requires internet for official crawling and may need API token; respect the repository's warning against high-frequency requests. | Not an adaptor interface. |

## Chosen emulator contract

The next phase should implement the following contract rather than depend on a GUI. It is deliberately close to the event data a real client would expose, while keeping action timing deterministic.

### Installation and process

- Use the supplied `/home/hemlix/GamingMaster-mh-tier2/runs/malody/.venv/bin/python` (Python 3.12.3). The parser can use the standard library (`json`, `math`, `random`, `dataclasses`, `hashlib`); no pip package, account, server, audio, GPU, or display is required.
- Keep chart fixtures or generated charts in a directory owned by the harness. Never fetch charts during an episode. Validate JSON before use: `meta.mode == 0`, `1 <= meta.mode_ext.column`, finite BPMs, legal columns, nondecreasing timing/hold endpoints, and no same-column impossible duplicate at one instant unless the chart format explicitly marks it.
- Expose a small in-process Python API or line-delimited JSON process. The exact transport is not important, but it must be resettable and must not sleep for real time.

### Reset and observation

`reset(seed, options)` should choose a deterministic generated or fixture `.mc`, judge profile (default C), mod (default none), offset, and starting chart time. It returns an observation like:

```json
{
  "episode": {"seed": 17, "mode": "key", "columns": 4, "judge": "C", "mod": "none", "time_ms": 0, "done": false},
  "chart": {"id": "seed-17", "bpm_segments": [{"start_beat": 0, "bpm": 150}], "offset_ms": 0},
  "upcoming": [{"id": 12, "column": 2, "start_ms": 800.0, "end_ms": null, "kind": "tap"}],
  "held_columns": [],
  "last_judgments": [],
  "stats": {"score": 0, "accuracy": 1.0, "combo": 0, "max_combo": 0, "misses": 0, "broken_holds": 0}
}
```

`upcoming` should include enough horizon for planning (at least 1--2 seconds or the next 64 notes), grouped by exact/coincident start times. It must include signed offsets and judgment results after resolution, not just a screenshot. A `peek`/`observe` call must not advance time.

### Actions

Accept either one action or a timestamp-sorted batch:

```json
{"type":"press", "column":2, "at_ms":800.0}
{"type":"release", "column":2, "at_ms":1180.0}
{"type":"advance", "to_ms":1000.0}
```

- `press` resolves the nearest eligible unjudged start in that column. Multiple presses at the same timestamp are a chord; a press in each required column is needed.
- `release` resolves the corresponding hold tail/body rule. The default emulator requires the head press in-window and a release no earlier than `end_ms`; it records a broken hold if the release is missing/early. If the real-client rule is later measured, make it a versioned option rather than changing old seeds.
- `advance` causes notes beyond the miss deadline to become misses and advances the logical clock. Do not silently auto-press notes.
- Reject malformed/out-of-order events with a typed error. Do not use wall-clock sleep; an adaptor can schedule exact chart times even while Astra is thinking.

### Judgment and scoring behavior

Use the C table in `brief.md` by default and make A--E a configuration. For a press, use signed error `at_ms - start_ms` and choose the most precise bucket containing its absolute value; after ±200 ms, mark M. Return `judgment`, `offset_ms`, `note_id`, `column`, and `combo_after` for every resolved object. For holds, return head/tail/broken-hold details separately.

Implement the documented accuracy weights (1.00/0.90/0.85/0.80/0.40/0.00) and expose a weighted mean over judged note heads. Implement the documented 200/332/466/600 D/no-mod Big-P combo bases and listed multipliers only as `score_model: "rm_proxy_2025"`; either apply the same accuracy factor to non-Big-P judgments or report score as a proxy with the formula in the episode metadata. Accuracy, histogram, misses, and combo are the authoritative evaluator outputs. This avoids presenting a community RM formula as an official Key formula.

## Credentials and environment variables

The chosen local emulator needs **no credentials and no required environment variables**. In particular, do not put a Mugzone UID/key/token in files. If the optional Store API is ever enabled, obtain `uid`, server-generated `key`, API version, and server base URL at runtime from a secret manager/environment; never hard-code them. Suggested optional names are `MALODY_STORE_URL`, `MALODY_UID`, `MALODY_API_KEY`, and `MALODY_API_VERSION`, all disabled by default and not needed by the evaluator.

## Real-client fallback setup (not selected)

For a future authorized calibration run: install Steam/Malody V on a Windows-capable host, create/login to a Mugzone account, launch a private/offline chart, and capture a replay with the same chart/judge/mod settings. On Linux, an experiment could use Wine/Proton + Xvfb + a virtual input device, but this is not a supported Malody API and must not be assumed to work. Use the local replay renderer with `python main.py --port 8766 --no-browser` only for post-hoc chart/replay visualization; it is not a way to feed actions to a live game.

## Sources

- [Malody V Steam listing](https://store.steampowered.com/app/1512940/Malody_V/)
- [Malody V App Store listing](https://apps.apple.com/us/app/malody-v/id1566873256)
- [Malody RM timing/scoring article](https://m.mugzone.net/wiki/2443?lang=sc)
- [Malody chart format reference](https://prefixaut.github.io/rconv/rconv/malody.html)
- [Malody-to-osu converter](https://github.com/Jakads/malody2osu/blob/master/convert.py)
- [malody-convert parser](https://raw.githubusercontent.com/kangalio/malody-convert/master/mc.py)
- [Malody V replay renderer](https://github.com/jk21lcl/malody-v-replay-renderer)
- [Malody hit-window reverse-engineering](https://github.com/Hxjjxg/malody-hitwindow)
- [Malody Store API](https://github.com/Mugzone/Malody_Store_API)
- [MorePracticeMalodyServer](https://github.com/RhythmCodec/MorePracticeMalodyServer)
- [Malody Data Studio](https://github.com/ChuanYuanNotBoat/Malody_Data_Studio)
