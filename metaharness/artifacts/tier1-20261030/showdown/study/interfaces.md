# Pokémon Showdown random battles: programmatic interfaces

Target: local `gen9randombattle` singles in the supplied checkout (`~/gm-tools/pokemon-showdown`). The evaluator has no GUI and must work offline, so the local simulator is the practical choice.

## Interface comparison

| Interface | Maturity and latency | Observation and control | Legal/ToS, cost, limitations |
|---|---|---|---|
| **Showdown simulator `BattleStream` (chosen)** | First-party simulator API in the Showdown repository; deterministic seeded games and usually sub-millisecond to a few milliseconds per engine step on this host. | Node `BattleStream` emits protocol text; `getPlayerStreams` splits omniscient, spectator, and per-player streams. A player receives public updates plus its own `sideupdate` containing `|request|{JSON}`. Write `>p1 move 1`, `>p1 switch 3`, or `>p1 move 2 terastallize`. | Runs entirely offline, no account or server. Showdown is MIT-licensed code ([LICENSE](https://github.com/smogon/pokemon-showdown/blob/master/LICENSE)); retain attribution and do not redistribute proprietary data. Requires the checkout's Node version (package currently declares Node >=22.18). A bridge must parse protocol and must never feed the agent omniscient output. |
| **Showdown server WebSocket / HTTP protocol** | Mature production protocol used by the web client; network round trips commonly 20–300 ms and server queue/timer behavior applies. | Connect to a server WebSocket (for example the public endpoint advertised by [pokemonshowdown.com](https://pokemonshowdown.com/)); receive battle protocol and send `/choose move 1|<rqid>` or `/choose switch 3|<rqid>`. | Requires network and usually a guest or registered account; ladder battles have clocks and server policy. Automated play can violate server rules, trigger anti-bot controls, or risk an account. Costs no software license but is unusable in the offline evaluator. |
| **`poke-env` Python bot library** | Widely used open-source educational bot wrapper; event parsing and battle objects are convenient, but it is primarily a server client and adds Python/network overhead (tens of ms plus network). [GitHub](https://github.com/hsahovic/poke-env), [docs](https://poke-env.readthedocs.io/). | Subclass `Player`, inspect `Battle`/`Pokemon`/`Move` objects, return `choose_move`; the library translates to Showdown WebSocket commands. It exposes a structured, partial-information state rather than raw protocol. | Free (MIT). Needs a compatible `poke-env` install and a running Showdown server; package versions and server protocol can drift. Server automation still has ToS/anti-cheat risk. It does not directly wrap the local `BattleStream`, so using it offline would require a custom server adapter. |
| **Local Showdown server or custom simulator plugin** | The repository can run a local server and plugins can hook rooms/battles; mature engine but high integration and maintenance cost. | WebSocket client gets normal player-visible protocol; a plugin can expose extra telemetry or seed/reset hooks. | Offline and free, but plugin hooks can accidentally reveal hidden state and produce a nonstandard game. More processes and protocol work than `BattleStream`; use only if a needed server feature is missing. |
| **Built-in Node bot classes / random-player example** | `sim/tools/random-player-ai.ts` and `sim/examples/battle-stream-example.ts` are maintained smoke-test examples; latency is local-engine speed. | The class consumes player stream requests and writes legal random choices. Source state is protocol text. | Free and offline, but random policy is a baseline, not a strong agent. Reuse its stream lifecycle rather than its policy. [example](https://github.com/smogon/pokemon-showdown/blob/master/sim/examples/battle-stream-example.ts) |
| **Replay/protocol log parser** | Very mature for offline analysis; effectively zero control latency. | Reads saved replay text and reconstructs public events. | Observation-only: cannot act or reset a live battle. Useful for tests and strategy data, not as an adaptor. [replay format](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md) |
| **Browser automation / screen capture + input injection** | Selenium/Playwright plus Xvfb is fragile and adds frame/click latency (roughly 50–500 ms). | Screenshots/DOM expose rendered battle UI; clicks or key events choose moves/switches. | Requires a GUI stack despite the headless constraint, is vulnerable to layout/theme changes, and online automation has the same ToS risk. Use only as a last-resort black-box test. |

## Chosen adaptor: direct `BattleStream`

The direct simulator is the only interface that is offline, deterministic, and exposes exactly the legal partial-information view needed by a player. It avoids server clocks, accounts, browser dependencies, and network variance. The simulator's own engine remains the authority for damage, effects, legality, and random-battle generation.

### Software and installation

* Existing checkout: `/home/hemlix/gm-tools/pokemon-showdown` (or set `SHOWDOWN_ROOT` to another checkout).
* Node.js version satisfying the checkout's `engines.node` (currently >=22.18.0); `npm install` in that checkout if dependencies are absent. No Python package is required for the protocol bridge.
* The adaptor may be a small Node process using `require('/path/to/pokemon-showdown')`, or it may run `pokemon-showdown simulate-battle` and speak newline-delimited stdin/stdout as documented in [SIMULATOR.md](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIMULATOR.md).
* The Python harness can launch one long-lived Node process or one subprocess per battle. The latter is simplest and isolates state; append `\n` to every command in text mode.

No account, API key, password, or network endpoint is required. If the adaptor is made portable, an optional environment variable is:

* `SHOWDOWN_ROOT` — checkout containing `pokemon-showdown`/`dist`; default `/home/hemlix/gm-tools/pokemon-showdown`; required only when the default is absent.

### Reset and observation contract

1. Start a fresh process (or call `stream.writeEnd()` and create a new `BattleStream`) for every `reset(seed)`; do not reuse a finished stream.
2. Send `>start {"formatid":"gen9randombattle","seed":[s0,s1,s2,s3]}`. Four integers make random-team generation and battle RNG reproducible. Omit `seed` only for an explicitly random episode.
3. Supply players with `>player p1 {"name":"Astra","team":null}` and `>player p2 {"name":"Opponent","team":null}`. In a random format, `team:null` asks the engine to generate legal teams. The protocol also accepts a packed/JSON team when a fixed scenario is desired.
4. Wrap the stream with `getPlayerStreams`. Feed only `streams.p1` to the agent. Parse each chunk as protocol lines. `update` messages contain public events; `sideupdate` for `p1` contains private requests and exact own information. A request is `|request|<JSON>` and may be a move request (`active`), forced-switch request (`forceSwitch`), team-preview request if a format adds one, or a wait request. Keep the latest request and all revealed public events; do not assume one request per turn because a faint, pivot, or Revival Blessing can create another request immediately.
5. The JSON request exposes legal move slots, PP and disabled flags, tera availability (`canTerastallize`/move flags), active Pokémon, own side's six slots and HP/status, and opponent-visible identities/HP/status/moves as they are revealed. Opponent unrevealed set details must remain unknown. `|split|`/omniscient data is for a referee or test oracle only.
6. On a move request, write exactly one legal command such as `move 1`, `move 2 terastallize`, or `switch 4` to the p1 stream. Choice syntax and errors are specified in [SIM-PROTOCOL.md](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md#possible-choices). On forced replacement, write `switch <slot>`; do not send a move. If a command is rejected, parse the following request and select again. Never use `default` as the strategic policy; it simply chooses the first legal option.
7. Read until the next `|request|` or `end`. At `end`, parse its JSON/log data and the referee's omniscient state only for score calculation. Report outcome, both sides' terminal HP fractions, HP margin, turn count, and whether the episode ended normally or was truncated.

The local text protocol is documented at [SIMULATOR.md](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIMULATOR.md), [SIM-PROTOCOL.md](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md), and the implementation of [`getPlayerStreams`](https://github.com/smogon/pokemon-showdown/blob/master/sim/battle-stream.ts).

## Credentials and environment

For the chosen local interface: **none**. Do not put credentials in source or study files. A server-backed alternative would need a server URL and, if using a registered account, username/password or a server-issued login token; those should be injected as environment variables at runtime and never committed.

## Practical risks

Keep the agent on its p1 stream: using `omniscient` or spectator output leaks hidden opponent sets and invalidates the task. Pin the checkout commit and format because random sets and protocol fields evolve. A Node/Python pipe must handle backpressure and blank-message framing. Seed values make starts reproducible but do not remove battle RNG; held-out seeds are still required. The standalone simulator has no wall-clock move timer, whereas a WebSocket server does.
