# Programmatic interfaces and an offline adaptor specification

## Decision, evidence, and immediate blocker

**Choose the EDOPro native duel core (`edo9300/ygopro-core`), embedded behind a small C/C++ bridge and driven by Python 3.12's standard-library `ctypes`.** Use a pinned matching card database and Lua script bundle. Run both players in one local worker; initially use a deterministic random-legal opponent. No client window, account, public server, paid service, GPU, or ML model is needed to run a duel. The bridge and adaptor are work for the next phase; none is implemented here.

Why this choice: it uses the actual automated rules engine, supports a headless process, and avoids betting on an old CPython extension wheel or a full training stack. A bridge compiled against the chosen header insulates Python from native structure layout and callback ABI changes. The tradeoff is substantial protocol parsing and observation filtering work. **The core is not a ready-made Gym environment, does not supply a general-purpose AI, and does not directly return a JSON legal-action list.**

Research was blocked by network/DNS restrictions on 2026-10-29. Requests for the ygo-agent README, EDOPro core repository, and official rulebook all failed with curl exit 6. No web pages or upstream source code were successfully fetched. URLs in this document are retrieval targets, not claims of inspection. Native API symbol families below reflect established EDOPro architecture; signatures, build commands, supported options, and wire layouts must be extracted from the **same pinned source revision**. No commit hashes, current versions, successful build, or Python-wheel compatibility are asserted.

**Local facts verified with the evaluator executable:**

| Item | Result |
|---|---|
| Python | `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/yugioh/bin/python`, 3.12.3 |
| Installed Python packages | Only pip 26.2.2 was listed |
| Standard library | `ctypes`, `sqlite3`, `struct`, and `json` import successfully; SQLite 3.45.1 |
| Platform | Linux x86_64, glibc 2.39 |
| Build tools | GCC/G++ 13.3.0, CMake 3.28.3, Ninja 1.11.1, Git, curl, pkg-config present |
| Native runtime libraries | Lua 5.4 and 5.2 shared libraries, SQLite and libstdc++ found by `ldconfig` |
| Development inputs | Python 3.12 include directory exists; no Lua include directory or SQLite header at the usual inspected paths |
| Game assets | No core, `cards.cdb`, `.ydk` decks, or relevant binding found in inspected workspace, gm-tools, and caches |
| Execution test | Not possible without core sources/library and assets; not attempted |

**Blocking dependency:** an offline source/asset bundle must be staged before the adaptor can build. A network-enabled acquisition environment can prepare it, but runtime here must stay offline. Do not silently replace the real engine with a toy combat model. This study sandbox permits writes only in this directory and `/tmp`; it cannot install into the specified venv under current permissions. No permission escalation is available. A later phase with the required write access must perform any venv installation. The chosen standard-library route needs no new Python package at runtime.

## Practical interface alternatives

All latency descriptions are architectural expectations, **not measured benchmarks**. For this workload, completeness and correctness matter more than microseconds.

| Interface | Maturity and what it exposes | Latency and dependencies | Cost / legal or service concerns | Decision |
|---|---|---|---|---|
| **EDOPro native core + local C bridge / ctypes** | Established simulator engine; binary event/request stream, state queries, scripted effects, response submission. Proposed Python adaptor is new. | In-process or local subprocess calls, normally much faster than model inference; Lua/effect work can vary. C++ compiler and matching Lua/core assets. | No service charge. Open-source code licenses must be retained and checked; card text/art are separate IP. No account/anti-cheat issue for local use. | **Primary.** Most controllable offline dependency chain and no CPython extension ABI dependency. |
| **`sbl1996/ygo-agent` / its ygoenv environment** | Research agent/environment route intended for machine policies; can offer numeric observations, action encodings, batching, and self-play. Exact coverage/configuration must be checked. | Native environment should be fast; training ecosystem may pull NumPy/JAX/PyTorch/EnvPool or build dependencies depending on revision. These are not all necessarily required for environment-only use. | Free source subject to its licenses and asset licenses. No remote account needed for local environment. | Worth inspecting first when sources arrive. Prefer it over writing a parser **only if** environment-only build succeeds on the exact Python 3.12.3 venv and action/observation completeness is demonstrated. Do not assert a PyPI package name or wheel exists. |
| **Legacy `Fluorohydride/ygopro-core` C API + ctypes/cffi** | Long-used older core family; functions such as `create_duel`, `process`, `get_message`, `set_responsei`, `set_responseb`, and query functions. | Small native boundary; straightforward FFI but requires complete protocol handling. | Same local/free-software considerations; inspect the selected license. | Viable for a coherent older-format bundle. **Not ABI-compatible** with modern EDOPro `OCG_*`; never mix new scripts with an arbitrary old core. |
| **WindBot + compatible local YGOPro/EDOPro server protocol** | Deck-specific scripted executors; practical stronger opponent for supported decks. Structured actions/field state exist inside bot. Not a universal rules engine or generic opponent for every deck. | TCP loopback plus a compatible duel host; C#/.NET or Mono requirements depend on revision. Local latency is not the bottleneck. | Free source; inspect license and shipped deck/assets. Offline local hosting has no platform account requirement. | Optional opponent upgrade. Requires a host and exact protocol/deck compatibility; do not add it to the minimum dependency set. |
| **Custom native bot or client linked to the core; local duel-server libraries** | Same underlying rules; server/client implementations can supply packet handling and visibility logic. YGOPro-family server protocols are implementation-specific, not an official Konami API. | Local IPC/TCP, extra host lifecycle and timers. Match core/client versions. | Free implementations exist; licenses vary. Public servers impose their own rules and are outside scope. | Useful parser reference, or for WindBot. Disabling room clocks is essential. |
| **EDOPro GUI client, single-player modes/puzzles, or source modification** | Mature graphical client with practical reference handlers; puzzle scripting initializes situations. Neither implies an off-the-shelf Python API. | GUI build brings graphics/windowing/assets. A source-level headless exporter is preferable to screenshots but requires a patch. | Free client code; retain licenses, distinguish proprietary card assets. | Reference implementation and optional human debugging. A puzzle is not a seeded two-deck benchmark by itself. |
| **Pure-Python simulators or accessible/text YGOPro clients** | Quality ranges from small rules subsets to clients backed by a native core. A text client can expose prompts but may rely on a server. | Usually fine for turn-based play; modern card correctness and Python 3.12 support are the risks. | Project-specific license; no cost if local/free. | No complete, maintained, Python-3.12-compatible pure-Python engine was verified here. Do not infer full rules support from “Yu-Gi-Oh simulator.” |
| **Screen capture + input injection (EDOPro under Xvfb)** | Can observe rendered cards/dialogs and click; loses hidden structure, precise IDs, and reliable legal-action mapping. | Screenshot/OCR/input round trips; X server, rendering stack, fonts, images, focus management. | Free tools possible; same card asset licensing issues. Online automation may violate service rules. | Poor fit for a server with no GUI; avoid unless every structured route fails. |
| **Emulated commercial game + memory/input hooks** | A licensed older game may include built-in AI, but its rules/card pool differ from EDOPro. | Emulator overhead and reverse-engineered state hooks. | Free emulator does not make ROM/game assets free. | Fails the intended free-software/data acquisition path; not selected. |
| **Official services or card-data web APIs** | Official card DB and YGOPRODeck provide card information; they are **not duel stepping APIs**. Commercial games are not local open engines. | Network/service dependencies. | Terms/rate limits/accounts depend on service; not offline. | Useful only for pre-staging references, not gameplay. |

Source targets: [EDOPro core](https://github.com/edo9300/ygopro-core), [EDOPro client](https://github.com/ProjectIgnis/EDOPro), [legacy core](https://github.com/Fluorohydride/ygopro-core), [ygo-agent](https://github.com/sbl1996/ygo-agent), [WindBot](https://github.com/ProjectIgnis/windbot), [legacy YGOPro client](https://github.com/Fluorohydride/ygopro), [YGOPRODeck API guide](https://ygoprodeck.com/api-guide/).

## Software and asset manifest

Acquire a **mutually compatible pinned snapshot**, preferably the versions/submodule commits used by one EDOPro release rather than independent latest branches:

1. EDOPro core source and its license; build a Linux x86_64 shared library or link it into the bridge. Primary: https://github.com/edo9300/ygopro-core .
2. Matching Lua implementation and headers required by that core's build. Do not assume the existing Lua 5.2 or 5.4 runtime is sufficient; headers and the expected ABI matter. Vendor/build the required version if necessary. Reference: https://www.lua.org/ .
3. Complete matching **CardScripts**, including shared helpers, startup/utility scripts, and individual card scripts. Reference: https://github.com/ProjectIgnis/CardScripts . Copying only `c<ID>.lua` for deck cards is insufficient: helpers and generated/referenced cards can be dependencies.
4. Matching **BabelCDB/card database** snapshot, including English names/text and effect description strings. Reference: https://github.com/ProjectIgnis/BabelCDB . Also retain matching system strings/format configuration from the client distribution. A `.cdb` supplies card metadata/text, not effect execution.
5. Reference client source from the same release for event decoding, response packing, startup/script loading, and visibility rules. Reference: https://github.com/ProjectIgnis/EDOPro . The client itself need not be built.
6. A benchmark manifest containing exact deck card IDs, counts, checksums, rule flags, any forbidden/limited-list snapshot, and opponent policy version. No images, sound, network updater, or asset downloader is necessary.
7. Build tooling: existing G++, Git, CMake/Ninja **or the upstream-selected build tool**. Some YGOPro-family projects use Premake; do not assume a CMakeLists exists. Stage the build-tool binary/source if the pinned project needs it. Standard Python `ctypes`, `sqlite3`, `struct`, `json`, and `multiprocessing`/`subprocess` suffice for the recommended runtime. No CUDA/JAX/PyTorch requirement.

Record SHA-256 hashes and commit/release identifiers for every source/archive, core/bridge binary, database, script bundle, deck manifest, and rule configuration. Freeze them for evaluation. Do not let an EDOPro launcher auto-update assets between seeds. Include software license texts; avoid shipping card art entirely. “Open-source engine” is not a blanket license to redistribute all Konami assets. Exact redistribution terms were not retrieved in this study.

## Configuration and credentials

**No user account, API key, remote server, or credentials are required.** The following are **proposed adaptor environment variables**, not upstream EDOPro variables. The adaptor should resolve them from a local manifest by default; the user need not supply them if the bundle uses the agreed directory layout.

| Variable | Proposed meaning | Required user input? |
|---|---|---|
| `YGO_ASSET_ROOT` | Absolute directory containing the pinned database, scripts, strings, deck manifest, and hashes | No; default to the staged bundle |
| `YGO_CORE_LIB` | Absolute path to the built compatible core/bridge shared library | No; derive from the bundle/build output |
| `YGO_SCENARIO_MANIFEST` | Absolute path to the fixed benchmark deck/rule/opponent manifest | No; default to benchmark manifest |
| `YGO_DECISION_TIMEOUT_S` | External agent-response deadline; proposed default 300 seconds, not an engine clock | No |

No secret values should appear in observations, files, or logs. A future WindBot variant could use a local socket address in its manifest; it still needs no public account.

## Setup sequence for the adaptor phase

1. **Stage assets offline.** Obtain the pinned bundle above. If no bundle or permitted acquisition channel exists, stop with an explicit missing-dependency error. The present host cannot download it in this study session.
2. **Read the pinned build and API files before coding.** In the core, locate `ocgapi.h`, its implementation, the message/constant definitions, and the build instructions. In the client, locate the handlers for each `MSG_*` selection and the single-player/core startup path. Common reference locations include `gframe/duelclient.cpp` and `gframe/single_mode.cpp`; confirm locations in the chosen tree. Build flags, struct sizes, seed layout, status values, and response encodings must come from these files. If using ygoenv instead, first inspect its environment examples, installation metadata, backend selection, action masks, truncation limits, and player-visibility behavior on Python 3.12.
3. **Build only native core and bridge**, with matching Lua and position-independent code. Use the checked-in upstream build system. Do not invent an untested `pip install ygoenv` or assume a current manylinux wheel. A bridge should expose a simple stable ABI to Python, own native structures/callbacks, and copy buffers with explicit lengths. Run an isolated worker process so a native assertion cannot kill the evaluator.
4. **Use the evaluator Python explicitly.** All imports and tests must use `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/yugioh/bin/python`. If any packages prove necessary, stage CPython 3.12 Linux x86_64 compatible wheels and install with that interpreter's `-m pip --no-index --find-links ...`; account for transitive build dependencies. Avoid system-Python success as evidence. If writing that venv remains prohibited, report the permission blocker rather than installing elsewhere silently.
5. **Load and validate the database.** YGOPro CDBs conventionally have `datas` and `texts` keyed by card ID. Inspect the actual schema. Card metadata includes alias, set membership, type, ATK/DEF, level/rank/Link-related data, race/type, and Attribute; text includes name/description and effect strings. Convert packed fields as the pinned core expects. Pendulum scales and Link markers are not ordinary level/DEF fields. Resolve name → ID once, preserve leading-zero printed passcodes as data correctly, assert every resolved card and required script exists, then freeze numeric IDs in the manifest.
6. **Implement and retain callbacks for the entire duel lifetime.** Card-data reader supplies native metadata; script reader loads the requested matching Lua text; log handler captures script/core errors. Include any corresponding native data-done/lifetime callback required by the pinned ABI. A native shim can preload metadata exported from Python's SQLite reader, avoiding a SQLite C build dependency. Python objects/buffers must not be freed while native callbacks still reference them. Missing metadata/scripts must fail explicitly, not turn effects into no-ops.
7. **Construct, seed, configure, populate, and start a duel.** Set the selected two-player Master Rule flags, 8,000 starting LP, five initial cards, one normal draw per turn, and selected starting player. Load required startup scripts using the matching client sequence. Add each player's Main/Extra Deck in the deterministic expected order, with correct ownership/controller/location. Use the core's documented shuffle/start semantics; ensure shuffling is done exactly as intended rather than accidentally performed twice or never.
8. **Build the event/request loop and semantic actions** described below. Keep authoritative state private. Derive both agents' observations through the same player-view filter, and seed the opponent independently. Run an initial random-vs-random smoke duel; then validate deterministic replay, all prompt types encountered by the pool, and hidden information before involving Astra.
9. **Freeze and run without networking.** No lookup, updater, account, or image retrieval should be attempted during reset or step. The returned observations must contain locally available card text. Save versioned action/event traces for reproducibility.

## Native interface boundary: do not mix generations

Modern EDOPro exposes an `OCG_*` API family. Expected operations include version querying, duel creation/destruction, new-card insertion, starting/processing a duel, getting messages, submitting a byte response, loading scripts, and querying card/field data. Common symbol names are `OCG_GetVersion`, `OCG_CreateDuel`, `OCG_DestroyDuel`, `OCG_DuelNewCard`, `OCG_StartDuel`, `OCG_DuelProcess`, `OCG_DuelGetMessage`, `OCG_DuelSetResponse`, `OCG_LoadScript`, and `OCG_DuelQuery*`. **Check their existence and exact signatures in the supplied header; this is not a binding declaration.**

Older cores instead expose APIs such as `create_duel(seed)`, `start_duel`, `process`, `get_message`, `set_responsei`, and `set_responseb`. Older `process` return-bit conventions, fixed response sizes, location widths, and selection encodings must not be copied into EDOPro merely because message names look similar. Build a bridge against one header, and use parsers/response writers from the matching client. Read API version information on load and reject a mismatch.

The host supplies card data and scripts because the core does not know where a card database lives. Core query results can reveal **both** players' secrets. They are an internal implementation resource, not a policy observation. Engine determinism is meaningful only with the same binary, rules, assets, seed expansion, initial insertion order, and responses.

## Required processing loop

The adaptor's unit of control is **one engine decision prompt**, not one whole turn.

1. Advance native processing and consume every emitted message in order, using the pinned API's `CONTINUE`/`WAITING`/`END` meanings rather than guessed numeric statuses. Copy borrowed buffers before the next native call invalidates them. A returned batch can contain several events; track lengths and bounds.
2. Update state from events and supported queries: turn/phase, zone changes, reveals/shuffles, summons, attacks, chain activation/resolution, LP, counters, relationships/materials, and victory. Resolve display strings locally. Do not call the model for mere animation/log events.
3. On a selection request, identify the **player actually asked**, decode the exact constraints, construct stable semantic options tied to that pending request, and stop processing that request until a response is supplied.
4. If the player is the scripted/random opponent, obtain a valid response from its filtered observation and continue. If it is the controlled player, return the public observation, current prompt, and opaque legal IDs. Mandatory singleton decisions can be automatically resolved and logged; do not automatically decline optional response windows.
5. Validate the returned ID or structured selection, encode the exact native response, submit it once, and resume. IDs are valid only for their prompt token/state version. An engine retry/invalid-response message signals a bug or bad input: do not consume RNG or choose an unrelated fallback silently.
6. On terminal outcome, translate winner seat to agent seat and save final public state. On unknown request, parse error, script failure, or impossible constraints, return an explicit technical error with a private diagnostic trace. Never guess an answer to an unknown message.

### Prompt coverage checklist

Names below are common YGOPro-family message names; the pinned constants and handlers decide actual names and layout. This table defines required semantic behavior, **not byte offsets**.

| Request family | Semantic action to expose | Critical details |
|---|---|---|
| `SELECT_IDLECMD` | Normal Summon, Special Summon, change position, monster Set, Spell/Trap Set, activate effect, enter available phase | Activation options need effect description IDs, not just card names. Repeated effects on one card are distinct. |
| `SELECT_BATTLECMD` | Attack with a specified monster, activate, go to Main Phase 2/End if offered | An attack choice may lead to a separate target or direct-attack decision. |
| `SELECT_CHAIN` | Activate a specific response effect or decline when allowed | Respect forced selection; “no chain” is not always legal. Never suppress prompts simply because it is the opponent's turn. |
| `SELECT_EFFECTYN`, `SELECT_YESNO`, `SELECT_OPTION` | Yes/no or one described option | Include source card, effect text, hint/context, and available option descriptions. |
| `SELECT_CARD`, `SELECT_UNSELECT_CARD` | Select required cards; toggle/finish/cancel where supported | Preserve eligible indices, min/max, selected set, and separate finish/cancel semantics. Counts may change after a toggle. |
| `SELECT_TRIBUTE`, `SELECT_SUM` | A constrained subset of candidates | Tribute values and alternative material values are not simply cardinality. Mandatory inclusions, exact/at-least total, min/max counts, and native encoding matter. Use an exact constraint solver in code. |
| `SELECT_PLACE`, `SELECT_DISFIELD`, `SELECT_POSITION` | Select legal zone(s) or battle position | Decode player-relative locations and masks; explicit legal choices prevent occupied/forbidden-zone mistakes. |
| `SORT_CARD`, `SORT_CHAIN` | A permutation or permitted “do not reorder” option | Preserve all candidates; simultaneous trigger order can be strategically important. |
| Counter and announcement requests (`SELECT_COUNTER`, `ANNOUNCE_*`) | Allocate counters; declare number, card, race/type, Attribute, etc. | Respect per-card available amounts and any card-declaration predicate/filter; some predicates use an encoded expression, not a simple list. |
| Rock-paper-scissors / starting-player / replay decisions | Seeded setup policy or a documented agent choice, if host emits these | Native embedding can usually set starting player outside network-room setup. Do not invent these messages if absent. |
| Retry, hints, confirmation, shuffle and win events | Update state, knowledge, or errors, not arbitrary model choices | A confirmation/reveal can carry information even when no response is required. |

Small selection sets can be expanded to complete legal actions. Large combinations must **not** be silently truncated to a fixed list or the first N options. Expose a validated selection object or stage typed subchoices with a final exact-constraint check; only submit the complete native response when valid. Enumeration limits, if any, must be explicit in diagnostics. Sampling a legal subset requires a constructive feasibility algorithm, not independent random card picks that violate the sum.

## Proposed evaluator observation/action contract

These field names belong to the new adaptor, not to an upstream API:

- `reset(seed)` returns the initial agent decision observation, or a terminal result if unusual setup ends immediately. `step(action)` handles one agent response, runs automatic events/opponent decisions, and returns the next agent prompt or terminal result. It may return another prompt in the same phase or turn.
- Observation identity: schema version, opaque episode ID, monotonic state version, prompt token, agent seat, acting seat, turn player, turn number, phase, whether the state is terminal, outcome/reason if public. **Do not expose the shuffle seed or RNG state to either policy.**
- Public board: LP, hand/deck/Extra Deck counts, face-up cards and their current stats, zones, positions, counters, equips/targets/material relationships where public, Graveyards, face-up banished cards, chain links and announced costs/targets, public history since the previous prompt. Unknown cards carry only legal public metadata such as location and face-down status.
- Private player view: own hand and own legally inspectable cards/Extra Deck/face-down cards, with locally loaded text, known card IDs, and legal effect descriptions. Include revealed opponent cards only to the extent justified by the event history. Opponent face-down banished identities stay hidden. Avoid unsupported assumptions about visibility of unusual overlays/attachments; mirror the reference client's per-player policy.
- Prompt: semantic kind, selecting player, reason/hint, candidate actions/cards, source/effect identity, selection constraints, allowed cancel/finish options. Provide current card text for known candidates and the chain context. Code-maintained counters/used-effects/known restrictions may supplement but must distinguish inferred information from engine facts.
- Action: an opaque `action_id` plus prompt token for ordinary choices; a typed constrained selection/permutation for large compound prompts. The adaptor stores native candidate indices privately. Reject stale IDs, unknown fields, duplicate selections, impossible values, and wrong-player responses.

Assign instance IDs to known in-play cards, but do not preserve traceable identities across hidden shuffles/moves that the player could not follow. Never dump raw engine messages, database code fields for hidden cards, debug logs, whole-state hashes, raw RNG seed, unfiltered queries, or internal opponent choices into the model context. Publicly knowing the pool/deck list does not entitle the policy to know current hidden locations.

## Fixed initial deck pool

Use a **custom three-copy maximum**, ordinary Master Rule duel, no tournament Forbidden/Limited list, no Side Deck, empty Extra Deck. This deliberately bounded baseline exercises summoning, setting, battle, chains, costs, targeting, flips, resurrection, and Tributes without requiring modern combo mastery. It is a benchmark proposal, not a recommended tournament deck list. Resolve these exact English names against the pinned CDB; assert unique intended cards, inspect their pinned text, then write IDs and counts to the manifest. Do not trust a model-generated numeric passcode.

All decks share this 16-card Spell/Trap package:

| Card | Copies |
|---|---:|
| Mystical Space Typhoon | 2 |
| Book of Moon | 2 |
| Fissure | 2 |
| Dark Hole | 1 |
| Monster Reborn | 1 |
| Rush Recklessly | 2 |
| Trap Hole | 2 |
| Mirror Force | 1 |
| Compulsory Evacuation Device | 2 |
| Call of the Haunted | 1 |

Add exactly one 24-monster package:

| Pool ID | Monster package (each listed card ×3) | Strategic character |
|---|---|---|
| `normal_pressure` | Alexandrite Dragon; Gene-Warped Warwolf; Gemini Elf; Luster Dragon; Vorse Raider; La Jinn the Mystical Genie of the Lamp; Battle Ox; Mystical Elf | Consistent Normal Summon pressure; protect damage and use removal efficiently. |
| `flip_control` | Alexandrite Dragon; Gene-Warped Warwolf; Gemini Elf; Mystical Elf; Giant Soldier of Stone; Man-Eater Bug; Penguin Soldier; Magician of Faith | Defensive sets and flip value; protect small monsters and trade removal. |
| `tribute_pressure` | Summoned Skull; Blue-Eyes White Dragon; Luster Dragon; Vorse Raider; La Jinn the Mystical Genie of the Lamp; Battle Ox; Mystical Elf; Giant Soldier of Stone | Deliberately higher variance; preserve bodies for Tributes, avoid dead high-level hands, exploit resurrection. |

Each is 24 monsters + 16 Spell/Trap cards = 40. No listed name exceeds three copies in a deck. No assertion is made that all three are equally strong: report results by matchup and starting seat, and use paired seeds. Names, scripts, legal placement, and game completion must be verified against the acquired assets before declaring the pool ready. If a card is unavailable, update and version the manifest explicitly; never substitute based on a similar name.

Later, optionally add pinned complete archetype lists and matching WindBot executors, or publicly documented historical formats. Expanding the pool requires new prompt-coverage and script tests. Do not introduce “current meta decks” from live web results during held-out evaluation.

## Opponent, seeding, and scoring

**Baseline opponent:** select uniformly among the currently offered complete legal actions when enumerable; for large constrained selections use a seeded, feasibility-preserving sampler and document its distribution. Optional response decline is a legal option only when offered. This is not globally uniform over duel trajectories. Random policies can make terrible plays, including repeatedly changing reversible options; detect/cap non-progress according to the benchmark safety policy, not by silently rigging actions. The opponent must answer all mandatory prompts and use the same private-information filter as the agent.

**Recommended stronger local upgrade:** a deterministic rule-based policy using the code priorities in `strategies.md`: forced choices, safe known lethal, prevent known lethal, profitable legal battle/removal, sensible summon/set, useful effect activation, then phase progression. This requires coding but no weights or service. Version it separately from random. WindBot is a further optional upgrade only for an exact supported deck and compatible host; the core has no built-in general AI that can simply be switched on.

`reset(seed)` must deterministically select the agent deck, opponent deck, first player/seat, shuffle streams, and opponent random stream. Use a documented fixed derivation (e.g. SHA-256 with domain tags) instead of Python's randomized built-in `hash`. Expand seed material to exactly the core's required seed representation. Separate selection, core/shuffle, and opponent RNG streams so changing policy random calls does not alter the opening shuffle. Keep seed and full hidden state in evaluator-private logs, never in observations. Frozen deck-pool contents can be public; seed-to-hidden-state mapping must remain private.

Use all ordered deck pairings, both starting positions, and many held-out shuffle seeds. Prefer a manifest/schedule that balances strata rather than relying on a few random draws. For paired comparisons, run the same scenario/opponent configuration for every candidate policy; optionally swap decks/seats in a second duel, treating it as a separately scored episode. Never choose seed difficulty based on an agent's actions or model response time.

Return native win/loss/draw plus terminal LP, and scalar `w + 0.1*clamp((LP_agent-LP_opponent)/8000, -1, 1)` for real terminal results. Use explicit benchmark policies for 300-second decision forfeits and 200-turn/10,000-decision truncations as defined in `brief.md`. Report legal-action failures and infrastructure errors separately. Score does not override the engine's winner, and LP payments must not be punished as intermediate losses.

For the proposed external decision deadline, specify when the clock starts (after the observation is delivered), whether model/API retries count (recommended: yes), and a monotonic clock source. Do not charge native opponent processing or asset loading against the agent's decision budget. Native processing has its own watchdog and produces an infrastructure error rather than an invented in-game outcome. Make these settings part of the manifest so slower-model comparisons use the same policy.

## Acceptance checks before claiming a working adaptor

1. **Real interpreter and offline startup:** import/load from the specified Python 3.12.3 venv in a process without network or DISPLAY; run at least one full seeded duel per ordered deck pairing. Missing assets or incompatible shared libraries must be visible errors.
2. **Legality and completeness:** every prompt encountered by every pool card has a tested handler; exercise summoning/setting, target/cost selection, Tributes, multiple candidates, chain decline/forced response, position/zone selection, and terminal events. Broader mechanics need controlled fixture duels for sums, permutations, announcements, and Extra Deck prompts before pool expansion. No silent legal-action truncation.
3. **Determinism:** same manifest, seed, and agent response sequence yields identical private event trace and terminal result across fresh processes. Different seeds vary shuffles/starts as expected. Capture all randomness, not just the first hand.
4. **Information boundary:** compare policies' views at each fixture step; no opposite hand, deck order, face-down identity, seed, or secret-derived stable tag appears. Reveal → shuffle → observe must correctly forget location-specific knowledge. Opponent observations receive the same treatment.
5. **Rule fixtures:** first player has five cards and no first-turn draw/Battle Phase; LP/battle accounting; Tribute costs; flip effects; resurrection; optional windows; reverse chain resolution; deck-out; seat-to-winner mapping; a mocked/fixture alternate-condition result must not be reclassified by LP.
6. **Failure semantics:** stale/invalid response rejection, core retry, missing script, parse error, native crash, response timeout, and nontermination cap each produce their distinct diagnostic category. Never reward a parser crash as a draw or unexplained terminal win.
7. **Baseline quality:** report completion rate and win/LP statistics of random versus random and scripted versus random, stratified by deck and starting seat. Require zero unknown prompts and no information leaks on the fixed pool before model evaluation.

## Source retrieval checklist

All links are **unretrieved in this session**. Inspect and archive these upstream sources to resolve the remaining build/protocol uncertainty:

- https://github.com/edo9300/ygopro-core — README/build files, license, `ocgapi.h`, API implementation, constants/message writers, seed and process semantics.
- https://github.com/ProjectIgnis/EDOPro — corresponding client readers/response writers, startup sequence, per-player visibility, release/submodule dependency versions.
- https://github.com/ProjectIgnis/CardScripts — helper/startup scripts and pinned per-card rules implementations.
- https://github.com/ProjectIgnis/BabelCDB — database layout and card metadata/text assets.
- https://github.com/sbl1996/ygo-agent — potential environment-only shortcut; inspect exact Python/native dependencies, observation/action conventions, supported formats, and tests.
- https://github.com/ProjectIgnis/windbot — optional scripted opponent, executor/deck coverage and host protocol requirements.
- https://github.com/Fluorohydride/ygopro-core and https://github.com/Fluorohydride/ygopro — coherent legacy fallback and reference protocol; do not cross-wire with modern EDOPro.
- https://www.lua.org/ — required Lua version/build/license once the core dependency is known.
- https://docs.python.org/3.12/library/ctypes.html and https://docs.python.org/3.12/library/sqlite3.html — Python 3.12 standard-library FFI/database interface.
- https://www.yugioh-card.com/en/rulebook/ and https://www.db.yugioh-card.com/yugiohdb/ — rules and official card text cross-checks.
