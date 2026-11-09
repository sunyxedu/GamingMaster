# Offline Yu-Gi-Oh! adaptor — opponent revision 3

`game.py:make_game()` implements Tier 1 with the **real EDOPro / ygopro-core engine**, its Lua card scripts, and **WindBot running on Mono**. Gameplay needs no GUI, account, credentials, network host, or live server. The preserved `game_v2.py` provided the working native-process plumbing. Its greedy opponent has been replaced; there is no evaluation difficulty switch.

## Setup and pinned inputs

The built Linux x86_64 bundle is staged under `runtime/`. Python gameplay uses only the standard library and `metaharness`. Build requirements are installed in `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/yugioh`. Mono 6.8.0.105 and its compiler were installed through the free Ubuntu `mono-devel` package.

```bash
# Provisioning only, if moving to a machine without Mono:
sudo apt-get install mono-devel
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/yugioh/bin/python -m pip install -r requirements.txt
./setup.sh                 # rebuild from staged sources, offline
./setup.sh --download      # restore missing pinned source repositories at build time
```

`env.example` contains only commented optional settings. No user configuration is required. `YGO_DECISION_TIMEOUT_S` changes the agent deadline; `YGO_PYTHON` selects the build interpreter. Neither changes opponent strength.

`assets.lock.json` version 3 hashes the native libraries, bridge sources, card database, metadata, all Lua scripts, exact deck recipes, WindBot library/bridge binaries, executor sources and opponent manifest. Construction verifies these hashes. `opponent.v3.json` separately versions the stronger opponent and its interface. Setup refuses changed upstream sources or mismatched revisions; no downloader runs during reset or play.

| Input | Pinned revision |
|---|---|
| EDOPro core, API 11.0 | `6864373c24d7313c8e451df17877c12c4f713c46` |
| Core's Lua submodule | `6e22fedb74cf0c9b6656e9fce8b7331db847c605` |
| CardScripts | `2ddefd0115a122f8cd3f494b6affbcbc27b4f974` |
| BabelCDB | `2e6f985a5537d9b16ec7396787b5279d94af419d` |
| EDOPro protocol reference | `d0f111d6f3e0d4ee529b88a869ae0a971954bae3` |
| Distribution strings | `54a6e2395c532648ff762540e9615319fac4f51b` |
| ProjectIgnis WindBot | `bffe6b62679c8b2fafea8f59740e03a132517da4` |

These independently pinned inputs were tested together; this is not claimed to be a packaged EDOPro release. Software licenses are retained in the vendor trees. Adaptor and bridge code use AGPL-3.0-or-later (`LICENSE`); WindBot is also AGPL-3.0-or-later. Card text/names remain their owners' IP; no art/audio is included.

## WindBot integration and strength

`windbot-fixed-pool-v3` uses the **unmodified upstream `GameAI` and `DefaultExecutor` library**, compiled as `ExecutorBase.dll`. `WindbotBridge.cs` supplies a persistent local-pipe transport instead of EDOPro's TCP room protocol. WindBot's main-phase/battle decisions, attack/target selectors, chain decisions, tribute selection and positions drive the real core. Its answers pass the exact same legality validator as agent answers. Bot crashes, illegal answers and timeouts invalidate a trial; there is no silent fallback to greedy or random play.

WindBot requires an executor for its deck. Upstream modern-deck executors cannot play these frozen recipes as supplied. `PoolExecutor.cs` registers **all 24 pool cards** with WindBot, using its default removal, trap, revival, battle, position and summon logic, with pool-specific handlers for the Flip monsters, Fissure, Monster Reborn, Book of Moon and Rush Recklessly. It prefers useful strong summons, sets Flip monsters, spends removal before combat, queues targets and preserves combat tricks for useful windows. This is a custom fixed-pool WindBot executor, not an assertion that an upstream modern tournament deck is being used. WindBot runs successfully here, so the fallback search opponent and ygo-agent checkpoint route were unnecessary.

The transport receives only the opponent's filtered view: its own hand, public field/current stats, public Graveyard, visible chain/targets, LP, counts, current battle, numeric prompt hints and legal candidates. It receives no agent hand, hidden face-down identity, deck order, raw engine messages or engine seed. The separate opponent RNG seed is used only to initialize WindBot's `System.Random`; upstream cosmetic chat is suppressed. Its decisions are deterministic for the seed. One fixed policy applies to every seed.

The historical greedy method is retained unchanged as `Duel.greedy_response()` solely for the comparison runner. `verify_opponent.py` plays it as the agent against WindBot on the same declared seeds, deck choices, seats and shuffles. It also compares the same aggressive baseline against old and new opponents on held-out scenarios. Results and all individual strata are recorded in `runtime/verification/opponent-v3.json`.

## Scenarios, rules and seeds

`decks.json` version 1 freezes the **exact three study recipes**, resolved against the pinned card database: `normal_pressure`, `flip_control`, and `tribute_pressure`. Each has 24 monsters plus the shared 16-card Spell/Trap package, exactly 40 cards, and empty Extra/Side Decks. Canonical card IDs were resolved by exact English name with artwork aliases excluded; no substitutions were made. This is a custom three-copy format without a banlist, **not a current tournament format**. The rules use Master Rule 5, 8000 starting LP, five initial cards, and the native first-turn draw/battle restrictions.

Each episode comprises **three independent duels**, advancing automatically after each result. A completed episode score is their mean. `SEEDS` declares 12 train (`1101–1112`), 12 held-out (`2201–2212`), 12 validation (`4401–4412`) and two development (`3301–3302`) seeds. The full splits each cover all **18 ordered deck-pair/agent-first-or-second strata twice**. The first six seeds retain revision 2's original scenarios; the second six add another balanced block. Development uses six exploratory strata. Other integer seeds also work.

`scenarios.py` creates the lexicographic 18-stratum list and shuffles it with Python `Random(int.from_bytes(SHA256("yugioh-benchmark-v1:0:schedule:" + split), "little"))`. Three consecutive triples go to each of the first six seeds. For the next six, it reshuffles that first permutation with the independent domain `schedule:{split}:block-2`. Seat 0 starts; the chosen seat determines whether the agent starts.

For seed `s` and zero-based duel `r`, derivations are SHA-256 of UTF-8 `yugioh-benchmark-v1:{s}:{domain}`:

- `round-{r}:engine`: four little-endian uint64 values initialize the native RNG.
- `round-{r}:shuffle-player-{p}`: little-endian digest integer seeds Python `Random` to shuffle each deck before insertion. Subsequent in-game shuffles use the native RNG.
- `round-{r}:opponent`: first four digest bytes, little endian and masked to 31 bits, initialize WindBot `System.Random`. The previous greedy comparison uses its original Python RNG from the full digest.
- `scenario-selection`: undeclared seeds independently select three deck/seat triples.

Seed, stream state, scenario mapping and hidden opponent deck selection stay in evaluator-only `info()`/private process memory. Policy observations never contain them. Response timing does not influence shuffles or opponent choices.

## Observation and actions

`describe()` provides rules, public recipes, card text availability, observation fields and the action grammar. `step()` blocks through native/opponent processing to the next agent prompt, including prompts during the opponent's turn.

- `choose TOKEN INDEX`: copy an exact `prompt.choices[].action`.
- `select TOKEN I,J,...`: choose distinct indices satisfying the displayed count/tribute constraints; `-` means empty.
- `order TOKEN I,J,...`: give a complete candidate permutation. Chain links resolve in reverse order.
- `noop`: wait at the current prompt without resetting its deadline.

Malformed, stale or illegal agent actions leave the decision pending and report `last_action_error`; they never reach the core. Mandatory singleton choices advance automatically. Prompts cover main/battle commands, yes/no, options, chains, card subsets, zones, positions, Tributes, select/unselect and ordering. Compound choices are represented by exact constraints, never silently truncated.

Strict-JSON observations expose seat, turn/phase, LP/counts, own hand and face-down identities, public zones/current stats, relationships, chain links/targets, last 40 public/own events, local card catalogue and the legal prompt. Unknown cards omit identity/stats. Slots identify current zones, not permanent hidden-card identities. Opponent hand/deck content, raw queries, logs and private diagnostics are excluded. A policy may remember past public events.

## Score, error handling and lifecycle

Each completed native duel scores `w + 0.1*clamp((agent_LP-opponent_LP)/8000,-1,1)`, where `w` is 1/−1/0 for win/loss/true draw. The engine's winner overrides LP, including deck-out/alternate wins. LP is clamped to zero on lethal damage. Episode score is the sum divided by three; unfinished duels contribute zero. `info().results` preserves native outcome/reason, LP, turns, decisions, deck/seat stratum, invalid-action count and flags. `score_breakdown()` separately reports win rate, outcome and LP terms, native wins/losses/draws, forfeits, truncations and technical errors.

The default **300-second agent-response deadline** starts when the observation is ready. Invalid actions/noops do not renew it; opponent/native processing is excluded. Expiry forfeits the duel on the next API call and is marked as a forfeit, never a native result. A 200-turn or 10,000-native-response cap produces a separate benchmark truncation with score zero, not a draw. Test-only kwargs can adjust caps/deadlines, but cannot weaken the opponent.

Native crashes, script errors, malformed engine data, engine retries, WindBot faults or watchdog expiry mark `info()["valid_trial"] = False`, with `invalid_reason`, and invalidate the whole episode. Default native-worker watchdog is 30 seconds; individual bot responses have a 10-second watchdog. A supervised worker owns each native duel and its Mono process. `close()`/reset close pipes, reap child processes, and kill the worker's process group if necessary. `close()` is idempotent. Generated outputs/crash files/logs go under `runtime/`; play does not write source or update assets.

## Validation and limitations

The requested `validate_adaptor(..., expected_tier=1)` command passed with no warnings using the evaluator Python. `verify_offline.py` runs the native integration suite under an inherited Linux seccomp filter that **denies socket creation**, including in every Mono child. This tests actual offline execution, not just a configuration claim. The suite covers all 18 deck/seat strata for each baseline, all 13 effect-bearing cards, multi-link chain order, hidden-information filtering, full episode deterministic replay, malformed responses, worker crash/watchdog handling, missing scripts, LP-zero/deck-out/tribute/Flip/revival rules, forfeits and truncations. Reports are under `runtime/verification/`.

Scope is the exact 24-card pool. Extra Deck mechanics, arbitrary modern decks, material-sum/announcement prompts, multiplayer and a TCP EDOPro client/server are outside this adaptor. Adding cards requires another manifest and fresh coverage validation. WindBot is a scripted engine, not a learned universal Yu-Gi-Oh champion. Measured win rates below concern these fixed recipes and opponents only.

## Measured results (2026-10-30)

Final revision 3 comparison: **WindBot won 106/108 duels (98.1%) against the previous greedy opponent**, including **36/36 held-out duels (100%)**. These 108 duels cover every ordered deck matchup and starting position six times across train, validation and held-out splits. All **216 comparison duels** completed natively, with zero technical errors, truncations or draws.

| Agent baseline | Opponent | Split | Opponent wins | Opponent win rate | Mean agent score |
|---|---|---|---:|---:|---:|
| greedy-tactical-v1 | windbot-fixed-pool-v3 | train | 35/36 | 97.2% | -1.0270 |
| greedy-tactical-v1 | windbot-fixed-pool-v3 | validation | 35/36 | 97.2% | -1.0322 |
| greedy-tactical-v1 | windbot-fixed-pool-v3 | heldout | 36/36 | 100.0% | -1.0894 |
| aggressive | windbot-fixed-pool-v3 | heldout | 31/36 | 86.1% | -0.7880 |
| random | windbot-fixed-pool-v3 | heldout | 36/36 | 100.0% | -1.0985 |
| aggressive | greedy-tactical-v1 | heldout | 12/36 | 33.3% | 0.3625 |

The aggressive baseline won **5/36** held-out duels against WindBot, versus **24/36** against the old greedy opponent on the identical scenarios. Random play won **0/36**. The benchmark therefore has substantial improvement headroom; a competent simple script does not saturate it. These are finite benchmark samples, not general tournament-strength estimates.

Per-stratum WindBot wins against greedy, pooled across all three full splits (six duels per cell):

| Greedy deck | WindBot deck | Greedy starts | WindBot starts |
|---|---|---:|---:|
| normal_pressure | normal_pressure | 6/6 | 6/6 |
| normal_pressure | flip_control | 6/6 | 5/6 |
| normal_pressure | tribute_pressure | 6/6 | 6/6 |
| flip_control | normal_pressure | 6/6 | 6/6 |
| flip_control | flip_control | 6/6 | 6/6 |
| flip_control | tribute_pressure | 5/6 | 6/6 |
| tribute_pressure | normal_pressure | 6/6 | 6/6 |
| tribute_pressure | flip_control | 6/6 | 6/6 |
| tribute_pressure | tribute_pressure | 6/6 | 6/6 |

Every held-out stratum was **2/2 WindBot wins** against greedy. `runtime/verification/opponent-v3.json` includes individual native results, LP, turns, decisions, scores, prompt coverage and seed/round identifiers for each stratum; it is evaluator-private.

The separate socket-denied integration run passed 36 baseline duels plus rule/failure/replay fixtures, exercised all 13 effect-bearing cards and nine multi-link chains, and observed a maximum public observation size of **15,635 bytes**. Two additional public-interface random episodes completed all six duels legally, with scores **−1.09958** and **−1.10000**. Normal closes and watchdog/crash tests left no running native workers or Mono bots.

```bash
# From the repository root:
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/yugioh/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('runs/yugioh/adaptor', expected_tier=1).to_markdown())"
# From this adaptor directory:
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/yugioh/bin/python verify_offline.py
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/yugioh/bin/python verify_opponent.py
```
