Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `Yu-Gi-Oh! EDOPro — offline three-duel benchmark` — Play three independent Yu-Gi-Oh! duels using the real EDOPro core (Master Rule 5), against WindBot (fixed-pool executor v3). Each duel starts at 8000 LP, five cards, empty Extra/Side Decks. The first player skips the first draw and cannot battle on turn one. This is a custom three-copy format with no tournament banlist. Normal/Tribute Summons, setting, positions, battle, Flip effects, targeting, costs and chains follow the card text and native rules. Reduce opposing LP to zero or win by another native rule (including opponent deck-out). Chain effects normally resolve last-in-first-out; destroying a source does not by itself negate its effect.
Each step answers one decision, not one entire turn. Answer even on the opponent turn. Copy an action string from prompt.choices: "choose TOKEN INDEX". For a subset prompt use "select TOKEN I,J,..." with distinct candidate indices satisfying the displayed constraint; use "-" for empty. Tribute selections require summed weights >= minimum and count <= maximum. A choice may explicitly cancel/finish. For an order prompt use "order TOKEN I,J,..." containing all candidate indices in desired order, first chain link / first deck position; chain links resolve in reverse order. Mandatory singleton choices advance automatically. "noop" waits; invalid or stale actions leave the decision pending and report last_action_error. Tokens identify the current round and prompt. Never invent native responses.
Observations include your seat; turn/phase; both players' LP, zone counts and public field/graveyard; your hand and face-down cards; current stats, positions, targets and equips; current chain; the last 40 public/own events; and the prompt with legal choices or exact constraints. card_catalog gives local text and base statistics for all 24 pool cards (type/race/attribute are engine bitmasks). Unknown cards have no identity. Opponent hand contents, face-down identities, deck order, seed and RNG state are private. Slot indices identify current zones, not permanent card identities. Own monster fields include current attack/defense. Effect hints explain targets and decisions. The deck pool is normal_pressure, flip_control and tribute_pressure; each has 24 monsters and the identical 16-card Spell/Trap package. The full public recipes are available in decks.json; no modern/Extra Deck cards are used.
After each duel the next starts automatically; completed_duels preserves outcomes. Final score is the mean of three duel scores: win=1, loss=-1, true draw=0, plus 0.1 times LP difference/8000 clipped to [-1,1]. Native winner overrides LP. A 200-turn or 10000-engine-decision cap is a benchmark truncation scoring zero. Each pending agent decision has a 300-second default response deadline, including retries; exceeding it forfeits that duel. Engine/opponent processing is excluded. Technical faults invalidate the trial. Win rate and per-matchup/seat results are recorded separately for evaluation.
Public fixed deck pool:
normal_pressure: 3x Alexandrite Dragon, 3x Gene-Warped Warwolf, 3x Gemini Elf, 3x Luster Dragon, 3x Vorse Raider, 3x La Jinn the Mystical Genie of the Lamp, 3x Battle Ox, 3x Mystical Elf, 2x Mystical Space Typhoon, 2x Book of Moon, 2x Fissure, 1x Dark Hole, 1x Monster Reborn, 2x Rush Recklessly, 2x Trap Hole, 1x Mirror Force, 2x Compulsory Evacuation Device, 1x Call of the Haunted
flip_control: 3x Alexandrite Dragon, 3x Gene-Warped Warwolf, 3x Gemini Elf, 3x Mystical Elf, 3x Giant Soldier of Stone, 3x Man-Eater Bug, 3x Penguin Soldier, 3x Magician of Faith, 2x Mystical Space Typhoon, 2x Book of Moon, 2x Fissure, 1x Dark Hole, 1x Monster Reborn, 2x Rush Recklessly, 2x Trap Hole, 1x Mirror Force, 2x Compulsory Evacuation Device, 1x Call of the Haunted
tribute_pressure: 3x Summoned Skull, 3x Blue-Eyes White Dragon, 3x Luster Dragon, 3x Vorse Raider, 3x La Jinn the Mystical Genie of the Lamp, 3x Battle Ox, 3x Mystical Elf, 3x Giant Soldier of Stone, 2x Mystical Space Typhoon, 2x Book of Moon, 2x Fissure, 1x Dark Hole, 1x Monster Reborn, 2x Rush Recklessly, 2x Trap Hole, 1x Mirror Force, 2x Compulsory Evacuation Device, 1x Call of the Haunted

Actions:
- choose: choose TOKEN INDEX: submit the exact action string from prompt.choices.
- select: select TOKEN I,J,...: choose a legal subset of prompt.candidates; - means empty.
- order: order TOKEN I,J,...: order all candidates, first to last.
- noop: Leave the current decision unchanged; the response deadline still runs.
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
One step answers one engine prompt, including responses on the opponent turn. Tokens reject stale choices. Bare parameterized action names are harmless invalid actions.

Model available through `ctx`: `ctx.think(prompt, session_id=...)` — Astra (System-2). Budget: 40 calls
per episode; one call may return a plan of several actions. `ctx.act(action)` applies one action and
returns the next observation. Each episode has a wall-clock deadline of 1800 s, including model
latency (`ctx.budget.remaining()["wall_s"]`); when it passes, or the call budget is used up, the episode
ends with the score reached. Each episode starts cold: fresh `ctx.memory` and fresh module globals.

Evaluation: mean score over the training seeds (the game's own score, see the description), then fewer
System-2 calls, then fewer actions. A candidate can only be selected if none of its training episodes
ends with an error (see the contract: e.g. a TimeoutError from a model call that the harness does not
catch).

What the harness must design: how observations are shown to Astra (full history, a diff, an ASCII map
with a legend, only what changed), what to remember between calls (rules learned, states visited,
plans), how to parse Astra's reply into actions robustly, how many actions to commit per call, when
cheap scripted search or bookkeeping replaces a model call, and how to notice that a plan or belief was
wrong. If the description says that rules or action effects are not given, discovering them by
experiment is part of the task.

A plan must be able to fail and recover. Whatever executes without the model (a compiled build order, a
convention policy, a solver) must check its own effects after each step — `ctx.progress()` counts the
actions since the score and the observation last changed and the actions the game rejected — and, when
the evidence says the plan has stopped working, hand the situation with that evidence to Astra for
re-planning, then verify the new plan the same way. Scenarios the training seeds never show are where a
scripted plan breaks: a candidate with identical scores on every training seed is a scenario-independent
script and will be judged on the unseen validation and held-out seeds.

**Evaluation scope.** The harness is selected on the training seeds of this game and then reported on
unseen seeds of the same game; there is no held-out game. Rules, strategies and search that the game
description or the study establish may be implemented as code once they are verified against
observations — but nothing specific to the training seeds.

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules, bookkeeping and search as code and put planning and adaptation into Astra prompts — but verify
every claim against observations, because sources can be wrong or outdated for this build of the game.

The full documents are readable in this archive under `prior/` (`prior/strategies.md`, `prior/brief.md`, `prior/interfaces.md`, `prior/study.json`, `prior/ADAPTOR.md`). Below: excerpts within a size budget; every cut is marked `…(truncated; full text in prior/<file>)`.

### study/study.json (summary)
- strategy_summary: Code must enforce legal responses, exact battle arithmetic, selection constraints, public knowledge tracking, and prompt ownership while the engine resolves rules. Astra should prioritize established wins and survival, then useful resource exchanges, meaningful interaction, and next-turn follow-up. In the initial pool, pressure decks use efficient low-level attackers, flip-control decks seek removal and recovery value, and tribute decks preserve bodies only when a larger threat justifies the investment. Sequence removal, summons, and attacks around plausible hidden responses, and remember that destroying an effect source usually does not negate an already activated effect. Preserve interaction for the threat that matters rather than activating every available card or overextending into mass removal. Jev is optional for typed role/threat/relevance judgments on known information; it should not plan combos, calculate lethal, or infer exact hidden cards. Evaluate improvements over held-out seeds and both starting positions using win rate and completion/error rates, not LP margin alone.
- scoring: Report native win/loss/draw and terminal reason, LP, turns, decisions, and error/truncation flags. For completed duels use w + 0.1*clamp((agent_LP-opponent_LP)/8000,-1,1), with w=1 for a win, -1 for a loss, and 0 for a true draw; also report win rate separately. Engine winner overrides LP, including deck-out and alternate wins. Proposed 300-second agent-response timeout is an identifiable forfeit, while engine/script/parser failures are technical errors. Proposed 200-turn or 10000-decision cap produces a separately reported benchmark truncation with score 0, not a claimed native draw.
- interface.observation: Proposed JSON: episode and prompt tokens, state version, agent/acting/turn seats, turn and phase, LP and zone counts, own legal private information, public field/Graveyard/face-up banished cards, current card text/stats and relationships, chain links/costs/targets, public event history, known restrictions, and the current request with legal actions or exact selection constraints. Unknown opponent hands, face-down identities, deck order, secret seed, raw native queries, and private logs are excluded. Native messages are binary and require an adaptor; the core does not directly supply this JSON.
- interface.actions: Choose one opaque legal action ID tied to the current prompt token, or a validated constrained subset/permutation for a compound prompt. Choices cover summon/set/position/activation/attack/phase progression, chain response or legal decline, costs/targets/tributes/materials, zones, positions, options, numbers, and announcements. The bridge packs the matching native response; step then processes events and opponent prompts until the next agent decision or terminal result, including decisions during the opponent's turn.

### study/strategies.md
# Strategy prior for Astra, optional Jev, and exact-rule code

## Scope, evidence, and division of work

This is a strategy prior for ordinary Yu-Gi-Oh! duels and the **three fixed beginner/intermediate decks in `interfaces.md`**. It is not a claim about the best tournament decks in October 2026. The initial pool deliberately omits Extra Deck combos, hand traps, and archetype search engines. Modern-game advice below is marked as an expansion path, not assumed to apply to the baseline.

Web retrieval failed in this session. The bibliography contains official and community reference URLs that must be retrieved and archived when access is available; no current article, tournament result, deck statistic, or source quotation is claimed to have been read. Rule-sensitive statements should be checked against the pinned card text and engine. Tactical recommendations are a synthesis of established game concepts and the proposed pool, not measured performance guarantees.

**Code owns exactness:** legal moves, state/knowledge bookkeeping, all arithmetic, constraints, costs/materials, effect-use tracking where observable, legal response encoding, seed handling, timeout policy, and score. The engine and matching scripts are the final rules authority. Search code, if added, must reason only from the agent's information set.

**Astra owns plans:** choose a win condition, identify the useful resource exchange, sequence plays around interaction, infer plausible opposing threats, choose which interruption matters, and preserve follow-up. It should explain a short plan to itself and return an action from the actual prompt rather than inventing a complete turn.

**Jev is optional:** this is Tier 1, with no need for a reflex controller. Use it only for independent typed judgments that reduce context cost, such as classifying a card's role, recognizing an exposed threat, or rating the relevance of a candidate interruption. Do not ask Jev to calculate lethal, solve material subsets, infer exact hidden cards, or plan a combo. All of its outputs are advisory and restricted to known cards and supplied legal candidates.

## Priority order and evaluation

1. **Code: obey the current request.** A mandatory cost/target/trigger request interrupts every broader plan. A legal optional decline can be best, but an illegal “pass” is never acceptable. Preserve effect IDs, source instance, selected targets, and prompt token. [S1, S2, S4]
2. **Code + Astra: take a genuinely established win.** Code evaluates legal damage or alternate-win opportunities from public/owned information, with explicit assumptions about unknown interaction. Astra chooses whether extra protection or an initial bait is worth delaying the attack. “Known board lethal” is not guaranteed lethal through an unrevealed Trap. [S1, S3]
3. **Astra: prevent a likely immediate loss.** Identify the opponent's actual route to lethal or a decisive lock, then use the least costly sufficient answer. Do not spend every interruption on the first effect simply because it is activatable. Code supplies projected damage and restrictions. [S2, S5, S6]
4. **Astra: improve usable resources and position.** Count what remains useful after the exchange: cards in hand/field, live Graveyard effects or revival targets, an unused Normal Summon, available zones, threats, defenses, and next-turn plays. Raw card count is informative but not decisive. [S5]
5. **Astra: preserve follow-up once the board is adequate.** An extra summon or effect activation that does not improve the expected outcome may expose another card to mass removal or consume next turn's starter. LP are a resource, not a reward to hoard at the expense of winning. [S5, S6]

A useful evaluation order is: terminal outcome → imminent survival/lethal → live threats and interaction → resource/follow-up quality → LP margin. Do not turn the benchmark's small LP tie-breaker into an instruction to heal, avoid all costs, or stall a winning game.

## Opening plans for the fixed pool

The first player starts with five cards, skips their first draw, and has no first-turn Battle Phase. Going first usually favors setting up useful interaction; going second offers an additional normal draw and a chance to attack. Which is better depends on the deck matchup and actual hand. The benchmark randomizes starting position, so Astra must handle both. **Code** supplies turn and first-turn restrictions; **Astra** chooses the plan. [S1]

| Deck | Opening and follow-up plan — Astra | Exact support — code | Optional Jev judgment |
|---|---|---|---|
| `normal_pressure` | Establish a strong low-level attacker; back it with one or more useful defensive interactions according to the hand and threat of mass removal. Going second, use removal to turn bodies into damage. Keep another summon in reserve if committing it adds little. | Compare actual ATK/DEF, calculate battle results and lethal assumptions, track Normal Summon and attack availability. | Which known opposing card most threatens the current attacker? Is this legal move pressure, protection, or unnecessary commitment? |
| `flip_control` | Against a stronger attacker, a useful flip Set can generate removal or recovery instead of losing a weak face-up monster. Protect it if doing so preserves a valuable next turn. Recover the Spell that supports the present plan, not automatically the flashiest Spell. Transition to attackers after trading resources. | Track public targets, own Graveyard Spell availability, whether a Set/position change is legal, effects already used, and cards pending destruction. | Does this known flip effect offer removal, bounce, recovery, or no immediate value? Which already-legal target is strategically valuable? |
| `tribute_pressure` | High-level monsters are not openers without bodies or an alternate legal summon. Keep low-level monsters alive for Summoned Skull or Blue-Eyes, and exploit legal revival when a large monster reaches a Graveyard. A Tribute Summon is worthwhile when its resulting threat is worth the bodies spent. | Count actual required Tributes and legal tribute candidates; compare before/after battle damage and resources. Do not assume every Level 7+ card has identical special conditions. | Which candidate tribute body is least useful next turn? Is a big monster a decisive threat or just a small ATK improvement? |

Concrete baseline examples, subject to the engine's actual offered choices:

- **Astra:** do not Tribute two productive attackers for Blue-Eyes merely because it has the highest printed ATK. **Code:** compare the existing legal attacks with the 3,000-ATK body, and account for lost cards, blockers, and remaining battle opportunities. If the existing monsters can attack profitably, attacking before a Main Phase 2 Tribute can preserve damage; if they would die or expose a fatal response, that line may be worse. [S1, S5, C1]

…(truncated; full text in prior/strategies.md)

**Astra evaluation:** compare win rate, completed-duel rate, illegal/stale response rate, truncation rate, and per-deck/seat results against the baseline over the same held-out seeds. Inspect failures such as MST-as-negation, spending a removal on a low-value target, missing known-board lethal, wasting a Normal Summon, losing flip value, or overextending into Mirror Force. Improvements should survive multiple opponent seeds and both first/second positions. LP margin alone is a poor measure of strategy quality. The correct deck plan may differ when the opponent changes from random to scripted.

## Reference URLs and how to use them

**None were fetched in this session.** These are stable reference targets from prior knowledge; availability and current content remain to be checked. Use official rules/card text and the pinned engine for rulings, community pages for explanations, and the explicit benchmark proposals above for engineering priorities.

- **S1:** Official rulebook — https://www.yugioh-card.com/en/rulebook/ . Basic rules, summon/battle/turn structure, Spell/Trap types.
- **S2:** Official Fast Effect Timing — https://www.yugioh-card.com/en/play/fast-effect-timing/ . Turn-player priority and response windows.
- **S3:** Community public/private knowledge overview — https://yugipedia.com/wiki/Public_knowledge . Verify detailed visibility against reference-client behavior.
- **S4:** Community Chain and PSCT references — https://yugipedia.com/wiki/Chain and https://yugipedia.com/wiki/Problem-Solving_Card_Text . Resolution order, costs/targets, conjunctions, and activation versus effect distinctions.
- **S5:** Community card-advantage concept — https://yugipedia.com/wiki/Card_advantage . A vocabulary for exchanges, not a complete evaluation function.
- **S6:** Community Damage Step reference — https://yugipedia.com/wiki/Damage_Step . Detailed battle timing and restricted activations.
- **S7:** Community hand-trap concept — https://yugipedia.com/wiki/Hand_trap . General interaction vocabulary, not a current tier list.
- **S8:** Official Forbidden/Limited lists — https://www.yugioh-card.com/en/limited/ . Freeze a dated list if adopting a tournament format.
- **S9:** Simultaneous effects — https://yugipedia.com/wiki/Simultaneous_Effects_Go_On_Chain . Check regional/version-specific details rather than hard-coding a remembered ordering rule.
- **S10:** Optional timing — https://yugipedia.com/wiki/Missing_the_timing . Confirm each actual card and trigger window.
- **S11:** Once-per-turn wording — https://yugipedia.com/wiki/Once_per_turn . Use the pinned actual text and core behavior.
- **C1:** Official card database — https://www.db.yugioh-card.com/yugiohdb/ . Search the exact card names in the tables, including MST, Mirror Force, Book of Moon, Trap Hole, Fissure, Compulsory Evacuation Device, Dark Hole, Monster Reborn, Call of the Haunted, Man-Eater Bug, Penguin Soldier, Magician of Faith, Rush Recklessly, and any expansion cards.
- **C2:** Script implementations — https://github.com/ProjectIgnis/CardScripts . Pin alongside the core; use implementation and regression fixtures to validate the intended rules rather than relying on prose alone.
- **C3:** Scripted-opponent source — https://github.com/ProjectIgnis/windbot . Once available, inspect a matching executor for concrete policy sequencing and supported decks; no executor was inspected here.

### study/brief.md
# Yu-Gi-Oh! — headless EDOPro benchmark brief

## Scope and evidence

Study date: 2026-10-29. Target: Linux x86_64, Python **3.12.3** at `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/yugioh/bin/python`, offline runtime, free software, no GUI. Recommend ordinary two-player Yu-Gi-Oh! Master Rule duels, not Rush Duel, Speed Duel, tag duels, or a live ranked service.

**Research limitation:** attempts to retrieve GitHub, raw.githubusercontent.com, and the official rulebook site failed with `curl: (6) Could not resolve host`. No browser/search connector is available. No relevant cached core, card database, scripts, or decks were found in the inspected local tool/cache directories. Source links below are reference targets from established documentation, **not pages retrieved in this session**. Rules are established game knowledge; current repository layouts, licenses, build recipes, and current tournament lists must be checked when sources are supplied. No installation or duel test was performed. `interfaces.md` distinguishes the proposed adaptor contract from upstream APIs.

## Objective and rules that affect decisions

- Each player normally begins at **8,000 Life Points (LP)** with five cards. Win by reducing the opponent to 0 LP, making them unable to draw when required, or satisfying a card's alternate victory condition. Simultaneous defeat can draw. Use the **engine's terminal result**, not an LP heuristic, to decide the winner. A Match usually consists of multiple Duels; this benchmark scores **single Duels**, with no side decking. [R1]
- Main Deck: 40–60 cards. Extra Deck: up to 15. Side Deck: up to 15 for matches. Normally at most three copies of a card name across all three, subject to a format's Forbidden/Limited list. The proposed initial benchmark uses exactly 40 Main Deck cards and empty Extra/Side Decks, with an explicitly custom three-copy format. It must not be described as current tournament legal. [R1, R4]
- Turns proceed through Draw, Standby, Main Phase 1, Battle, Main Phase 2, and End Phase, subject to rules/effects. The first player skips the first draw and cannot conduct a Battle Phase on their first turn. Main Phase 2 follows a conducted Battle Phase; do not invent a Main Phase 1 → Main Phase 2 transition. The normal hand limit is six at the end of the turn; effects can change rules. [R1]
- Normally one Normal Summon **or Normal Set** per turn. Level 5–6 monsters usually require one Tribute; Level 7+ usually require two. Special Summons have their own conditions and restrictions, not a universal once-per-turn limit. Attack-position monsters are face-up; face-down monsters are normally in Defense Position. Links cannot be in Defense Position. [R1]
- Extra Deck mechanics, if the pool is later expanded: Fusion uses its specified materials and a suitable summoning method; Ritual usually uses a Ritual effect and required Tributes to summon a Ritual Monster from the Main Deck/hand; Synchro generally combines a Tuner and other listed materials whose Levels total the summoned monster's Level; Xyz generally stacks the listed same-Level materials under a monster with a Rank; Link uses listed materials whose contribution satisfies its Link Rating. Pendulum Summoning uses scales and has location/zone restrictions, especially for face-up Extra Deck monsters. These are only summaries: card-specific methods, alternate materials, and the chosen Master Rule revision control legality. Current ordinary Master Rule placement differs between Fusion/Synchro/Xyz and Link or face-up Extra Deck Pendulum summons. Use the core's exact options rather than deriving material/zone legality from this paragraph. [R1, R6]
- Monsters, Spells, and Traps implement a much larger rules system than their printed ATK/DEF. Costs, activation conditions, targets, effects, continuous modifiers, restrictions, and once-per-turn clauses matter. The core and the matching Lua scripts are the authority. A language model must never validate legality from card prose alone. [R1, R3, R6]
- Battle: attacking an Attack Position monster normally compares ATK; the weaker monster is destroyed and its controller takes the difference. Attacking a Defense Position monster compares attacker ATK with defender DEF; destruction and damage differ, and piercing is an effect rather than a universal rule. Direct attacks usually require an empty opposing monster field. Effects and Damage Step rules can change these outcomes. Have code calculate projected damage, and the engine resolve it. [R1, R5]

…(truncated; full text in prior/brief.md)

No intermediate LP reward is required. Paying LP for a strong effect can be correct; LP gain alone is not strategic progress. Maintain separate counts of successful turns, legal responses, and terminal completions for adaptor diagnostics. Proposed deterministic safety caps are 200 turns or 10,000 response decisions; label exceeding either `truncated`, report a benchmark draw with score 0, and publish its rate separately. Never convert an unsupported prompt, missing script, or crash into a game draw. Benchmark timeouts may score as forfeits but must remain identifiable.

## Sources

- **R1 — Official rulebook landing page:** https://www.yugioh-card.com/en/rulebook/ — foundational rules, turn phases, zones, card types, summoning, battle, chains. Obtain a local copy before final rules validation.
- **R2 — Official Fast Effect Timing:** https://www.yugioh-card.com/en/play/fast-effect-timing/ — open game state and response opportunities.
- **R3 — Community Problem-Solving Card Text reference:** https://yugipedia.com/wiki/Problem-Solving_Card_Text — conditions, costs/targeting, resolution language; use official card text/rulings for a disputed interaction.
- **R4 — Official Forbidden & Limited Lists:** https://www.yugioh-card.com/en/limited/ — relevant only if a dated tournament format is adopted.
- **R5 — Community Damage Step reference:** https://yugipedia.com/wiki/Damage_Step — restricted activation windows and battle timing.
- **R6 — EDOPro native core:** https://github.com/edo9300/ygopro-core — authoritative native lifecycle, processing, query, and response interface once pinned and inspected.
- **R7 — Official card database:** https://www.db.yugioh-card.com/yugiohdb/ — card text and card-specific information; not an offline duel service.

### adaptor/ADAPTOR.md
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

…(truncated; full text in prior/ADAPTOR.md)

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

### study/interfaces.md
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

…(truncated; full text in prior/interfaces.md)

- https://github.com/ProjectIgnis/CardScripts — helper/startup scripts and pinned per-card rules implementations.
- https://github.com/ProjectIgnis/BabelCDB — database layout and card metadata/text assets.
- https://github.com/sbl1996/ygo-agent — potential environment-only shortcut; inspect exact Python/native dependencies, observation/action conventions, supported formats, and tests.
- https://github.com/ProjectIgnis/windbot — optional scripted opponent, executor/deck coverage and host protocol requirements.
- https://github.com/Fluorohydride/ygopro-core and https://github.com/Fluorohydride/ygopro — coherent legacy fallback and reference protocol; do not cross-wire with modern EDOPro.
- https://www.lua.org/ — required Lua version/build/license once the core dependency is known.
- https://docs.python.org/3.12/library/ctypes.html and https://docs.python.org/3.12/library/sqlite3.html — Python 3.12 standard-library FFI/database interface.
- https://www.yugioh-card.com/en/rulebook/ and https://www.db.yugioh-card.com/yugiohdb/ — rules and official card text cross-checks.


## Measured model latency (at archive creation, 2026-10-30 21:32)
- System-2 (Astra), one-word answer: 35 s
- System-2, an answer of ~1029 output tokens: 78 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).