Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `Dominion (offline pyminion)` — Play standard two-player Dominion, Base second-edition cards, against the disclosed
BMU-1 (pyminion BigMoneyUltimate) opponent. Each player starts with 7 Copper and 3 Estate, shuffled,
and draws five cards. Seat 0 acts first. Each turn starts with 1 Action, 1 Buy, $0.
Playing an Action spends one Action and resolves all its text; +Actions enable further plays.
Copper/Silver/Gold produce $1/$2/$3 and are automatically all played after Actions.
Buying costs coins and a Buy and normally gains to discard. Gains from Actions do not spend
coins or Buys. Cleanup discards hand and played cards and draws five; shuffle discard only
when the draw deck is exhausted. Attacks may ask you to discard or reveal Moat during an
enemy turn. Moat protects only when in hand and revealed for that attack.
Finish the entire turn before checking for an empty Province pile or any three empty piles.
Most VP wins; equal VP favors fewer turns taken, otherwise draw. All owned cards count,
including Curses (-1 each) and Gardens (floor(total cards/10) per copy). Objective: win and
maximize final VP margin; score is the unweighted VP margin, win_score is reported separately.
Supply has 8 of each Victory (including Gardens), 10 other Kingdom cards, 10 Curses,
46 Copper, 40 Silver, 30 Gold after starting decks. Exact card costs/types/text are in supply.

Each step answers one pending prompt, then advances to your next choice or game end.
Use `choose DECISION_ID OPTION_ID ...`, selecting distinct option IDs within prompt min/max;
`choose DECISION_ID` or `pass` declines only if min=0. `wait` is a true no-op.
Alternatively send {"decision_id": integer, "option_ids": ["0", ...]}.
IDs identify indistinguishable card occurrences locally to this prompt, not persistent cards.
False/True choices are explicit labeled options. Sentry inspected cards are listed top first;
its final True swaps the two remaining cards. Forced sole responses resolve automatically.
Invalid/stale actions leave the game unchanged and set last_error; there is no automatic fallback.

Observations: viewer and active/decision player, seat, phase, per-player turns STARTED,
active player's resources, kingdom/opponent IDs, exact supply with rules, own hand/discard
multisets, public played cards, hand/deck/discard counts, public ownership and exact known VP,
trash, ordered public_log_delta since the last decision, prompt, derived pile/VP arithmetic,
and terminal result. Repeated observe() is stable; log deltas are not consumed by observe().
Opponent hand, both unrevealed deck orders and RNG/seed are hidden. episode_id is a constant
non-secret label; decision_id increases within an episode. Do not reuse responses across resets.
BMU-1 is the engine's BigMoneyUltimate policy in every scenario. It plays Smithy.
Its first affordable nonempty buy priority is: Province if total Treasure money in its deck
exceeds $15; Duchy at <5 Provinces; Estate at <3; Gold; Duchy at <7;
Smithy if Smithies owned < Treasure cards owned / 11; Silver; otherwise stop.
It can own multiple Smithies. It uses upstream defaults for forced choices, including
discarding the first required cards in its hand; it does not use hidden opponent information.
Safety limits end as truncated, never as a Dominion victory. There is no decision time limit.

Actions:
- choose: choose DECISION_ID OPTION_ID ... : select distinct prompt options in order.
- pass: Decline an optional prompt (min=0); otherwise report an illegal action.
- wait: Leave the pending decision and game state unchanged.
(real-time games: ticks apply 'wait' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
One step answers one decision, including reactions during the opponent turn. Use prompt option IDs, not card names. choose ID with no options means pass. JSON {"decision_id": ID, "option_ids": ["0", ...]} is also accepted.

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
- strategy_summary: Analyze each kingdom for thinning, draw, action capacity, payload, extra buys or gains, attacks and scoring routes before selecting an opening. Efficient money decks with a little terminal draw are a useful baseline, while strong engines require balanced components and a fast enough payoff. Trash early junk while preserving functional economy, and adapt to Witch or Militia pressure rather than treating defense as permanent immunity. Sequence actions to preserve action capacity and improve draws, and monitor shuffle timing without inspecting hidden deck order. Switch from building to scoring according to remaining Provinces, three-pile threats and opponent capability, with exact ending and tiebreak calculations in code. Astra owns strategic planning and adaptation; code owns rules, arithmetic and bookkeeping; optional Jev questions only recognize patterns or rank already legal choices under the plan.
- scoring: Report agent_vp, opponent_vp, integer vp_margin, outcome and win_score (1 win, 0.5 draw, 0 loss). Final VP includes Curses and variable scoring such as Gardens across all owned zones. Resolve equal VP by fewer turns taken, then shared draw if turns are equal; zero margin can therefore be a win. Track end reason, turns, illegal/fallback counts and truncations separately. Aggregate win rate and mean VP margin across paired seats, kingdoms and opponents; do not replace the requested objectives with money, deck size or a fabricated weighted scalar.
- interface.observation: Proposed adaptor JSON, not an asserted upstream API: episode and decision IDs; viewer/turn owner/decision player; phase and resolving effect; Actions, Buys and money; exact kingdom and remaining Supply; own hand, discard and played cards; deck counts and publicly inferable ownership; opponent counts and public events only; legally known reveals/top cards; a typed pending prompt with legal option IDs and selection constraints; code-computed visible VP and pile pressure. At genuine termination expose full final scores, turns and result. Do not reveal hidden orders, opponent hand or seed.
- interface.actions: Answer one pending prompt using its decision ID and legal option IDs, with ordered selections where required. Cover Action play/pass, Treasure play or validated Base-only play-all, buy/stop, constrained gain, trash, discard, topdeck/order, optional reaction and effect-specific mode/target choices. Validate before mutation, return the same prompt on invalid input, then advance forced effects and scripted opponents until the next agent decision or terminal state. Game rules remain owned by the simulator.

### study/strategies.md
# Dominion strategy prior for Astra and the harness

Scope: two-player Base-set Dominion, preferably second edition, with a tested card whitelist. This is strategic guidance, **not gameplay code**. Claims synthesize established community knowledge; network access was unavailable, so cited pages were not read live and no win-rate or simulator-performance claim is made. See `sources.md`. Exact card text and legality always outrank a heuristic.

## Division of responsibility

- **Code:** rules, legal option enumeration, shuffle/zone bookkeeping, card counts, cost/VP arithmetic, turn counts, public-event memory, exact visible-state endgame calculations, serialization/privacy, deterministic baseline policies, and optionally legal rollout/search. Never ask Jev to count cards, add money, determine legal costs, or compute a win from scores.
- **Astra prompts:** kingdom analysis, choosing money versus engine versus alternate scoring, selecting an opening, acquisition order, balancing components, opponent adaptation, deciding when to stop building and score, and revising the plan after draws/attacks. Tier 1 permits Astra at every meaningful choice.
- **Jev questions (optional):** recognize a situation or rank **already legal** options from supplied features and Astra's current plan. It does not invent a multi-turn plan, implement rules or calculate. Batch independent judgments about the same frozen observation; do not batch questions whose answers change what is subsequently drawn/revealed. Prefer no Jev dependency in the first working adaptor.

Astra should maintain a small plan record: chosen archetype; desired next acquisitions in order; draw/action/payload deficits; trash priorities and stopping rule; threats from opponent; likely ending; and conditions to switch to scoring. Update it at least each own turn and on a significant public attack/pile change. Give Astra current card text, legal options, counters, ownership ledger, remaining piles and prior plan. Never require it to reconstruct the game from raw prose alone.

## Read the kingdom before buying

Inventory the board's **trashing**, **draw**, **+Actions**, **+Buy/gaining**, **payload** (money, attacks, direct gains), **defense**, and **scoring/end conditions**. A Village is not an engine by itself; draw must reach the useful payload and enough Actions must let it be played. An engine can be weak if it draws well but buys only one card or cannot produce money. Conversely, an efficient money deck needs few moving parts and wins against slow unfocused construction. [S1, S2, S3]

**Code:** expose role tags and exact effects for implemented cards, terminal-action count, card counts and public pile pressure. **Astra:** choose a coherent build based on combinations and expected game speed. **Jev:** classify a supplied choice as supporting draw, action capacity, payload, thinning, or scoring; these tags can normally be static code instead.

Before committing to a long build, check: can Witch flood the deck with Curses; can the opponent race Provinces; can Workshops/Gardens empty three piles; is there enough +Buy/gaining to exploit a large turn? A board with Chapel + Village + Smithy + Market supports a different plan from a board with Smithy and little useful support.

## Openings and the first reshuffle

The initial two hands normally produce a 4/3 or 3/4 split, or a 5/2 or 2/5 split. This follows from seven Coppers distributed over two five-card hands. The *order* matters for timing and occasionally effects: two purchases usually first become drawable in the first reshuffle, but early effects can change that. Do not hard-code identical openings for every kingdom. [S1]

| Available tools / money | Practical opening prior | Why and caveat | Owner |
|---|---|---|---|
| Chapel, 4/3 | Chapel + Silver is a strong default | Early removal of Estates/Coppers improves future shuffles. A powerful attack or special synergy can change the companion buy. | Astra chooses; code executes legal buys |
| Chapel + Witch, 5/2 | Witch + Chapel is an attractive default | Early cursing plus thinning; not a universal theorem, assess pace/support. | Astra |
| Witch, 5/2 without Chapel | Witch on $5; consider useful $2 support or pass | Early Curse delivery is often decisive. Buying a weak $2 card just to spend money can hurt. | Astra |
| Militia, 4/3 | Militia + Silver is a solid attack/economy default | $2 payload with hand-size attack. Competes with Smithy and other terminals for the one starting Action. | Astra |
| Smithy money plan, 4/3 | Smithy + Silver | One early terminal draw card can improve money density; do not buy many Smithies without actions/cycling support. | Astra; explicit baseline in code |
| No useful action plan | Silver + Silver on 4/3 | Reliable economy; $5 opening often still buys Silver if no worthwhile $5 card. | Code baseline; Astra can improve |
| 5/2 with Sentry | Consider Sentry on $5 | Cantrip thinning/deck control; strength depends on support and competing attacks. | Astra |
| Any opening | Avoid an early Estate or speculative Village with no terminal demand | Unproductive cards slow cycling; cheap does not mean useful. | Astra; optional Jev flags |

A legal $5 hand need not spend $5. Likewise, no-buy is often correct on $2. Initial shuffle luck is not a justification for changing starting decks or supplying extra money. [S4–S8]

## Thinning: remove bad cards while preserving function

Chapel can trash **up to four** hand cards. Early priorities are typically Curses, Estates, then surplus Coppers, but preservation of buying power matters. Trashing all Coppers without a way to gain or buy useful economy can strand a deck. A Chapel opening aims to obtain stronger economy and repeatedly remove junk; trashing four weak cards can be worth missing a purchase when the deck will recover. The exact choice depends on remaining hand, deck composition and next shuffle—not a fixed "trash four" rule. Late in the game, keep needed VP if there is no time for thinning to repay its cost. [S4]

Sentry draws a card and gives +1 Action, then examines the top two cards for trash/discard/keep with ordering choices; it can improve a deck without using up the action chain. Remodel converts a trashed card into one costing **up to $2 more**; Mine optionally upgrades a Treasure to one costing **up to $3 more**, gained to hand. They are not identical to Chapel and are not mandatory upgrades whenever available. Removing Estates via Remodel to useful $4 cards can build; Gold-to-Province Remodel can score or finish; action cost and lost economy must be weighed. [S8; card pages via S12]

**Code:** enumerate valid trash sets/upgrade targets, calculate resulting composition, track known cards and prices. **Astra:** choose a target deck and how far to thin; adapt the priority near the finish. **Jev:** given the current plan and already computed resource-survival features, score legal trash candidates as keep/low/medium/high priority. Jev must not infer exact affordability after trashing from raw card names.

Common failure: treating all Coppers as permanently bad while still needing them, keeping every Estate until they clog repeated shuffles, or using a trashing action after the last useful reshuffle. Thin for future useful hands, not for its own sake.

…(truncated; full text in prior/strategies.md)

| `chapel_engine` | Chapel, Village, Smithy, Market, Festival, Laboratory, Militia, Moat, Remodel, Merchant | Thin/draw/payload planning, engine versus money |
| `curse_race` | Witch, Chapel, Sentry, Village, Smithy, Market, Moat, Militia, Festival, Laboratory | Early attacks, thinning and engine resilience |
| `gardens_rush` | Gardens, Workshop, Cellar, Village, Smithy, Market, Militia, Moat, Remodel, Merchant | Gains, deck-size scoring and three-pile endings |

Use `base_intro` for the first complete adaptor. Only expand to all supported Base cards after covering special prompts (Sentry, Vassal, Library, Bandit, Throne Room, Artisan, etc.). Do not replace unsupported cards randomly inside a named scenario: fail reset clearly or publish a new scenario ID. Include both seats and varied shuffle seeds for every fixture; hold out entire combinations in addition to seeds. Report per-opponent results so a BM exploit is not confused with general competence.

## Optional Jev typed questions

Inputs should include the current legal option list, Astra's plan, exact derived counts/resources, card descriptions, and ending flags. Candidate generation belongs to code. Examples of **independent questions at a frozen state**:

| Question | Typed output | Intended use |
|---|---|---|
| Which current bottleneck is most evident given these computed features? | choice: draw / actions / payload / scoring urgency / none | Feature suggestion to Astra, not an autonomous plan |
| How well does each legal acquisition support the stated plan? | one ordinal score per option: 0 / 1 / 2 / 3 | Cheap candidate ranking; Astra handles strategic exceptions |
| Which of these prevalidated discard sets best preserves the stated next-turn plan? | one candidate ID | Local choice; code has already checked size and supplied playable-card/spend features |
| Is this supplied pattern a terminal collision? | yes / no | Recognition only; code still enforces Actions |
| Which of these legal Action plays most supports “draw while retaining action capacity”? | one option ID | Local ordering suggestion, validated before execution |

Do not ask "How much money is in hand?", "Who wins if three piles empty?", "What is the legal Remodel gain limit?", or "What should we buy for the next five turns?" of Jev. The first three belong in code, the fourth in Astra. Jev failures must not affect rules legality or reveal hidden state.

## Community references

All links below are unverified live in this session; use the official rulebook for rules conflicts.

- [S1] Strategy overview: https://wiki.dominionstrategy.com/index.php/Strategy
- [S2] Big Money: https://wiki.dominionstrategy.com/index.php/Big_Money
- [S3] Engine construction: https://wiki.dominionstrategy.com/index.php/Engine
- [S4] Chapel: https://wiki.dominionstrategy.com/index.php/Chapel
- [S5] Smithy: https://wiki.dominionstrategy.com/index.php/Smithy
- [S6] Witch: https://wiki.dominionstrategy.com/index.php/Witch
- [S7] Militia: https://wiki.dominionstrategy.com/index.php/Militia
- [S8] Sentry: https://wiki.dominionstrategy.com/index.php/Sentry
- [S9] Throne Room: https://wiki.dominionstrategy.com/index.php/Throne_Room
- [S10] Gardens: https://wiki.dominionstrategy.com/index.php/Gardens
- [S11] Workshop: https://wiki.dominionstrategy.com/index.php/Workshop
- [S12] Card index and individual card FAQs: https://wiki.dominionstrategy.com/index.php/Card_list
- [S13] Library: https://wiki.dominionstrategy.com/index.php/Library
- [S14] Council Room: https://wiki.dominionstrategy.com/index.php/Council_Room

### study/brief.md
# Dominion: study brief

Scope: two-player, offline **Dominion (deck-building card game)**, initially the Base set, second edition. Tier **1**. Written 2026-10-29. The target interpreter was checked directly: `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python` is **Python 3.12.3**. No game package is installed.

**Evidence limitation:** this session has no web/browser tool and outbound HTTPS requests failed DNS resolution for PyPI, GitHub raw content, and Dominion Strategy Wiki. No local simulator source was found. Rules and strategy below are established game knowledge, accompanied by references for verification; these URLs were not successfully retrieved in this session. Software APIs, release support, licenses, and current service terms remain unverified. See `sources.md` for the evidence ledger. Do not interpret citations as claims of live browsing.

## Objective and setup

Build a deck with the greatest number of victory points (VP) **when the game ends**, not the most money or cards. Each player begins with 7 Coppers and 3 Estates, shuffles, and draws 5. Players take alternating turns; the shared Supply normally contains basic Treasures, basic Victory cards, Curses, and exactly 10 Kingdom piles. For two players, basic and Kingdom Victory piles normally contain 8 cards; ordinary Kingdom piles contain 10; the Curse pile contains 10. Starting Estates are additional to the Supply's 8 Estates. Initial Copper supply is 60 minus starting Coppers (46 remaining in a two-player game), Silver 40, Gold 30. Exclude expansions and special pile rules initially. [R1, R2]

| Basic card | Cost | Effect / final value |
|---|---:|---|
| Copper | $0 | $1 when played |
| Silver | $3 | $2 when played |
| Gold | $6 | $3 when played |
| Estate | $2 | 1 VP |
| Duchy | $5 | 3 VP |
| Province | $8 | 6 VP |
| Curse | $0 | −1 VP; Curse is its own type, not a Victory card |

Gardens is a Base Kingdom Victory card: **each copy** is worth floor(total cards owned / 10) VP at game end, counting itself and all owned cards. The engine must calculate variable VP. Cards in hand, deck, discard, play, and any other owned zone count; trashed cards do not. [R1, R3]

## A turn and all decision types

1. **Action:** start with 1 Action. Playing an Action consumes one Action, then resolve its printed instructions in order, as far as possible. Some cards draw, give more Actions, give money or Buys, gain cards, trash cards, or attack. The player may stop playing Actions even if more are legal. An Action that adds no Actions is commonly called a terminal Action.
2. **Buy:** start with 1 Buy. Play Treasures to produce money; Action-generated money also counts. Spend money and a Buy per purchase. Unused money/Actions/Buys normally disappear at turn end. Base-only Treasures can be resolved automatically once the player enters Buy; Merchant's first-Silver bonus must still be implemented by the engine. Do not assume arbitrary Treasure ordering is harmless when adding expansions.
3. **Clean-up:** discard hand and cards in play, draw 5 for the next turn. If a draw needs more cards than remain in the deck, draw those remaining, shuffle the discard to make a new deck, then continue drawing. Do not shuffle just because the deck became empty if no draw is requested. If both deck and discard run out, draw as many as possible. [R1, R2]

A gain normally places the card in the discard pile; a purchase is a gain, but gaining is not necessarily buying. Exceptions are on the card: Mine gains a Treasure to hand, Artisan gains to hand then topdecks a hand card. A gain without buying does not consume a Buy or money. Card costs, not produced money, govern Mine/Remodel/Workshop/Artisan limits. Trashing removes cards from ownership rather than discarding them. [R1, R3]

The adaptor needs prompts **within** a turn as well as for action and purchase choices: choose cards to discard, trash, gain or topdeck; order a sequence; pick a mode; optionally reveal a reaction; select an Action to play twice. Militia's victim discards to 3 during the opponent's turn. Moat's reaction is optional and does not play the Moat. Throne Room plays the chosen Action twice without spending another Action; mandatory effects happen twice, with intermediate choices resolved separately. Playing Throne Room again is not a new turn. [R1, R3]

**Edition boundary:** second-edition Base includes Artisan, Bandit, Harbinger, Merchant, Poacher, Sentry, Vassal. First-edition cards Adventurer, Chancellor, Feast, Spy, Thief, Woodcutter are not part of that second-edition pool. Match the manifest to actual simulator support instead of silently mixing editions. [R2]

## End, score, and progress

…(truncated; full text in prior/brief.md)

Useful progress indicators are Province count, number of empty/nearly empty piles, public VP ledger, deck size/composition, deck cycling speed, payload (money/gains/attacks) per shuffle, and ability to buy multiple scoring cards. These are descriptive observations, not replacements for terminal score. Early VP is frequently a poor progress reward because buying green cards weakens future hands.

## Time and controls

This is **tier 1**: a synchronous offline simulator changes state only when a chosen action resolves or the simulator advances forced effects. There is no physics tick, built-in decision deadline, or deterioration while Astra thinks. Human online clocks and tournament slow-play rules are irrelevant to this chosen offline setup. Astra may take seconds to a minute on every nontrivial decision without in-game penalty; Jev is optional.

`study.json` uses `decision_latency_s = 86400.0` as a **conservative representable lower bound (one day allowed), not a rules-imposed timeout**. The actual rules-level limit is unbounded, which strict JSON cannot encode as Infinity. Any evaluator/model timeout must be tracked separately as infrastructure policy. Disable a game clock; permit suspension at every nested prompt.

Programmatic controls will be legal option IDs passed to the simulator's decision hooks through the adaptor, not keystrokes. `interfaces.md` defines the intended JSON contract and installation/verification path.

## References

- [R1] Rio Grande Games, official Dominion second-edition rulebook: https://www.riograndegames.com/wp-content/uploads/2016/09/Dominion-2nd-Edition-Rules.pdf ; publisher product page if the PDF path has moved: https://www.riograndegames.com/games/dominion-2nd-edition/
- [R2] Dominion Strategy Wiki, Base set / editions: https://wiki.dominionstrategy.com/index.php/Dominion ; general rules: https://wiki.dominionstrategy.com/index.php/Gameplay
- [R3] Card reference index: https://wiki.dominionstrategy.com/index.php/Card_list ; example: https://wiki.dominionstrategy.com/index.php/Gardens
- [R4] End conditions: https://wiki.dominionstrategy.com/index.php/End_of_game

### adaptor/ADAPTOR.md
# Dominion offline adaptor

`game.py` exposes a Tier 1 `TurnGame` backed by pyminion 0.5.0. The game and both players run locally; no account, GUI, network service, credentials, or runtime download is used. `reset(seed)` selects a fixed scenario and reproducible shuffle stream. Each accepted decision resumes the engine through the next agent prompt or the end of the game.

## Install and provenance

Run `./setup.sh` from this directory. It installs the included wheel offline into the supplied Python (by default the evaluator's Dominion venv) and checks dependencies. `env.example` documents that no environment variables or secrets are needed.

- Engine: pyminion distribution 0.5.0, MIT license.
- Upstream commit: `d51e94604894fd76eecfbbd4fe74d1fe875c0be1`.
- Vendored source and tests: `vendor/pyminion/`; original archive: `vendor/pyminion-source.tar.gz`.
- Wheel: `vendor/wheels/pyminion-0.5.0-py3-none-any.whl`, SHA-256 `15dc35d85f00e986a6638dd7f38370246fc2a42a77371ea40aaf00134465dd79`.
- Adaptor/manifest version: 2.0.0. `game_v1.py` is retained from the previous adaptor. `game.py` is the current public entrypoint; the engine bridge and tested repairs are in `worker.py` and `engine_patches.py`.

The wheel metadata reports 0.5.0. Its package-level `pyminion.__version__` is stale and says 0.3.0; use distribution metadata for the installed version.

## Opponent and measured strength

Every declared game scenario uses `BMU-1`, pyminion's built-in `BigMoneyUltimateDecider` from the pinned source. It plays Smithy when available. Its first affordable, nonempty purchase is Province when total deck Treasure money exceeds $15; Duchy when fewer than five Provinces remain; Estate when fewer than three remain; Gold; Duchy when fewer than seven remain; Smithy when owned Smithies are fewer than owned Treasures divided by eleven; Silver; otherwise it stops. It uses pyminion's default responses to forced choices and has no access to the agent's hidden zones.

We measured it on the actual patched engine over 50 shuffle seeds per prior train/held-out kingdom/policy matchup, pairing both player seats (600 complete games total). Against the previous adaptor's policies, BMU-1 won 299/600 games (49.8%); including half credit for draws, win score was 52.3%, and mean VP margin was +0.19. Results by previous opponent were:

| Previous opponent | Games | Raw wins | Win score (draw = 0.5) | Mean VP margin |
|---|---:|---:|---:|---:|
| BM-1 | 200 | 147 (73.5%) | 75.8% | +4.95 |
| SM-1 | 200 | 102 (51.0%) | 55.3% | +0.70 |
| WM-1 | 200 | 50 (25.0%) | 25.8% | -5.08 |

This shows a clear weakness against early Witch pressure; the combined figure should not hide it. As a cross-check against the vendored example bots, the same BMU-1 policy won 1,701/2,000 games (85.1%) across eligible tested kingdoms against pyminion's `BigMoney`, `BigMoneySmithy`, and `ChapelBot` deciders, with a +7.45 mean VP margin. The batch is a reproducible local strength check, not a claim about expert human play. `OptimizedBotDecider` is an abstract strategy helper without buy/action priorities, not another complete AI. The Bandit example requires an unsupported Bandit pile and is excluded. Among the applicable complete built-ins, BMU-1 beat each alternative individually: win scores of 95.56% versus BigMoney (800 games), 79.44% versus BigMoneySmithy (800), and 80.13% versus ChapelBot (400). These are head-to-head measurements, not a global optimality claim.

Raw per-game reports are `runtime/previous-opponent-benchmark.json` and `runtime/builtins-benchmark.json`. Each records per-kingdom, per-opponent and per-seat aggregates. To reproduce from this directory, run the evaluator Python on `benchmark_opponents.py --pairs 50 --suite previous --output runtime/previous-opponent-benchmark.json` or use `--suite builtins --output runtime/builtins-benchmark.json`. Benchmark seeds start at 900000 plus 10000 times the kingdom's manifest index; each seed is paired across both seats. These seeds are independent of the evaluator splits and cannot be passed to public `reset()`.

## Rules and supported scenarios

Two players start with 7 Copper and 3 Estate, draw five, and use normal two-player piles and exactly ten tested Base kingdom piles. Each turn starts with one Action, Buy, and $0. Ordinary Copper/Silver/Gold are automatically played after Actions. Buying spends money and a Buy; card-effect gains do not. Turns finish before Province/three-pile endings are checked. Final VP counts all owned cards, including Curses and Gardens (`floor(total cards / 10)` per Gardens). Higher VP wins; a tie goes to the player who took fewer turns, otherwise it is a draw.

The supported kingdom cards are Cellar, Chapel, Festival, Gardens, Laboratory, Market, Merchant, Militia, Mine, Moat, Remodel, Sentry, Smithy, Village, Witch, and Workshop, plus basic Supply cards. All four named study fixtures are included exactly. Two held-out kingdom combinations (`heldout_base_draw`, `heldout_chapel_sentry`) add draw/thinning variation while keeping to the tested card whitelist. Unsupported cards and seeds fail explicitly; no kingdom is silently substituted.

Scenario assignments are in `scenarios.py`. For each even seed, the following odd seed uses the same shuffle seed and the opposite agent seat; seat 0 goes first. The pairs and kingdom configuration are:

| Split | Seed pairs | Kingdoms |
|---|---|---|
| train | 1100/1101, 1200/1201, 1300/1301, 1400/1401 | base_intro, chapel_engine, curse_race, gardens_rush |
| heldout | 2100/2101, 2200/2201, 2300/2301, 2400/2401 | heldout_base_draw, heldout_chapel_sentry, heldout_witch_gain, heldout_thin_gardens |
| dev | 3100/3101 | gardens_rush |
| validation | 4100/4101, 4200/4201, 4300/4301, 4400/4401 | base_intro, chapel_engine, curse_race, gardens_rush |

All splits are disjoint. Train and held-out use different kingdom combinations as well as shuffle streams; each has eight seeds and both seats. The engine's process-local Python RNG is initialized from the pair seed before setup and supplies later shuffles in normal engine call order. A seat swap shares the initial seed but is not promised identical later draws after strategies diverge. Replay files record seed, scenario, seat, opponent, engine commit, adaptor version, and accepted actions; the seed is never shown in observations.

## Actions, observations, and scoring

The action vocabulary is `choose`, `pass`, and `wait`. Use `choose DECISION_ID OPTION_ID ...` for ordered selections of distinct option IDs within the prompt's `min` and `max`; `choose DECISION_ID` or `pass` declines only when zero choices are allowed. A JSON object with `decision_id` and `option_ids` is also accepted. `wait` leaves the pending decision unchanged. IDs refer only to options in the current prompt. Invalid, stale, malformed, or duplicate selections do not advance the engine and are reported as `last_error`; no fallback is made.

Prompts include Action play, buy, trash, discard, gain, yes/no reactions, effect ordering, and the nested choices needed by the whitelist. They also occur when the opponent attacks. Forced sole answers resolve automatically. Ordinary Treasures are played automatically; there are no supported Treasure-order interactions.

Observations are compact strict JSON: seat and active player, phase and turn counts, current resources, exact kingdom/Supply card text and counts, the agent's hand/discard, public play zones and ownership/VP ledgers, opponent public counts and ownership, trash, public event changes, derived pile/VP state, and the pending prompt. They do not expose either unrevealed deck order, the opponent's hand, RNG state, reset seed, or private engine logs. Repeated observation is stable. The largest observation in the stress run was under 5 KB.

`score` is the integer VP margin as a float. `info()` separately reports both final VPs, margin, outcome, `win_score`, turns, ending reason, truncation, illegal/fallback counts, scenario and replay metadata. `win_score` is 1/0.5/0 for win/draw/loss. `score_breakdown()` reports the numeric score components. A 400-turn or 20,000-decision safety limit marks a truncation; it is not a Dominion ending and must not count as a win/loss sample. Engine failure marks `valid_trial=False` with a reason.

## Engine repairs and limits

`engine_patches.py` narrowly repairs four upstream edge cases used by the whitelist: duplicate Moat reactions; Remodel/Mine trash-before-gain timing; Sentry's inspected-card ownership and top-first ordering; and Workshop when no eligible pile remains. The engine continues to own turn flow, card resolution, resources, piles, shuffles, termination, and winner calculation. Additional regression tests check starting decks and pile sizes, all enabled cards, nested attacks, Gardens/Curses, gain endings, tie-breaking, hidden-zone privacy, deterministic replay, malformed actions, resets, truncation, and process cleanup.

This is a Base-card simulator, not a live digital client. Throne Room, Bandit, Library, Artisan, Vassal, expansion cards, alternate starting decks, and unsupported kingdom cards are not enabled. Public ownership is tracked from observable engine events; hidden order is intentionally not inferred. `close()` terminates and reaps the worker and closes all streams; it is safe to call repeatedly.

## Verification

Verified with `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python`:

- Generic `validate_adaptor(..., expected_tier=1)`: **PASS**, no warnings; includes two seeds, action checks, malformed action, determinism, and cleanup.
- Vendored pyminion tests plus adaptor tests: **596 passed**.
- Offline money-policy and randomized legal-action batches: 26 games each, all completed without invalid trials, illegal choices, fallbacks, or truncation. Maximum observed JSON was 4,515 bytes for money play and 4,378 bytes for random play.
- Score headroom: the public-observation SM-1/WM-1 smoke policy averaged 48.08% win score and 0.00 VP margin; random play averaged 0% and -70.31. Neither saturates the objective across the scenario suite.
- Opponent comparisons: 600 games against the previous adaptor's policies and 2,000 against the vendored example bots, with both seats paired and no truncations.

To rerun the generic smoke test from the repository root:

```sh
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('runs/dominion/adaptor', expected_tier=1).to_markdown())"
```

From this directory, reproduce the conformance suite with:

```sh
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python - <<'PYTEST'
from engine_patches import apply
import pytest
apply()
raise SystemExit(pytest.main(['-q', 'vendor/pyminion/tests', 'test_adaptor.py']))
PYTEST
```

Run `evaluate.py` and `evaluate.py --random` with that interpreter for full public-interface episodes. Replay a current-version episode with `replay.py runtime/episode-.../replay.jsonl`. Replays from the previous adaptor version are explicitly rejected because its manifest and opponent differ. Runtime artifacts are the only files written during play.

### study/interfaces.md
# Programmatic interfaces and adaptor specification

## Recommendation and readiness

Use **a local Python bot-library adaptor around pyminion**, initially two players and a tested whitelist of Base cards. Its intended niche—simulating Dominion with programmable players—fits this task better than controlling a client. Keep Astra outside the rules engine: the engine owns card resolution, legality, shuffles, score, and termination; the adaptor supplies choices and filtered observations.

**This selection is provisional, not an installation or compatibility claim.** The target Python is confirmed as **3.12.3** at `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python`. It currently contains pip only. Neither pyminion source nor dependencies are available locally. DNS/network access failed and this study sandbox cannot write the target venv. No upstream API, source commit, dependency versions, license, supported cards, or successful match has been verified. The source-inspection gate below is mandatory. Do not build against guessed method signatures.

Sources: https://github.com/evanofslack/pyminion ; https://pypi.org/project/pyminion/ . All external URLs here are reference targets, not successfully retrieved pages; see `sources.md`.

## Practical interface survey

Latency figures below are **engineering expectations, not measurements**. Engine computation should be small relative to Astra's seconds-to-minute inference; measure actual callbacks after installation.

| Route | Observation and control | Maturity / likely latency | Cost, licensing/ToS, and suitability |
|---|---|---|---|
| **Pyminion local Python simulator / bot extension** | In-process game/card/player objects; adapt player decision functions into JSON prompts. Potential access to all state requires deliberate hidden-information filtering. Exact hooks and card coverage require source inspection. | Small community project; packaging, tests, maintenance and 3.12 compatibility unverified. Local callback/queue operations expected sub-second. | Intended free/open-source candidate; inspect LICENSE and bundled assets before pinning. No remote account, anti-cheat, subscription, or runtime network. Best integration fit. |

…(truncated; full text in prior/interfaces.md)

5. **End/scoring:** Province ending, three-pile ending, finishing all current-turn effects, Curse penalties, Gardens floor-per-copy, VP tie/fewer-turn win and true draw. End through gains as well as purchases.
6. **Privacy:** observations with identical public information but different opponent hands/deck permutations contain no distinguishing private fields. No seed/order leak in IDs or logs. Own draw events reach only their owner until publicly revealed.
7. **Determinism:** same config/seed and response sequence reproduce the same prompts/results. An invalid action leaves state unchanged; reset leaves no stale prompt or worker. Pausing a prompt does not change state.
8. **Baseline games:** a modest batch of completed legal games, both seats and all curated kingdoms; inspect trace and scores. No benchmark win-rate claims until measured. A high safety limit on turns/decisions is an infrastructure truncation, reported separately, not Dominion's end condition.

All implementation work above is for the next phase. This study contains specifications and findings only; it does not contain a working adaptor or gameplay code.


## Measured model latency (at archive creation, 2026-10-31 04:11)
- System-2 (Astra), one-word answer: 39 s
- System-2, an answer of ~922 output tokens: 72 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).