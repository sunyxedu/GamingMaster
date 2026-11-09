Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `Dominion (offline pyminion)` — Play standard two-player Dominion, Base second-edition cards, against the disclosed
BM-1, SM-1 or WM-1 opponent. Each player starts with 7 Copper and 3 Estate, shuffled,
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
BM-1 buys Province, then Duchy at <=4 Provinces, Estate at <=2, then Gold, Silver, else passes.
SM-1 inserts one affordable Smithy before Gold/Silver while >=5 Provinces remain. WM-1 instead
inserts one Witch while >=5 Provinces and some Curses remain. Both play and retain their one
terminal when possible. Opponents otherwise retain highest Treasure money then VP on discards.
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

…(truncated; full text in prior/strategies.md)

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
# Dominion offline Tier 1 adaptor

This adaptor runs real **pyminion 0.5.0** Dominion rules in a separate local Python
process and pauses its live call stack at each agent choice. No account, GUI,
network service, credentials, or runtime downloads are used. The opponent is a
scripted player in the same engine, not a simulated score function.

## Installation and provenance

Run `./setup.sh` from this directory. Its default interpreter is
`/home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python`; an alternate
Python path can be supplied as the first argument. The script installs the
bundled wheel with `--no-index --require-hashes`, then checks dependencies.
`requirements.txt` pins the wheel; there are **no pyminion runtime dependencies**.
`env.example` intentionally declares no environment variables.

Upstream: https://github.com/evanofslack/pyminion

- Commit: `d51e94604894fd76eecfbbd4fe74d1fe875c0be1` (distribution version 0.5.0).
- License: MIT, retained in `vendor/pyminion/LICENSE` and the wheel.
- Unmodified source and upstream tests: `vendor/pyminion/`.
- Original source archive: `vendor/pyminion-source.tar.gz`, SHA-256
  `2b571120b92a1a0c015afabbbada9594751d67d8a528bfb20cdd22d9cfd212b3`.
- Wheel: `vendor/wheels/pyminion-0.5.0-py3-none-any.whl`, SHA-256
  `15dc35d85f00e986a6638dd7f38370246fc2a42a77371ea40aaf00134465dd79`.
- Adaptor/patch/manifest version: 1.0.0. Runtime patches are explicit Python
  method replacements in `engine_patches.py`; they never rewrite source files.

Source was downloaded and inspected during provisioning. Setup and all gameplay
now work offline using the staged artifacts. Verification used the specified
CPython 3.12.3 environment, not system Python.

## Real engine interface mapping

| Capability | Actual upstream API / integration |
|---|---|
| Setup and Supply | `Game(players, expansions, kingdom_cards, random_order=False)`, `Game.start()`; exact ten supplied cards, no random replacements |
| Turn execution | `Game.play_turn(player)` and `Player.take_turn()`; finish cleanup before `Game.is_over()` |
| Action / buy choices | `Decider.action_phase_decision`, `buy_phase_decision` return a card or None |
| Ordinary Treasures | `treasure_phase_decision` returns all Copper/Silver/Gold automatically |
| Nested choices | `trash_decision`, `discard_decision`, `gain_decision`, `binary_decision`; each exposed for the agent, including attack victims |
| Effect ordering | `effects_order_decision` bridge; current whitelist's simultaneous automatic effects commute |
| Zones | `Player.hand`, `.deck`, `.discard_pile`, `.playmat`, `.set_aside` |
| Public bookkeeping | `PublicPlayer` instruments successful `buy/gain/trash/play/discard/reveal`; ownership starts at 7 Copper/3 Estate |
| Randomness | Upstream Python module-global `random`; private per-process state seeded once before setup |
| Scoring and ending | `Player.get_victory_points`, `Game.get_winners`, `Game.is_over`; public-ledger score independently checked against engine at completion |

The synchronous subprocess uses dedicated newline JSON stdin/stdout. It emits a
snapshot and blocks reading a reply. The parent validates the pending prompt
before writing that reply. It waits for the next snapshot, so `step()` returns
only after resolution reaches another decision or game end. There are no agent
background threads. Reset terminates and reaps the previous worker; `close()` is
idempotent and closes every process and pipe. Unexpected worker exit, broken
pipe, or 30-second engine-advance timeout sets `info().valid_trial=False` and an
`invalid_reason`; the timeout never applies while the policy is thinking.

## Supported cards and focused engine repairs

Only the union of the six declared kingdoms is enabled: Cellar, Chapel, Festival,
Gardens, Laboratory, Market, Merchant, Militia, Mine, Moat, Remodel, Sentry,
Smithy, Village, Witch, Workshop, plus the seven basic Supply cards. These cards
pass upstream and additional conformance tests. No expansion cards, alternative
starting decks, or undeclared kingdoms are accepted. Throne Room, Bandit,
Library, Artisan, Vassal and the other Base cards are not enabled; their special
choice APIs have not been certified for this bridge.

The adapter applies these narrow repairs to upstream 0.5.0:

1. **Moat:** use one attack reaction that checks the defender's actual hand.
   Upstream removes every registered Moat reaction when just one of several
   Moats leaves hand. The fix also avoids redundant prompts for identical Moats.
2. **Remodel/Mine:** actually trash before requesting the constrained gain;
   allow the effect to finish when no eligible pile remains. Mine remains
   optional and gains to hand. Remodel's sole mandatory trash resolves directly.
3. **Sentry:** hold inspected cards in the owned `set_aside` zone, apply each
   trash/discard before the next choice, and supply remaining cards explicitly
   to the swap prompt in first-to-draw order. Upstream's Boolean ordering hook
   omits this context and keeps temporary cards outside owned zones.
4. **Workshop:** no gain when no eligible card remains, rather than asserting
   that the nonexistent selection contains one card.

These repairs are regression-tested. Turn rules, draws, shuffling, card effects,
Supply mutations, resource accounting, cleanup, end conditions, and final
winner calculation otherwise remain upstream engine operations.

## Actions and observations

`make_game()` implements TurnGame; `TIER=1`, `DETERMINISTIC=True`.
`describe()` and `action_spec` provide the policy with rules and the vocabulary.

- `choose DECISION_ID OPTION_ID ...` selects distinct occurrence IDs from the
  current prompt, subject to `min` and `max`. Identical copies have separate
  local option IDs; IDs reveal no persistent card identity or hidden ordering.
- `choose DECISION_ID` or `pass` declines only an optional prompt (`min=0`).
- `wait` is the noop and leaves the game frozen.
- Equivalent JSON: `{"decision_id": 7, "option_ids": ["0", "2"]}`.

Invalid, malformed, duplicate, or stale choices do not advance the worker. The
next observation reports `last_error` and increments `illegal_action_count`.
There is no fallback strategy; `fallback_count` is always zero. IDs reset with
each fresh episode. `episode_id` is the constant `episode`, deliberately not a
seed-derived token; responses must not be reused across resets.

Prompts cover playing Actions, buying, trashing, discarding, gaining and yes/no
reactions. Sentry's options/relevant cards are top first; True at the final
prompt swaps the two remaining cards. Optional zero-card choices remain choices.
Forced sole responses resolve automatically. Ordinary Treasure play is automatic;
no supported card benefits from withholding or reordering these Treasures.

Observations include the exact kingdom, opponent family, agent seat, active
player, decision player, phase, active player's Actions/Buys/coins, and turns
**started** for both players. Supply includes counts, costs, types and rules.
The agent sees its own hand/discard as multisets, both public played zones,
hand/deck/discard counts, trash, exact public ownership and VP, pile pressure,
and a public event delta since the previous decision. Repeated `observe()` does
not consume the delta. Terminal results include final VP and outcome.

Both unrevealed deck orders, the opponent's hand, reset seed, RNG state, file
paths, and future decisions are excluded. Public ownership uses a separate
ledger of observable acquisitions and trashing, not hidden card identities.
Private Sentry inspection is shown only in the agent's own prompts. The policy
can remember top cards it ordered; no speculative future deck order is supplied.
Engine prose logs (which include private draws) are never observations.
The public event delta omits private draws and cleanup hand discards; aggregate
ownership remains exact. Engine draw events visible only to the owner are

…(truncated; full text in prior/ADAPTOR.md)

commit, adaptor version, and accepted decisions. `info()` includes those metadata
and the replay path; none is leaked through observation. Replay with:

```
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python replay.py runtime/episode-.../replay.jsonl
```

Replays reproduce decisions/results; rejected actions and waits are not part of
the game trace, so their error counters are not reproduced. Runtime files may be
deleted after evaluation. No source file is generated or changed during play.

## Scores, limits and validation

`score` is the **integer VP margin exposed as float**, never a weighted blend.
`score_breakdown()` separately reports agent VP, opponent VP, margin, win_score,
truncation and error/fallback counts. `info()` additionally reports outcome,
turns, end reason, kingdom, opponent, seed, version, and trial validity.
Gardens counts all owned cards; Curses subtract VP. At equal VP, fewer turns wins;
equal turns gives a shared draw. Thus zero margin can still be a win.
`win_score` is 1/0.5/0 for a genuine win/draw/loss. Before completion it is zero
with outcome `ongoing`, so it must be aggregated only for completed games.

Default limits are 400 turns per player and 20,000 decisions. `make_game` accepts
`max_turns` and `max_decisions` for testing. Limits are safeguards, not Dominion
end conditions: they set `truncated=True`, outcome `truncated`, win_score 0,
and a distinct end reason. Truncated games must be excluded from legitimate
win-rate/margin aggregates and counted separately; they are not worker failures
and do not trigger infrastructure retries. Current margin is available when an
external evaluator stops an episode early; consumers must check completion.

`evaluate.py` runs public-observation money policies over every declared seed;
`evaluate.py --random` performs randomized legal-action stress games. Reports
under `runtime/` contain per-episode results and separate mean VP margin and
win rate (draws count half) grouped by kingdom, opponent and seat. These are
small integration checks, not estimates of general playing strength. No policy
is given the hidden reset seed or engine state.

Verification records are saved under `runtime/`: upstream tests, patched-engine
and adaptor tests, baseline/random reports, and generic validator output.
Additional tests cover starting decks/Supply, resource accounting, all enabled
cards, nested attacks, duplicate-Moat protection, Mine/Remodel timing, Sentry
ordering, empty gains, Gardens/Curses, gains ending games, tiebreaks, hidden-state
permutation privacy, deterministic replay, malformed-action nonmutation, reset,
close, truncation and worker failure. The test runner uses pytest and pytest-mock;
these are development tools, not gameplay dependencies.

Verified results for this revision: generic `validate_adaptor(...,
expected_tier=1)` **PASS**, with no warnings; 554 unmodified upstream tests
passed; all 594 patched-upstream plus adaptor tests passed. Twenty baseline and
twenty randomized legal-action episodes completed, with no illegal choices,
fallbacks or truncations. Largest measured observation was below 5 KB. Baseline
integration-policy win score averaged 0.65 with mean margin +2.05; random play
averaged 0.0 and -66.9. Exact per-episode records are in the reports; these small
suite results should not be interpreted as broad Dominion win-rate claims.

Reproduce conformance checks in the prepared evaluator environment:

```sh
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python - <<'PY'
from engine_patches import apply
import pytest
apply()
raise SystemExit(pytest.main(['-q', 'vendor/pyminion/tests', 'test_adaptor.py']))
PY
```

Run the generic validator from the repository root:

```sh
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('runs/dominion/adaptor', expected_tier=1).to_markdown())"
```

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


## Measured model latency (at archive creation, 2026-10-29 22:50)
- System-2 (Astra), one-word answer: 17 s
- System-2, an answer of ~897 output tokens: 76 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).