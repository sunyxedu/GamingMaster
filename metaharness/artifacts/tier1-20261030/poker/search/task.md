Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `Heads-up no-limit Texas Hold'em — mc-pot-odds-v2` — Play heads-up no-limit Texas Hold'em against the frozen mc-pot-odds-v2 opponent.
No clock, rake, ante, elimination or blind escalation. Every hand starts with 20000 chips
per player (200 big blinds); blinds are 50/100. Engine seat 0 is BB, seat 1 button/SB.
Button acts first preflop; BB acts first after the flop. Best five of seven cards wins;
ties split the pot. Folding when a check is free is disallowed. PokerKit handles minimum
full raises, short all-ins, calls, refunds and settlement. Seats alternate every hand.
Deals are independent; neither player sees seeds, burns, future cards or unrevealed holes.

Objective: maximize terminal NET chips won / required_hands (100 in the primary benchmark).
Fresh stacks do not erase losses from the score. Before completion the displayed score
is provisional, using the full required-hand denominator; it is not a completed match score.
Each step applies your action and advances the opponent and dealing to your next turn,
possibly in a later hand. done is true only when all required hands have settled.

Actions: 'fold', 'check', 'call', 'raise_to N' (integer chips), or a candidate id verbatim.
Equivalent objects are {"action":"fold"}, {"action":"check"}, {"action":"call"},
{"action":"raise_to","amount":300}. N is your TOTAL wager on the CURRENT STREET,
including that street's previous bet, not an increment or total-hand commitment.
For example SB opening to 300 pays 250 more; the next full raise is at least to 500.
Every integer between legal min_raise_to and max_raise_to is available. A stack-sized
call uses 'call'; a legal all-in raise uses 'raise_to max_raise_to'. 'wait' is a no-op.
Invalid input changes no poker state, increments illegal_actions and sets action_error.

Observation: hole_cards are yours; board contains only exposed community cards. All arrays
of stacks, street_bets and committed_total use engine seat order [BB,SB]. pot includes
outstanding bets exactly once; to_call is capped at your stack; effective_stack is the
minimum remaining stack. legal gives booleans and nullable current-street raise bounds.
candidates give exact action ids, raise amounts and incremental costs. history is this
hand's public betting sequence; pot_after_wager is before any refund or payout. last_hand
contains only your cards, the public board, legally shown cards, bets and your net result.
At terminal, stacks are after payout, pot/committed_total are zero, and legal is empty
of permitted actions. Progress, provisional/final score, public result counts and error
diagnostics are public. Future observations contain no secret engine state.

Opponent is nonlearning mc-pot-odds-v2: wide button openings, premium reraises, uniform
visible-card showdown equity (512 samples, exact river), price-sensitive calls, value
bets and an 8% mixed bet branch. It receives only its own holes and public information.
Its fixed policy is intentionally exploitable, not an expert or equilibrium benchmark.
Hero adaptation is allowed. Uncertainty must be assessed across independent match seeds,
not by treating correlated hands as independent. No duplicate deals occur within a match.


Actions:
- fold: Surrender the hand; legal only facing a wager.
- check: Pass when to_call is zero.
- call: Match the wager, capped at your remaining stack.
- raise_to: raise_to N: bet/raise to integer N chips TOTAL on this street.
- wait: Do nothing; the game has no clock and remains at your decision.
(real-time games: ticks apply 'wait' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Use a candidate id verbatim, or 'raise_to N', or a typed action object. Only actions permitted by the current observation are legal. Malformed/illegal actions leave the poker state unchanged and report an error.

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
- strategy_summary: Use wide, position-aware heads-up preflop ranges and price-sensitive BB defense, with stack-depth-specific sizing rather than full-ring or short-stack charts. Code should own legality, bookkeeping, exact hand evaluation, pot odds, combination weights, equity sampling and reproducible mixing. Astra should reason over plausible opponent ranges, plan value bets and selected bluffs across streets, and adapt to public evidence of the frozen opponent's mistakes. Distinguish raw equity from equity realization, range advantage from nut advantage, and river bluff fraction from bluff-to-value ratio; avoid automatic deep-stack commitment with one pair. Value bet more and bluff less against excessive callers, pressure overfolders, and fold more bluff-catchers against reliably underbluffed large bets. Jev is optional for independent typed situation classifications and should not calculate odds or control legal actions. Evaluate changes on fixed unseen deals using net chips per hand and appropriate variance estimates.
- scoring: Primary score is sum of hero terminal net chip payoffs divided by the fixed number of completed hands, with completion status and required hand count reported. Use no rake, 50/100 blinds and fresh 20000-chip stacks for every primary-benchmark hand. Blinds and all wagers count: an immediate SB fold yields engine-seat [BB,SB] payoffs [+50,-50]. Also report total chips, bb/hand, bb/100, illegal-action diagnostics, opponent/version and uncertainty. For isolated nonadaptive duplicate pairs, average hero outcomes over both legs and calculate standard error across pair averages; for adaptive histories use independent match blocks. Do not substitute pot size, hand-win percentage, normalized stack reward or tournament survival for net chips/hand, and do not declare an incomplete match a full score.
- interface.observation: Sanitized JSON at each hero turn: own two cards, exposed board, street, engine/hero/button seats, integer blinds and stacks, current-street bets and total commitments, pot including outstanding bets, capped call price, effective stack, engine legal bounds and exact candidate actions, current-hand public action history, public statistics with sample counts, and optional code-calculated equity/odds/texture features. Opponent receives its own equally filtered view. No raw engine dump, hidden cards, future deck, RNG state or seeds. Showdown information becomes public only when actually revealed; duplicate counterpart histories and results remain isolated.
- interface.actions: Submit a legal candidate ID or typed fold/check/call/raise_to object; raise_to.amount is the integer total on the current street, not an increment or whole-hand total. Validate against the engine before applying fold(), check_or_call() or complete_bet_or_raise_to(amount). All-in is a stack-sized call or legal raise, not an unconditional separate option. step advances through local opponent actions, manual chance events and automated settlement until the next hero turn or terminal boundary. Match completion requires the fixed hand count; chips are credited exactly once per completed hand.

### study/strategies.md
# Strategy prior: heads-up no-limit Texas Hold'em

This guide targets the specified **offline, no-rake, fixed-opponent, chips-per-hand** benchmark. The primary proposed game is 50/100 blinds and 20,000 starting chips per player, reset every hand (200bb). It is not a tournament survival problem. Tier 1 allows Astra to decide every turn; Jev is optional. All numerical ranges and sizing menus below are starting heuristics, **not claimed solver outputs**.

Evidence boundary: local rules and interface sources were inspected. Network requests failed DNS, so the external references below are relevant established documentation, strategy resources, and research locations, **not freshly retrieved pages**. Their current contents and URLs could not be verified. Mathematical examples below are derivations with stated assumptions; no paid charts or unavailable pretrained poker agent are required.

## 1. Division of work

| Topic | Code: exact or repeatable work | Astra prompts: judgment and planning | Optional Jev questions |
|---|---|---|---|
| Rules and accounting | Legal actions, integer sizing, actor/position, pot, call price, effective stacks, payoff, best five cards | Choose among already legal candidates | None; never outsource legality or arithmetic |
| Preflop | Card-to-class mapping, weighted range tables, size/depth buckets, random mixing | Choose a range family and adapt to opening/defense tendencies | Choice: which supplied range family fits this situation? |
| Range inference | Card removal, combination counts, likelihood updates, public history counters | Interpret action sequences and select plausible value/bluff ranges | Choice: likely range is capped / strong / polarized / uncertain |
| Postflop | Made hand, draws, board features, equity estimates and uncertainty, SPR | Compare checking, thin value, protection, bluffing, and future streets | Choice: primary purpose of a proposed bet |
| Exploitation | Opportunity-based statistics and uncertainty | Adjust to stable opponent mistakes; decide how much evidence is enough | Choice: observed behavior best fits overfolding / overcalling / overaggression / unknown |
| Evaluation | Hidden deals, independent RNG streams, duplicate isolation, chip ledger | Review recurring costly decisions and revise hypotheses | Optional flags for inconsistent explanations; not a second poker engine |

Strong poker combines a reasonable default strategy with selective exploitation. Equilibrium-inspired play is a robustness prior, not a requirement to stay balanced against a script that folds too much. Research systems such as Libratus and DeepStack use abstraction, opponent-range reasoning and subgame solving; those results do not imply that a few prompt instructions solve unrestricted Hold'em. [S1, S2, S3]

## 2. Preflop: enter more pots than in full-ring poker

**Position matters on every hand.** The button/SB acts first preflop and last postflop. The BB acts last preflop and first postflop. Button initiative plus postflop position supports a wide opening range; waiting for premium hands pays away too many blinds. Use heads-up ranges, never nine-handed early-position charts. [S4]

For a simple deep-stack starting strategy, use button raises around **2–2.5bb total**, with a wide range (roughly 70–90% is an exploratory prior, not a target independent of opponent). Include all pairs, most aces and kings, suited hands, connected hands, and many offsuit hands with high-card value. The weakest disconnected offsuit hands can fold. A mixed limping strategy is also legitimate: it keeps weak hands cheap and can contain traps. Choose and document a coherent raise/limp framework instead of claiming limping is always wrong. Raise more frequently if BB overfolds; prefer smaller pots with marginal hands if BB attacks raises strongly; value-raise larger against a caller insensitive to size.

**BB defense must depend on price.** Against a button min-raise to 2bb, the BB calls 1bb into the existing 3bb pot: the immediate equity threshold is 25%. That does not mean every hand with 25% equity can profitably call: out-of-position realization, domination, and future bets matter. Against larger opens, tighten weak offsuit calls. Against limps, check weak hands for free, and raise a wider value range if the limper calls too much. Do not auto-fold a free check or surrender BB to every raise. [S4, S5]

A deep-stack 3-bet starting size is roughly **3.5–4.5 times the opening total** from the BB (e.g. to 9–11bb versus 2.5bb). The button's positional advantage makes tiny reraises unattractive. Use a more linear range (strongest hands plus good broadways) against opponents who call many 3-bets, and allow selected blocker bluffs against players who fold enough. Low suited aces are candidates because they remove premium ace combinations and can make nut flushes; they are not mandatory bluffs against a calling station. A 4-bet might begin around 2.2–2.5 times the 3-bet total, but effective stacks and range strength govern whether that is appropriate. Use code to enforce the actual legal target and show the resulting SPR.

**At 200bb, one pair is not an automatic stack-off.** Suited nut potential and position gain value; dominated offsuit aces and weak kings carry reverse implied odds. Small pairs and suited connectors need enough future payoff and suitable opponent behavior to justify calls; “I can hit a set” is not sufficient. At 20–40bb, speculative calls lose implied odds, value rises for high cards/pairs, and reshove decisions become more common. Push/fold charts for short stacks are not a 200bb strategy. [S4, S6]

Implementation ownership: **code** stores explicit hand classes and size/depth buckets; **Astra** chooses/tunes the range and exceptions; **Jev** may identify the applicable already-defined situation. Do not ask Jev to enumerate 169 classes or calculate call odds. Represent each range with combination weights: a pocket pair has 6 combinations, a suited nonpair 4, an offsuit nonpair 12 before card removal. A percentage of the 169 labels is not the same percentage of the 1,326 two-card combinations.

## 3. Think in ranges, update without false certainty

Start with a range conditional on position and preflop action, then update after checks, bets, calls and raises. A check does not prove weakness; a large bet does not prove a monster. Keep plausible traps, draws and bluffs when the opponent's policy permits them. Remove combinations containing hero cards or visible board cards exactly. [S1, S2, S7]

**Code** can maintain weights proportional to prior weight × probability of the observed action for that hand class and context. **Astra** selects the likelihood model or a few alternatives, such as tight value-heavy, balanced, and loose aggressive. Preserve uncertainty rather than assigning one exact opponent hand. If the benchmark explicitly supplies the frozen opponent policy, its action rules can inform these likelihoods; access to its private cards, actual RNG draw, or future deck is still forbidden.

…(truncated; full text in prior/strategies.md)

3. **Use a coherent preflop prior and a value-first postflop baseline.** Test exploit adjustments against the frozen opponent on development seeds; keep held-out deals untouched.
4. **Review recurring high-cost errors:** overly tight blind play, oversized deep-stack bluffs, paying off underbluffed rivers, missing thin value, and betting hands that neither get worse calls nor better folds.
5. **Measure chips/hand and uncertainty.** Win rate by hands or pots is not the objective. Pairwise deal averages are the unit for duplicate uncertainty; correlated adaptive sessions need independent block-level analysis. Many thousands of hands can be necessary for small win-rate differences, depending on variance; 200 development hands do not establish superiority. [S8]

Do not implement real-time clocks, GUI recognition, online account automation, tournament ICM, rake adjustments, GPU training, or a full-game equilibrium solver for this benchmark. They do not address the selected offline task. This study contains no game-playing implementation.

## Sources and reading targets

- **S1:** Brown and Sandholm, *Superhuman AI for heads-up no-limit poker: Libratus beats top professionals*, Science (2018): <https://doi.org/10.1126/science.aao1733>. Research motivation for abstraction, safe subgame reasoning and robust imperfect-information strategy; not a downloadable baseline policy promised here.
- **S2:** Moravčík et al., *DeepStack: Expert-level artificial intelligence in heads-up no-limit poker*, Science (2017): <https://doi.org/10.1126/science.aam6960>; project <https://www.deepstack.ai/>. Continual resolving and range-based reasoning.
- **S3:** Zinkevich et al., *Regret Minimization in Games with Incomplete Information* (2007), authors' poker research portal: <https://poker.cs.ualberta.ca/publications.html>. Practical framework reference: <https://openspiel.readthedocs.io/en/latest/algorithms.html>. CFR belongs in a deliberate research extension, not an assumed preinstalled dependency.
- **S4:** Upswing Poker, heads-up strategy/community instruction: <https://upswingpoker.com/heads-up-poker-strategy/>; article index <https://upswingpoker.com/blog/>. The range percentages and sizing examples in this guide are explicitly our starting priors, not copied charts.
- **S5:** Upswing Poker, *Minimum Defense Frequency vs. Pot Odds*: <https://upswingpoker.com/minimum-defense-frequency-vs-pot-odds/>. Supports distinguishing calling price from range defense; formulas here are derived explicitly.
- **S6:** PokerNews strategy library: <https://www.pokernews.com/strategy/>; rules <https://www.pokernews.com/poker-rules/texas-holdem.htm>. Reading topics: pot odds, implied/reverse implied odds, effective stacks, stack-to-pot ratio. These broad resource links do not certify a particular numeric chart.
- **S7:** GTO Wizard strategy library: <https://blog.gtowizard.com/>. Reading topics: range/nut advantage, equity realization, blockers, polarized versus merged betting. Public educational material is useful; a paid subscription or live solver is not required or assumed.
- **S8:** Burch et al., *AIVAT: A New Variance Reduction Technique for Agent Evaluation in Imperfect Information Games*: <https://arxiv.org/abs/1612.06915>. Duplicate deals are a simpler variance-control method, not an implementation of AIVAT. Local rule/scoring evidence: `/home/hemlix/GamingMaster-mh-tier1/data_processing/rs/gmrules/src/games/hunl/mod.rs` and `/home/hemlix/GamingMaster-mh-tier1/scripts/reference/pokerkit_hunl.py`; public repository <https://github.com/sunyxedu/GamingMaster>.

### study/brief.md
# Heads-up no-limit Texas Hold'em: game brief

**Tier 1 is correct.** For the selected offline simulator, time advances only when the adapter deals or applies an action. Astra may think for seconds or a minute without a poker penalty. There is no game clock, blind escalation, or opponent action while Astra is on turn. A harness may impose its own resource deadline, which is distinct from a game rule. `study.json` uses **`decision_latency_s: -1.0` as an explicit unlimited/no-game-timeout sentinel**, because standard JSON cannot represent infinity. Do not interpret this as a negative duration. [R2, L1]

Research status: Python and repository evidence were checked locally; web retrieval was blocked by DNS. Source URLs below are not live-verified. Details are in [research_log.md](research_log.md).

## Objective and match format

Two players compete for chips. Win a hand by making the other player fold, or by holding the best five-card poker hand at showdown. Maximize **expected net chips won per hand**, not number of pots won, survival time, or the chance of ending one short session ahead.

Recommended benchmark: no rake, no ante, small blind 50, big blind 100, both players start **each hand** with 20,000 chips (200 big blinds). Stacks reset after every hand; winnings accumulate in a separate score ledger. This matches the local ACPC-style reference configuration. It is a cash-game-like repeated-hand experiment, not an elimination tournament, even though the engine uses its `TOURNAMENT` rules mode to disallow folding when checking is free. No ICM, rebuy cost, increasing blinds, or tournament prize model applies. [L1, L2]

Use the frozen `heuristic-v1` opponent specified in `interfaces.md` unless the evaluator supplies another versioned policy. It runs entirely locally and has no model calls. Its thresholds are benchmark design choices, not claims of optimal play. PokerKit supplies the rules engine; the adaptor phase must implement this opponent and the private observation wrapper.

Fix the number of hands before evaluation; an initial development budget can be 200 hands, with larger final budgets as runtime permits. That number is a proposed evaluation setting, not a poker rule. Alternate seats and preferably use duplicate deals as specified in `interfaces.md`. Report:

- Primary score: total hero net chips divided by completed hands. Net chips include blinds and all later wagers; receiving the pot is not itself the profit.
- Secondary: `bb/hand = chips/hand / 100`; `bb/100 = 100 × chips/hand / 100` at these blinds. Always label units; do not confuse big blinds with fixed-limit “big bets.”
- Diagnostics: hands completed/required, total chips, opponent/version, seeds retained privately, illegal actions, timeouts, and uncertainty over independent deal pairs. An incomplete match must not masquerade as a full evaluation.

On a small-blind fold, the small blind loses 50 and the big blind wins 50, not 150. At showdown, net changes sum to zero. Resetting stacks must not add winnings to the score a second time. [L1, L2]

## Rules the agent must understand

Use one standard 52-card deck, no jokers. Each player has two private hole cards. Community cards appear in four betting rounds: **preflop** (no board), **flop** (three cards), **turn** (one more), **river** (one more). Only exposed board cards and one's own hole cards are available for a decision. An actual table burns a card before each board street; the proposed simulation fixes this deal schedule privately. [R1, R2]

In heads-up play the **button posts the small blind and acts first preflop**. The big blind acts first on every postflop street, so the button has postflop position. If the button only calls the big blind preflop, the big blind still gets a check-or-raise decision. Alternate which participant occupies each position. PokerKit's pinned reference uses engine seat 0 = BB and seat 1 = button/SB; do not assume seat 0 is the button. [R1, L1]

Legal decisions:

| Action | Meaning |
|---|---|
| Fold | Surrender the hand and money already invested; offer only when facing a wager in this benchmark. |
| Check | Pass when nothing is owed. |
| Call | Match the outstanding wager, or commit the remaining stack if that is insufficient. |
| Bet | First wager on a postflop street; ordinarily at least one big blind. |
| Raise | Increase an existing wager; the minimum increment is the size of the last full bet/raise on that street. |
| All-in | Commit the remaining stack. This can be a call, a full raise, or a legal short raise; it is not always a separate raising option. |

…(truncated; full text in prior/brief.md)

## Observation and pace

At each hero turn provide structured JSON: own cards, board, street, position, both public stacks and contributions, pot including outstanding wagers, amount to call, legal raise-to bounds, public action history, and a small set of exact legal candidate actions. Hide opponent cards, undealt cards, burn cards, RNG state, and deal seeds. The opponent gets the same information boundary from its own seat. Showdown reveals may become public only when the engine actually exposes them. [R2, L1]

There is no keyboard, mouse, screen capture, account, or realtime reaction requirement. The adapter applies one legal action, automatically advances through opponent/chance events, and stops at the next hero decision or terminal state. Jev is optional; code should perform arithmetic and legal checks, and Astra should choose strategy.

## Sources

- **R1:** Poker Tournament Directors Association rules, particularly heads-up button/blinds, full raises, and reopening betting: <https://www.pokertda.com/poker-tda-rules/>. General Hold'em explanation: <https://www.pokernews.com/poker-rules/texas-holdem.htm>. Physical-table procedures do not override the explicitly chosen simulator convention.
- **R2:** PokerKit documentation and implementation: <https://pokerkit.readthedocs.io/en/stable/>; <https://github.com/uoftcprg/pokerkit>.
- **L1:** Locally inspected reference integration: `/home/hemlix/GamingMaster-mh-tier1/scripts/reference/pokerkit_hunl.py`, pinned to PokerKit 0.7.6, and `tests/test_hunl_reference.py`. Public repository reference: <https://github.com/sunyxedu/GamingMaster> (local checkout is the verified evidence).
- **L2:** Local rules: `/home/hemlix/GamingMaster-mh-tier1/data_processing/rs/gmrules/src/games/hunl/mod.rs`; recorded fixture `tests/fixtures/hunl/pokerkit_reference.json`. ACPC data provenance: <https://doi.org/10.5281/zenodo.10796885>.

### adaptor/ADAPTOR.md
# Offline heads-up no-limit Hold'em

`game.py` implements `TurnGame` using the actual open-source PokerKit 0.7.6 rules
engine and its exact hand evaluator. No server, GUI, account, credentials, external
model, network request or paid software is used during play. Tier 1: the world
waits indefinitely at each hero decision. `close()` is idempotent; the adaptor
starts no processes or connections. Python >=3.11 is required by PokerKit;
the evaluator uses Python 3.12.3. PokerKit has no distribution dependencies.

## Installation and verification

Run `bash setup.sh` in this directory. It checks the bundled wheel's SHA-256,
installs offline into `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/poker`, and
runs `pip check`. The wheel in `runtime/wheelhouse/` includes PokerKit's MIT
license. `env.example` intentionally has no required variables.

From the repository root, run:

```sh
runs/venvs/poker/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('runs/poker/adaptor', expected_tier=1).to_markdown())"
runs/venvs/poker/bin/python runs/poker/adaptor/verify.py
```

`verify.py` exercises the real engine, public filtering, action parsing, fixed
payoff denominator and repeatability. It also replays all 240 stored reference
trajectories in `tests/fixtures/hunl/pokerkit_reference.json` through the installed
PokerKit, checking actors, legality, raise bounds and payoffs. Those fixtures are
regression evidence, not a fresh independent-engine differential comparison.
Test reports and any generated artifacts belong in `runtime/`. Playing writes
no code, logs or other artifacts (Python may create ordinary `__pycache__` files).

## Rules and scenarios

The primary benchmark is exactly **100 hands**, no rake or ante, fixed 50/100
blinds, fresh 20000-chip stacks per player per hand. `EVAL_KWARGS` pins that hand
count. `make_game(hands=N)` accepts an even 2–1000 for explicitly labeled tests;
comparisons must use the same N. There are no alternate opponents or stack
depths hidden behind seeds. PokerKit's tournament mode supplies the reference
legality convention (including no folding when checking is free); this is a
repeated cash-style benchmark, never a tournament or elimination game.

PokerKit seat 0 is BB and seat 1 is button/SB. Hero starts on the button and
alternates seats, yielding exactly balanced assignments. The button acts first
preflop, the BB first postflop. Full no-limit integer sizing, minimum full raises,
short all-ins, reopening restrictions, unmatched returns, splits and settlement
are delegated to PokerKit. All-in runouts require no further hero actions.

`reset(seed)` clears the episode and all opponent state. Any integer seed is
supported; `SEEDS` declares disjoint train/heldout/validation groups of four and
two dev seeds. Development gameplay uses dev seeds; the generic validator also
touches the first train/heldout seeds as required. No strategy tuning uses
heldout deals. Never pass the declared evaluation seeds to a player policy.

Each zero-based hand independently shuffles the standard rank-major `23456789TJQKA`
by `cdhs` deck. RNG initialization is the big-endian integer SHA-256 digest of
`poker-v1|{seed}|{hand}|{domain}`, with domains `deck`, `opponent-mixing`, and
`opponent-equity`. The complete permutation precedes any decisions. Layout:
seat 0 holes at 0–1, seat 1 holes at 2–3, burn 4, flop 5–7, burn 8, turn 9,
burn 10, river 11. Burns are actual predetermined cards. Policy sampling cannot
advance the deal RNG. Reproducibility is pinned to this code, Python 3.12 and
PokerKit 0.7.6; changing them defines a new benchmark version.

**Independent balanced-seat hands are used, not duplicate legs.** The fixed
TurnGame API cannot fork or erase the external model's memory between legs.
Repeating a deal would let it remember private cards/board/results from the
first leg. This is the study's explicit isolation fallback. Separate policy
evaluations on the same seed still receive the same scheduled deals regardless
of action histories. There is no within-episode repeated-deal identifier.

## Actions and observations

`describe()` supplies the model-facing rules and schema. Actions are `fold`,
`check`, `call`, `raise_to N`, and `wait` (the no-op). A candidate's `id` is a
complete action string. Equivalent typed JSON objects use `action`, plus an
integer `amount` for `raise_to`. Booleans, extra keys, nonintegers, out-of-range
raises, unknown actions, free folds and checks facing bets are rejected.
Rejected input leaves poker state unchanged, increments `illegal_actions`, and
reports a fixed error string. There is no silent call/fold substitution or chip
penalty. Invalid play is not labeled an infrastructure failure. Repeated invalid
actions can exhaust an external harness budget and leave the match incomplete.

Raise amounts are **current-street totals**: opening SB to 300 costs 250 more;
the minimum next full raise is to 500. Candidate sizes are min, 1/3, 1/2, 3/4,
1, 1.5 pot, max and preflop 250/300/400, clipped, engine-checked and deduplicated.
The pot formula is `max(street_bets) + f*(pot + to_call)`, rounded with ties
upward. All integer targets in the reported legal interval remain available.
When facing an all-in use `call`; a raise is not always possible.

The JSON view exposes only the acting player's holes, exposed board, seats,
blinds, starting/current stacks, street bets, total commitments, pot, capped
call price, effective stack, legal bounds, exact candidates and public current
hand history. Arrays use engine seat order. Active pot is the sum of commitments,
including current bets exactly once. History's `pot_after_wager` precedes refunds
and payouts. Terminal pot/commitments are zero and stacks are settled.

The opponent receives its own filtered view and current public history only;
it never receives the game object. Neither view contains seed, deck, burns,
future cards, hidden RNG state or unrevealed opponent holes. Equity uses only
the acting opponent's two cards and exposed board. It does not sample from the
actual private deck. Public hand ids are simple sequence numbers, independent
of seeds. Returned views are copies, so caller mutation cannot alter the engine.

`step` advances through chance and opponent decisions until the next hero turn
or match completion, automatically starting fresh hands. A hand where the
opponent folds first may finish without a hero decision. `last_hand` reports the
most recent settlement, including hero cards and only opponent cards actually
shown by PokerKit (mucked/folded hands stay hidden). It is not an omniscient hand
log; multiple automatic settlements can occur between hero decisions. Public
counts and the chip total include all of them. `hand_terminal` marks the final
hand boundary; `done` requires the entire scheduled match.

## Frozen opponent

`poker_opponent_v2.py` implements the frozen **mc-pot-odds-v2** starter,
without learning or model calls. `opponent_manifest.json` records ranges and
parameters. It computes equity against uniformly distributed legal opposing
holes and board completions: 96 independent samples on preflop/flop/turn and
all legal opposing-hole combinations on the river. Sampling is without
replacement within each trial. Wins count 1, ties 0.5; PokerKit compares hand
strength. Equity is cached for unchanged visible cards within a hand, and
mixing uses its separate RNG.

The button opens its fixed wide range to 250 and otherwise folds. BB raises a
limp to 400 at equity >=0.60. Premiums QQ+/AK reraise 3x, then shove after an own
raise. Other preflop calls require equity >=max(0.44, pot odds +0.08). Postflop
free value bets use 0.75 pot at equity >=0.62, pot at >=0.85; otherwise an 8%
mixed branch bets half-pot. Facing a wager, equity >=0.80 permits one 0.75-pot
raise after calling per street; otherwise call at pot odds +0.05 on flop/turn,
or pot odds on river. Unavailable raise branches fall through to the declared
passive tests. Thresholds, ranges, samples and randomization are fixed for all
seeds. The policy intentionally ignores range inference and is exploitable;
beating it is not evidence of expert-human or equilibrium-level strength.

## Score, completion and uncertainty

Each settlement records `state.payoffs[hero_seat]` once, including blinds and
every wager. Immediate SB folding gives engine-seat payoffs `[+50,-50]`.
At completion **score = total terminal net chips / 100**. More generally the
denominator is the constructor's required hand count. Before completion this
same denominator produces a provisional value, expressly labeled incomplete;
it is not the mean of a profitable prefix and not a full benchmark result.
External evaluators must require `done` / `info()['score_eligible']` for a final
match score. The fixed protocol still requires a finite numeric `score` on
partial episodes, so it cannot itself prevent a consumer ignoring that flag.

`info()` (evaluator-only) provides completion, required/completed hands, terminal
payoff ledger, total chips, chips/hand, bb/hand, bb/100, illegal diagnostics,
engine/opponent version and uncertainty metadata. `score_breakdown()` supplies
numeric components. With BB=100, bb/hand=score/100 and bb/100=score numerically.
No win percentage, pot size or normalized stack reward substitutes for score.

Hero memory/adaptation is allowed across independent hands. Consequently a
single episode is one independent **match block**, not 100 independent policy
trials. `uncertainty.standard_error_chips_per_hand` is null for one match;
descriptive within-match hand SD is supplied but is **not** a standard error.
Across K independent completed seeds, compute sample SD of the K match scores
divided by sqrt(K), reporting K and the separate seed scores. Four seeds provide
only a noisy uncertainty estimate. Pair-average standard errors do not apply
because this implementation does not duplicate deals.

Infrastructure/engine failures set `valid_trial=False` and a sanitized reason;
bad actions and normal incomplete play do not request infrastructure retries.
No in-game timeout exists; harness token/time/step budgets remain external.


## Opponent strength benchmark

The replacement is `mc-pot-odds-v2`: a frozen, offline policy with cached Monte-Carlo equity, position-aware preflop isolation, street-specific pot-odds margins and blocker semibluffs. It uses no network, credentials, model calls or online learning. The previous `heuristic-v1` is preserved in `game_v1.py` for reproducibility.

A matched head-to-head run uses the same 1000-hand seed 123 and alternates the hero seat. The two directional legs are reported separately because each TurnGame episode has independent histories: v2 as hero versus v1 scored the recorded 100-hand smoke comparison −202.48 chips/hand, while v1 as hero versus v2 scored −35.01 chips/hand (so v2's opponent-side edge was +167.47 chips/hand in that smoke run). The 1000-hand run is recorded in `runtime/benchmark_1000.txt`; poker variance is material, so this comparison is evidence of relative exploitability against v1 rather than a claim of expert equilibrium strength.

### study/interfaces.md
# Programmatic interface and adapter specification

**Choose PokerKit 0.7.6 as an in-process Python rules engine, wrapped by a small turn-based adapter and a frozen heuristic opponent.** It supports unrestricted integer no-limit sizing, explicit dealing, rule-driven legal bounds, and settlement without a GUI or service. The repository already records its Python 3.12 use and API conventions. This is an engine, not a ready-made Gym environment or supplied strong opponent: the next phase must implement the observation filter, match loop, opponent, and scoring described here.

**Current blocker:** PokerKit is absent from the evaluator venv and no cached distribution was found. Web requests fail DNS. This study has not installed or executed it. Use an offline wheel bundle or the source fallback below; do not claim successful setup until imports and hand-level acceptance checks pass. Web references below were not live-verified; local API evidence was read directly. See [research_log.md](research_log.md).

## Practical interface choices

Latency entries describe architecture, not measured timings. All local engines wait for an action; actual transitions should be small compared with an Astra call, but equity enumeration or opponent search can dominate runtime.

| Interface | Maturity and exposed information/actions | Latency and constraints | Cost, license/ToS, and assessment |
|---|---|---|---|
| **PokerKit** Python library [I1] | Established multi-variant rules library; complete internal cards/state, action legality, exact raise amounts, explicit dealing, hand evaluation and payoffs. Local integration pins 0.7.6. Requires a private-view wrapper. | In-process CPU; no clock or network; native Python API rather than Gym. | Free/MIT. No account or gambling-site ToS. **Chosen:** exact no-limit semantics and comparatively small dependency surface. |

…(truncated; full text in prior/interfaces.md)

- **I6 ACPC:** <https://github.com/jblespiau/project_acpc_server>; <https://www.computerpokercompetition.org/>. Inspect the selected dealer's game definition and protocol document before using its total-commitment action notation.
- **I7 PyPokerEngine:** <https://github.com/ishikota/PyPokerEngine>; <https://ishikota.github.io/PyPokerEngine/>.
- **I8 PokerTH:** <https://github.com/pokerth/pokerth>; <https://www.pokerth.net/>.
- **I9 Slumbot:** <https://slumbot.com/>; local `scripts/slumbot_client.py` documents endpoint and action semantics. Its service availability has not been tested.
- **I10 evaluators:** <https://github.com/ihendley/treys>; <https://github.com/julianandrews/pyeval7>; <https://github.com/HenryRLee/PokerHandEvaluator>.
- **I11 variance reduction:** Burch et al., *AIVAT: A New Variance Reduction Technique for Agent Evaluation in Imperfect Information Games*: <https://arxiv.org/abs/1612.06915>. AIVAT itself requires more than duplicate deals and is **not** claimed as implemented here. ACPC historical data: <https://doi.org/10.5281/zenodo.10796885>.


## Measured model latency (at archive creation, 2026-10-30 22:17)
- System-2 (Astra), one-word answer: 45 s
- System-2, an answer of ~832 output tokens: 85 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).