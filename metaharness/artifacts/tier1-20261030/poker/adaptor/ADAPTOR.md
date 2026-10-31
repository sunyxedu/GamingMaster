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
