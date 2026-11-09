# Programmatic interface and adapter specification

**Choose PokerKit 0.7.6 as an in-process Python rules engine, wrapped by a small turn-based adapter and a frozen heuristic opponent.** It supports unrestricted integer no-limit sizing, explicit dealing, rule-driven legal bounds, and settlement without a GUI or service. The repository already records its Python 3.12 use and API conventions. This is an engine, not a ready-made Gym environment or supplied strong opponent: the next phase must implement the observation filter, match loop, opponent, and scoring described here.

**Current blocker:** PokerKit is absent from the evaluator venv and no cached distribution was found. Web requests fail DNS. This study has not installed or executed it. Use an offline wheel bundle or the source fallback below; do not claim successful setup until imports and hand-level acceptance checks pass. Web references below were not live-verified; local API evidence was read directly. See [research_log.md](research_log.md).

## Practical interface choices

Latency entries describe architecture, not measured timings. All local engines wait for an action; actual transitions should be small compared with an Astra call, but equity enumeration or opponent search can dominate runtime.

| Interface | Maturity and exposed information/actions | Latency and constraints | Cost, license/ToS, and assessment |
|---|---|---|---|
| **PokerKit** Python library [I1] | Established multi-variant rules library; complete internal cards/state, action legality, exact raise amounts, explicit dealing, hand evaluation and payoffs. Local integration pins 0.7.6. Requires a private-view wrapper. | In-process CPU; no clock or network; native Python API rather than Gym. | Free/MIT. No account or gambling-site ToS. **Chosen:** exact no-limit semantics and comparatively small dependency surface. |
| **RLCard** `no-limit-holdem` [I2] | Established card-game research toolkit. `rlcard.make`, `reset`, `step`, legal action IDs, extracted/raw observations and payoffs. Usually fold, check/call, half-pot, pot, all-in abstraction; confirm exact version. | In-process; convenient ready agent loop. Coarse built-in bet choices and observation/reward conventions can differ from this contract. | Free/MIT. No service. Good quickest coarse-action baseline; do not assume it exposes every raise size or chips as reward. Python 3.12 install not verified here. |
| **PettingZoo** `texas_holdem_no_limit_v6` [I3] | Maintained AEC wrapper around RLCard, `agent_iter`, `observe`, `step`, masks, rewards, termination. An environment API, not an independent poker rules implementation. | In-process; turn-based, rendering optional. Adds wrapper/version dependencies and generally preserves RLCard's sizing abstraction. | Free open-source (PettingZoo MIT; inspect dependencies). No account. Useful for multi-agent tooling, unnecessary here. |
| **OpenSpiel** `universal_poker` [I4] | Mature research framework; ACPC-based configuration, chance outcomes, legal actions, information states/observation tensors, returns. Configure two-player **no-limit** Hold'em explicitly, including stacks and full-game sizing; `kuhn_poker` and `leduc_poker` are different games. | Compiled backend, fast transitions. Wheel/build availability and configuration are more involved; full legal action lists can be large. | Free/Apache-2.0 framework with third-party components to audit. No credentials. Strong alternate engine/search framework; do not install a large training stack just to run a hand. |
| **Existing local `gmsim.State("hunl", config)`** [I5] | Source available now; ACPC HUNL, exact raises, Python frames/candidates, recorded PokerKit cross-check fixtures. Not a globally published poker standard. API and rules inspected locally. | Rust/PyO3 backend; requires building extension plus NumPy. No built extension found. Frames encode values rather than friendly JSON. | Repository source Apache-2.0; README says combined gmsim binaries are GPL-3.0-or-later because of linked chess code. Free offline use is suitable. **Fallback** if local build dependencies can be supplied. |
| **ACPC dealer and protocol** [I6] | Long-used competition C dealer with game definitions and socket clients. Private `MATCHSTATE` messages contain position, betting history, visible cards; actions encode fold/call/raise-to. | Local subprocess/TCP overhead; match timeout rules must be disabled/raised for Astra. More protocol and build work; exact chip semantics differ from PokerKit. | Free/open-source; inspect chosen dealer fork's LICENSE. No remote account when self-hosted. Good interoperability/duplicate infrastructure, overkill for this adapter. |
| **PyPokerEngine** [I7] | Older Python bot framework: callbacks such as `declare_action(valid_actions, hole_card, round_state)`, JSON-like public state, raise intervals. Emulator companion supports simulations. | In-process; Python 3.12 compatibility and heads-up edge cases need testing, maintenance status not checked live. | Free/MIT. Low ToS risk locally. Convenient opponent callbacks, less attractive than locally evidenced PokerKit. |
| **Small purpose-built engine** | Full control over integer chips, decks, view filtering and arbitrary raises. Can reuse a licensed hand evaluator, but an evaluator alone is not a betting engine. | Standard-library-only is possible on Python 3.12; seven-card brute force evaluates 21 five-card subsets. Engineering/verification cost is the main latency. | No fee; preserve licenses of copied material. Viable last resort offline, but short all-ins, return of excess, and action order need independent tests. No such code is written in this study. |
| **Self-hosted poker server / PokerTH / custom web client** [I8] | Server protocols, source modifications, browser DOM, or plugins can expose public tables and accept actions. Bot interfaces vary by project; PokerTH is GPL software. | Headless server may work; GUI client requires display emulation. Clocks, reconnects and server internals add complexity. | Free options exist; check individual licenses and server rules. Offers no benefit over a direct engine for this task. |
| **Slumbot public bot API** [I9] | Local repository includes a client; receives action strings, own cards, board, position, winnings; sends poker actions. Strong external benchmark. Optional login for leaderboard in that client. | Network service and rate limits; remote availability/timeouts not guaranteed. Cannot run offline or control deals freely. | Public API described as free by local client, current terms unverified. **Excluded by offline requirement**, not the selected opponent. |
| **Commercial poker sites, screenshots/OCR + input injection, browser automation, emulator memory hooks** | Can theoretically read visible cards/chips and click actions. No universal official Hold'em bot API. Hand-history readers are retrospective and usually cannot act. | GUI/display or browser needed, parsing errors, network, clocks; hooks are client-specific. | Many sites prohibit bots/real-time assistance and may require real money/accounts. Incompatible with offline/free benchmark. No reason to use or evade their controls. |

Supporting libraries such as **Treys**, **eval7**, and **phevaluator** can accelerate hand evaluation/equity but do not implement the betting rules or match. Open-source CFR tools and solvers address strategy, not observation/control; no solver license, pretrained checkpoint, GPU, or online chart subscription is required. [I10]

## Installation and reproducibility

Use **only** `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/poker/bin/python` for install and validation. It was verified as CPython 3.12.3 on the supplied Linux environment. Do not accidentally install into the system Python, a temporary `uv run` environment, or another game's venv.

Required software: existing Python/pip, **`pokerkit==0.7.6`**, plus whatever dependencies its pinned wheel metadata actually declares. The core library is Python-based; no renderer, Torch, CUDA, browser, display server, or paid solver is required. Optional later acceleration is a separately pinned evaluator, only after a Python 3.12 import/performance check. Do not present optional packages as installation prerequisites.

Provision a `wheelhouse/` directory in this study directory from a trusted connected build machine, preserving licenses and recording SHA-256 hashes. Download the exact PokerKit version and its dependencies for Python 3.12/Linux x86_64; a universal pure-Python wheel needs no CPU-specific build. Verify distribution metadata (`Requires-Python`, `Requires-Dist`) rather than assuming today's latest release has the same API. A staging command on a connected compatible machine is `python3.12 -m pip download --dest wheelhouse 'pokerkit==0.7.6'`; this is a setup instruction, not a command executed during this study.

Then install on this machine, when the adapter environment has write access to the venv:

```sh
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/poker/bin/python -m pip install --no-index --find-links=./wheelhouse 'pokerkit==0.7.6'
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/poker/bin/python -m pip check
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/poker/bin/python -c 'import importlib.metadata as m; from pokerkit import Automation, Mode, NoLimitTexasHoldem; print(m.version("pokerkit"))'
```

Keep wheel hashes, installed versions, adapter version, opponent parameters, and rule configuration in a manifest. A package upgrade requires rerunning legal-action and payoff checks. Missing network is not solved by blindly retrying pip.

**Credentials and environment variables: none required.** No account, server address, API key, or model credential belongs in the game adapter. Supply hand count, seed, stack and opponent configuration through reset/config arguments. Model service setup, if any, is outside the game engine; “offline” here means the simulator and opponent need no external service. Running Astra/Jev themselves locally would require separately available model infrastructure, not provided by a poker package. Optional Slumbot credentials mentioned in its source are irrelevant and must not be requested for this task.

## Exact engine contract, evidenced by the local reference

Local authority: `/home/hemlix/GamingMaster-mh-tier1/scripts/reference/pokerkit_hunl.py` and `tests/test_hunl_reference.py`. These use the following symbols in 0.7.6; this study read them but did not import the missing package. [I1, I5]

Create `NoLimitTexasHoldem.create_state` with arguments in this order:

| Argument | Value |
|---|---|
| automations | `ANTE_POSTING`, `BET_COLLECTION`, `BLIND_OR_STRADDLE_POSTING`, `HOLE_CARDS_SHOWING_OR_MUCKING`, `HAND_KILLING`, `CHIPS_PUSHING`, `CHIPS_PULLING` from `Automation` |
| ante trimming | `True` |
| antes | `0` |
| blinds/straddles | `(50, 100)` |
| minimum bet | `100` |
| starting stacks | `(20000, 20000)` |
| player count | `2` |
| mode keyword | `Mode.TOURNAMENT` |

The mode chooses engine rule behavior; do not implement tournament elimination. Leave hole dealing, board dealing, and card burning **manual** so the adapter controls every card. PokerKit reverses the two blinds for heads-up: **engine 0 = BB, engine 1 = SB/button**. Verify immediately that posted bets are `[100, 50]` and, after hole dealing, the actor is 1. Both engine seats are stable; map hero identity to them per hand.

Useful interface members:

| Purpose | Member / operation |
|---|---|
| Hand active? / actor | `state.status`, `state.actor_index` (may be `None` while resolving chance) |
| Public accounting | `state.starting_stacks`, `state.stacks`, `state.bets` (current-street contributions) |
| Deal readiness | `can_deal_hole()`, `can_deal_board()`, `can_burn_card()` |
| Explicit chance | `deal_hole(card_text, seat)`, `deal_board(card_text)`, `burn_card(card_text)` |
| Legal passive actions | `can_fold()`, `can_check_or_call()` |
| Legal raising | `can_complete_bet_or_raise_to()`, then `min_completion_betting_or_raising_to_amount`, `max_completion_betting_or_raising_to_amount` |
| Apply | `fold()`, `check_or_call()`, `complete_bet_or_raise_to(amount)` |
| Terminal result | `state.payoffs`, after automated settlement completes |

Card text is two characters, e.g. `Ac`, `Td`, `7h`; ranks `23456789TJQKA`, suits `cdhs`. The reference demonstrates explicit `deal_hole` with a seat and `deal_board` one card at a time. It burns the unknown card token `"??"`. For the new adapter, use actual privately predetermined burn cards; verify this normal explicit-card path after installation. Do not copy the reference's unknown-burn convention and simultaneously consume additional real burn cards from the same deck.

Maintain a private card ledger and public history yourself. That avoids relying on undocumented serialization of `state.__dict__`, internal operation objects, or a full engine dump. The ledger records only successfully applied operations. Do not parse `repr(state)` for observations.

## Deterministic deals and event loop

1. `reset(seed, config)` establishes a fixed number of hands, rule/opponent version, and independent random streams for deals, opponent randomization, and policy mixing. Use stable derivation such as SHA-256 of a domain label plus seed and hand index, not Python's process-randomized `hash()`. Keep seeds private from both policies.
2. Generate the whole 52-card permutation **before actions**. One acceptable layout is seat 0 holes = indices 0,1; seat 1 holes = 2,3; flop burn = 4; flop = 5,6,7; turn burn = 8; turn = 9; river burn = 10; river = 11. Grouping hole cards by seat has the same deal distribution as alternating them. Document and freeze the chosen layout.
3. Create a fresh engine state, post blinds through automations, and explicitly deal the two cards to each seat when allowed. A private ledger tracks which part of the fixed schedule is next. Future cards never depend on how many random calls the opponent or policy made.
4. Repeatedly resolve pending burn/deal operations and the selected automations until `actor_index` exists or `status` becomes false. On a flop, finish all three flop cards before returning a decision. A called all-in can require several consecutive board streets without further player actions.
5. On an opponent turn, build its sanitized observation and call the frozen local policy. On a hero turn, return a fresh JSON observation and pause indefinitely until `step(action)`.
6. Validate and apply the hero action; record action type, street, engine seat, street raise-to, incremental cost, and pot before/after. Continue step 4. Return only at another hero turn or at hand completion. No model calls for chance events.
7. At terminal, finish collection/payout, record the hero's `payoffs[hero_seat]` exactly once, expose only legally revealed showdown cards, and move to the next scheduled hand with fresh stacks. `done` for a match means the predeclared hand count is complete. Expose a separate `hand_terminal` marker if useful.

Treat an active state with no actor and no recognized pending operation as an adapter error; do not spin forever or silently force a check. Exceeding an internal defensive event bound should report failure, not settle an unresolved hand as zero.

## JSON observations and actions

The following is a **proposed adapter schema**, not PokerKit's own output. At a hero decision include:

| Field group | Required content |
|---|---|
| Identity/progress | `schema_version`, opaque `hand_id`, completed/required hands, `street`, `hero_seat`, `button_seat`, `actor_seat` |
| Cards | `hole_cards` exactly two; `board` length 0/3/4/5. Never future board or opponent holes. |
| Money, integers | `small_blind`, `big_blind`, `starting_stacks`, `stacks`, `street_bets`, `committed_total`, `pot`, `to_call`, `effective_stack` |
| Legality | `can_fold`, `can_check`, `can_call`, `can_raise`, nullable `min_raise_to` and `max_raise_to`; both bounds explicitly **street totals** |
| Candidate actions | Unique legal action IDs with exact amounts and incremental costs; no hidden random state. |
| Context | Full public betting history for the current hand; public prior-hand statistics with sample counts; optional code-calculated hand/draw/equity features and uncertainty. |
| Results | Public cumulative score, completed-hand result when applicable, terminal/match flags. No evaluator-only omniscient logs. |

At live decision points before payout, derive total committed for seat `i` as `starting_stacks[i] - stacks[i]`. Pot `P` is the sum of both commitments, including uncollected current bets. Never add `bets` to this again. Derive prior-street commitment as `committed_total[i] - bets[i]`. The amount needed to match the wager is `max(bets) - bets[actor]`, capped at the actor's stack for an all-in call. Baseline equal starting stacks avoid unmatched-amount complexity, but the engine still owns refunds. Do not reuse these formulas after payouts as a substitute for `payoffs`.

Use either a candidate ID or one typed object: `{"action":"fold"}`, `{"action":"check"}`, `{"action":"call"}`, or `{"action":"raise_to","amount":300}`. A raise-to amount includes that player's **current-street** contribution already paid, not earlier streets. Validate JSON type (reject booleans as amounts), integer chip units, action membership, actor, bounds, and engine legality before mutation. A check is allowed only at zero call price; a call is allowed only at positive price. Both dispatch to `check_or_call()`.

Preflop example: SB has paid 50; BB 100. Opening to 300 pays another 250. The minimum full reraise is to 500. Postflop example: earlier streets each cost 300, current bets are `[0, 0]`; a bet to 200 passes **200** to PokerKit, but is a **500** total-hand commitment in an ACPC/gmsim interface. Mixing those conventions corrupts both legality and scoring.

Candidate sizing: let `b` be the actor's street contribution, `m` the highest street contribution, `C = m-b` the full call price, and `P` the pot including outstanding bets. For a legal raising situation, a fraction-`f` pot raise is a street target `m + f × (P+C)`, rounded to integer chips, clipped to the engine bounds, then revalidated and deduplicated. Its incremental cost is target minus `b`. With no wager this reduces to a bet of `f×P` (on a new postflop street `m=0`). Prefer candidates min-raise, 1/3, 1/2, 3/4, 1, 1.5 pot, and maximum, plus relevant preflop targets in big blinds. Fractions refer to a **raise increment after calling**, not a final total. If call consumes the stack or the opponent is all-in, omit raises. Keep full integer raises available if the adapter contract permits, even if the model usually selects candidates.

Do not silently convert an illegal raise to a call. During development raise a validation error; during evaluation follow a declared policy (e.g. one retry within the harness budget, then legal check else fold), record the failure, and apply the same rule to all harnesses. Invalid actions are a diagnostic, not an invented chip reward. Repeated failures make the evaluation invalid.

## Privacy, duplicate deals, and scoring

Filtering is mandatory: the engine is omniscient. Neither Astra, Jev, nor the opponent may receive the raw state, opponent holes, burns, undealt cards, deck order, deterministic seed, engine snapshots, or debug histories containing chance cards. Equity calculation samples possible opponent hands from the **visible** state; it must not use the engine's actual private cards. Never expose hidden cards after a fold merely because the engine retains them. Public showdown history is permitted only when it would truly be revealed.

For duplicate pair `j`, create two separate episodes with the **same role-labeled private deal**: seat 0 keeps its cards and BB role, seat 1 keeps its cards and button role, and the board is identical. Put hero in seat 0 in leg A and seat 1 in leg B; the opponent takes the other seat. Average both hero payoffs. The two legs need not have the same betting sequence; variance is reduced, not eliminated, and their sum is not generally zero.

**Prevent duplicate memorization:** fork the policy/opponent public memory from the same pre-pair snapshot, run each leg in a separate context, and do not deliver leg A's transcript/results to leg B. This prevents the second leg knowing the other player's cards or the board. Mask any pair identifier that announces a repeated deal. In this forked-pair design, both legs must also receive the same pre-pair cumulative-score/statistics snapshot; revealing leg A's payoff through an updated scoreboard would leak information. Evaluator alone aggregates the results after both legs. A simple default resets strategic memory for each pair and does not merge its transcripts into future pairs. If the harness cannot isolate/reset policy context, use ordinary independent hands with balanced seats instead of claiming fair duplicate dealing.

An alternative that supports adaptation is two isolated match lanes, initialized identically: each lane retains only its own prior-hand public history and lane-specific score, seats swap between lanes, and corresponding hands share the role-labeled deal. Lane histories can then differ; this is an alternative to cloning the same pre-pair history. Never merge lane transcripts during evaluation. For adaptive lanes, estimate uncertainty across independently seeded whole matches, not by treating individual pairs as independent.

Use an even fixed hand count `H=2M`; store hero chip outcomes `x_jA, x_jB`. Primary score is `sum(x)/H`. Pair values `y_j=(x_jA+x_jB)/2` have mean equal to this score; for fixed nonadaptive policies, estimate standard error as sample standard deviation of the `y_j` divided by `sqrt(M)`. If adaptation creates serial dependence, use independent match/seed blocks and report uncertainty across blocks. Convert to bb/100 only with the declared big blind. Compare harnesses on the same hidden evaluation deals; do not select new seeds based on losses or stop after reaching a profit target. [I11]

Main benchmark holds blinds, 200bb stacks, opponent policy and hand count fixed; seed varies unseen deals and fixed-policy randomization. Optional 20bb/40bb/100bb/200bb and different frozen opponent profiles are **separate labeled strata**, not an undocumented seed-dependent change to the primary game. Give public stack parameters to the policy; keep opponent profile identity hidden unless the benchmark explicitly makes it public. Hold out seeds and, for a separate robustness score, hold out parameter profiles. No training on held-out deals.

## Frozen opponent specification for the adapter phase

PokerKit supplies rules, not a suitable opponent. Build one deterministic/versioned heuristic policy with its own private seat view and seeded mixing. Freeze all thresholds before comparing harnesses. It must choose legal actions only and have bounded CPU cost. A defensible baseline specification is:

- Preflop: a fixed 169-class ranking/range table, wider button opening/limping range, tighter BB reraises, stack-aware calls; publish the chosen ranges and sizes in evaluator configuration. Do not use an unexplained oracle trained on test deals.
- Postflop: equity from its own cards against a declared visible-information range, pot-odds calls with an explicit realization margin before the river, more value bets with stronger equity, and a small fixed bluff/semibluff component. Use public board texture, action history and stack-to-pot ratio. Equity sampling uses a separate fixed stream and declared sample budget.
- No live learning or model API calls in the opponent for the primary benchmark. Randomization still describes a fixed policy. Public history can condition actions within a hand without changing policy parameters.
- Freeze value thresholds, aggression/bluff rates, preflop table, legal size menu, and RNG derivation in the versioned manifest. Always-check/call and always-fold policies are useful plumbing tests but too exploitable to be the sole benchmark opponent.

Optional tight-passive, calling-station and aggressive profiles should be separate tests. Do not tune the benchmark opponent while tuning the hero, and do not silently switch to a random policy if opponent provisioning fails.

### Fully specified starter opponent: `heuristic-v1`

The following is an implementable **benchmark design**, not a claim of expert or equilibrium poker. It replaces the general suggestions above if no other frozen opponent is supplied. It has deliberate, stable weaknesses for Astra to discover. It requires no trained weights or external chart. Implement it during the adaptor phase, not as part of this study.

Define `premium = {QQ, KK, AA, AKs, AKo}`. Define `open_range` as the union of all pairs, all suited nonpairs, all offsuit aces, all offsuit kings, and `{Q7o,Q8o,Q9o,QTo,QJo,J8o,J9o,JTo,T8o,T9o,98o,87o}`. `s` means suited; `o` means offsuit. These sets are explicit, not percentile labels. Track the number of voluntary bets/raises already made **by this opponent** on the current street (blinds do not count).

At each decision compute `e`, its showdown equity against uniformly distributed legal opponent hole cards and board completions. Use 512 independent Monte Carlo samples on preflop/flop/turn (sampling without replacement within each sample), counting wins as 1 and ties as 0.5. Enumerate all legal opponent combinations on the river. The evaluator must be the verified PokerKit hand evaluator or another validated exact evaluator; consult the installed version's evaluation API rather than guessing its rank-number direction. This equity input intentionally ignores betting-range inference; document this weakness. Only the acting opponent's own hole cards and exposed board may enter the sampler. Sample RNG is separate from deck RNG and action-mixing RNG. Reuse an equity result for unchanged visible cards within a hand; never condition samples on actual hidden cards or burns.

Evaluate these branches in order. Any desired raise is rounded to the nearest integer (ties upward), clipped to the engine's legal min/max, and verified. If raising is unavailable, skip that raising branch. A fallback with nothing owed checks; a fallback with a positive price uses the specified call/fold test. Compare thresholds inclusively (`>=`).

| Context | Fixed behavior |
|---|---|
| Button preflop, no voluntary raise yet, first action | If in `open_range`, raise to 250 chips; otherwise fold. If raise unavailable, call if `e >= 0.50`, otherwise fold. |
| BB preflop facing only a limp (no raise), free check available | If `e >= 0.60`, raise to 400; otherwise check. |
| Other preflop decisions | With `premium`, raise to three times the current highest street bet if it has not voluntarily raised this street; after one previous own raise, shove if legal. Otherwise call if `e >= max(0.44, q+0.08)`, where `q=C/(P+C)`; else fold. If nothing is owed, check. Premium hands that cannot raise use the same call test. |
| Postflop, nothing owed | If raising is legal and `e >= 0.62`, bet 0.75 pot; if `e >= 0.85`, use 1 pot instead. Otherwise bet 0.5 pot with probability 0.08, if legal; check otherwise. This last branch is an intentionally simple mixed bet, not necessarily a pure bluff. |
| Postflop, facing a bet | If `e >= 0.80`, no previous own raise on this street, and raising is legal, raise by 0.75 pot **after calling**, using the formula in the sizing section. Otherwise call if `e >= q+margin`; else fold. `margin=0.05` on flop/turn and 0 on river. |

Use the common minimum of the two remaining stacks for effective-stack features; the engine still controls legal all-in targets and refunds. All-in calls use the actual contestable pot. No opponent action uses Astra or Jev. Action-mixing draws are reproducible and independent of how many equity samples were used. Freeze these exact sets, thresholds, sample count, RNG domain labels and rounding convention in the opponent manifest. Use this version only for the stated 200bb primary configuration unless a separate depth-specific version is declared. Optional stronger opponents can use range inference from `strategies.md`, but their scores are separate benchmarks.

This specification makes all decisions defined, but does not guarantee good poker: uniform equity particularly overcalls against strong ranges, and rare random bets do not create a well-balanced bluff range. Report the opponent version next to results; beating it is not evidence of beating human experts.

## Acceptance checks before trusting scores

Run these in the **evaluator Python**, after provisioning, as adapter verification work (not performed in this study):

1. Import/version and known-card showdown checks: board plays, wheel straight, flush/straight ordering, full house, kicker and split pot. Compare final engine payoffs with independent known expected values.
2. An immediate SB/button fold yields engine-seat payoffs `[+50,-50]` (seat 0 = BB, seat 1 = SB); map that result to hero identity. Limp/check reaches a flop with a 200-chip pot; button/BB order is correct on all streets.
3. Open to 300 gives a full minimum reraise to 500; no-limit maximum is stack-constrained. Short all-ins, all-in calls, no raises against an all-in opponent, refunds, and tie settlement are correct. New-street raise totals start from zero contributions, not whole-hand commitments.
4. Called preflop all-in automatically reaches showdown without further hero decisions. Every completed hand conserves total chips; zero-sum payoffs; stack reset and score accumulation happen once.
5. Determinism: same seed/config/action history reproduces observations and payoff. Different policy RNG calls cannot change the deal. Duplicate legs preserve seat-labeled cards and board while swapping policies.
6. Privacy: alter only opponent cards/future deck while holding the visible prefix fixed; hero observation and calculated features at that prefix must be identical. Repeat from opponent's perspective. Ensure no seed leak and no duplicate-leg transcript sharing.
7. Replay the local PokerKit reference fixture or selected trajectories using its **unknown-burn** convention. It contains 240 recorded hands across ACPC, older blind units, short-stack and odd-blind configurations. Keep this test convention separate from the adapter's actual-burn deal layout. Do not merely count recorded successes as a new test pass.
8. Complete a small fixed match without rendering/network; verify hand count, illegal-action reporting, and chips/hand versus manually summed ledger. Measure CPU cost for engine transitions and equity calls separately from model latency.

## Offline fallback if PokerKit cannot be staged

The repository already has a source implementation; prefer verifying/reusing it over inventing another betting engine. `data_processing/rs/gmsim/pyproject.toml` declares Python >=3.10, `numpy`, and a `maturin>=1.7,<2` build backend. The Rust extension uses abi3 for CPython >=3.10, which includes the evaluator's 3.12.3. Stage compatible NumPy (the repository recommends `<2` on older CPUs), maturin, Rust/Cargo and required Cargo crates, or a prebuilt trusted Linux x86_64 abi3 wheel. **These artifacts were not found/verified here**, so this is not a claim that the fallback is already runnable. Build outside the read-only source checkout if necessary; install the resulting wheel into the evaluator venv. Avoid a root-project `uv sync`, which would install unrelated Torch/model dependencies.

Locally inspected fallback API: construct `gmsim.State("hunl", {"small_blind":50,"big_blind":100,"stack":20000})`; `turn()` gives `("chance",[])`, `("act",[seat])`, or `("terminal",[])`. Use `apply_chance(card_id)` where ID = rank index in `23456789TJQKA` times 4 plus suit index in `cdhs`. Deal seat 0 twice, seat 1 twice, then board 3/1/1; this engine omits burn events, so choose a matching documented deck layout. `legal_actions(seat)` returns fold 0, check/call 1, and raises >=2 as **total-hand commitment**. Apply `apply_action(seat, id)`, and read terminal `returns()` in chips. Read `frame(seat)` only for that seat; `decode(seat,[candidate_index])` translates its 15-entry abstract action mask to an exact canonical action. Retain your own public money/history ledger rather than reversing lossy normalized tensors. Never forward `history()` or full state strings: chance history contains private cards. The fallback's action cap (63 already-recorded street actions prevent further raising) is a declared ACPC implementation detail; test and document differences from PokerKit.

If neither package nor build dependencies are available, the adapter phase can implement a small standard-library engine and validate it against the local golden trajectories. That is additional implementation/verification work, not an installed dependency or completed result of this study.

## Source URLs

- **I1 PokerKit:** <https://github.com/uoftcprg/pokerkit>; <https://pokerkit.readthedocs.io/en/stable/>; distribution <https://pypi.org/project/pokerkit/0.7.6/>. Local reference, not the mutable latest documentation, establishes APIs used above. The installed distribution's own metadata must establish its actual Python requirement and dependencies.
- **I2 RLCard:** <https://rlcard.org/>; <https://rlcard.org/games.html>; <https://github.com/datamllab/rlcard>; no-limit source <https://github.com/datamllab/rlcard/tree/master/rlcard/games/nolimitholdem>.
- **I3 PettingZoo:** <https://pettingzoo.farama.org/environments/classic/texas_holdem_no_limit/>.
- **I4 OpenSpiel:** <https://openspiel.readthedocs.io/en/latest/>; <https://github.com/google-deepmind/open_spiel/tree/master/open_spiel/games/universal_poker>.
- **I5 local engine/reference:** <https://github.com/sunyxedu/GamingMaster>; verified local files listed in `research_log.md` and above.
- **I6 ACPC:** <https://github.com/jblespiau/project_acpc_server>; <https://www.computerpokercompetition.org/>. Inspect the selected dealer's game definition and protocol document before using its total-commitment action notation.
- **I7 PyPokerEngine:** <https://github.com/ishikota/PyPokerEngine>; <https://ishikota.github.io/PyPokerEngine/>.
- **I8 PokerTH:** <https://github.com/pokerth/pokerth>; <https://www.pokerth.net/>.
- **I9 Slumbot:** <https://slumbot.com/>; local `scripts/slumbot_client.py` documents endpoint and action semantics. Its service availability has not been tested.
- **I10 evaluators:** <https://github.com/ihendley/treys>; <https://github.com/julianandrews/pyeval7>; <https://github.com/HenryRLee/PokerHandEvaluator>.
- **I11 variance reduction:** Burch et al., *AIVAT: A New Variance Reduction Technique for Agent Evaluation in Imperfect Information Games*: <https://arxiv.org/abs/1612.06915>. AIVAT itself requires more than duplicate deals and is **not** claimed as implemented here. ACPC historical data: <https://doi.org/10.5281/zenodo.10796885>.
