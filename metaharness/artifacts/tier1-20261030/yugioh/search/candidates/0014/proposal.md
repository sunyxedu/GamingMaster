parent: 0007

## Hypothesis

Combat search needs a conditional action policy across native timing windows, rather than either refusing unsupported attacks (0008) or spending the combat trick before declaring them (0007). Reserve a currently legal Rush Recklessly, declare the supported attack, observe opponent interaction, and commit Rush only in a verified Damage Step window when its marginal battle outcome still warrants a card. This tests timing and option preservation, not deployment weights or the leader's exchange recipe.

Concrete public trace evidence:

- Leader 0008, worst seed 1111, steps 13 and 166: Luster Dragon (1900 ATK) and legally available Rush face Gene-Warped Warwolf (2000 ATK), but the controller exits Battle. It spends Rush on the subsequent opponent turn instead.
- Parent 0007, seed 1104, steps 191–195: `choose r3p12 0` spends Rush on Luster before attacking; `select r3p16 0` then exposes Mirror Force, which destroys Luster. Rush has already been consumed.
- Parent 0007, seed 1103, steps 133–137: Rush is consumed before Gemini Elf attacks; Book of Moon then turns that attacker face-down and prevents combat. Again the combat trick is gone before the attack can benefit.
- Native timing is directly observable: parent 0007 seed 1105 steps 177–180 offers Rush at attack declaration, `Battle Phase`, `Start of the Damage Step`, and `Before damage calculation`. The top-level phase remains `Battle start`; inspecting phase alone cannot implement delayed commitment.

These are observed resource losses and timing opportunities, not claims that preserving Rush guarantees a counterfactual win.

## Change from parent

Preserve 0007's legal menu validation, bounded visible-board combat search, position semantics, guarded Astra transport and per-action progress checks. Replace proactive Rush execution with a reserve/declare/observe/commit state machine. The search still requires a currently offered spell and attacker, a known face-up opponent, and enough projected gain to pay for one card. Its first action is the attack, with the intended defender rebound to the live target menu.

Rush is then evaluated at public damage-step markers, including on the opponent turn. Recompute current fighters and public pending Rush/Book/removal effects; decline if a pending boost already suffices, the fighter is face-down, the battle was cancelled, or raising ATK cannot improve a defending monster's outcome. Bind any actual activation to the surviving friendly fighter. Never invent a response or future token.

Audit the reservation after every action alongside ctx.progress(). An interrupted attack conserves Rush. A missed promised damage window or failed supported battle records expected/observed evidence, disables speculative attacks for that duel, clears intentions, and requests Astra repair. The planner sees the reservation and its observed outcomes. Model timeouts remain 235 seconds with an independent 210-second wait guard, both above the observed 81-second recent maximum and 120-second archive maximum. Exceptions and malformed text recover to legal local play.

## Confirmation and refutation

Confirm the mechanism if the archived pre-buff positions instead attack while retaining Rush, attack-declaration Book/Mirror Force leaves Rush unspent, and surviving combat activates one sufficient Rush at a displayed damage window. Refute if timing markers are missing, the reserved spell ceases to be available, buffs target the wrong fighter, or unsupported attacks lose monsters. Actual strength requires more native wins across training and unseen scenarios without errors or forfeits. Replay alone establishes neither a win-rate improvement nor hidden-card predictions.

## Validation

Final preflight passed: native `game_done`, 455 actions, no exception. The stand-in run answered no model calls, so explicit fixtures additionally checked garbage text, wrong JSON types, empty answers, TimeoutError, RuntimeError, stale tokens, duplicate subsets, weighted Tributes and incomplete orders.

Public-observation replay in `checks.py` covered **7,063 decisions** from parents 0007 and leader 0008 with **zero invalid actions**. Four archived pre-buff/missed-attack positions now choose the legal supported attack. Native-schema fixtures check declaration-versus-damage timing, redundant pending Rush, face-down fighters, cancelled attacks, failed combat recovery, the observed +700, and target cleanup after resolution.

Both allowed development evaluations completed:

| Seed | Score | Native wins/losses | Actions | Wall time | Astra requests |
|---|---:|---:|---:|---:|---:|
| 3301 | 0.3785417 | 2 / 1 | 473 | 218 s | 1 |
| 3302 | 0.3720833 | 2 / 1 | 251 | 216 s | 1 |

All six duels completed natively with no invalid actions, errors, forfeits or truncations. Mean development score is **0.3753125**. Both initial model requests reached the 210-second independent wait guard, so no Astra answer was applied. Neither development seed exercised a new combat reservation or a useful late Rush commitment. These runs establish completion only; the timing mechanism is presently supported by archive replay and fixtures, not a native counterfactual win.

The first development run preceded the same-window effect audit and the check that preserves Rush instead of spending Book/Compulsory merely to cancel a battle Rush can win. The second run included those changes but preceded final target-cleanup hardening. These final changes passed fixtures and final preflight; no third development evaluation was run. The final cleanup prevents a completed Rush from binding a later attack-target prompt, and is not a new strategy hypothesis.

No full training evaluation, game implementation source, private state, held-out or validation result was read or run. The evaluator's training and unseen-seed results must confirm or reject playing-strength gains.

## Limits and next evidence

This is a deliberately narrow conditional combat controller. It retains the parent's other spell/tribute policies and does not solve their known resource-allocation weaknesses. The visible combat search does not know hidden interaction, future opponent draws or summons. Declaring a supported attack still risks losing the fighter to Mirror Force; the intended benefit is preserving the unspent spell when that happens. Observed timing failure disables supported attacks for that duel and escalates the evidence to Astra, within the remaining call/time budget.
