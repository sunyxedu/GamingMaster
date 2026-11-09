parent: 0014

# Hypothesis

The endgame scheduler in 0014 promotes clues by action type even when they create
no executable play. This turns a useful rule (communicate before the recipient's
last turn) into compulsory unproductive clue spending. Restoring the distinction
between actionable and non-actionable clues should improve late deck access and
clue availability without changing the communication convention or risk policy.
This tests a scheduler correction on a different parent from leader 0013.

# Evidence

Leader 0013's weakest training episode ends at 21 with all lives and stacks
R4/Y4/G3/W5/B5. Its final three plays, turns 70-72, are a late yellow prerequisite
chain; surviving Y5/G4/G5 still cannot be completed. This motivates investigating
turn access separately from the R5 loss at turn 53 (already explored by other
proposals).

Parent 0014's seed-14 trace provides a specific implementation failure. At turn
63 (deck 4) it chooses REVEAL_RANK 0 1, value -0.55, with `new_safe_slots: []` and
`save_unique_slots: []`. At turn 65 (deck 3), it chooses REVEAL_COLOR 0 B, value
-0.63, again creating no safe slot or save. Public-promise replay of these archived
observations reconstructs an available DISCARD 4 with `p_dead: 1.0`, zero expected
loss, and base value 1.702 at both turns. The scheduler assigns every non-clue a
priority of -1e9, so even a zero-benefit clue wins. No rollout overrode that parent
choice in these two actual decisions. This is direct evidence of the ranking bug,
not proof of a counterfactual score increase.

# Change from parent

Retain 0014's exact finite-population beliefs, public clue/discard protocol,
masked paired rollouts, phase boundary (six deck cards or final round), certain
play ordering, final-action handling, separate model sessions and caught timeout
fallbacks. When no own certain play exists in the short-horizon phase, promote a
clue only if it creates a new certain partner slot, the partner has no existing
certain play, and at least one partner action remains. Among promoted clues and
among all other actions, use the existing scalar value. Literal preparation and
save clues remain legal candidates and can still win by their base value or
paired rollout evidence; they simply receive no unconditional type priority.

Also repair a menu-construction defect in that same scheduler: it copies safe
entries and adds an extra priority key, then uses dictionary equality to remove
them from the original list. The unequal copies leave duplicate PLAY actions.
Filter the remainder by action dictionaries instead. This prevents duplicate
options from consuming the two-per-family search shortlist. No posterior,
threshold, protocol or early-game candidate menu is changed. The corrected
continuation also runs inside rollouts, so early search decisions can still
change when their estimated terminal values change.

# Confirmation and refutation

Confirm the mechanism if archive replay promotes no clue without a new executable
partner slot, keeps one entry per legal action, and restores the ordinary discard
ranking at the cited turns. Scoring support requires better terminal scores or
fewer unproductive late clues without additional critical-card losses or misplays.
Refute if those apparently unproductive clues provide valuable delayed deductions
that the inherited base scorer/search cannot recover, or if drawing earlier starts
the countdown before a prerequisite chain can execute. Replay is a local ranking
audit, not a new evaluation or evidence that the same subsequent draws occur.

# Validation

Harness and proposal were written before experiments. Initial preflight passed
before refinement; final preflight, archive replay, explicit adviser fallback
checks and any permitted dev evaluation will be recorded below. No full training
or held-out evaluation is run.

## Completed checks

Final required preflight passed: natural game completion, 64 legal actions, no
error, and no model calls. Its 23-point stand-in result is a contract check only.
The explicit adviser fixtures separately passed malformed/out-of-range replies,
valid JSON selection, RuntimeError, backend timeout, outer queue timeout and
separate resumed sessions for both seats.

Observation-only replay of all four parent training traces checked 273 menus.
All 217 early-game menus match the parent exactly. Twenty endgame top rankings
change; all menus retain the same set of protocol-valid actions, contain no
duplicate actions, and use only legal dictionaries. Twenty-two parent menus had
duplicated safe plays. At the two cited turns the new top candidate is exactly
DISCARD 4, value 1.702 and certain-dead probability 1.0, in place of clue values
-0.55 and -0.63. Public-promise replay and face-erased discard codebooks remain
identical. Synthetic scheduler checks cover actionable/empty clues, already-safe
partners, unique safe entries and remaining recipient turns. Inherited fixtures
for final-draw counting, immediate bomb-out/perfect endings and five-token
recovery also pass. These checks are retained in `validate.py`.

## Dev result

The permitted dev evaluation on seed 201 completed **25/25** in **65 actions**
with all three lives, seven clues and no episode error. It used one Astra call;
the first adviser request timed out safely and autonomous policy continued. This
is one unseen-to-this-proposal dev deal and does not establish training or held-out
performance, but it is consistent with the scheduling hypothesis: the episode
reached a perfect terminal state without a late unproductive-clue failure.
