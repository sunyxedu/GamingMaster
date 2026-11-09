# Candidate 0023 — preserve the landing, finish with a rotation

## Parent and one hypothesis

Parent: **0017**, the best evaluated candidate (mean 63,216.67). This is a
refinement, not a new placement strategy. Keep its immediate guardian,
background one-preview/depth-two refinement, evaluator weights, hold handling,
Jev roles, Astra modes, and tick ownership.

**Hypothesis:** part of the score deficit is in execution, not placement
selection. A T placement already chosen by 0017 can lose its scoring
classification when its last successful maneuver is a translation. Preserve
the exact selected landing and prefer a short SRS route whose final successful
maneuver is a rotation. Never rerank placements for spin value, force a cavity,
or change the board to obtain a spin.

## Evidence from the weakest parent traces

The actual lowest parent score is seed **5**, not seed 4:

* `0017/trace_seed5.jsonl`, final observation: `pieces: 200`, `lines: 79`,
  `holes: 0`, `score: 56834`, and score components `line_clear: 48100`,
  `drop: 6834`, `combo: 1900`, `back_to_back: 0`. Survival is already solved.
* At ticks **93–100**, seed 5 uses `hold`, four `right` inputs,
  `rotate_ccw`, `right`, `hard_drop`. The lock at tick **100**, piece **24**,
  is T at `[[9,2],[8,1],[9,1],[9,0]]`, `lines: 1`,
  `clear_kind: "none"`, `points: 100` (trace line 133). The pre-lock board and
  these cells establish three occupied/wall diagonal corners around its center.
  The final `right`, not the landing geometry, prevents a rotation-ending lock.
* The same pattern occurs at seed-5 ticks **170–175**, piece **41**:
  three `left` inputs, `rotate_cw`, `left`, `hard_drop`; the one-line T lock
  reports `clear_kind: "none"`, `points: 200` (trace line 228).
* In the second-lowest episode, seed **4**, ticks **804–809**, piece **186**,
  repeat three `left` inputs, `rotate_cw`, `left`, `hard_drop`. Its lock is
  `[[0,2],[0,1],[1,1],[0,0]]`, `lines: 1`, `clear_kind: "none"`,
  `points: 800` (trace line 1055), again with three diagonal corners.

These are not assertions that a new route will necessarily remain reachable
under gravity. They identify already-selected scored-clear candidates for a
bounded execution-only experiment.

Generation 6 cautions against changing the objective at the same time:
0018 lost survival; 0019 repeated rotation cycles; and 0020 recognized spins
accurately but changed placement preferences, increasing no-line spins from
65 to 99 without improving mean score. This candidate does **not** add any
spin bonus or spin role. It isolates route realization from placement choice
and gives route progress an explicit finite limit.

## Targeted change

Only the execution of an already-selected T landing is eligible for refinement.
Code checks the public settled board and exact landing cells, then searches a
short SRS route with a final rotation. The resulting drop must reproduce the
selected cell set exactly. The router tracks the actual successful last
maneuver from observations, including hold and translation invalidation.

A per-piece finite allowance prevents the fresh-rotation loops seen in 0019.
The route is abandoned on nonprogress, an incompatible target/hold state,
unverified geometry, insufficient gravity/lock slack, or exhausted allowance.
The parent's ordinary router remains the fallback, and its urgent-drop rule
keeps priority. No extra model call is needed for this mechanical decision.

This first draft is written before validation. Implementation details and
short, archive-only checks will be recorded below; no long game experiment or
model-backend call will be run by the proposer.

## System-1 state and answer-to-action bridge (unchanged)

Each request carries a strict-JSON object with:

* `ruleset`, Astra `policy`, and `piece_instance`;
* board height/hole/surface buckets, lines, level, combo-active flag, and B2B;
* active type/orientation/x, hold permission, held piece, and visible queue;
* descriptions of code-simulated `safe`, `clear`, `future`, `recovery` roles,
  `fallback_role: "safe"`, and candidate count.

One asynchronous `ctx.decide` request batches `pick_a` and `pick_b`, independent
Choices with opposite option orders. Both must select the same valid role with
selected-label probability at least **0.45**. Confidence is not substituted
for probability. Malformed, uncertain, incompatible, or other-piece answers
leave the code-selected plan in force. The inherited timeout is **0.60 s**;
the parent measured about **0.25–0.32 s** median per episode and a roughly
**0.57 s** observed successful maximum, with five timeouts. No latency-policy
change is bundled into this experiment.

Jev selects a role, code selects its exact landing, and the executor converts
that landing to legal inputs. The new route refinement sits after that same
bridge, applies identically to guardian/search/Jev landings, and never changes
the Jev questions or the placement's numeric value.

Astra is still a nonblocking reviewer after six pieces and on a later danger
transition. A leading `BALANCED`, `RECOVER`, or `WELL` token adjusts only future
evaluator weights. Its **65 s** timeout is retained; malformed or cancelled
text is ignored. This experiment does not claim a new model-cooperation gain
or ask Astra to rewrite Jev's interface. A later interface revision would be
justified by measured same-piece role overrides improving or harming board
outcomes, not merely by model-call counts.

## Confirmation / refutation

Confirmation requires no contract errors, full 200-piece survival, zero illegal
actions, and near-parent stale ticks. The mechanism is supported when traces
show a route-only refinement reaching the identical chosen cell set with a
successful final rotation, followed by the engine awarding a line-clearing
spin. Mean score should exceed 0017 through line-clear/B2B points without
worsening lines or holes. Engine lock events, not predicted spin labels, are
the source of truth.

It is refuted if refinements are rarely reachable, change landing cells,
introduce long rotation sequences, reduce survival, or lose more drop/clear
value through delay than they gain. More no-line spins alone would not confirm
a difficult-clear gain. Because wall-clock timing can change when background
searches arrive, equal placement-ranking code does not guarantee identical
whole-episode board trajectories; route audits must distinguish local
realization from that timing effect.
