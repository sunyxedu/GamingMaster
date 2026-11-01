# Candidate 0003: placement bridge for Jev

## Parent and targeted hypothesis

Parent: `candidates/0000-tier2_jev_only/`.

The parent’s failure is not that Jev cannot name a legal action; it is that a
primitive-action question gives it no stable way to finish a piece. In the
worst archived episode (seed 3), the trace shows Jev selecting `hard_drop` on
all five calls, with probabilities for that choice of 0.59, 0.51, 0.50, 0.45,
and 0.52. The resulting ticks repeatedly re-applied `hard_drop` while the
model was thinking: ticks 4--14 contain stale `hard_drop` actions. The episode
ended after only 11 pieces with score 190, zero lines, 33 holes,
`top_out_height: 20`, and `terminal_reason: top_out`. The parent’s aggregate
summary confirms the same pattern across seeds: 202 mean score, zero lines,
23.67 holes, and 73% stale ticks. There is no `lessons.md` in the archive.

**Hypothesis:** if Jev chooses a final horizontal placement rather than a
single key, and code turns that choice into verified one-tick movement followed
by a hard drop, repeated stale drop actions will stop building the same central
column. A short placement bridge should improve survival and produce line
clears without spending Astra calls. This isolates the answer-to-action bridge;
it does not add a new model or revise the parent’s Jev-only scheduling.

## What changed

There is still one batched System-1 Choice per decision and no System-2 call.
For the current orientation, the harness derives the active-cell offsets from
the observation, enumerates every horizontally legal anchor, simulates its
landing and line removal, and computes holes, aggregate height, bumpiness,
maximum height, and line clears. Jev receives these named `xN` placement
alternatives plus the compact board/queue/hold state. The best code-ranked
candidate is the deterministic fallback.

After the answer, the harness re-observes the game. If the answer is late,
malformed, refers to an old piece, or is no longer legal, it uses the best
current candidate. It sends one `left`/`right` action per tick, verifies the
observed x position after each tick, and then sends `hard_drop`. Thus a model
call cannot cause the next piece to inherit an unexamined `hard_drop` stream.
No rotations are synthesized in this candidate: limiting the bridge to the
observed orientation is intentional, so SRS coordinate assumptions are not
introduced as a second change.

## Jev state and questions

The state contains:

* the settled 20x10 board and exact current active type, orientation, x, y,
  and cells;
* ghost cells, hold/can-hold, queue, pieces, lines, level, and current stack
  holes/aggregate height/bumpiness/top-out height;
* one object per legal horizontal placement with label, target x, landing y,
  lines, holes, aggregate height, bumpiness, and maximum height.

The one `Choice` asks: choose the safest legal final placement, preferring a
line clear, then no holes, then a low and flat stack. Its criteria repeat the
candidate facts in semantic buckets (no/few/many holes and low/moderate/high
stack measures), while retaining exact values in the JSON state. Jev is not
asked to count, invent SRS inputs, or calculate a score.

The harness accepts a selection only when its label is present for the latest
piece. There is no confidence gate because a rejected/uncertain choice has a
fully executable code fallback; the answer is advisory rather than a lethal
primitive. `TimeoutError` and ordinary `RuntimeError` use the current best
placement. `BudgetExceeded`/`EpisodeOver` end normally, and
`ModelUnavailable` is allowed to propagate as required by the contract.

## Fallback and freshness policy

A candidate is refreshed after the synchronous Jev request. The piece
signature `(pieces, active.type, active.orientation)` must still match before
using the selected label. Each movement is checked against the next
observation; a blocked move stops the sequence and drops at the current legal
x rather than looping. If the piece locks while a sequence is in flight, the
sequence is abandoned. All posted actions are valid named actions, and every
wait is bounded by the real-time game tick.

## Confirmation or refutation

The hypothesis is confirmed if training episodes have fewer stale ticks and
survive beyond the parent’s roughly 10--12 pieces, with nonzero lines and
lower final holes/top-out height at comparable System-1 latency. It is
refuted if the placement bridge still makes zero lines and tops out at the
same height, or if Jev’s selections are consistently worse than the
code-ranked fallback. A later iteration should then test rotation-aware SRS
placement or a richer multi-piece evaluator, but not attribute that result to
Astra planning.
