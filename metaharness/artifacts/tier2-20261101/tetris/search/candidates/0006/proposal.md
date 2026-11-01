# Candidate 0006 — beam-planned placements with gated System-1 arbitration

## Parent and diagnosis

Parent: `candidates/0002-tier2_idle/`.  I also used the strongest evaluated
comparison, `candidates/0004/`, and the Jev placement bridge in
`candidates/0003/`.

The idle parent proves only that surviving without posting an action is not a
useful objective: its six episodes scored zero, cleared zero lines, and topped
out after roughly 10--12 pieces.  The important failure in the best active
candidate is visible in its traces, not in its model-call count.  In
`candidates/0004/trace_seed0.jsonl`, the lock events put the first pieces at
`[0..1]`, then mostly the middle columns: by piece 11 the board has 14 holes
and aggregate height 58, the following central I creates 28 holes and height
76, and the episode tops out on piece 13.  The same controller does better on
seed 1, but its only line clear is at piece 15 and it then ends with 29 holes
and height 83.  The lock events are almost all orientation 0; the one-piece
ranker and the late candidate answer did not establish a sustainable surface.
The two representative Astra requests were cancelled at game end, and
`Astra->act` was 0%, so its initial policy token was not evidence for the
score.  In 0004, Jev changed only about 20% of posted actions; the deterministic
first candidate was doing most of the work.

This suggests a structural failure rather than a timeout tweak: 0004 searches
one future placement, presents seven similar candidates to Jev, and then
allows a candidate selected for a stale falling piece to compete with the
current code fallback.  It has no multi-piece commitment to a surface or to a
near-complete row.  A safe placement can therefore be locally ranked while
making the next two pieces impossible to place cleanly.

## Hypothesis

**A short code-owned beam over the known queue will beat one-ply placement
ranking, provided the model is only a gated tie-breaker among already safe
Pareto alternatives.**  The controller should optimize a three-piece
continuation (current piece, next preview, and the following preview), with
hold branches for the current piece, rather than ask Jev to discover a
rotation, a destination, or arithmetic.  A richer board score explicitly
penalizes new/covered holes and top-out height, rewards completed and nearly
completed rows, and retains bumpiness/well/surface features.  Jev is most
useful for recognizing whether the code's top few alternatives are a
reasonable clear, a smooth stack, or a recovery placement; it should not be
allowed to invent a placement or override a high-risk code plan.

This also changes the real-time division of labour.  The code selects and
starts the best beam plan immediately, so a 0.2 s Jev answer is never on the
critical path.  Jev asynchronously judges only three or four named Pareto
plans and can replace the code plan only when two reversed-order Choices agree
with sufficient probability while the same piece is still falling.  Astra is
not called at episode start: 0004 showed that a 28--70 s reviewer cannot
inform a two-second standard episode.  It is started only after a genuinely
long survival or a critical stack event, and its single-token risk-mode output
changes the weights for *future* code searches.  Thus a late Astra answer is
useful on a long episode but cannot stall or corrupt early control.

## What changed

### Code owns exact mechanics and planning

* Convert the public top-to-bottom board to bottom-origin cells and compute
  heights, aggregate height, maximum height, holes, hole depth, bumpiness,
  wells, row/column transitions, and near-complete-row counts.
* Enumerate all visible hard-drop results for the four SRS orientations (one
  for O), for both the current piece and a legal current hold.  The board
  simulator removes full rows in the correct bottom-origin order and rejects a
  result that still occupies the visible ceiling.
* For each first placement, search the best few legal placements of the next
  two known queue pieces.  The returned first plan is ranked by immediate
  safety plus discounted future surface quality.  This is a bounded beam, not
  seed-specific scripting; it uses only the current observation and previews.
* Generate only short verified controls.  Hold is issued at most once, each
  rotation is sent one tick at a time, horizontal movement is re-read after
  every tick, and a blocked rotation or translation falls back to the best
  target in the observed orientation.  A hard drop is posted only when the
  observed type/orientation/x matches the plan, so a stale answer cannot drop
  the replacement piece into the old geometry.
* Post the code fallback immediately and overwrite a previous hard-drop action
  with `noop` before doing a new search.  The game clock therefore never waits
  for either model and a slow search cannot accidentally hard-drop every new
  piece.

### Jev is a gated recognizer, not a planner

At a piece boundary the state contains the ruleset, piece instance, compact
stack buckets, active/hold/queue facts, the current policy mode, and three
named code-generated alternatives.  Each alternative includes semantic
buckets for clear type, holes, maximum height, surface roughness, future
support, and whether it uses hold.  Exact arithmetic stays in code.

One `ctx.decide` request batches two independent `Choice` questions:
`pick_a` has the options in code/Pareto order and `pick_b` has them reversed.
Both ask for the best safe balance of line progress and a low, hole-free,
future-friendly surface.  A model answer is accepted only if both values are a
valid current label, they agree, and each selected label has probability at
least **0.40**.  Since every option has already passed code's legality and
ceiling filters, this is a moderate gate; disagreement, a diffuse answer,
a timeout, malformed stand-in output, an old piece instance, or a changed
hold state leaves the code beam plan in control.  A Jev answer can change the
plan only before a hold has been sent and before the executor has committed a
hard drop.  All action generation is still deterministic and revalidated.

### Astra is an event-triggered risk-mode reviewer

The first Astra call is deliberately suppressed.  On a long episode, or in a
higher-level scenario, code may start one background review when the board is
critical or the controller has survived a substantial number of pieces.  Its
prompt contains only public derived metrics, recent lock/clear facts, the
current queue, and the current mode; it asks for exactly `BALANCED`,
`RECOVER`, or `CLEAR`.  Parsed output changes a mode version and is included in
future Jev states and beam weights.  Unparseable, late, cancelled, or failed
output leaves `BALANCED`/the current mode unchanged.  Astra never specifies a
keypress or destination and never blocks the executor.

## Jev state schema, questions, and thresholds

The strict JSON state sent to Jev is shaped as follows:

```text
{
  "ruleset": "guideline-marathon-v1",
  "policy": "BALANCED | RECOVER | CLEAR",
  "piece_instance": integer,
  "board": {
    "height": "low | moderate | high | critical",
    "holes": "hole-free | shallow | several | deep",
    "surface": "smooth | uneven | rough",
    "near_rows": "none | promising | many",
    "lines": integer, "level": integer,
    "combo": "inactive | active", "back_to_back": boolean
  },
  "active_piece": string, "hold": string|null, "can_hold": boolean,
  "next_queue": [string],
  "fallback": "P0",
  "candidates": {"P0": description, "P1": description, "P2": description}
}
```

`pick_a` and `pick_b` are reversed-order `Choice`s over the candidate keys.
Their instructions explicitly say that candidates are legal final SRS-shaped
placements, that Jev must not calculate or invent input, and that a clear is
preferred only when it does not create a dangerous hole/top-out surface.  The
0.40 per-answer probability gate and exact agreement gate are the only model
confidence thresholds.  Questions are launched with a **0.55 s** timeout in a
background thread: the measured Jev p90 was 0.17 s, while the prior controller
observed a 0.426 s tail timeout at 0.45 s.  The slightly wider budget reduces
avoidable failed calls without ever blocking the executor.  A timeout is an
ordinary fallback, never an episode error.

The deterministic fallback is always the highest-ranked beam plan.  While a
Jev future is pending, the executor continues that plan; it does not spend
warm-up ticks waiting.  If the current piece changes, if `can_hold` changes,
if the answer names an absent candidate, or if a movement/rotation does not
change the observed pose, the answer is discarded and the executor stops
trying that target: it hard-drops the current verified pose rather than
repeating a blocked wall input.  Valid named actions are used throughout, so
malformed model text cannot create an illegal game action.

## Evidence that would confirm or refute the hypothesis

Confirmation would be a material rise above 0004's mean 337.3, especially
more than one line per episode and survival beyond 16 pieces, while final
holes/top-out height fall.  Traces should show rotations and placements spread
across the board, beam alternatives with future support, and no stale repeated
hard drops after a lock.  Jev should sometimes select a non-default Pareto
plan, but score should remain good when it times out; an Astra mode update on
a long episode should precede a later change in beam choices.

The hypothesis is refuted if exact rotation-aware execution still repeats the
central orientation-0 tower, if a three-piece beam creates more holes than the
one-ply fallback, or if the code's chosen target is frequently unreachable and
causes illegal actions.  If the score improves but Jev never changes a plan,
that confirms the code-search part and says the next iteration should reduce
System-1 calls.  If score improves only when Jev overrides the code fallback,
then the Pareto descriptions or beam weights need revision rather than a
larger raw action question.
