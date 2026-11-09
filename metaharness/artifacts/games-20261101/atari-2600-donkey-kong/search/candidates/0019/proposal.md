# Candidate 0019 — observed-height route with bounded connector recovery

## Parents and hypothesis

Parents: `candidates/0017/` and `candidates/0014/`.

The two parents expose complementary mechanisms:

* `0017` is the strongest scorer (mean 4,000). Its committed, same-direction
  jump transaction is durable enough to turn a dangerous encounter into a
  100-point jump rather than immediately losing the Mario. Its code-owned FIRE
  restart and no-model action loop also has excellent timing and no failed
  calls.
* `0014` makes the ladder problem explicit: it uses the observed ten-ladder
  slots, bounded alignment variants, alternate connectors, and a progress
  confirmation rather than treating an apparent sprite overlap as proof of a
  climb. It is a useful recovery mechanism even though its jump experiment
  died before it could validate the route.

The concrete failure in `0017` is not primarily its jump score mechanism. In
all six traces its route event remains `phase=0` throughout the run. For
example, `candidates/0017/trace_seed0.jsonl` records the first crossing at tick
194 with Mario later at `(99,148)`, then repeated crossings at ticks 285, 344,
403, and onward, always with `phase=0`. Its snapshot at tick 220 is `(95,153)`
and still `climb=true`; after the crossing displacement, the controller keeps
its old climb transaction instead of reacquiring and completing the route. The
same trace is at `(105,154)`, `climb=true`, and `phase=0` at tick 1800 after
2,100 points of repeated jumps. The relevant phase-advance check was nested in
an alternate-edge branch that is never entered for the nominal first
connector. Thus 0017 farms jumps while remaining on the first upper girder.

The hypothesis tested here is:

> Preserve 0017's committed same-lane jump/landing primitive, but make ladder
> progress a separate transaction. If every nominal connector advances only
> after observed upward y movement, and a failed connector gets only bounded
> alignment/alternate probes from 0014, the agent will stop farming at the
> first girder and reach the rivet screen without sacrificing real-time safety.

This is a structural fix rather than a threshold-only change. A `level` is the
current walkway, `connector` is a candidate from the fixed ladder graph, and
`probe` is one of three nearby x alignments. The level changes only when Mario
has moved upward by at least eight observed pixels, reached the next y band, and
settled for two observations. A jump that displaces x cancels the climb
transaction and reacquires the same connector. The phase-advance test is
outside the alternate-connector code path.

## What changed

1. **Ramp route transaction.** The public ladder slots are grouped into six
   upward connectors. Nominal player-edge targets are `(106, 46, 106, 46,
   106, 30)`. The controller uses the visible ladder boxes when present, with
   the documented fixed coordinates as fallback. It requires measured y
   progress and a short settle before changing `level`.
2. **0014-style bounded recovery.** After 24 no-progress observations, code
   tries nearby alignment probes and then the second connector on a split row;
   it never wanders indefinitely or changes level because a timer elapsed. A
   Jev recovery judgment can choose among those already legal operations, but
   the deterministic probe runs immediately if Jev is late.
3. **0017-style jump ownership.** A closing current-row hazard at the
   overlap/commit distance starts a directional FIRE transaction. FIRE remains
   posted until the RAM jump byte has been observed active and cleared, then
   the same direction is held through a five-tick landing tail and a short UP
   reacquisition tail. No Jev/Astra answer can interrupt this lethal window.
4. **Lifecycle repair.** FIRE is posted once from idle and sparsely after a
   death transition only when a Mario remains. The final Mario is still played
   (a zero `lives_remaining` value does not mean that the currently visible
   Mario is absent), while an unrestartable transition is bounded at 90 ticks
   so the harness cannot spend the 900-second deadline posting noops.
5. **Rivet continuation.** If the ramp succeeds, raw row bytes drive a
   bottom-to-top serpentine sweep. A row-byte change creates a directional gap
   jump; ordinary walking never crosses a cleared gap. The same hazard guard
   remains authoritative.

## Evidence that would confirm or refute it

Confirmation is a trace with `ramp_advance` events at levels 1, 2, and beyond,
not merely a high jump score at level 0, followed by `stage=rivet` and a change
in at least one `rivet_rows` byte. A particularly diagnostic confirmation is a
crossing that earns 100 points, temporarily displaces Mario away from its
ladder, then reacquires the connector and advances on observed y. A reduction
in repeated `jump_begin` events with the same level, plus positive screen or
rivet progress, would support the hypothesis even before full completion.

The hypothesis is refuted if the new controller still produces many committed
jumps with a fixed `level` while no connector stalls are logged, or if it
advances levels without measured y change. It is also refuted if the nominal
connector reaches the next y band but the same-direction landing tail
systematically kills Mario where the parent survives. If ramp progress is
confirmed but no rivets change, that is evidence against the separate rivet
route (not against the repaired ladder transaction), and would motivate a
later rivet-only iteration.

## Tier-2 model bridge

The action loop never waits for inference. Every tick waits for the game clock,
observes, applies hard lifecycle/jump/legality guards, and posts a valid
literal action. Astra and Jev are spawned only as advisory work; a late,
malformed, cancelled, or failed answer is discarded. There are no background
Python loops, and model futures are allowed to be cancelled when the episode
closes.

### Jev state schema

The state is strict JSON and contains derived semantic buckets rather than raw
RAM arithmetic:

```json
{
  "game": "Atari 2600 Donkey Kong NTSC retail port",
  "stage": "ramp|rivet",
  "player": {
    "horizontal_band": "left|middle|right",
    "vertical_band": "bottom|middle|upper",
    "motion": "up|down|steady"
  },
  "connector": {
    "level": 1,
    "candidate": 0,
    "candidate_count": 2,
    "alignment_probe": 0,
    "climb_age": 12,
    "vertical_progress_seen": false
  },
  "route": {
    "row": 0,
    "row_status": [0, 6, 12, 18],
    "gap_crossing": false,
    "jump_committed": false
  },
  "hazard": {
    "present": true,
    "side": "left|right|none",
    "motion": "left|right|still|none",
    "distance_bucket": "overlapping|near|approaching|none",
    "urgency": "overlap|commit|watch|clear"
  },
  "last_action": "up",
  "lives_remaining": 1
}
```

Code computes coordinates, distance buckets, current row, legal connector
candidates, and all BCD/rivet arithmetic. Jev is not asked to count, compare
numbers, infer action indices, or invent a route.

### Questions and control thresholds

A ramp recovery request batches:

* `recovery`: `Choice` over `retry_alignment`, `alternate_connector`, and
  `continue`. A returned value is accepted only when its option probability is
  at least 0.55. `retry_alignment` advances the nearby probe,
  `alternate_connector` selects the second fixed slot, and `continue` gets at
  most two extra short attempts. Any absent/low-confidence/late answer uses
  the code fallback: probe all nearby alignments, then the alternate slot.
* `danger`: a diagnostic `Noul` asking whether the supplied hazard is an
  overlap/commit current-row threat. It is never allowed to suppress or create
  a jump; the deterministic jump guard owns that consequence.

A rivet request batches:

* `sweep`: `Choice` over `left`, `right`, and `hold`. A side is used only when
  the current row still has that side and no climb, gap, or jump transaction is
  open. It changes a preference, not the immediate action.
* `danger`: the same explicit current-row Noul, used only for diagnostics.

Requests are asynchronous. Connector recovery can request one judgment per
unique `(level, candidate, probe, lives)` key; rivet requests are spaced 16
observations on a clear row or five near a commit/overlap hazard. A future is
accepted only if it is complete, type-correct, probability-gated, and still
relevant to the current route. `ctx.decide` exceptions, arbitrary stand-in
answers, and malformed values are caught and logged.

### Astra role

On the first verified rivet screen, Astra receives only the current row bytes,
Mario's side, and the goal of choosing the first sweep side. Its prompt
requires exactly `{"first_side":"left"}` or `{"first_side":"right"}`.
The JSON parser accepts plain or fenced output defensively. The result may
change only the first-row preference before rivet progress; it cannot select a
frame action, override a gap jump, change ladder alignment, or delay the actor.
A slow or missing Astra answer therefore falls back to the deterministic
left-first serpentine policy.
