# Candidate 0023 — do not convert the first climb into a score lane

## Parent and targeted hypothesis

Parent: `candidates/0018/`.  I keep its observation-ledger controller, calibrated
player-edge connector graph, RAM-confirmed jump leases, sparse restart pulses,
and asynchronous advisory bridge.

The targeted hypothesis is that the parent's first-life safety fallback is
triggered one transition too early.  In `_mark_climb`, immediately after the
first observed y transition from the bottom walkway to the next walkway, the
parent sets `farm_mode=True`.  That makes the first Mario stop using the
connector graph at phase 1 and deliberately hold the right-side lane for jump
points.  The controller consequently never tests the upper connectors or the
ramp goal with the first life.  This is a route-state decision, not evidence
that the upper route is unsafe.

The parent trace evidence is unusually direct:

* `candidates/0018/trace_seed3.jsonl` records `ramp_advance` only once, at
  tick/clock roughly 70 (`phase=0 -> 1`, `x=105`, `y=154`), immediately after
  `score_fallback_enabled` with reason `first nominal climb confirmed`.  Its
  later snapshots at ticks 3100, 3120, 3140, and 3200 all still report
  `stage=ramp`, `phase=0` after the later life reset, `x=105/97`, `y=154`, and
  `action=up`; the episode ends in the zero-bonus transition shell with score
  4000 and `41` jumps, `screens_completed=0`.
* The same mechanism appears in `trace_seed0`: after the one
  `ramp_advance` at `(105,154)`, the final snapshots remain at
  `phase=0`, `(105,154)`, while `jump_begin` repeats at ticks 3087, 3146,
  and 3214.  It ends at score 4100 without a screen change.
* `trace_seed1` shows the parent doing the same first climb, then after the
  life boundary logs `score_fallback_enabled` and spends its final Mario at
  `phase=0`, `(105,154)`, finally ending at score 4200.  Across all six parent
  episodes, `screens_completed=0` and `rivets_removed=0`.

There is positive control evidence that the route ledger can progress when it is
allowed to remain nominal: the predecessor `candidates/0015/trace_seed2.jsonl`
observed `ramp_phase` transitions at `(105,154)`, `(47,126)`, and `(105,98)`.
Thus this iteration does not retune hazard thresholds, jump direction, or
connector coordinates.  It removes exactly the `farm_mode` assignment attached
to the first nominal climb.  `farm_mode` remains available for the parent's
already-established later-life fallback after an actual life boundary.

## What changed

The only behavioral change from parent 0018 is in `_mark_climb`:

* before: confirming `old phase 0 -> phase 1` also enabled `farm_mode`;
* now: that confirmation only records the new phase, clears the climb
  transaction, selects the phase-1 connector, and continues the nominal route;
* after a real death/life boundary, the existing `_sync` logic may still enable
  `farm_mode`, preserving the parent's conservative recovery behavior.

All hard lifecycle, legality, and safety behavior remains code-owned.  The
actor still starts with one FIRE edge, uses `ctx.wait_for_tick()`, keeps FIRE
leased only while the documented jump byte confirms an airborne jump, detects
same-row closing hazards arithmetically, and never treats `lives_remaining` as
terminal.  It still stops on explicit game-over/context completion or the
parent's exhausted-bonus transition shell.

The expected isolating result is a trace with `ramp_advance` events for phases
1, 2, and beyond on the first Mario, fewer repeated jump rewards, and a stage
change to `rivet` or a completed ramp.  If the first Mario reaches the upper
route but dies, the later-life fallback still supplies the parent's score
floor rather than causing an error or an uncontrolled loop.

## Confirmation and falsification

Confirmation requires more than a higher jump score.  A confirming trace should
show, before the first life boundary, the sequence of observed y gains through
phase 1 and phase 2 (ideally phase 3 or the top), with `ramp_advance` events
whose y values are lower than the preceding walkway.  Strong confirmation is
`stage=rivet`, a positive `screens_completed`/`rivets_removed`, or at least a
completed ramp with a banked bonus and no long phase-1 jump-farming loop.

The hypothesis is falsified if removing the premature fallback still produces
only the phase-0-to-phase-1 event on every seed while the controller is visibly
aligned at the phase-1 connector, or if it repeatedly reaches the same upper
connector and loses a life without any nominal route progress that the parent
also lacked.  A score increase consisting only of more `jump_begin`/`+100`
events while y and phase remain fixed is not confirmation.  A model response,
late answer, or malformed Astra text is also not evidence: the nominal code
path must remain safe without it.

## Tier-2 System-1 interface and fallback

Jev remains asynchronous and event-triggered exactly where the parent uses it:
for a connector stall or an immediate hazard.  It never blocks the tick loop or
owns a route coordinate.  Each request carries a strict JSON state containing
only derived semantic facts:

```json
{
  "game": "Atari 2600 Donkey Kong NTSC retail cartridge",
  "event": "connector_stall|hazard",
  "stage": "ramp|rivet",
  "player": {
    "horizontal_band": "left|middle|right",
    "vertical_band": "bottom|lower|middle|top",
    "motion": "up|down|steady",
    "jump_committed": false
  },
  "route": {
    "phase": 1,
    "selected_connector": 0,
    "connector_options": 2,
    "alternative_available": true,
    "climb_open": false,
    "fallback_action": "left",
    "rivet_row": 0,
    "rivet_rows": [0, 6, 12, 18]
  },
  "hazard": {
    "present": true,
    "slot": 2,
    "side": "right",
    "motion": "left",
    "distance_bucket": "near",
    "closing": true,
    "urgency": "immediate"
  },
  "progress": {
    "lives_remaining": 1,
    "screens_completed": 0,
    "rivets_removed": 0
  },
  "last_action": "up",
  "macro": {"connector": "adaptive", "hazard": "jump", "first_side": "left"}
}
```

Code computes all coordinates, distance/closing tests, connector legality,
row-byte decoding, and jump state.  Numeric values are semantic buckets before
they reach Jev; Jev is not asked to count, compare distances, or invent an
action mapping.

The request batches two independent `Choice` questions:

1. `route` has `hold`, `nominal`, and `alternate`.  `alternate` is valid only
   when `route.alternative_available` is true during a genuine connector stall;
   `hold` is the safe answer during a live climb or hazard.
2. `escape` has `keep_climbing`, `jump_left`, and `jump_right`.  A jump label is
   considered only for the supplied same-row `overlap`/`immediate` hazard.

The deterministic action is posted first.  A returned value is accepted only
if it is one of the declared options, its option probability is at least
`0.60` (the conservative `hold`/`keep_climbing` fallback may be accepted when
probabilities are unavailable), and its event stamp still matches stage, phase,
branch, rivet row, hazard slot/urgency/motion, and deadlock count.  It expires
after three control ticks.  A stale, low-confidence, timed-out, malformed, or
failed call leaves the deterministic action unchanged.  There is at most one
outstanding Jev future, with the parent's short `0.65 s` timeout; a garbage
stand-in answer therefore cannot escape the harness.

Astra remains a nonblocking macro reviewer, started after repeated connector
deadlocks or on rivet entry.  It receives the same compact state and is asked
for optional JSON labels `connector=nominal|alternate|adaptive`,
`hazard=climb|jump`, and `first_side=left|right`.  Parsed labels are applied
only at a later revalidated route boundary; invalid or plain-text output is
ignored.  Astra cannot undo the targeted first-life nominal-route change and
cannot override the hard jump/lifecycle guards.

The actor posts one valid action per real-time observation and has a complete
code fallback, so late inference increases neither stale ticks nor episode
errors.  The experiment isolates whether the parent's premature first-climb
score fallback—not the graph geometry or hazard lease—is what prevented screen
progress.
