# Candidate 0017 — lane-committed crossings with an idle-safe lifecycle

## Parents and hypothesis

Parents: `candidates/0012/` and `candidates/0002-tier2_idle/`.

0012 supplies the only consistently useful positive mechanism so far: the
player-edge route `(106,46,106,46,106,30)`, observed-y climb confirmation, and
raw-RAM jump bookkeeping.  The idle parent supplies an important robustness
property: it never blocks on inference, never throws an action/model error, and
can remain in the emulator indefinitely.  I combine those properties, but do
not keep its no-FIRE policy once play is known to be idle.

The concrete failure hypothesis is:

> 0012 is getting a credited jump but dying during the landing/reacquisition
> window, because it reverses direction after the jump.  Its trace begins a
> rightward crossing at a right-side hazard and then posts `left` immediately
> after the FIRE/RAM portion.  The safe lane is the side beyond the hazard,
> so the landing should continue right until Mario has exited that sprite.
> A second, smaller problem is that the `distance <= 22` emergency threshold
> leaves too little rising time.  Start a same-row crossing in the calibrated
> 22–34 pixel band, but do not interrupt a climb for a merely early warning.

The new controller therefore models a crossing as `FIRE -> same-direction
landing -> bounded hazard lock`, rather than as independent per-tick jump
choices.  It still uses 0012's edge targets and raw jump byte, but its crossing
state and lifecycle are new.

## Evidence and changes

The strongest positive trace is `candidates/0012/trace_seed2.jsonl`.  It
confirms the route through phase 2: 0012 reaches approximately `(105,154)` and
`(47,126)`.  At tick 157 it records a right-side, left-moving hazard at
`distance=21`, starts `rightfire`, and earns its first 100 points.  The next
boundary observation at tick 163 is Mario `(89,117)` with the hazard at
`(90,131)` and `raw_ram[5]=8`, so the jump was real and was still active.  The
action stream then is `rightfire` through tick 165 followed by `left` at tick
166, immediately before the ramp enters transition.  The same seed repeats the
pattern on the next life: crossings at ticks 333, 365, and 397 earn points and
reach phase 3, but the final transition follows the later leftward landing
segments.  Seed 5 shows the same first crossing at 157 and the same first death;
its final life ends in a transition with zero lives and the old harness keeps
posting `noop` to tick 13,499 until the 900-second deadline.

This supports two separable tests:

* **Landing lane:** select the hazard's side (`right` for a hazard to Mario's
  right, `left` for one to his left) and keep that direction through the
  landing tail.  This is not a velocity-only choice: velocity says whether the
  object is closing, while side says where the clear lane is after contact.
* **Start window:** use a deterministic `commit` bucket through distance 34,
  rather than waiting for 21 or less.  A warning on another walkway is ignored,
  and a warning during a climb is ignored unless it is already overlapping.
  Thus this does not repeat 0013's blanket 45-pixel threshold.

The ramp route is still a finite, observation-closed transaction.  A phase
changes only after at least eight pixels of measured upward motion and arrival
near the next y row.  If the rung is quiet for 30 observations, the controller
tries one one-pixel edge probe and then re-enters the same transaction.  It
never advances because an action count elapsed.

The lifecycle is also code-owned.  One initial FIRE leaves idle.  Transition
FIRE is sparse and only allowed when a life remains.  When a last-life
transition is explicitly observed, the loop returns instead of spending the
wall deadline on a known unrestartable shell.  All malformed model output,
missing RAM, unknown stage values, and failed advisory calls fall back to valid
code actions; no model is on the action critical path.

The rivet implementation retains the public raw-row interpretation and
bottom-to-top serpentine route from 0012.  It treats every observed row-byte
change as a gap crossing, not as permission to walk through the cleared cell.
That part is deliberately present so a ramp completion tests the next stage
rather than returning at the first screen boundary.

## What would confirm or refute the hypothesis

Confirmation is a trace that repeats the known phase-2 alignment but survives
the first `(rightfire, right landing)` crossing, reaches the next ladder or
phase 3, and ideally changes stage to `rivet`.  A more diagnostic confirmation
is a trace where the 100-point jump remains while the player is visibly on the
far side of the hazard after `raw_ram[5]` clears.  A fall in transition deaths
with the same route and similar jump count would support the landing mechanism
without requiring a screen completion.

The landing hypothesis is refuted if the same-direction tail repeatedly dies
while the old reversal survives, or if a jump starts in the commit bucket but
Mario is killed before the hazard is vertically crossed.  The early-window
hypothesis is refuted if commit-band starts consistently waste jumps or die
more often than immediate starts.  The route hypothesis is refuted if Mario
holds the documented edge `(106/46)` with a measured y change absent any
hazard, or if all losses occur in the transition state rather than in a
crossing.  Rivet progress would separately test the row-mask/gap logic; lack
of rivets after a ramp death says nothing about that later component.

## Tier-2 model interface

Normal ramp play intentionally uses zero model calls: the public geometry and
verified RAM facts are exact, while Jev's measured ~0.15–0.2 second latency is
not a safe substitute for a four-frame jump guard.  Advisory calls are enabled
only after the state is safely in the rivet route.  This preserves the idle
parent's no-blocking/no-error behavior while still defining a model bridge for
an unverified later-stage choice.

### Jev state schema

Each request contains strict JSON with only these fields:

```json
{
  "game": "Atari 2600 Donkey Kong NTSC retail port",
  "stage": "rivet",
  "player": {"horizontal_band": "left|middle|right",
              "vertical_band": "bottom|lower|middle|top",
              "motion": "up|down|steady"},
  "route": {"ramp_phase": 0, "climb_open": false,
             "rivet_row": 0, "rivet_rows": [0,6,12,18],
             "crossing": false},
  "hazard": {"present": true, "side": "left|right|none",
             "motion": "left|right|still|none",
             "distance_bucket": "overlapping|near|approaching|far|none",
             "closing": true, "urgency": "commit|overlap|warning|far|clear"},
  "progress": {"lives_remaining": 1, "screens_completed": 0,
                "rivets_removed": 0},
  "last_action": "left"
}
```

Code computes all coordinates, row-byte meanings, distance buckets, current
row, action legality, and collision priority before the request.  Jev is not
asked to count rivets or do arithmetic.

One request batches two independent judgments:

1. `sweep`: `Choice` over `left`, `right`, and `hold`.  It recognizes the
   already legal side for the current raw row; `hold` means preserve an open
   crossing or climb transaction.
2. `danger`: `Noul` asking whether the named hazard facts explicitly describe a
   `commit` or `overlap` closing object on Mario's row.  This is diagnostic and
   never replaces the deterministic hazard guard.

Requests are spawned every 16 ticks on a clear rivet route and every 5 ticks
near a named hazard, with a 0.65-second timeout.  A result is collected only
if its future is complete and its answer is one of the expected strings.  A
side is accepted only when its returned option probability is at least 0.55
(or when a stand-in has no probability map), and only if the raw row still
supports that side and no climb/crossing is open.  Low confidence, stale,
malformed, timed-out, or failed results are discarded.  The hard fallback is
always the raw-row serpentine controller plus the crossing and lifecycle
guards.

### Astra role

Astra is started at most once on entry to a verified rivet screen, in a
background thread, with the compact state above.  Its prompt requests only
`{"first_side":"left"}` or `{"first_side":"right"}`.  JSON is parsed from
plain or fenced text defensively; any other text is ignored.  The answer can
change only the first sweep preference before the first row is acted on.  It
cannot select a frame action, override a gap jump, or delay the actor.  On
ramp failures Astra is deliberately not called: the test is about a concrete
crossing primitive, and a 49–177 second System-2 answer cannot rescue a
four-frame collision.  A future iteration should revisit that boundary only
if traces show a surviving Mario stalled with no hazard, not merely another
jump death.
