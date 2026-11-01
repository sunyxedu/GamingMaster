# Candidate 0020 — observed-height ledger and connector leases

## Parent and hypothesis

Parent: `candidates/0016/`.  I also used `0017` as the strongest evaluated
negative/positive comparison because its score is high but its trace exposes a
route-state defect.

The hypothesis is:

> Candidate 0017's apparent success is not upward play.  Its route phase is
> permanently locked at band 0 because the code that can advance the phase is
> nested under an alternate-connector branch that does not exist on the first
> band.  A controller that derives the current walkway from repeated observed
> y positions, then treats the next ladder as a separate measured connector
> transaction, will stop farming jumps at y≈148 and resume the alternating
> upward route.

This is a structural state-representation test, not another distance threshold.
The route ledger can also recognize a jump that lands on a higher band, but only
from repeated non-airborne observations; a jump arc cannot advance the route in a
single tick.

## Evidence and diagnosis

The `0017` traces are unusually diagnostic.  In `trace_seed0`, the controller
logs no `ramp_advance` event at all.  Its snapshots repeatedly report
`phase=0`, `x=105/99`, `y=154/148`, while `cross_begin` is logged at ticks 194,
285, 344, and then roughly every minute of emulated play.  The first score
observation at tick 199 is Mario `(99,148)` with an active jump byte and a
barrel at `(100,162)`; later observations stay on the same upper edge and earn
100 points over and over.  The final score of 4000 is therefore forty jump
credits, not screen progress: `screens_completed=0`, `rivets_removed=0`, and
all six episodes remain on `stage=ramp`.

The corresponding code path confirms the cause.  `0017.ramp_action` checks
climb progress only inside `if alternate is not None`.  Band 0 has no alternate,
so after its first climb reaches y=154 the method repeatedly calls `open_climb`
and returns `up`, but never tests the next y row or increments `phase`.  The
high score is a local loop, not evidence that the same-direction crossing
solved the board.

`0016` provides the contrasting evidence that an observed climb can move the
route: its seed-0 trace confirms `(105,154)`, then `(47,126)`, and later reaches
`(105,98)`/phase 3 before losing a life.  However, its fixed internal graph can
stall at `(85--87,111)` while still aiming at x=106 and repeatedly interrupts
that location with hazard jumps.  This motivates keeping a short alternate
connector, but only after horizontal non-response has actually been observed.

## What changed

* `0020` uses an **observed-height ledger**.  It commits band 1 at the repeated
  y≈154/148 observations, band 2 at y≈126/120, and so on.  The candidate must
  see three consecutive non-jump observations before promoting a band, so the
  raw jump state prevents a high arc from being mistaken for a climb.
* A band has an explicit connector lease.  The nominal player-edge targets are
  `(106,46,106,46,106,30)`; split bands also expose `(78,86,66,74)` alternatives.
  A connector changes only after ten observations of genuine horizontal
  non-response, and after both options fail it reacquires the nominal edge
  instead of oscillating on a timer.  Entering a connector starts UP; observed
  band promotion closes it.  This is different from 0017's nested climb code
  and from 0016's phase/branch state being the sole source of vertical truth.
* The jump controller is an action lease: FIRE continues until RAM `$85`
  confirms the airborne state, the original direction is held for a short
  landing tail, and a bounded post-landing UP tail precedes route control.
  The first route attempt uses the named hazard side/counter-motion primitive
  rather than independently reversing from hazard velocity.  A current-row
  overlap/commit is the hard first-attempt interrupt; an already progressing
  climb ignores a warning on the old girder.
* A life-consuming upper-band attempt activates a **risk fallback** for later
  Marios.  The fallback does not claim screen progress: it holds the proven
  scoring lane, does not promote the observed-height ledger, and waits for the
  tighter overlap window before jumping.  This prevents a failed route
  hypothesis from spending every remaining life on the same upper-band
  collision while retaining the route attempt as the diagnostic first trial.
* The lifecycle remains code-owned and accepts the documented zero-spare,
  nonterminal transition.  It sends one start FIRE and sparse transition FIRE
  edges, never blocks on a model, and keeps posting actions until the real game
  ends or the context closes.
* The rivet controller is complete enough to test the next screen: it reads
  all four row bytes, uses bottom-to-top serpentine targets, jumps across an
  observed cleared-rivet gap, and uses visible ladder geometry for row changes.
  It does not walk through a recorded hole.

## Confirmation and refutation

The strongest confirmation is a trace containing `band_confirmed` events
`0→1→2` (and ideally through band 5), followed by `stage=rivet` or a positive
`screens_completed` count.  A more local confirmation is that the first
0017-style `(x≈105,y≈154)` observation is followed by a walk toward x≈46,
not another sequence of jumps at x≈99,y≈148.  At the split-band failure,
`connector_switch` followed by a new band would support the measured
non-response rule.

The hypothesis is refuted if `band_confirmed 0→1` occurs but the cartridge
still refuses the x≈46 connector with no hazard, or if the new ledger reaches
no higher than 0017 while its snapshots show the next target being actively
walked.  If the ledger reaches the rivet screen but rivet bytes do not change,
that refutes the later sweep heuristic only; it does not refute the route-state
hypothesis.  A score made of repeated jump rewards with `band=0` would be
negative evidence even if the raw score exceeds 0017.

## Tier-2 System-1 interface

Normal ramp play deliberately makes zero inference calls.  Code has exact
verified facts (the fixed ladders, raw jump byte, row bytes, stage, and
arithmetic hazard tests), while a 0.15--0.2 s Jev response is too slow to own a
four-frame collision.  Astra is spawned only after rivet entry and cannot
change a ramp action.

When the rivet screen is active, Jev receives this strict JSON schema:

```json
{
  "game": "Atari 2600 Donkey Kong NTSC retail port",
  "context": "rivet sweep",
  "stage": "rivet",
  "player": {
    "horizontal_band": "left|middle|right",
    "vertical_band": "bottom|lower|middle|top",
    "motion": "up|down|steady",
    "jump_committed": false
  },
  "route": {
    "ramp_band": 3,
    "rivet_row": 1,
    "rivet_rows": [6, 0, 18, 18],
    "gap_jump_pending": false
  },
  "hazard": {
    "present": true,
    "side": "left|right|none",
    "motion": "left|right|still|none",
    "distance_bucket": "overlap|near|approaching|far|none",
    "closing": true,
    "urgency": "overlap|commit|soon|watch|near|clear"
  },
  "progress": {
    "lives_remaining": 1,
    "screens_completed": 0,
    "rivets_removed": 2
  },
  "last_action": "left"
}
```

Code supplies semantic bands and the already decoded row statuses; Jev is not
asked to count, compare coordinates, or invent an action mapping.  One request
batches two independent questions:

1. `side`, a `Choice` over `left`, `right`, and `hold`, selects the next legal
   rivet sweep side.  `hold` preserves a climb/gap transaction.
2. `hazard`, a `Noul`, asks whether the supplied hazard is explicitly a
   current-row overlap/commit.  It is diagnostic only; the code's hard guard
   owns the actual jump.

A request is spawned approximately every 14 ticks on a clear rivet route and
every 5 ticks near a named hazard, with a 0.55 s timeout.  A side is accepted
only if it is one of the declared values, its probability is at least 0.55 when
probabilities are present, and the exact stage/row/mask/life stamp still
matches.  Any late, low-confidence, missing, malformed, or failed answer is
discarded.  Even an accepted side can only alter the next row preference when
that side is still reported as the remaining side and no jump is active.

Astra receives the same compact state on rivet entry and is asked for only
`{"first_side":"left"}` or `{"first_side":"right"}`.  Fenced, plain, or
otherwise garbage output is parsed defensively; only those two strings can
change the first-row preference.  Astra never emits a joystick sequence,
never delays the actor, and cannot override a gap jump, hazard guard, ladder
transaction, terminal state, or restart edge.

The deterministic fallback is always executable: FIRE in idle, neutral during
transition except for sparse restart edges, ledger/connector actions on the
ramp, RAM-synchronized directional FIRE for hazards, and raw-row serpentine
movement plus gap jumps on the rivet screen.  Thus arbitrary stand-in model
text in pre-flight cannot cause a parse exception or an illegal action.
