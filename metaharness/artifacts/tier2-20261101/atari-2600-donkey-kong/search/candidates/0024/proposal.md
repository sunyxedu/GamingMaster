# Candidate 0024 — phase-aware post-jump hazard lease

## Parents and hypothesis

Parent: `candidates/0013/` as requested by the generation directive.  The
working route and geometry are refined from the later positive controls
`candidates/0018/` and `candidates/0022/`; this is not a seed-specific replay.

The archive's best score is not route completion.  `0018` earns 4,000–4,200
points by holding the first upper girder and repeatedly collecting jump credit,
while `0022` is the strongest evidence about the actual upper-route failure.
In `candidates/0022/trace_seed4.jsonl`:

* the observed route reaches phase 2 at `(47,126)`;
* a jump begins at tick 529 and ends at tick 541 at `(85,123)`, then the
  landing is adopted as the phase-2 alternate connector `(85,121)`;
* the next climb is confirmed at tick 560 as `ramp_advance phase 2 -> 3` at
  `(85,98)`;
* the same tick's snapshot says `hazard: "commit"` and `row_error: 1.5`, but
  the action is `left`, not a new escape jump; and
* the game enters `transition` at tick 566.

That ordering is a concrete controller failure.  The prior jump had installed a
24-quantum `jump_lock`.  The lock was intended to prevent FIRE spam, but it
also suppressed the only hazard guard during the newly entered phase.  The
route then started walking left across phase 3 while a current-plane hazard was
already marked `commit`.  The route check also ran before the phase advance and
was not repeated after the new phase became visible.  This is different from
merely choosing an earlier or later distance threshold, and it is not the
already-tested sloped-plane hypothesis: the trace explicitly has a small
current-plane row error.

**Hypothesis:** a jump cooldown should be a same-object duplicate-action lease,
not global collision immunity.  If a newly entered ramp phase gets an immediate
second hard hazard check, and that check can preempt the lease for a different
or phase-new critical obstacle, the upper route will survive more often.  A
normal same-slot hazard in the same phase remains suppressed briefly so the
change does not turn one crossing into FIRE spam.

## What changed

The harness is a self-contained finite-state controller with the following
isolated change:

1. It retains a short-lived grounded ramp-plane fit, observed-height connector
   transactions, the fixed connector graph, raw jump-byte synchronization, and
   the mask-driven rivet continuation.  It retains the demonstrated
   first-life route attempt and restores the known score-preserving lane after
   an actual life boundary; the phase-aware hazard change is therefore tested
   on the initial route rather than hidden by an unvalidated second attempt.
2. `jump_phase` and a `(slot, side, motion)` `jump_signature` are recorded when
   a jump begins.  After landing, `jump_lock` suppresses only the same object in
   the same phase, except for a true overlap.  A different hazard or a phase
   change can start a fresh jump immediately.  The lock is therefore not used
   as a blanket safety claim.
3. `advance_phase` sets `recheck_hazard`.  `choose` runs the ordinary route
   action, then repeats the code-owned hazard guard on the same observation if
   the route just proved a new phase.  A phase-3 `commit` hazard cannot be
   hidden behind the previous jump's tail/lock.
4. Route jumps retain the deepest public route's counter-motion steering
   (barrel moving left -> `rightfire`, barrel moving right -> `leftfire`),
   rather than changing jump direction in this experiment.  The observed
   alternate-connector adoption around the public `(86,120)` connector is
   retained.  The score fallback uses the same counter-motion direction and
   only the close hazard window, so the candidate has a floor if the upper
   experiment still fails.
5. Rivet control is still deterministic: row bytes select the remaining
   target, every newly cleared rivet creates a pending gap jump, and the
   bottom-to-top serpentine route climbs only after the row is complete.
   Malformed fields fall back to the safe literal route.

No training seed, private emulator state, ROM data, or game implementation
source is embedded.

## Confirmation and falsification

A confirming trace should contain the exact sequence `ramp_advance phase 2 ->
3`, an immediately following `jump_begin` (or a safe no-hazard continuation)
while `jump_lock > 0`, then further phase progress rather than the
`transition` at tick 566.  The strongest confirmation is a ramp-to-rivet stage
change, a banked screen bonus, or a nonzero `screens_completed`/`rivets_removed`
value.  A useful intermediate confirmation is that the second hazard check
fires only after a phase advance and that repeated same-slot hazards in the
same phase do not cause a rapid duplicate jump.

The score-floor tradeoff is confirmed if the first active route still
preserves the post-phase hazard behavior and a later life logs `farm_enabled`
with enough time to collect the established close-window jump points.  If the
phase-aware recheck never fires despite a small current-plane `row_error` and
the same upper-ramp death recurs, or if the new override causes repeated jumps
on the same object and lowers survival, the central hypothesis is falsified.
A higher score consisting only of farm-mode jump credit, with no extra phase or
screen progress, confirms only the fallback floor and not the central route
hypothesis.  The rivet mechanism is independently confirmed only by an actual
stage change and row-byte movement; it has not been claimed as validated by
prior traces.

A one-minute live smoke trial of the final candidate on seed 4 completed 899
quanta with score 900, zero stale ticks, no model calls, and no harness error.
The trace enters `farm_enabled` at the first life boundary but records no
`ramp_advance`; it ends at phase 0 with one life and the bonus still running
because the try was time-capped. This validates only the bounded fallback and
real-time loop. It does **not** confirm the phase-aware hypothesis: the first
route death prevented the phase-3 case from being exercised in this short trial.

## Tier-2 System-1 interface and fallback

### State schema

Jev is used asynchronously only after a measured connector stall.  One request
contains this strict, derived JSON shape (no raw RAM and no inaccessible
history):

```json
{
  "game": "Atari 2600 Donkey Kong NTSC retail cartridge",
  "event": "stall",
  "stage": "ramp",
  "player": {
    "horizontal_band": "left|middle|right",
    "vertical_band": "bottom|lower|middle|top",
    "motion": "up|down|steady",
    "jump_committed": false
  },
  "route": {
    "phase": 2,
    "branch": 0,
    "connector_count": 2,
    "alternate_available": true,
    "climb_open": false,
    "plane": "learned|uncertain",
    "rivet_row": 0,
    "rivet_rows": [0, 6, 12, 18]
  },
  "hazard": {
    "present": true,
    "slot": 3,
    "side": "right",
    "motion": "left",
    "distance_bucket": "near",
    "relative_motion": "closing",
    "closing": true,
    "urgency": "commit|overlap|soon|watch|near|clear"
  },
  "fallback_action": "right",
  "progress": {
    "lives_remaining": 1,
    "screens_completed": 0,
    "rivets_removed": 0
  },
  "last_action": "right"
}
```

Code computes coordinates, plane membership, closing motion, row masks,
connector legality, action strings, jump leases, and lifecycle.  Jev is not
asked to compare numbers, calculate a direction from x coordinates, or invent
an action mapping.

### Questions and answer bridge

The one request batches two independent judgments:

* `branch`: a `Choice` over `hold`, `nominal`, and `alternate`.  `alternate`
  is accepted only when `route.alternate_available` is true and the code has
  already measured a stall.  A response with an option probability below 0.65
  is discarded.
* `current_row`: a `Noul` asking whether the supplied normalized hazard is a
  current-walkway threat.  It is diagnostic only; it cannot suppress the hard
  jump guard or create a jump.

The actor switches to the deterministic alternate before waiting for Jev.  A
valid high-probability branch answer can affect only the still-current stall
signature; `hold`, a missing probability, a timeout, malformed stand-in model
output, a stale future, or a failed call leaves the code route unchanged.  At
most one future is outstanding.  The future is collected only when complete,
and all values are checked with `getattr` before use, so arbitrary pre-flight
answers cannot escape as an exception.

The hard fallback on every tick is:

1. one `fire` start edge;
2. sparse transition `fire` edges, including when the spare-life byte is zero;
3. an in-flight directional FIRE lease synchronized to RAM byte `$85`;
4. a current-plane overlap/commit guard, including the phase-aware post-jump
   recheck;
5. observed connector walking/climbing and, after a life boundary, the
   established close-window reward lane; and
6. raw-mask rivet routing with committed jumps across cleared gaps.

The loop posts one valid action after every `wait_for_tick`; it never blocks on
Jev and never returns merely because a model is late.  It stops only when the
context/game ends or after the documented zero-bonus, zero-spare transition
shell has had a short terminal grace period.

### Astra policy

This candidate deliberately makes no Astra call.  Astra's measured minimum
latency is several seconds and its normal answer latency is tens of seconds,
while this hypothesis concerns a six-frame/one-phase hazard boundary that is
already fully represented in observations.  There is no useful slow output to
turn into a timely joystick action here.  The finite macro facts that a slow
planner would review (`phase`, connector candidates, plane status, rivet mask,
lives, and current hazard bucket) are already serialized in the Jev state, and
all high-stakes execution remains code-owned.  Avoiding an unnecessary Astra
call also leaves no slow background task to outlive a game-over episode.
