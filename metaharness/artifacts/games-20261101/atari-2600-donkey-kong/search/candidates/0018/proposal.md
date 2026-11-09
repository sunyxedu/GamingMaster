# Candidate 0018 — separate route phase from connector choice

## Parents and hypothesis

Parent: `candidates/0015/` (the last-life/lifecycle refinement).  The strongest
comparison is `candidates/0017/`, which combined that route family with a
same-direction jump policy and is the current leaderboard best.

The concrete failure hypothesis is a controller-state bug, not a lack of jump
points:

> The best candidate couples the route-phase update to its alternate-connector
> detection.  On the nominal phase-0 connector there is no alternate, so the
> code never executes the climb-progress and phase-advance logic.  Mario can
> earn a jump reward indefinitely while the controller still believes it is on
> the bottom walkway.

This is visible without inferring anything from the score.  In
`candidates/0017/trace_seed0.jsonl`, snapshots at ticks 60, 80, 100, 120,
140, 160, and 180 report `phase=0`, `x=105/106`, `y=154`, and `action="up"`.
There are no `ramp_advance` events in that trace.  The trace then records 40
`cross_begin` events, all with `phase=0`, and finishes in transition at score
4000.  The same pattern appears in the other seeds: the 3900--4100 reward is
repeated jump credit, not vertical or screen progress.  Thus adding more jump
threshold tuning would preserve the failure.

There is positive evidence for the missing mechanism.  The parent trace in
`candidates/0015/trace_seed0.jsonl` reaches phase 1 at `(105,154)` and phase 2
at `(47,126)` after observed y motion.  The adaptive experiment in
`candidates/0016/trace_seed0.jsonl` reaches phase 3 at `(105,98)` on its final
Mario.  Those traces establish that observed-y confirmation and the nominal
player-edge route work at least through the upper ramp; they do not establish
that a fixed horizontal target is sufficient after a jump landing.

## What changed

This candidate is a structural rewrite of the control split:

* **Code owns an explicit connector graph.**  `phase` is the walkway that the
  latest observation proves Mario occupies.  `branch` is only the connector
  selected on that walkway.  Climb bookkeeping (`climb_origin_y`, best y,
  measured gain and stall) is evaluated on every climb observation, outside
  connector-discovery conditionals.  A phase can advance only when the player
  has moved upward by at least eight pixels and is near the next row.
* **Connector discovery is observation-driven.**  The split rows use the
  verified player-edge options `(46,78)`, `(106,86)`, `(46,66)`, and `(30,74)`.
  A jump landing near a nonselected connector can adopt that connector only
  on the immediately following confirmed ramp-jump landing; a normal walk
  passes many visible ladders and must not silently turn the nominal route
  into a detour.  A horizontal or vertical no-progress window tries the
  other graph edge, records the failed edge, and asks Jev about the route
  event while the deterministic alternate is already running.  After the
  first nominal climb is confirmed, the controller deliberately protects the
  first Mario with the proven right-ladder lane; after a life boundary it keeps
  that explicit overlap-only score-preserving fallback rather than repeating
  an unvalidated upper-ramp experiment.  This specifically handles the
  previously observed `(x≈87,y≈111)` upper-ramp landing rather than walking
  forever toward x=106.
* **Jump and route are separate transactions.**  A hazard jump is directional
  FIRE until the documented RAM jump byte confirms it, followed by a short
  route-directed landing tail and a fresh connector reconciliation.  The
  hazard guard does not interrupt a climb that has visibly gained height unless
  the object overlaps Mario.  This keeps the useful close crossing behavior
  while preventing jump reward from becoming the route objective.
* **Lifecycle remains code-owned but is corrected for all lives.**  Initial
  FIRE and sparse transition FIRE edges are posted without waiting for a model.
  The explicit terminal flags, not `lives_remaining`, stop the actor.  In
  particular the zero-spare transition that `0015` proved playable still gets
  its restart pulse.
* **The model division is moved to route events.**  Unlike the parent, Jev is
  not reserved for an untested rivet screen.  It receives a compact event
  state at a genuine connector stall or immediate hazard and can select an
  already-described alternate connector or escape label.  It cannot invent a
  coordinate, suppress a hard jump, or delay an action.  Astra is started
  asynchronously after repeated connector deadlocks or on rivet entry to
  review macro connector preference, climb-vs-jump policy, and initial rivet
  side.  Its parsed labels are applied only at a later, revalidated boundary.
  Normal nominal ramp play therefore remains zero-latency code, while the
  models have a useful, falsifiable role when the finite route disagrees with
  observations.
* **The rivet route is implemented as the same kind of observable state
  machine.**  Code decodes row bytes, detects which rivet disappeared, jumps
  over a newly created gap, follows the bottom-to-top serpentine side order,
  and climbs only from the row-side ladder.  Astra can change only the first
  side preference; Jev can advise a live hazard/route event.  Garbage model
  text and missing fields leave this fallback unchanged.

No seed-specific trace, emulator state, or private game source is used.

## Confirmation and falsification

The primary confirmation is a trace containing `ramp_advance` events with
phases 1, 2, and 3, followed by a stage change to `rivet`, while the number of
jump rewards is lower than the 0017 jump-farming trace.  A more diagnostic
confirmation is a jump landing near `(87,111)` followed by a
`connector_adopted` event, `climb_begin` on the alternate x≈86 connector, and
subsequent measured y progress.  A completed ramp should bank the remaining
bonus and make `screens_completed` or `rivets_removed` nonzero rather than
ending with a zero-bonus transition shell.  The lifecycle fix is confirmed if a
zero-spare transition receives an edge and returns to active play, as it did in
0015 seed 5.

The phase-separation hypothesis is refuted if nominal phase 0 still remains
at y≈154 despite `climb_begin`, or if phase advances without the stated y gain.
The connector hypothesis is refuted if a verified `(87,111)` landing repeatedly
fails to respond to UP at x≈86 and the alternate branch is never accepted.  The
jump policy is refuted if close, RAM-confirmed crossings still kill Mario
before any route progress while the same trace would progress with no jump.
The model bridge is not considered a success merely because a Jev/Astra call
returns: a late or low-confidence answer must have no effect, and model
failures must leave the episode error-free.

A local four-minute trial on training seed 2 reached score **4100** at 3236
real-time ticks with zero stale ticks and no screen completion.  It logged the
nominal phase-0-to-phase-1 y-confirmation, then used the guarded overlap-only
lane; after the final transition the `current_bonus==0` score shell ended the
harness at 4100 instead of waiting for the deadline.  This confirms the
controller is at least competitive with 0017's 4000-point score-farming
behavior and confirms the endpoint guard, but it also refutes any claim that
this iteration has solved the upper ramp or rivet screen.  The final harness
uses a 0.65-second Jev timeout; all such calls remain asynchronous and late
answers still fall back to code.

## Tier-2 System-1 contract and fallback

### State schema sent to Jev

Every Jev request is strict JSON and contains only derived, semantic fields:

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
    "phase": 2,
    "selected_connector": 0,
    "connector_options": 2,
    "alternative_available": true,
    "climb_open": false,
    "fallback_action": "right",
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

Code computes coordinates, row-byte meanings, distance buckets, current
connector, jump state, and legality.  Jev is never asked to compare numbers or
calculate a direction from raw x values.

### Questions, thresholds, and application

One request batches two independent typed judgments:

1. `route`: a `Choice` over `hold`, `nominal`, and `alternate`.  `alternate`
   is meaningful only on a split walkway during `connector_stall`; `hold` is
   the safe answer during a live climb or a hazard event.
2. `escape`: a `Choice` over `keep_climbing`, `jump_left`, and `jump_right`.
   It is advisory recognition of an explicit same-row immediate/overlap hazard,
   not a replacement for the code guard.

The actor starts the deterministic fallback first.  A response is accepted
only if the option is valid, its returned option probability is at least 0.60,
and its event signature (stage, phase, branch, hazard slot/urgency, and
deadlock count) is still current.  It expires after three control ticks.  A
valid alternate can switch only a currently split connector; a jump answer
can start only a currently immediate/overlap hazard when no jump is committed.
A response with absent probabilities is accepted only for the conservative
`hold`/`keep_climbing` values.  Timeout, malformed answer, failed future, low
probability, stale signature, or unavailable budget all use the already-valid
code action.  There is at most one outstanding Jev future.

Requests are event-triggered rather than periodic: a connector request is
made after the no-progress switch, and a hazard request is made when the hard
immediate guard fires.  This avoids spending the 0.15--0.2 second Jev latency
on routine four-frame walking and makes a Jev-informed action diagnostically
meaningful if it arrives while the event is still current.

### Astra bridge

Astra is spawned, never awaited, after two route deadlocks or on entry to the
rivet screen, with the current state schema and the public route facts.  It is
asked for one compact JSON macro review:

```json
{"connector":"nominal|alternate|adaptive",
 "hazard":"climb|jump",
 "first_side":"left|right"}
```

Parsing accepts plain or fenced JSON and ignores every other response.  A
connector label changes only the next branch selection; `first_side` changes
only a fresh rivet row; `hazard` is a recorded macro preference and cannot
override the overlap guard.  The actor continues posting actions throughout
an 85-second Astra call.  If the game ends first, the future is allowed to be
cancelled by the context.  This is intentionally a macro-revision role, not a
slow action policy like the failed Astra-only candidates.

## Runtime safety

The loop posts one valid action per observed real-time tick and uses
`ctx.wait_for_tick()`.  It starts with FIRE, never uses reset as an action,
continues through a nonterminal zero-spare transition, and stops only on
`ctx.done`, explicit game-over/terminal state, or a game-side end.  All model
parsing and action selection has defensive fallbacks, so arbitrary stand-in
model text during preflight cannot escape as an exception.
