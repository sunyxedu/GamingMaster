# Candidate 0010: ladder-settle transactions and velocity-opposed crossings

## Parent and hypothesis

Parent: `candidates/0008/`; `0007` is the positive control for the fixed-route
part.

The parent fixed the opening route, but its trace exposes a more specific
failure than “the route is wrong.” In `trace_seed0`, Mario reaches the first
ladder at `x=106`, climbs to `y=156`, and the controller records
`ramp_level=1`; from ticks 70 through 170 it posts `left` while the logged
state remains `x=106,y=156`. It has changed horizontal intent before the
cartridge has finished settling the ladder transaction. At the first death,
the hazard facts are an obstacle on the left moving right (`x=83 -> 103`) and
the controller repeatedly starts `rightfire` at ticks 180 and 189. The action
is away from the next route target and, at the right edge, does not cross the
incoming obstacle. The same still/`rightfire` pattern repeats after each
restart. The parent consequently reaches only phase 1, loses all lives, and
scores 0.

`0007` is useful evidence for the diagnosis: its simple action trace holds UP
through tick 79, does not turn left at the first visually-near y, and later
reaches `y=126` and earns a 100-point jump. Its scored jump is especially
informative: at `n=360`, the hazard is at approximately `x=34` with `dx=+2`
while Mario is near `x=50`, and the controller posts `leftfire`—the jump
opposes the obstacle's measured motion. By contrast, 0008 repeatedly sees an
obstacle moving right from `x=83` toward `103` and posts `rightfire`. It does
not establish a complete solution, but together these traces support the
following testable hypothesis:

> A climb must be treated as an edge-triggered transaction: keep UP until the
> new walkway is reached and vertically stable for several quanta, then turn
> toward the next ladder. When an obstacle closes, one committed jump should
> oppose its observed horizontal velocity (using the route side only when the
> obstacle is still), and remain latched until the obstacle has passed.
> Repeatedly firing an away-moving or same-direction jump is the failure mode.

This is a structural control change, not another Jev cadence or tolerance
change.

## What changed

* The ramp is an explicit six-phase route (`108,48,108,48,108,32`). A phase
  does not advance at the first y threshold. The controller keeps `UP`, tracks
  observed y/dy, and requires a minimum climb duration plus eight consecutive
  settled observations before it changes horizontal target. It never sends a
  horizontal action while the climb transaction is open.
* Hazard control is a separate crossing primitive. When a hazard has measured
  left/right motion, the jump direction is the opposite direction; if it is
  still, the next route side is the fallback. FIRE is held for eight quanta,
  followed by a ten-quantum non-FIRE landing tail. The interrupt uses a narrow
  same-row band and waits for a near/closing hazard rather than spending the
  jump at a 53-pixel approach. A 36-quantum hazard lock survives the FIRE tail
  and releases early only when the hazard clears, so a lingering sprite cannot
  create a new jump every eight ticks.
* A jump can land away from its rung. If that moves Mario more than two pixels
  from the ladder target, the climb transaction is aborted and horizontal
  alignment is reacquired before UP is attempted again. Exact one-pixel
  alignment is required to open a new climb; the earlier three-pixel shortcut
  produced a stationary UP hold at `x=51` for a ladder at `x=48`.
* Lifecycle remains code-owned: one FIRE starts idle play, transitions release
  input and send sparse restart edges after the observed death animation, and a
  zero-life transition returns without burning the wall deadline.
* The rivet controller keeps the raw row bytes as its progress oracle, sweeps
  the lowest incomplete row in a serpentine order, and uses the same crossing
  primitive after a byte change. It does not trust a prose interpretation of
  whether 6 or 12 names the left side: the side changed by the observed sweep
  is retained as the gap-crossing direction.
* Astra is asynchronous and low-authority. It receives the normalized current
  state and may choose only the first rivet side; malformed or late text is
  ignored. Jev receives a compact tactical state and can label `route`,
  `cross_left`, `cross_right`, or `pause`, plus an independent closing
  judgment. Neither answer can suppress the deterministic ladder settle,
  imminent crossing, legality, or restart fallback.

## Evidence that would confirm or refute it

Confirmation is a trace that stays at the first ladder for a short settle
interval, then moves left from approximately `x=106` rather than remaining
stuck, reaches at least the second/third ramp phase, and survives an incoming
obstacle without a sequence of same-direction/retriggered FIRE actions. A
100-point jump reward on a velocity-opposed crossing, more than 100 score, a
screen completion, or a rivet byte change would be progressively stronger
confirmation. A post-death transition followed by active play also confirms
that lifecycle did not mask the control experiment. The short live trials
already confirm the first part: this harness reaches phase 2 (`y≈126`) and
scores 100–200 on five seeds without stale ticks, whereas 0008 stalls at phase
1 and scores 0.

Refutation is a trace in which the UP/settle transaction is observed at the
correct ladder but the cartridge never accepts the next horizontal action, or
in which a velocity-opposed one-shot crossing still dies while the hazard is
well outside the collision box. If the controller reaches the rivet screen but
cannot change a row byte, the ramp hypothesis may be right while the rivet
side/gap interpretation is wrong; that would be a separate follow-up rather
than evidence for more model calls.

## System-1 interface and fallback policy

Every Jev request is strict JSON containing only derived, semantic facts:

* `stage`, player horizontal/vertical bands, motion, and whether a crossing is
  committed;
* route kind, ramp phase and target side, climb-active/settling flags, or the
  current rivet row/row bytes;
* one same-row hazard with side, motion, named distance bucket, closing flag,
  and urgency (`clear`, `watch`, `soon`, `immediate`); and
* lives, screens, rivets removed, and the last posted action.

The batched questions are:

1. `tactic`: a `Choice` among `route`, `cross_left`, `cross_right`, and
   `pause`. It recognizes the immediate label; code has already computed
   geometry and legality.
2. `closing`: a `Noul` asking whether the named hazard is an urgent same-row
   closing obstacle. It is logged as an independent diagnostic and is not
   allowed to override arithmetic.

Requests are spawned asynchronously every 12 ticks, or every 4 ticks while a
hazard is soon/immediate. A completed answer is tagged with the current stage
and consumed once. A hint can agree with the already computed crossing, but a
late, low-confidence, malformed, or failed answer changes nothing. The hard
fallback runs every tick: route transaction in active play, committed jump
when the hazard guard fires, neutral transition handling, one FIRE in idle, and
NOOP for terminal/dead states. Thus the measured Jev latency (including its
roughly two-tick tail) cannot stall the real-time actor.

Astra's prompt is a compact review of this same state and asks for one JSON
field, `rivet_first`. It is spawned only on lifecycle/screen changes and is
never awaited by the actor; the fixed ramp route and crossing direction remain
code-owned because a 50-second System-2 answer cannot safely decide a four-
frame quantum. The accepted field is whitelisted and can affect only a future
rivet sweep, so arbitrary stand-in output is harmless during pre-flight.
