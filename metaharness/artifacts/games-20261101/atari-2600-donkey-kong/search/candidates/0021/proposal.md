# Candidate 0021 proposal

## Parents and hypothesis

Parents: `candidates/0018/` and `candidates/0019/`.

`0018` supplies the strongest score mechanism: a committed, RAM-synchronized
jump transaction, a sparse FIRE lifecycle, and a score-lane fallback that
continues to play after the spare-life byte reaches zero.  Its traces also show
why its score is misleading: after the first observed climb it enables
`farm_mode`, and `trace_seed3` records `phase=1`, `x≈99`, `y=148` for dozens of
successive 100-point jumps.  It reaches 4,000 points without leaving the first
upper girder.  The score is durable, but it is not route progress.

`0019` supplies the useful route abstraction: observed y movement advances a
finite connector graph, rather than counting UP actions.  In its best route
attempts it confirms phase `0→1→2` at `(105,154)` and `(46,126)`.  Its concrete
worst failure is at the next hazard: at tick 158 in `trace_seed3`, Mario is on
phase 2 and a closing object is 22 pixels to the right; the controller starts a
jump, then the trace has `uprightfire` through tick 166 and returns to a
transition.  The same event repeats after restart at ticks 334 and 536.  The
controller keeps branch 0 (target x≈106), even though the successful jump
observation in the earlier route family places Mario around `(87,118)`, next to
the split connector at x≈86.

The hypothesis is:

> The fatal upper-ramp mistake is not merely “jump too late” or “choose more
> Jev actions.”  It is a transaction-boundary mistake.  A jump can land on a
> different usable connector on the current walkway.  If the controller waits
> until the jump tail ends and then resumes the old target, it walks away from
> that connector and into the approaching obstacle.  Resolve the landing
> immediately, adopt the nearest fixed split edge, and use a short
> opposite-direction tail to settle onto it.  Give this route experiment one
> life; if it fails, retain 0018's proven score-preserving lane on subsequent
> lives instead of repeating the same upper-ramp risk.

This is a structural crossover rather than a jump threshold tweak.  It tests
whether the `(87,118)` landing is actionable route evidence, while retaining a
bounded fallback so a failed route experiment does not consume the whole
episode.

## What changed

* The ramp is an explicit six-row connector graph:
  `((106,), (46,78), (106,86), (46,66), (106,), (30,74))`.  `phase` is only
  advanced after an observed upward y gain and arrival at the next row.
* A jump owns FIRE until raw RAM byte `$85` is observed active and then cleared.
  On the first post-airborne observation, code checks the current phase's
  already-known connector edges.  A landing near x=86 at phase 2 adopts branch
  1 before the landing tail.  The route tail is opposite the FIRE direction
  (the earlier horizontal-FIRE route traces use `rightfire` followed by a
  short `left` tail); score-lane jumps retain their same-direction tail.
* Ramp hazards use horizontal directional FIRE rather than the diagonal
  `uprightfire` used by 0019.  This specifically tests the earlier public trace
  in which `rightfire` reached `(89,117)` and allowed the route ledger to
  observe the upper row.  Code, not Jev, selects the literal action.
* A split connector is tried once, then the other fixed edge is tried after a
  measured horizontal or vertical no-progress window.  No timer alone can
  advance a phase.  A route stall enables a bounded farm fallback; a life
  boundary arms the durable phase-1 score lane for the next Mario.  Thus the
  first Mario tests progress, while later Marios use 0018's survival/reward
  mechanism rather than repeating an empirically lethal branch.
* The lifecycle sends one FIRE from idle and sparse FIRE edges during every
  nonterminal transition, including `lives_remaining == 0`; only explicit
  terminal/game-over flags stop play.  Malformed observations and missing RAM
  bytes degrade to a safe action.
* The rivet controller is included as an observation-driven continuation: it
  decodes 0/6/12/18 row bytes, jumps across a newly cleared gap, serpentine
  traverses rows, and climbs from the side ladder.  It is not allowed to walk
  through a detected gap.
* Astra is a nonblocking macro reviewer after a route stall/life boundary or
  rivet entry.  It may propose only `branch`, `first_side`, and `risk` labels.
  Its result is parsed defensively and applied only at a fresh route boundary.

No training seed, private emulator state, ROM data, or held-out result is used.

## Confirmation and falsification

The central confirmation is a trace with an event such as
`landing_connector_adopted(phase=2, branch=1)` immediately after a hazard jump,
followed by `climb_begin` at x≈86, `ramp_advance` to phase 3, and eventually
`stage=rivet`.  A stronger confirmation is a lower count of repeated jumps at
one fixed phase and a nonzero `screens_completed` or `rivets_removed` value.
The risk-budget mechanism is confirmed if a failed first route attempt causes
later active play to log `farm=True` and earn points without retrying the same
phase-2 edge.

The landing hypothesis is refuted if the observation is near x≈86/y≈118 but
adoption is never followed by vertical progress, or if the opposite tail
systematically kills Mario where the old tail survived.  It is also refuted if
all traces still repeat phase-2 jumps with branch 0, or if a route reaches the
rivet screen but the raw row bytes never change.  A high score with only
`phase=1`, repeated jumps, zero screens, and zero rivets would confirm only the
fallback and would not count as route success.

## Tier-2 System-1 schema and bridge

Jev is used only for a measured connector stall or an immediate hazard.  The
actor posts a deterministic fallback before any answer can arrive, waits for
the next game tick, and keeps at most one outstanding request.  The strict JSON
state sent to Jev has this shape:

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
    "selected_branch": 0,
    "branch_count": 2,
    "alternate_available": true,
    "climb_open": false,
    "fallback_action": "right",
    "row": 0,
    "rivet_rows": [0, 6, 12, 18],
    "farm_mode": false
  },
  "hazard": {
    "present": true,
    "slot": 3,
    "side": "right",
    "motion": "left",
    "distance_bucket": "near",
    "closing": true,
    "urgency": "commit"
  },
  "progress": {"lives_remaining": 1, "screens_completed": 0, "rivets_removed": 0},
  "last_action": "right"
}
```

Code computes all coordinates, distance/closing buckets, row-byte decoding,
legality, score arithmetic, and the connector candidates.  Jev is not asked to
count, compare raw numbers, infer action indices, or invent a coordinate.

Each request batches two independent typed questions:

1. `route`: a `Choice` over `hold`, `nominal`, `alternate`, and `farm`.
   `alternate` is accepted only on a split row with a measured stall;
   `farm` only changes the already-described score-lane mode.  A returned
   option must have probability at least `0.60` and the observation signature
   (stage, phase, branch, hazard slot/urgency, and route-failure count) must
   still match.  The code fallback tries the alternate edge immediately and
   then enables the farm lane; Jev never delays that action.
2. `danger`: a `Noul` asking whether the supplied object is an explicit
   commit/overlap current-row threat.  It is diagnostic; the hard hazard guard
   owns all lethal jump decisions.

Late, failed, malformed, low-probability, or stale answers are discarded.
Stand-in model text can therefore not escape the harness or turn into an
illegal action.  An answer expires after three control ticks.

Astra receives the same compact derived state asynchronously and is prompted
for one JSON object with optional `branch`, `first_side`, and `risk` labels.
`branch` is used only at a later connector selection, `first_side` only before
rivet-row progress, and `risk` only records the macro preference; no Astra
output can interrupt a jump, suppress a restart, or post an action.  The slow
call is never awaited by the actor loop and may be cancelled when the episode
ends.

## Runtime safety

The loop sends a valid action immediately, then uses `ctx.wait_for_tick()` and
observes before every next post.  It never calls a game reset action, never
runs a background polling loop, and catches model/future parsing failures.
The only background work is a finite `ctx.spawn` inference call that is
cancelable by the episode context.
