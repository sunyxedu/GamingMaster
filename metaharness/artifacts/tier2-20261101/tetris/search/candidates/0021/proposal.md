# Candidate 0021 — transactional difficult-clear crossover

## Parents and diagnosis

Parents: `candidates/0017/` and `candidates/0019/`.

`0017` supplies the score and robustness baseline: its cheap guardian, gated
depth-two queue refinement, SRS-aware verified executor, and background work
completed all 1,200 training pieces with zero illegal actions and zero stale
ticks, for a mean **63,216.67**.  Its weakness is that the rank is mostly an
ordinary surface/line heuristic: the mean B2B bonus is only **333.33**, and
five of six episodes earn no B2B bonus.

`0019` adds the useful missing mechanism: it carries combo/B2B and conservative
T-spin geometry into the placement rank and describes difficult candidates to
Jev.  This produced real difficult clears in its better traces (for example
seed 1 reports a Tetris with a **1,600** B2B bonus), but its execution change
was not safe.  The worst trace, seed 4, has the same score **11,372** from
piece 75 through tick 719: roughly every fifth tick is `left`, the intervening
ticks are `rotate_cw`, and only later does it hard-drop.  The piece never
advances while the actions are all fresh.  Seed 4 therefore tops out at 125
pieces and 38 lines despite zero stale ticks.  This is a routing-progress
failure, not evidence that the tactical rank itself is bad.

The likely trigger is the spin-specific BFS: it requires a route to end in a
rotation but recomputes that route after every tick.  A route can be legal and
still make no transactional progress toward the target; repeated rotations
are then counted as fresh progress by the clock watchdog.  `0017`'s ordinary
route never exhibited this long spin loop.

## Hypothesis

A **transactional difficult-clear layer** can retain `0019`'s score-state
recognition while recovering `0017`'s survival if a spin target is treated as a
bounded transaction rather than an indefinitely re-planned goal.  The code
will still prefer a verified full/corner-supported T-spin or Tetris only when
its simulated board is safe, but it will abort the target after repeated poses,
blocked inputs, too many rotations, or a short maneuver budget and immediately
hard-drop the current piece.  This specifically tests whether the 0019 score
loss came from execution livelock rather than from difficult-clear utility.

A secondary conservative change is that only a rotated, three-corner
(full-spin) candidate is allowed to force a rotation-ending route.  A
one-line/two-corner mini hint may remain a descriptive tie-breaker, but it
cannot trap the executor in a spin transaction.  The engine's observed lock
event remains authoritative; geometry is only a search hint.

## What changed

* Retain `0017`/`0019`'s code-owned board simulation, hold alternatives,
  SRS kick tables, guardian, bounded queue beam, background search, reversed
  Jev role check, and asynchronous Astra policy review.
* Retain `0019`'s combo/B2B-aware tactical utility and T-spin corner facts, but
  distinguish `spin_route` (full, rotated, line-clearing or explicitly safe
  full-spin setup) from a weaker `tspin_hint`.  Only `spin_route` changes the
  router; ordinary and mini candidates use the proven route.
* Add a per-piece executor transaction ledger.  It records the active pose,
  repeated `(orientation, x, grounded)` poses, movement/rotation attempts, and
  whether the last maneuver changed the pose.  A repeated grounded pose,
  blocked maneuver, more than four rotations, or more than twelve maneuver
  actions aborts the target and posts `hard_drop`; the abort is sticky until
  the next lock.  This prevents the exact `left`/`rotate_cw` loop in 0019
  from consuming hundreds of ticks.
* Remove the special case that invents an extra rotation when a target pose is
  already reached.  A successful final rotation is sufficient; if the target
  orientation equals the current orientation, the candidate is not a spin
  route.
* Log `transaction_abort`, maneuver counts, and spin-route counts so the
  evaluation can distinguish a safe fallback from a silently failed plan.

## System-1 state and bridge

Jev receives a fresh, strict-JSON derived state for a current shortlist:

```text
{
  "ruleset": "guideline-marathon-v1",
  "policy": "BALANCED | RECOVER | WELL",
  "piece_instance": integer,
  "board": {
    "height": "low | moderate | high | critical",
    "holes": "hole-free | few shallow holes | several holes | many or deep holes",
    "surface": "flat | uneven | rough",
    "lines": integer, "level": integer,
    "combo": "active | inactive", "back_to_back": boolean
  },
  "active_piece": {"type": string, "orientation": integer, "anchor_x": integer},
  "hold": string | null, "can_hold": boolean,
  "next_queue": [string],
  "fallback_role": "safe",
  "roles": {"safe": description, "clear": description,
            "future": description, "recovery": description},
  "candidate_count": integer
}
```

The role descriptions are generated only from legal simulated placements and
state their clear size, projected height/hole/surface buckets, and whether the
candidate is a difficult clear, combo continuation, B2B-preserving clear, or
full-spin route.  Jev chooses a role, never a coordinate or primitive input.
Two independent `Choice` questions use reversed option order.  A response is
accepted only when both values agree, the selected-label probability is at
least **0.45** in each answer, the answer is for the current `(pieces,
active.type)` snapshot, and hold compatibility still holds.  Otherwise the
code-ranked plan remains active.  A Jev timeout or malformed stand-in answer
is a normal fallback, not an exception escaping the episode.

Astra is asked asynchronously at the opening and on a later danger
transition to return one of `BALANCED`, `RECOVER`, or `WELL`.  Its answer only
changes future evaluator weights.  It cannot revise an in-flight route or
post an action; failed/late text leaves the current policy unchanged.

## Real-time fallback policy

Every tick has a code-owned action.  A new piece first receives a fresh noop
and then an immediate coarse guardian; search and models run in the
background.  Search results are tagged by piece and discarded after a lock or
hold mismatch.  The latest observation is rechecked before every action.

For a normal plan, the bounded SRS BFS routes toward the target.  For a
`spin_route`, the same BFS must include a genuine target-orientation change
and end with a rotation; its transaction budget is still finite.  If the
active pose repeats, an attempted move/rotation does not change `(x,
orientation)`, the route exceeds four rotations or twelve maneuvers, the
transaction is marked aborted and `hard_drop` is posted immediately.  A
stale/late model or worker result cannot clear the abort flag.  Hold is used
at most once and only when `can_hold` is true.  Near lock, hard drop wins over
any pending answer.

## Confirmation or refutation

Confirmation requires all six episodes to reach 200 pieces with zero illegal
actions and near-zero stale ticks, while mean score recovers at least the
**63,216.67** parent level.  A stronger confirmation is increased line-clear,
combo, difficult-clear, or B2B points without the 0019-style long gap between
locks.  Traces should show occasional `spin_route` selections and, if a route
is bad, a bounded `transaction_abort` followed by a lock within a few ticks;
no episode should contain hundreds of fresh rotations for one piece.

The hypothesis is refuted if the watchdog causes frequent premature drops and
loses 0017's lines/score despite no livelock, if engine events do not validate
spin routes, or if the controller again tops out with a long no-lock action
run.  If survival returns but B2B remains at 0017 levels, the routing diagnosis
is supported but the tactical layer did not create useful multi-piece spin
setups.  If score falls while aborts are rare, the rank/geometry utility—not
the 0019 executor loop—needs revision.
