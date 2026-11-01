# Candidate 0010 — temporal deconfliction and learned lethal edges

**Parent:** `candidates/0008/` (with the best execution evidence from `candidates/0007/`).

## Hypothesis

The remaining ceiling is not another first-board palette-boundary bug.  It is a
short-horizon control failure: the controller commits to a legal edge when the
observation looks clear, then the Q*bert/enemy animation advances several ticks
before the collision is reported.  A late observation can contain no live enemy
at all, so a policy that only blocks currently occupied `safe_moves` still dies.

The archive gives a concrete repeated pattern:

* In `0007` seed 0, the controller had four lives through inferred board 4.
  At tick 2312 it chose `UP` from node 4 with no tracked threats; the lesson
  records that a Purple Ball was adjacent by tick 2322 and a life was lost.
  Similar losses followed at ticks 2222 and 2422 while the controller was
  launching ordinary hops.
* The event trace shows the losses arriving during the same temporal window as
  a newly launched hop, not at a stable observation with an occupied landing.
  For example, seed 0 landed `7 -> 4` at tick 2311, reported the clear node at
  tick 2312, and lost the life at tick 2322.  In seed 4, a Purple Ball was
  visible near node 8 at tick 1964, the controller reached node 18 at tick
  1973, and still lost the life at tick 1987.  The existing four-tick hazard
  memory and one-tick post-hop release did not cover this window.
* `0007` repeatedly revisited bottom/return edges (for example `14 <-> 20`
  around the fourth and sixth inferred clears).  A collision is therefore
  evidence about the attempted edge and its local timing, but the previous
  controller did not retain that evidence or change its route mode after a
  life loss.
* `0008` improved the completion handoff but still lost all lives before it
  could establish a later-board survival policy.  Its false handoff/ledger
  failure is already explicitly refuted in the lessons and is not repeated
  here.

**Test:** use a longer but decaying enemy shadow, add a bounded neutral dwell
only after a *confirmed* landing, and learn the edge/endpoint associated with
each observed life loss.  After a loss, change the coverage sweep (mirrored
sweep, center/escape-biased sweep, then nearest) rather than replaying the same
route timing.  This is an online, seed-agnostic adaptation: no seed or action
prefix is stored.

The intended prediction is fewer losses per board and more score after the
fourth inferred clear, even if the controller spends a few extra ticks per hop.
If this is the wrong diagnosis, it should either preserve the same death ticks
while reducing coverage, or produce large stalls because the enemy shadow is
mistaken for a permanent blockage.

## What changed

### 1. Board-local palette model, with a separate handoff mode

The harness learns the destination from a local positive score/color transition,
uses the current raw palette only to derive unfinished nodes, and detects the
large whole-board flash.  It does not trust `completed_rounds` or fixed palette
numbers.  A stable all-destination view enters `handoff`; the controller posts
neutral input until the flash ends, then starts a fresh epoch.  Immediately
after the flash it ignores a stale all-zero `needed_landings` vector until a
local scored landing calibrates the new board, so the new epoch cannot be
silently emptied by the previous board's decoder frame.  A bounded 60-tick
recovery prevents a decoder omission from turning into an infinite wait.  This
preserves the useful part of 0007/0008 but is a separate implementation and
keeps the temporal experiment isolated from the old one-cube target trap.

### 2. Short-lived enemy shadows and graph-aware risk

`RiskMemory` stores each visible lethal object with its estimated node, motion
and age for up to 16 ticks.  It projects Purple/Red Balls one and two downhill
edges, gives Coily a pursuer shadow, and decays the cost as the evidence ages.
The current adaptor `safe_moves` remains authoritative: the harness never
posts an edge absent from it.  The shadow is only a cost/risk signal, not a
model-generated action.

The key new feature is `edge_failures`.  On a life decrease, code selects the
most recent launched edge in a 22-tick collision window (falling back to the
most recent landing), increments that edge's failure count, and gives its
endpoint a temporary 110-tick danger shadow.  This is not a permanent ban: if
it is the only certified way to an unfinished cube, its risk can still be
accepted.  It also handles the documented case where the enemy sprite is absent
by the time the life counter changes.

### 3. Temporal landing dwell and route perturbation

The motor emits an edge for at most two action ticks and neutral input during
hop animation.  It confirms a landing only when the player is standing at the
expected node.  It then dwells neutrally for one or two ticks; lower rows,
recent hazard evidence, and the optional cautious policy add at most two more.
The dwell is bounded and is never used as a general idle controller.

Targets remain hysteretic until their useful color is observed, but candidate
edges are ranked by route distance, projected risk, escape-space, failed-edge
cost, recent backtracking, and a generated sweep rank.  The first life retains
the validated nearest-coverage behavior so the experiment does not sacrifice
the reliable opening.  After life losses the route changes to the mirrored
sweep, then a center/escape-biased ranking, then nearest coverage.  Thus the
same board and seed do not force every life through the same enemy timing.

The loop continues until the actual game terminal signal; reported `lives == 0`
is a risk fact, not permission to stop, because the archive showed useful score
continuing after that counter reached zero.

## System-1 interface and bridge

At most one asynchronous Jev request is started every six standing decisions.
The state is a strict, derived JSON object:

```json
{
  "clock": {"tick": 123, "observation_age": 0},
  "position": {"node": 7, "motion": "standing",
                "safe_moves": ["UP", "DOWN", "RIGHT"]},
  "progress": {"board_epoch": 4, "unfinished": [2, 5, 9],
                "target": 5, "route_style": "right_sweep"},
  "candidates": [
    {"action":"UP", "landing":"unfinished", "danger":"clear",
     "escape_space":"multiple", "route":"near", "useful":true,
     "code_preferred":true}
  ],
  "hazards": [{"kind":"purple_ball", "node":8,
                "risk":"near", "age":2}],
  "recent_edge_failures": [[7, 4, 1]],
  "proposed": "UP"
}
```

Code supplies legal candidates, graph distance buckets, usefulness, hazard
shadow and failure facts; Jev never counts nodes or invents joystick mappings.
Each request batches:

1. `choice_a`: a `Choice` among the supplied directions, bounded `NOOP`, and
   `unknown`;
2. `choice_b`: the same judgment with reversed option order, reducing the
   documented first-option bias;
3. `danger`: a `Noul` asking whether the proposed edge has a concrete
   immediate lethal threat before its landing.

The bridge accepts a choice only when the two choices agree, both selected
probabilities are at least **0.62**, the answer is no more than five ticks old,
and the current node, board epoch, life epoch, hazard signature and current
safe set still match.  A selected edge must be within 1.8 cost units and 0.7
risk units of the deterministic fallback.  `NOOP` is accepted only when the
fallback itself is meaningfully threatened.  A high Jev danger answer can only
select the next code-certified candidate; it cannot veto into an illegal edge.
Missing probabilities, arbitrary stand-in answers, disagreement, timeout or
malformed output all fall back to the deterministic planner.

The Jev timeout is 0.8 seconds and calls run through `ctx.spawn`; a late answer
never blocks `tick`.  The loop posts a valid action or neutral fallback and
calls `ctx.wait_for_tick()` every cycle, so stale ticks should remain near zero.

## Astra role and revision trigger

Astra is deliberately off-clock and capped at two asynchronous calls.  It sees
only score/lives, inferred board epoch, unfinished nodes, recent hazard facts,
learned failed edges, and the current route mode.  It returns a tiny JSON policy
with `style` (`left_sweep`, `right_sweep`, `center_escape`, or `nearest`) and
`dwell` (`short` or `long`).  The code validates both fields; unsupported or
non-JSON text is ignored.  An initial policy can set the route style, while a
life-loss audit can request longer dwell; after a loss, the deterministic
failure-response schedule has priority over an old Astra suggestion.  Astra
never supplies an action, edge, disc exit, FIRE, or arithmetic.

A later revision should be triggered if the trace shows many life losses with
no recent launch in the attribution window, or if the same edge fails despite
its temporary penalty.  That would mean the missing variable is a broader
hazard phase/timer model rather than local edge timing.  Conversely, repeated
life preservation through epochs 4–6, fewer repeated edge failures, and a
higher score with only a modest action-time increase would support this
interface and hypothesis.

## Evaluation accounting

Expected confirmation is a higher mean score and longer game survival, not
optional bonus points.  Stale ticks should remain close to zero because all
model work is asynchronous.  S1 calls are sparse and S2 calls are limited to
policy audits.  The implementation uses only standard library modules and
`metaharness.models.jev`, named six-action controls, the public 21-node graph,
and public observations; it does not use raw-RAM writes, save states, images,
network services, or seed-specific routes.
