# Candidate 0016 — corridor certificates and a sticky-action handshake

**Mode:** rewrite. **Parent:** `candidates/0009/` (with `0007` as the score baseline).

## Hypothesis

The surviving controllers have mostly solved *where a single hop is legal*, but not
whether the landing is a survivable place from which the next hop can be started.
The late deaths are not explained by a missing choice from Jev or by a single stale
sprite record.

The strongest evidence is in the parent/best traces:

* In `0007/trace_seed0.jsonl`, the controller remained at four lives through the
  fourth inferred clear at tick 2017, then repeatedly traversed the same outer
  return.  It selected `DOWN` from node 14 at tick 2219 while Coily was estimated
  at node 9 and lost a life at tick 2222.  After recovery it selected `UP` from
  node 4 at tick 2312 with `hazards=[]` and `purple_ball_state=36`; the life loss
  arrived at tick 2322.  Another loss occurred at tick 2382 with the same empty
  hazard list.  Thus current `safe_moves` and a fresh one-hop answer were not a
  sufficient future-safety certificate.
* The same trace shows a motor/observation mismatch after the first of those
  losses: the controller posted `LEFT` from node 14, but the observed landing was
  node 19 rather than its expected node 9 at tick 2260.  The parent logged the
  landing and continued.  `0009` made landing verification explicit, but its seed
  4 still lost at 1783/1792 after the intended landing was verified, so verification
  alone is not the missing mechanism.
* `0013` logged many `hidden_ball_guard` and `occlusion_decision` events, yet still
  lost in empty-hazard/state-36 windows.  A timer-only directional bias is therefore
  not a reliable enemy decode.  Conversely, `0009`'s broad radar and waits reduced
  the mean to 15,454, and `0014`'s shelter logic reduced it further; blanket
  conservatism is not the answer either.

The testable hypothesis is that the controller should certify a **short corridor**
for every first edge: immediate landing risk plus the number of usable exits from
the landing under a bounded public-object time envelope.  It should also treat the
joystick as a two-stage transaction: clear the prior sticky edge, post a new edge
only while standing, and neutralize after the observed landing.  After a life loss,
the controller should spend a bounded recovery interval re-establishing the top /
standing state instead of immediately reusing the old route context.

This is deliberately not a claim that the public timer identifies a hidden ball.
The timer and scheduler can add a small uncertainty cost, but no enemy node is
invented from them.

## Structural changes from 0009

### Code owns a corridor, not a one-edge score

`ColourBook` retains the useful palette-epoch idea but keeps all progress evidence
local and explicitly inferred.  `EnemyField` stores only rendered lethal objects
and short-lived memory.  For each legal edge, `CorridorPlanner` computes:

1. collision risk at the takeoff and intended landing;
2. a bounded rolling/pursuit possibility set for the next hop;
3. how many legal exits from the landing remain below the same risk threshold;
4. distance to an unfinished cube, repeated-node cost, and whether the landing is
   useful.

The result is a ranked set of corridor records (`edge_UP`, etc.), not a queued
macro.  A closed corridor is strongly disfavoured but is never made an illegal
edge by local code; the adaptor's `safe_moves` remains the hard legality filter.
The planner uses a stable, port-specific 21-node coverage preference only as a tie
breaker and re-evaluates every edge after each observation.

The public `object_scheduler` is used only to mark a short spawn-uncertainty
window after a large reset.  `purple_ball_state` is reported to the strategic
summary but cannot create a fabricated position.  This is a narrower and more
falsifiable use than 0009's broad graph shadow or 0013's state-36 climb guard.

### The motor owns sticky-action hygiene

`HandshakeMotor` is folded into `Controller.motor` as a small transaction state:

* at most two posts of an edge while still standing;
* `NOOP` during the animation;
* accept a hop only after observing a standing node, logging a mismatch and
  resynchronizing if it differs from the expected destination;
* two neutral ticks after every landing, and a longer bounded neutral/recovery
  window after a life loss or mismatch.

No model can extend a direction through a hop.  No resource counter stops the
loop; only the real game/context terminal signal does, preserving the lesson that
reported zero lives can precede additional scored ticks.

### Jev changes level of abstraction

Earlier Jev bridges asked for an immediate joystick direction.  Here Jev receives a
small strict JSON state of **corridors**:

```json
{
  "position": {"node": 14, "safe_moves": ["LEFT", "DOWN"]},
  "progress": {"epoch": 4, "unfinished": 18, "target": 19},
  "corridors": [
    {"key":"edge_LEFT", "action":"LEFT", "landing":"unfinished",
     "danger":"clear", "escape":"open", "time_window":"ordinary",
     "code_preferred":true},
    {"key":"edge_DOWN", "action":"DOWN", "landing":"transit",
     "danger":"concrete", "escape":"narrow", "time_window":"spawn_uncertain",
     "code_preferred":false}
  ],
  "hazards": [{"kind":"coily", "visibility":"recent_memory",
               "proximity":"near", "motion":"pursuit"}]
}
```

Code computes all node arithmetic, risk, escape counts, route distance, timer
buckets, and candidate membership.  Jev does not count, decode RAM, infer the
board, or generate an action.  One request batches four independent typed
judgements:

* `corridor_a`: `Choice` over the supplied corridor keys plus `wait`;
* `corridor_b`: the same `Choice` with reversed option order, to reduce first
  option bias;
* `proposed_safe`: `Noul`, asking whether the proposed corridor has immediate
  supported lethal danger;
* `keep_escape`: `Noul`, asking whether the proposed landing retains an exit.

The bridge accepts Jev only if the two Choices agree, the selected-option
probabilities are both at least **0.60**, the selected record is still present,
its computed risk is within **0.35** of the deterministic fallback, its cost is
within **2.0**, and it is not closed.  The two Noul gates require at least 0.62
support for safety and 0.55 for an escape.  This makes Jev a reviewer of a
code-generated plan rather than an unsafe source of joystick commands.

The request is asynchronous, no more often than every eight standing ticks, with
one outstanding request and a 0.9-second timeout.  Its metadata includes tick,
node, life epoch, palette epoch, policy version, and the complete hazard/scheduler
signature.  Advice is collected only within eight ticks and executed only within
three ticks; any stale, malformed, arbitrary stand-in, disagreement, timeout, or
missing probability falls back to the freshly ranked corridor.  While hopping,
model freshness is irrelevant and the motor posts neutral.

### Astra changes from a style audit to a durable operating posture

Astra is never on the action clock and is capped at two asynchronous calls.  It
gets the current score/lives, inferred epoch, unfinished count, recent colour
transitions, semantic hazards, and the reason for review.  It may return only:

* `posture`: `upper | spine | frontier`, a tie bias in the code's unfinished-cube
  ordering; and
* `safety`: `certified | shelter`, a small preference for open exits after a loss
  or late uncertainty.

It cannot return a route, action, macro, FIRE, node list, or raw-RAM instruction.
The first call is an initial posture review; the second is requested after a life
loss or a late-board scoring stall, subject to a 300-tick cooldown.  Unsupported
or prose-only output leaves the current policy unchanged.  A policy change bumps
`policy_version` and invalidates pending Jev advice.  This is a different
allocation of labour: Astra chooses the mode of the deterministic corridor
planner, Jev recognizes local corridor safety, and code executes/searches.

## Fallback and terminal policy

The deterministic fallback hierarchy is:

1. neutral during palette flashing, falling, recovery, a hop transaction, or a
   post-landing gate;
2. a represented rolling hazard wait when the current cube is safer and the
   offered edge has concrete risk;
3. the lowest-risk supplied corridor, preferring an open exit and unfinished
   landing;
4. neutral when the adaptor supplies no safe edge.

Only named directional actions in the current `safe_moves` list can arm a hop.
`FIRE` is absent from every criterion and cannot be returned by the bridge.  The
loop runs until `game.done` or `ctx.done`, never to an internal step limit and
never stops merely because `lives` reaches zero.  Model calls are wrapped in
background jobs; late calls are harmless and are not joined on the game clock.

## Confirmation and refutation

The hypothesis is supported if the official evaluation preserves the parent's
first four inferred clears while showing fewer repeated losses in the late
empty-hazard windows, fewer `corridor_mismatch` events, and more scoring epochs
or longer survival.  A useful trace should show decisions changing because an
edge has a closed/narrow corridor, not merely because a model answered.  The
opening is a negative control: if the new planner loses before the first clear,
the corridor costs are too conservative or the handshake is wrong.

It is refuted if corridor certificates and the recovery handshake work but the
same post-fourth-epoch loss count/timing remains, or if the neutral gates cause
large progress stalls without reducing loss rate.  In that case the missing
mechanism is likely a validated flying-disc/Coily escape or a more exact later
level cycle, not another local hazard classifier.  A result near 0007 with zero
Jev overrides would also show that the code structure, rather than System-1,
carried the improvement; that is still a valid test of the stated controller
hypothesis and is logged explicitly.
