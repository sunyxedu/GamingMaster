# Candidate 0022 — causal collision memory and a no-revisit escape brake

## Parent and diagnosis

Parent: `candidates/0019/`, with `0016` as the best comparator. The parent is the
route-goal/corridor executor, not a synchronous model controller. I read its worst
seed-1 trace and the Generation-6 lessons before changing it.

The parent reaches a lot of dots, but its claimed collision memory is dead code:
`_remember_control_edge` is defined but never called, `last_control_node` is never
populated, and `edge_failures` is only read by the route search and never incremented.
Thus a life loss teaches the next life nothing about the edge/action that was actually
being attempted. The evidence is especially clear in `trace_seed1.jsonl`: it loses at
frames approximately 1932 and 5204, then on the final life repeatedly accepts route
leases whose computed margin is negative (for example the final sequence switches
between dot/retreat goals with margin `15.31` down to `-15.54`) and ends at frame 8769
with the player `[11,99]` and a ghost `[13,99]`. The final ghost location is not itself
a collision coordinate, but the last pre-loss control context is available before the
respawn and can be used causally.

The lessons explicitly identify this mechanism as untested: the earlier candidate's
pre-death-edge and recovery helpers had no call sites, while `0019` still leaves its
edge-failure counter unused. This is different from merely reserving a power pill or
adding another projected-risk threshold.

## Hypothesis

**A life loss is useful local supervision only if it is attributed to a physically
resolved approach shortly before the loss and then made directional, epoch-local
evidence.** The route council currently recomputes from the same inferred graph after a
respawn and can send the new life through the same exposed approach. A small, decaying
causal edge memory should remove repeated fatal approaches without freezing the whole
route: penalize only the directed edge and its immediate action at the resolved node,
keep the warning for this maze epoch, and reconsider it after expiry or when it is the
only exit. If the graph executor is otherwise productive, this should reduce repeated
deaths while preserving dot coverage and may allow a first maze/refill.

## What changed

This candidate keeps the parent's continuous cardinal executor, route-goal leases,
resource epoch handling, inferred graph, and asynchronous model bridge, but adds a
structural **pre-loss causal ledger**:

* every posted action stores its observation position, inferred graph node, action, frame,
  route risk, and a pending resolution in a short ring buffer;
* the following observation must show plausible cardinal displacement before that record
  becomes a resolved edge. Stationary commands, teleports, respawns, and a life-change
  observation are not motion evidence;
* a death transition consumes the most recent resolved motion from the old life, rather
  than treating the respawn coordinate as the collision point. It is learned only when
  the resolved sample had a nearby ghost or a high computed hazard class;
* the implicated directed edge/action is recorded with a bounded finite penalty for the
  current epoch; nearby reverse edges are not blacklisted automatically;
* weighted search, the parent BFS route call site, packet construction, and the final
  cached-goal action boundary all apply the warning. A warned first edge is rejected
  when an equally or more safe live alternative exists, but remains available as the
  only exit or when every alternative is worse;
* after a warning changes a route, the executor logs a bounded `causal_detour` and
  returns to the ordinary dot/resource mission. This is a live branch choice, not a
  fixed map route or a blind post-respawn turn;
* a repeated loss at a distinct edge is recorded separately, so the controller does
  not infer a global maze rule from one collision. Epoch renewal clears old geometry
  and old causal evidence.

The code still owns legality, arithmetic, timing, and the final action. Jev can only
choose among current route goals after the causal brake has removed dominated packets;
Astra can suggest a rubric but cannot command movement.

## Confirmation and refutation

Confirmation would be trace evidence of resolved `collision_edge_learned`, a later
`causal_detour`, fewer repeated pre-loss edge signatures, and a higher fraction of
runs surviving the parent’s first/second death frame with comparable dot counts. The
strong result is a detected dot repopulation/resource epoch or fewer than three deaths;
score improvement without those trace events is not sufficient attribution.

Refutation is either (a) the ledger never receives a correctly attributed pre-loss
record, (b) it removes productive edges and lowers coverage without changing the death
pattern, or (c) the same life still repeatedly dies after the remembered edge has been
avoided. In the latter case the approximation of the collision horizon—not resource
scheduling—is the next target. A post-respawn coordinate must not be counted as causal
evidence.

## Tier-2 System-1 bridge

### State schema

Each Jev request is strict JSON and contains:

* `rules`: score/progress objective, cardinal-only execution, inferred geometry, and
  the fact that a power pill is an escape resource rather than proof of frightened
  immunity;
* `plan`: epoch, life phase, current route-goal kind, rubric, and the current goal
  semantics;
* `options`: at most four already executable route goals. Each has purpose, semantic
  risk (`clear`, `watch`, `interception_possible`, `interception_likely`), exit branch,
  dot yield, continuity, reversal, and a `collision_memory` label saying whether this
  route contains a temporary suspected approach;
* `recent`: score progress, current hazard class, number of learned causal edges,
  whether collision memory is available, and whether a causal detour has been applied.

Raw RAM, wall discovery, numeric path calculation, and ghost-colour frightened-state
inference are excluded.

### Questions, thresholds, freshness

One asynchronous request batches two reordered `Choice`s over the same route-goal
shortlist plus a `Noul` asking whether the supplied evidence indicates a stalled route.
The two choices must agree; the chosen option must have probability at least `0.70`.
The Noul is only a diagnostic trigger for the next Astra audit and never posts an
action. Answers are tagged with frame, epoch, death count, rubric version, option
signatures, and the request clock. They are dropped after 48 native frames, a life or
epoch change, a changed rubric, missing goal, newly worse risk, or a causal warning on
the first edge. Unknown/malformed answers, disagreement, low confidence, timeouts,
and budget failures leave the deterministic fallback in control.

Requests are interval/event-gated (junction, hazard, causal route replacement, or at
least 24 native frames since the last request), one in flight, with a 2.5-second
timeout. This is longer than the normal roughly 0.2-second Jev response but short
enough to avoid accumulating stale decisions. Astra is at most three background
interface audits with a 100-second timeout. `ModelUnavailable` is allowed to
propagate as required by the contract; ordinary model failures are safe fallbacks.

### Fallback policy

Every loop iteration observes, updates bookkeeping, polls futures, recomputes the
causal-risk route, and posts a valid cardinal action before waiting for the next tick.
A late answer never pauses the game. Code rejects a remembered fatal first edge,
chooses an aligned branch with the greatest graph safety margin, and holds a selected
escape for a bounded interval to avoid left/right oscillation. If no route is known it
continues the last valid cardinal direction. All actions are revalidated against the
latest observation before posting.
