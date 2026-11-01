# Candidate 0009 — radar-guarded epoch controller

**Mode:** crossover. **Parents:** `candidates/0007/` and `candidates/0008/`.

## Hypothesis

The parents solved the first-board boundary but fail for a different reason after they
have learned to clear several boards: their motor can commit to a hop whose *intended*
edge is not the edge that the observation later shows, and their hazard memory treats a
missing rendered enemy as absence.  The late losses are therefore not fixed by another
coverage tie-break or by more Jev calls.

The evidence is concrete.  In `0007/trace_seed0.jsonl`, the controller was still at
four lives after the fourth inferred epoch at tick 2017.  At tick 2187 the observation
showed Q*bert near node 20 with only `LEFT` safe and Coily at estimated node 13.  At
tick 2219 the controller selected `DOWN` from node 14 while its short hazard memory
reported Coily at node 9, and a life was lost at tick 2222.  More diagnostically, after
the respawn the controller logged a pulse from node 14 whose **expected** destination
was node 9, but at tick 2260 it accepted node 19 as a landing (`unexpected` in the
observation sequence, although `0007` called it `hop_landed`).  It then continued as
if the route transaction had succeeded.  At tick 2312 it selected `UP` from node 4
with no tracked threats; the next life loss was at tick 2322.  The observations in
that interval had no visible hazard but `timers.purple_ball_state` was 36.  Thus
`safe_moves` and a four-tick memory were not a sufficient safety horizon.

`0008` supplies a useful complementary lesson: an explicit post-clear handoff and a
persistent target/detour ledger prevented the first-board flash from immediately
recreating a one-cube target list, but its independent ledger still produced a false
handoff (seed 3, tick 1367) and it stalled on a goal trap.  I retain its *barrier,
verified target hysteresis, and neutralized one-hop idea*, but do not make its ledger
the sole completion authority.  `0007`'s palette-epoch calibration and temporal wait
were the stronger route mechanism and are retained.

The test is therefore: combine palette epochs with an explicit handoff barrier and
persistent goals, then add a small time-expanded shadow for recently visible lethal
objects.  A hop is not counted as successful until the observed node equals the
intended destination.  When a high purple-ball timer remains while the sprite is
suppressed, the controller waits or chooses an escape-space route rather than
assuming the branch is empty.  This is a structural test of late-round survival,
not a seed route or a parameter-only change.

## What changed

### 1. Palette epochs plus a real handoff barrier

`PaletteEpochs` learns a destination from a local scored colour change and recomputes
unfinished cubes from the current palette.  A simultaneous whole-board recolouring
enters `flashing`; all movement is neutral during the flash.  It rebases only after
several quiet frames and either a return to the apex or a bounded timeout, then adds
an eight-tick neutral barrier before the first new hop.  An all-zero `needed_landings`
vector is never allowed to resurrect a one-cube target list during that interval.
If a board remains visibly complete without a flash, a bounded phase probe avoids an
infinite wait; this is labelled a probe, not a claimed round completion.  Sam or a
normal colour reversal is seen through the current palette and makes the affected
cube unfinished again.

### 2. Persistent targets and detours

`RoutePlanner` holds a target until it is no longer unfinished instead of selecting
the nearest current cube on every standing frame.  If the target is the current cube
but needs another landing, it first chooses a safe waypoint and returns.  Immediate
backtracking, recently repeated nodes, zero-escape corners, and radar-shadow landings
are penalized.  BFS still supplies only a one-edge suggestion; it is recomputed from
the current 21-node graph and current `safe_moves` on every decision.  No arcade
28-node route, FIRE, RAM write, or disc-exit guess is used.

### 3. Predictive hazard radar

`HazardRadar` records the last observed node and age of each lethal object.  A recent
purple/red ball shadows its current node and one or more downward child branches as
it ages; Coily shadows a small graph radius.  The shadow expires (18 ticks for a
ball, 12 for Coily), so it cannot become a permanent wall.  The router scores a
candidate by shadow occupancy, adjacency, and escape-space, while the adaptor's
current `safe_moves` remains the hard legality filter.  Missing RGB/palette evidence
is not treated as proof of disappearance.  A high, nonzero
`timers.purple_ball_state` with no visible ball creates a soft `occluded` posture and
slightly favours lateral/downward escape over blindly climbing.  If every route edge
has a materially higher shadow risk than the current cube, it waits for at most 12
ticks and then resumes the safest certified edge.

### 4. Verified hop transactions and debouncing

The motor sends a direction for at most two posts and then sends neutral input while
Q*bert is animated.  After the animation it checks the actual node against the
intended destination.  An unexpected landing is logged, the planner is resynchronized
to the observed node, and the source/destination pair is temporarily penalized; it is
never silently counted as the intended landing.  A two-tick neutral gate follows a
landing, life recovery, or palette rebase.  Falling, invisibility, and an absent node
produce NOOP.  This is the main crossover change beyond either parent: it protects
both the 0007 coverage engine and the 0008 handoff engine from sticky-action route
assumptions.

### 5. Advisers remain off the action clock

Astra is asynchronous and may make at most three small policy reviews.  Its bounded
JSON can select only `route: nearest|frontier` and `radar: long|balanced`; malformed,
late, or unsupported text leaves the deterministic defaults unchanged.  It cannot
name an action or change legality.  The default radar posture is `long`, and reviews
are requested initially, after a life loss, and after a later epoch subject to a
cooldown.  Jev is asynchronous and advisory; every tick has a deterministic action
or NOOP even when either model is late.

## System-1 interface and bridge

The state sent to Jev is a strict, derived JSON object.  It contains:

```json
{
  "game": "Atari 2600 Q*bert Atari port; 21-node six-row graph",
  "position": {
    "node": 14, "standing": true,
    "safe_moves": ["LEFT", "DOWN", "RIGHT"],
    "last_posted": "NOOP"
  },
  "progress": {
    "epoch": 4, "inferred_clears": 4,
    "unfinished_count": 12, "goal": 9, "waypoint": null
  },
  "candidates": [
    {"action":"LEFT", "landing":"unfinished", "danger":"clear",
     "escape_space":"multiple", "route":"short", "reverse":false,
     "code_preferred":true}
  ],
  "hazards": [
    {"kind":"purple_ball", "visibility":"recent_shadow",
     "proximity":"near", "motion":"descending_shadow"}
  ],
  "proposed_action":"LEFT",
  "observation_event":"cube"
}
```

Code supplies graph legality, candidate destinations, distances, danger buckets,
escape counts, palette arithmetic, freshness signatures, and the action bridge.  Jev
is not asked to count, decode colours, infer the 21-node graph, or generate a route.

One request is launched no more often than every seven standing ticks and is usually
pipelined during a hop.  It batches three independent judgments:

* `route_a`: `Choice` over the code-generated safe edges plus `NOOP`;
* `route_b`: the same semantic choice with criteria in reverse order, reducing
  first-option bias;
* `danger`: `Noul` asking whether the proposed action has immediate supported lethal
  collision or edge-fall risk.

The two Choices must agree, name a supplied option, and have a minimum probability of
0.58.  A danger probability of 0.80 can veto the proposed edge only in favour of a
currently certified alternative whose computed risk is within 0.4.  An accepted
alternative must be no more than 0.65 risk units above the deterministic best and
must not lose more than one escape option.  These gates deliberately make Jev a
recognizer/safety reviewer rather than a source of unsafe creativity.

Advice is tagged with tick, node, palette epoch, life epoch, policy version, and the
radar shadow signature.  It is discarded after seven ticks at collection and after
three ticks at execution, or whenever any tag changes.  Missing probabilities,
unknown values, disagreement, timeout, malformed stand-in output, or a stale answer
falls back to the code-selected edge.  While a hop is in progress, a loss of model
freshness changes nothing: the transaction motor posts NOOP.

The fallback hierarchy is:

1. neutral during flash/barrier, falling, a verified hop, or the bounded radar wait;
2. the lowest-risk code-generated candidate in current `safe_moves`;
3. another current safe edge with the best escape space;
4. NOOP when no certified edge exists.

FIRE can never be returned by the bridge.  This also makes arbitrary pre-flight model
answers harmless.

## Astra prompt and revision rule

Astra receives the reason for review, score/lives, inferred epoch/clears, unfinished
count, recent local colour evidence, and semantic radar facts.  It is explicitly told
that all joystick edges and arithmetic belong to code and Jev.  It returns only the
two policy fields described above.  A valid policy change increments a policy version
and invalidates pending Jev advice.  An empty or prose-only answer is an audit with
no change.  Astra is never synchronously awaited by the game loop; its 105-second
timeout is absorbed by the real-time fallback.

Evidence would support the hypothesis if the full evaluation shows fewer losses per
inferred epoch, fewer unexpected landings, and more post-fourth-epoch scoring than
0007, while stale ticks remain near zero.  In particular, the trace should show
`unexpected_landing` being resynchronized rather than turning into a repeated branch,
and `recent_shadow`/`occluded` radar facts before a formerly invisible-ball death.
A score near 0007 with the same loss times despite correct landing verification would
refute the motor part and point to a missing disc/Coily escape strategy.  A large
score drop with no unexpected landings and long periods of radar waiting would refute
the shadow thresholds as too conservative; a later generation should then shorten
only the shadow, not remove transaction verification.

The primary selection objective remains score and survival, not bonus chasing.  All
progress counts in logs are explicitly `inferred`; official score/reward and terminal
signals remain evaluator-owned.
