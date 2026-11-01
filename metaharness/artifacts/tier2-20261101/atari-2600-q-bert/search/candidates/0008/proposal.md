# Candidate 0008 — completion-aware closed-loop ledger

**Parents:** `candidates/0005/` (primary), with the guarded execution evidence from
`candidates/0004/`.

## Hypothesis

The plateau in 0005 is not primarily a Jev classification problem or a missing
Astra route.  It is a deterministic state-machine failure at the first pyramid
boundary: the controller clears all 21 cubes, then treats the cartridge's
post-clear animation as ordinary cube progress.

The evidence is unusually repeatable.  In the 0005 seed-0 trace, tick 220 has
`destination_cubes=21`, `incomplete_cubes=0`, `event_counts.cube=21`, and score
850.  From ticks 222–259 the same Q*bert remains at node 20 while the decoder
alternates between all-S and all-D cube arrays; `last_event` is repeatedly
`green_ball`, score rises by 100 at a time, and `completed_rounds` remains zero.
At tick 259 the score is 3950 and all 31 of those 100-point events have occurred.
The controller then resumes with a route that believes only one to three cubes
are unfinished (ticks 267, 275, and 417), oscillates around the top, and never
finishes another board.  0004 shows the same boundary in a different controller:
it reaches all D at tick 443, then stays at node 20 through a long sequence of
alternating all-S/all-D observations and eventually loses its remaining lives.
This makes the failure a shared transition/ledger issue rather than a random
Jev answer.

The proposed test is that a controller which explicitly waits through that
flash, resets its own board ledger after the handoff, and maintains target
hysteresis will convert the repeatable first-board score into additional round
progress.  It also tests whether the 0005 score was being supplied by the
route executor rather than Astra: this candidate deliberately keeps slow Astra
off the action path, while preserving a compact Jev safety interface.

## What changed

### 1. A completion handoff state machine

The controller has `play` and `transition` modes.  Seeing all 21 destination
cubes (or an all-zero `needed_landings` vector) enters `transition` immediately.
During the bounded transition window it posts only `NOOP`; an alternating
all-D/all-S palette view with repeated `green_ball` events cannot recreate a
one-cube target list.  It exits only when Q*bert is visibly moving, a real cube
event occurs, the node leaves the completion node, or a 55-tick safety bound is
reached.  On exit it starts a fresh board ledger.  A decoded level/round counter
change is also accepted as an authoritative handoff.

This is intentionally different from merely selecting a safer action at node
20: 0004 already used neutral actions for most of the stuck interval, while
0005 used a direction.  The new invariant is that neither stale palette phase
nor a stale strategic target can survive the handoff.  A bounded fallback keeps
the controller from waiting forever if the decoder suppresses the first active
frame.

### 2. An independent landing ledger with audit-only observation merging

The adaptor's `needed_landings` remains valuable, but positive observations are
merged rather than allowed to erase code-owned obligations during a flash.  A
new cube event updates the ledger using the level-specific finite cycle:

- level 1: S→D;
- level 2: S→I→D;
- level 3: S→D→S;
- level 4: S→I→D→I;
- level 5: S→I→D→S.

After a completed-board handoff the ledger starts every node at the first
useful landing count for the current level (`1,2,1,2,2`).  Sam/life-loss
observations can add positive obligations back.  A short per-node ignore window
prevents the same landing's transient decoder frame from undoing the update.
This is a targeted bookkeeping change, not seed-specific data.

### 3. Target hysteresis and detours

The old route selection could repeatedly choose the current cube after the
board transition, producing the observed UP/DOWN alternation.  0008 retains a
goal until the goal's landing is observed.  It excludes the just-landed cube
from immediate reselection; if that cube still needs another landing (levels
2–5), it chooses a certified neighboring waypoint and returns.  If it is the
only unfinished cube, it explicitly leaves and comes back.  BFS recomputes a
single first edge from the current 21-node graph at every standing decision,
with a fixed upper-region tie-breaker and no blind multi-hop macro.

### 4. One-hop input pulses

A direction is posted only at a standing node and is followed by neutral input
while the hop animation is active.  If sticky-action realization leaves Q*bert
standing, the same certified pulse may be retried twice; it is never held past
that retry limit.  A one-tick neutral release follows each hop.  This tests a
second concrete control invariant without depending on a model's ability to
remember when an edge became unsafe.

### 5. Jev is advisory and fresh, not the route planner

There is at most one asynchronous Jev request per several standing decisions.
The compact JSON state contains:

```text
mode
position.node, position.standing, position.safe_moves, position.legal_moves,
position.proposed
progress.level, progress.round, progress.lives,
progress.unfinished_nodes, progress.goal, progress.next_waypoint
hazards[kind, lethal, risk, distance, node]
recent.event, recent.last_action, recent.qbert_motion
```

Raw RAM, coordinates, score arithmetic, and the complete observation are
omitted.  Code supplies the candidate actions and the graph facts.

Each request batches three independent typed questions:

* `move_a`: a `Choice` over the current certified safe edges plus `NOOP`;
* `move_b`: the same immediate choice with the criteria in reverse order, to
  reduce first-option bias;
* `danger`: a `Noul` asking whether the proposed edge has an immediate lethal
  collision or edge-fall risk.

A Jev answer is usable only if both choices agree, the selected name is still
in the latest `safe_moves` (or is `NOOP`), the node and ledger generation are
unchanged, the answer is at most four ticks old, and the selected probability
is at least 0.48.  A danger probability of at least 0.72 vetoes a non-forced
proposed edge; code then chooses another current safe edge.  A low-confidence,
missing, malformed, timed-out, or late answer is ignored.  Jev can never press
FIRE, bypass transition mode, or manufacture an edge.

The fallback hierarchy is: current code-computed route edge; another edge in
current `safe_moves`; `NOOP` while moving, uncertain, transitioning, or with no
certified edge.  Every call is spawned with a 0.36-second timeout so the game
clock never waits for Jev.  Model exceptions are contained in the adviser job;
garbage stand-in answers therefore reduce to the deterministic pulse policy.

## What would confirm or refute the hypothesis

Confirmation would be a score above the 0005 mean of 4025, especially if the
trace shows a second `destination_cubes=21`/completion handoff and a nonzero
`completed_rounds` or a second full 21-cube landing sweep.  Reduced life losses
at the handoff, no long UP/DOWN oscillation on the top three nodes, and a
similar result when Jev answers are rejected would further support the
code-owned diagnosis.  The important event-level check is that the controller
resets its ledger after the alternating all-S/all-D interval instead of
retaining the transient one-to-three-cube view.

The hypothesis is refuted if 0008 still reaches the same first clear and then
stalls despite the explicit transition mode, if it loses lives during the
neutral handoff before a new board is playable, or if it completes no more
coverage than 0005 while its own ledger reports the correct full target set.
A score increase with identical transition behavior but only different Jev
choices would instead point to the local safety adviser as the missing piece.

## Astra decision

No Astra call is made in this candidate.  The archive shows that Astra cannot
be on the action clock (23.5–69 seconds per call), 0004's seven calls all
failed or timed out, and 0005's six calls preceded only 0.035% of actions.
Removing it here is an intentional control: the test isolates whether the
shared deterministic completion/ledger bug explains the plateau.  If this
controller reaches later boards, a later generation can add an asynchronous
Astra plan whose only allowed output is a target-order preference, without
putting it in the hop or transition path.

## Selection criteria and runtime accounting

The primary expected gain is more completed pyramids and preserved lives, not
optional Green Ball/Sam points.  Stale ticks should remain near zero because a
valid action or neutral fallback is posted every tick and model calls are
backgrounded.  Jev calls are deliberately sparse and fresh; System-2 calls
are zero.  The controller never uses FIRE, the 28-node arcade map, a save
state, raw RAM writes, or seed-specific routes.
