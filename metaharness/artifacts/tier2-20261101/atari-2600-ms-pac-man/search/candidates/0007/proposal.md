# Candidate 0007 — graph route with progress memory and conservative edge recovery

## Parents and diagnosis

Parents: `candidates/0004/` and `candidates/0003/`.

`0004` supplies the winning mechanism: the game must be driven continuously by
code. Its observation-derived pill graph posts an action on every tick, and Jev
only arbitrates at branches or hazards; this reached 620–1,070 points with 0%
stale ticks. `0003` supplies a useful robustness idea that `0004` did not carry
far enough: compare successive player positions and treat a repeated
no-progress action as evidence against that action, rather than trusting a
model answer or a nominal route forever.

The concrete remaining failure is visible in `0004`'s traces. Every training
episode loses all three lives. In seeds 0, 3, and 5 the first loss occurs at
roughly the same opening-maze event: after reaching 25 dots and `[81,99]`, the
controller continues `right` while three ghosts are clustered around
`[81,80–81]`; the death observations are frames 864–944 and scores 250–260.
The second loss similarly recurs around `[61,109]` while the route continues
through the lower branch. On the last life, seed 0 alternates `left/right`
near the same danger (accepted Jev overrides at frames 1816–1804 in the trace)
and still dies at `[42,99]`; seed 3 finally dies at `[29,40]` after a long
uninterrupted `up` run. The graph keeps collecting dots, but its route is
stateless with respect to a life-ending directed edge, and its ghost guard only
calls a situation immediate when velocity says a ghost is closing. A ghost
near the next graph node can therefore be treated as harmless when it is about
to turn.

## Hypothesis

The score ceiling is now primarily caused by repeating locally hazardous graph
edges, not by insufficient Jev/Astra reasoning. If the continuous graph
controller (a) uses graph proximity as a hazard even when the decoded velocity
is sideways or stale, (b) records the directed route edge at a life loss and
avoids it on the next attempt, and (c) imports `0003`'s observed-progress
fallback to break a wall/turn stall, it should preserve the 0%-stale score
engine while converting repeat deaths into alternate corridors or power-pill
escapes.

This is one structural change: a graph controller with a small online hazard
memory. It does not import an arcade map, assume ghost personalities, or ask a
slow model to steer.

## What changed

Starting from `0004`, the harness adds:

* **Progress feedback.** It stores the last player position and direction.
  During a non-opening run, a cardinal action that leaves the player stationary
  for two observed ticks is marked blocked for the current node. The current
  route is invalidated and a different graph neighbor is selected. A life
  change resets the transient stall counter but does not erase learned bad
  edges.
* **Life-local edge memory.** At a death increment, code records the nearest
  graph node and the route's next directed edge (plus the action if no edge is
  available). For the next few route selections that edge is taboo, with a
  longer penalty on the last life. A taboo is a route cost/avoidance signal,
  not a claimed wall: if it is the only reachable option the fallback can still
  use it rather than stop. The memory is cleared on a maze change.
* **Velocity-independent local danger.** Ghosts are still filtered for the
  decoder's opening sentinel, and their RGB is never treated as frightened.
  In addition to the existing short projected-lane test, a ghost within a small
  graph-distance band of the candidate node is hazardous even if its reported
  `dx/dy` is zero, sideways, or about to turn. Emergency branch scoring prefers
  the neighbor with the greatest graph separation and a reachable power-pill
  branch.
* **Less oscillatory Jev use.** The two reversed-order Choice questions remain
  at graph junctions/hazards. A Jev override is now rejected if it points at a
  taboo edge, a newly blocked action, or a stale node; its answer never defeats
  the deterministic last-life guard. Jev still only chooses among code-computed
  cardinal options.
* **Astra remains asynchronous and strategic.** One background Astra call
  chooses `survival`, `balanced`, or `score`; it cannot delay a tick. The last
  life and a newly learned death edge force survival locally. Malformed Astra
  text and arbitrary stand-in answers leave deterministic defaults intact.

## Jev interface and bridge

The state is compact strict JSON and contains derived facts only:

```text
{
  observation: {frame, tick, score, lives, deaths, visible_pills, maze_clears,
                player_xy, player_direction, motion, stalled_ticks},
  plan: {risk_mode, fallback_action, legal_actions, node, graph_degree,
         policy_version},
  threats: {summary, nearest_distance_band, graph_near_count,
            closing_same_lane_count, immediate},
  options: {action: {continues_route, corridor_risk, graph_escape,
                     power_escape, taboo}},
  escape_option: action-or-none
}
```

Code computes coordinates, graph paths, graph distances, ghost proximity,
progress, taboo edges, and all numeric comparisons. `Choice(move_a)` and the
same `Choice(move_b)` with reversed option order ask which supplied legal
cardinal branch is safer while preserving dots; an independent `Noul(escape)`
only decides whether the already-computed power-pill escape should replace the
fallback. Jev is not asked to discover walls, count pills, calculate a path,
or infer frightened state from RGB.

A returned move is accepted only if both Choices agree, the answer is one of
the current legal options, the request is at most 24 native frames old, the
death count and graph node remain compatible, and both selected-option
probabilities are at least 0.34 in a normal branch or 0.50 during a hazard.
The escape Noul is used only for an already computed escape and requires a
probability of at least 0.50. Low confidence, missing probabilities,
malformed stand-in values, disagreement, timeout, or stale futures are ignored.
The action fallback is always the graph route/ghost guard, never parsed model
text.

## Fallback and scheduling

The loop posts `ctx.act` and waits for exactly the next tick on every iteration,
including while Astra or Jev futures are pending. Thus stale ticks remain zero.
At a normal corridor it holds the route direction. At a junction, near a
measured ghost, on a taboo edge, or after no progress it re-plans. Immediate
hazard uses a power-pill route when its first edge is not itself dangerous;
otherwise it chooses the safest computed branch, preferring continuity only
when risks are comparable. If graph construction is unavailable or an
observation is malformed, it holds the last valid cardinal direction (then
`left`), and never crashes on arbitrary model output.

## Evidence for confirmation or refutation

Confirmation would be a score above 0004's 778.3 mean with fewer repeated
life-loss locations, a lower number of locally repeated directed edges after a
demise, and ideally a remaining life or maze clear. Traces should show
`edge_taboo`/`progress_replan` events followed by alternate graph branches,
while actions remain fresh every tick.

Refutation would be the same three canonical deaths despite taboo edges and
velocity-independent graph hazards, or a substantial fall toward the idle
baseline caused by over-conservative escape choices. If the graph edge memory
is never exercised, the hypothesis was wrong about repeated edges; if it is
exercised but ghosts still reach the player at graph distance 1–2, the next
revision should improve corridor-time projection rather than add model calls.
