# Candidate 0009 — reserve an escape pill and remember failed directed edges

## Parents and diagnosis

Parents: `candidates/0008/` and `candidates/0007/`.

`0008` supplies the current score engine: an observation-derived pill graph, continuous
cardinal actions, explicit power-pill checkpoint routing, and two reordered Jev choices.
It reached **1,608.3** mean with zero stale ticks and consumed roughly 126.5 visible
collectibles.  Its traces also identify the remaining failure: checkpoint coverage did
not buy a life.  In seed 2 all four checkpoints were consumed by the first route
(`power_checkpoint` at `[8,147]`, `[8,15]`, `[148,14]`, and `[148,146]`), then the
first life ended at frame 2008 / `[81,99]` while holding `right`, the second at frame
2804 / `[61,109]` while holding `down`, and the last at frame 3225.  Seed 3 has the
same first two death locations (frames 1936 and 2668).  Thus 0008 spends all of its
explicit safety resources before the repeated exposed dot sweep reaches its lethal
edges; life rotation cannot help once no pill remains.

`0007` supplies the useful complementary mechanism: compare actual motion with the
posted direction, and make a directed edge seen at a life loss temporarily taboo.
However, its progress fallback was too eager.  In the worst seed it emitted
`progress_replan` on almost every tick at `[72,103]` and alternated `left/right` for
do¬zens of ticks, and similarly alternated `up/down` at `[124,151]`.  The lesson is to
retain the evidence memory without letting a stationary observation become a
per-tick oscillating controller.

## Hypothesis

The score ceiling is caused by **resource exhaustion plus repeated local edges**, not
by a lack of model calls.  A controller that (1) keeps one power pill in reserve for a
later life/emergency, (2) records the pre-death directed edge rather than the
post-death sprite position, and (3) commits to a candidate recovery direction for a
short bounded hold after no progress should preserve 0008's coverage while avoiding
at least one of its canonical death routes.  The Jev/Astra interface stays subordinate
to this deterministic state machine.

## What changed

Starting from 0008, this harness keeps its continuous graph executor and checkpoint
normalization, but adds:

* **One-pill reserve.** On the first life at most three ordinary checkpoint goals are
  visited.  The fourth remains visible for a later life or for the immediate hazard
  escape routine.  After a death the quota is reset for the new life and the remaining
  checkpoint is selected before returning to dot clearing.  A last-life mode is always
  survival; it never invents a new checkpoint if none is reachable.
* **Pre-death edge memory.** Before every `ctx.act`, code stores the current graph node,
  intended next node, action, and life/maze version.  When the next observation reports
  a death, that predecessor edge/action is penalized and the route search avoids it on
  the next attempt.  This avoids 0007's ambiguity from recording the respawn/death
  animation coordinate as the bad edge.  The taboo is soft: if every route is blocked,
  the least-bad legal branch is still used.
* **Bounded progress recovery.** A stationary player after the opening invalidates the
  route only once per recovery attempt.  It marks the attempted edge, chooses an
  untried legal graph neighbor (or the observed-direction reverse as a last resort),
  and holds that cardinal action for several ticks before trying another.  It cannot
  alternate on every tick, and Jev overrides are ignored during the hold.  Movement
  clears the attempt.  This retains 0007's feedback without its `[72,103]` and
  `[124,151]` oscillation.
* **Conservative checkpoint accounting.** Any power-pill disappearance near the player
  (including a pill reached by emergency escape) is marked consumed; malformed or
  stand-in model output cannot make the controller believe a checkpoint was reached.
  The emergency graph guard remains authoritative over the quota.

Astra still makes at most two asynchronous maze-level policy calls and can only select
an explicit `MODE=survival|balanced|score`.  It never blocks a tick.  The first and
last-life code rules override a score-oriented answer.  Jev still chooses only among
already computed legal cardinal branches at junctions/hazards; it cannot veto a
checkpoint route or invent a diagonal.

## System-1 interface, thresholds, and fallback

Every Jev request is strict JSON and contains only derived facts:

```text
observation: {frame, tick, score, lives, deaths, visible_pills,
              visible_power_pills, pill_active, maze_clears, player_xy,
              player_direction, stalled_ticks}
plan: {risk_mode, goal_kind, goal_xy, life_phase, checkpoint_count,
       checkpoint_quota, fallback_action, legal_actions, graph_degree,
       policy_version}
threats: {summary, nearest_distance_band, closing_same_lane_count,
          immediate}
options: {action: {continues_goal_route, corridor_risk,
                   reaches_checkpoint, branch_degree, taboo}}
escape_option: action-or-none
```

One request batches:

* `move_a`: `Choice` over the legal cardinal actions;
* `move_b`: the identical `Choice` with criteria in reverse order, countering the
  documented first-option bias;
* `escape`: `Noul` asking only whether the already computed power-pill escape is safer.

A move is accepted only when both choices agree, the selected value is a current legal
cardinal action, both selected-option probabilities are at least **0.40** in an
ordinary branch or **0.55** for a hazard, the request is no more than 18 native frames
old, deaths/policy version/node are compatible, and no recovery hold is active.  An
escape Noul may select only the code-computed escape action and needs probability
**0.60**; immediate hazard code remains authoritative.  Missing probabilities,
malformed stand-in values, disagreement, timeout, stale futures, or a model exception
are ignored.

The fallback posts a valid cardinal action every tick: current route direction, an
immediate graph/power-pill escape, or a bounded recovery direction; before the graph is
usable it holds the observed cardinal direction and then `left`.  There is never an
intentional `noop`, synchronous Jev wait, or Astra wait.  All futures are polled and
all work naturally stops when `ctx.done`.

## Evidence that would confirm or refute it

Confirmation would be zero stale ticks plus a score at least near or above 0008's
1,608 mean, with a trace showing a visible power pill surviving the first life,
`predeath_edge`/`edge_taboo` followed by a different route, fewer than three deaths,
or a maze clear.  A recovery trace should show one chosen direction held rather than
an alternating action pair.

Refutation is the same three canonical deaths after all four pills are effectively
spent, no route change after a recorded pre-death edge, or a large score collapse from
prematurely abandoning dot coverage.  If the reserve helps but directed-edge memory
never changes a route, the next revision should focus on graph-to-sprite alignment;
if bounded recovery still stalls, the observed graph's turn timing—not Jev
uncertainty—is the likely defect.

Pre-flight cause: the recovery-method insertion accidentally removed the `_direction_order` definition, so route search raised `AttributeError` before the first action; it has been restored.
