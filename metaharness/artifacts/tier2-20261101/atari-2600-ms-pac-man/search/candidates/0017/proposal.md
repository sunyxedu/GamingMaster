# Candidate 0017 — resource epochs plus a code-owned emergency lock

## Parents and diagnosis

Parents: `candidates/0012/` (the high-scoring graph/checkpoint/reserve controller) and
`candidates/0007/` (continuous graph execution, progress feedback, and life-local edge
memory).

The useful mechanism in both parents is code-owned continuous movement.  `0012` reached
2,141.7 mean points, 0% stale ticks, and 163.8 visible collectibles per episode by
routing through the observed pill graph and making the four power-pill coordinates
explicit goals.  `0007` supplied the right robustness direction: compare actual player
positions, remember a directed edge around a life loss, and let a stall trigger a route
change instead of asking a slow model to recover.

The failure is concrete in the `0012` worst traces.  In seed 4 the harness repeatedly
recorded hazard-2 Jev moves selecting `right` (for example around ticks 348, 356, 359,
362, 365, 368, 372, and 375).  The player was at `[51,75]` at frame 1400 while ghosts
were already at `[53,79]` and closing; the episode then reached the first death at frame
1632.  The same pattern appears in seed 2: near the first death, the player travelled
through `[121,99]`, `[121,109]`, `[121,119]`, and `[121,144]` while ghosts occupied the
same central/lower lanes, and the harness continued to accept a Jev `right` override.
The parent’s fallback had computed a hazard, but `_choose_action` allowed the late model
answer to replace it.  Thus the slow recogniser was not merely stale; it actively
repeated the route that the deterministic guard was trying to escape.

A second independent issue is visible in the successful seed-2 tail: the first dot set
was exhausted and a new strawberry-level dot set appeared, but the adaptor still
reported `maze_number=1` and `mazes_cleared=0`.  `0012` therefore retained consumed
power-target flags and did not reliably start a fresh resource/map epoch.  Finally,
`0007`’s death memory used the respawn observation as the location of the loss.  The
archive explicitly warns that this observation is later than contact, so 0017 records
recent pre-death edges and their nearest observed ghost distance instead.

## Hypothesis

The graph route is already a good score engine; the primary remaining failure is an
authority inversion at lethal moments.  If the deterministic controller owns an
emergency lock—Jev can arbitrate ordinary branches, but cannot override a current or
projected collision—then the repeated hazard-2 direction should stop causing all three
lives to be lost.  Combining that lock with an actual dot-repopulation resource epoch,
pre-death edge memory, and bounded no-progress recovery should preserve 0012’s coverage
while making later lives and later mazes usable.

This is a structural change rather than a timeout or probability tweak: model output is
removed from the irreversible emergency path, and the controller’s state machine has a
new epoch boundary that does not depend on the unreliable maze counter.

## What changed

* Retained an observation-derived sparse corridor graph, cardinal action queue, two
  routine power-pill goals, and a protected reserve route.  Actions are posted every
  tick while Jev and Astra run in background futures, so the realtime fallback remains
  fresh.
* Added a **hazard lock**.  A physical/short-horizon ghost collision, or a graph-near
  ghost with a close physical projection, invokes `_escape_action`.  `_choose_action`
  clears any Jev override and returns that fallback for hazard level 2.  `_poll_jev`
  also rejects a response judged from an emergency state, even if it arrives within the
  freshness window.  Jev can still choose a safe normal junction.
* Added a **resource epoch detector**.  A transition from at most 12 visible ordinary
  dots to at least 40 resets the dot graph, power targets, route, and edge penalties,
  even if `mazes_cleared` and `maze_number` remain unchanged.  This is designed for the
  observed empty-grid-to-strawberry transition.
* Added **pre-death edge memory**.  The harness keeps a short history of directed graph
  edges and the nearest physical ghost distance.  On a life increment it penalizes the
  closest recent edge, not the respawn coordinate, and searches around it when another
  route exists.  If all alternatives disappear it remains a soft taboo: the fallback
  can still use the edge rather than stopping the game.
* Added a bounded progress recovery.  Two stationary observations mark the current
  direction at the current node as blocked; one alternate branch is held for a short
  recovery window rather than alternating `up/down` or `left/right` every tick.
* Normal BFS excludes unspent power-pill nodes as accidental intermediate goals.  A
  power pill is selected routinely only twice, then only for a physical emergency, a
  forced post-death escape, or survival on the last life.  A consumed pill and a new
  epoch are tracked separately, so resources renew after repopulation.

## Jev interface and bridge

Every request is strict JSON and contains only derived facts:

```text
{
  observation: {frame, score, lives, deaths, player_xy, visible_dots,
                visible_power_pills},
  plan: {risk_mode, goal, fallback, legal_actions, node, policy_version},
  threat: {level, label, nearest_bucket, physical_immediate,
           closing_lanes},
  options: {action: {keeps_code_route, risk, blocked_by_memory, branch_degree}}
}
```

Code computes graph legality, path search, node/ghost distances, collision projections,
resource status, and action validity.  Jev receives two reversed-order `Choice` questions
(`move_a` and `move_b`) to reduce documented first-option bias, plus an independent
`Noul` asking whether an already computed escape is preferable.  The choice is accepted
only when both choices agree, the answer is in the still-legal option set, the request is
within 24 native frames, the epoch/death/policy version is unchanged, and both selected
probabilities exceed 0.40 in a normal branch or 0.55 near a nonlethal threat.  The escape
answer can only select an existing code-computed escape; it cannot invent a direction.

For a level-2 or physically immediate threat, Jev is deliberately ignored regardless of
confidence.  A late, malformed, missing, disagreeing, or low-probability answer is
silently discarded.  The fallback is always a cardinal route action, a computed escape
branch, a held progress-recovery action, or the last valid cardinal direction.  Arbitrary
stand-in model values therefore cannot crash preflight or put a non-cardinal command on
the wire.

## Astra role and scheduling

One Astra call is launched asynchronously at opening.  It sees a compact summary of the
current resource epoch, score, lives/deaths, visible dots/pills, and player direction,
and is asked for only `MODE=survival`, `MODE=balanced`, or `MODE=score`.  It cannot block
the tick loop and malformed text is ignored.  Code forces survival on the last life;
Astra never overrides the emergency lock or performs per-tick steering.  With the
measured long System-2 latency, the harness treats an absent or cancelled Astra answer
as normal rather than waiting for it.

## Evidence that would confirm or refute the hypothesis

Confirmation would be hazard-2 periods in traces showing `escape_commit` or a safe branch
without intervening `jev_branch`, fewer immediate post-hazard deaths, and a rise in
surviving lives, maze repopulation events, or maze clears while stale ticks remain zero.
A fresh epoch should show new `map_epoch` and `power_goal` events after the first grid is
exhausted.

Refutation would be the same close-ghost deaths despite the emergency lock, or a large
score collapse from choosing escapes too early.  If `death_edge_memory` is repeatedly
recorded but the same physical collision persists, the next revision should improve
corridor-time projection rather than add more model calls.  If the epoch detector never
fires on a full-grid return, the adaptor’s progress signal needs a different verified
transition detector.
