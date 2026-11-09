# Candidate 0008 — committed macro exploration

## Parent and hypothesis

Parent: `candidates/0004/` (with the execution evidence from `0003` and `0005`).

The failure is not primarily that Jev lacks a Doom rule.  The motor interface is
being asked to solve a long-horizon coverage problem one action at a time, and
its late or option-order-biased answer becomes an absorbing motion.  The traces
make that concrete:

* `0003` seed 1002 spent **15,102** ticks on `forward` and **14,204** on
  `forward_left`; at tic 1900--3000 it remained around x=112, y=1230 with
  eight items and no kills or exit.
* In `0003` seed 1004, the action counts were **15,703 `forward`** and
  **15,531 `forward_left`**, while observations stayed around x=688, y=-200
  and the episode timed out at full health.  The anti-stall watchdog only
  changed which local maneuver was followed by another long forward hold.
* The more elaborate target servo in `0004` did the opposite failure: seed
  1007 alternated **83 `turn_left` / 82 `turn_right`** actions, fell to health
  1, and died before acquiring a weapon.  `0005` seed 1008 similarly spent
  **2,246 left and 2,075 right turns** and ended at health 11 after making
  partial item/kill progress.

Thus freshness (and even a large observed-resource memory) is not enough.  The
controller needs a persistent, map-agnostic search primitive that continues to
translate after a wall, and USE must be exercised during that search.  My
hypothesis is that a deterministic sweep/escape automaton, with model answers
used only as semantic boundary hints, will cover more reachable corridors and
activate more doors/exits than raw Jev action selection or a point-chasing
servo, while preserving the idle controller's survival when no threat is
visible.

This is deliberately not a route memorized from the training seeds.  It uses
only the current visible labels, measured position/angle, observed displacement,
counts, and the action specification.  The 64-unit cells are a visited-position
memory, not a supplied map.

## What changed

1. **Code owns the motor clock.**  Every fresh observation gets a legal action.
   When a model is pending, the last deterministic controller still posts a
   forward, combat, USE, turn, or escape action.  There is no synchronous Jev
   action-to-tick dependency.
2. **Committed sweep search.**  In an empty area the controller runs forward for
   a bounded macro, turns approximately 90 degrees, and repeats.  A measured
   collision causes a bounded `backward -> turn -> forward` escape.  A long
   absence of new observed cells changes the turn hand and performs the same
   escape rather than polling a model in place.  Repeated goals expire after a
   finite lack of odometric progress.
3. **Interaction is part of locomotion.**  Visible exit/switch/door/teleporter
   labels outrank pickups and are approached with bearing code and a one-tic
   USE pulse.  In addition, a periodic USE pulse and one bounded USE probe per
   blocked face test unseen doors/lifts without holding USE indefinitely.
4. **Combat is a safety override, not a route objective.**  Code prioritizes
   close/hitscan/major/body-blocking labels, aims from world bearing, strafes
   while firing, retreats at contact, avoids uncertified splash weapons, and
   switches only to an owned weapon with observed ammo.  Distant routine
   monsters do not steal the exit search.
5. **Astra authors only a bounded macro policy.**  One background Astra review
   can change `hand=left|right`, `sweep=short|medium|long`, and
   `combat=clear_blockers|avoid_optional`.  Garbage, lateness, or a policy with
   no legal field is ignored.  It never outputs a route or button, so its
   roughly-minute latency cannot freeze the game.
6. **Jev judges local semantic evidence.**  Jev may bias the next macro turn,
   request one USE pulse, or choose a dodge side.  It cannot select a raw
   action, invent a target, or override code-level combat/collision safety.
   High-level answers expire after 55 Doom tics; timeout/invalid/stale answers
   leave the deterministic policy unchanged.

## Evidence that would confirm or refute it

Confirmation would be a material rise in `exit_success`, plus traces showing
increasing visited cells, bounded repeated-cell runs, USE pulses near blocked
faces or visible interactions, and fewer thousands-long forward/left-right
runs.  Health should remain closer to the idle baseline except during genuine
combat.  The hypothesis is refuted if the square sweep repeatedly cycles the
same cells, spends its time probing harmless walls without reaching a label,
or dies more often than `0003` without improving exit completion.  In
particular, more kills/items alone would not confirm it: the decisive measure is
map exit activation.

## Jev contract and fallback (tier 2)

The fixed JSON state sent on each request is:

```text
{
  task, rules,
  health: critical|wounded|healthy,
  armor: none|present,
  damage_recent: boolean,
  motion: {phase, blocked, novelty, visited_cells, same_cell_visits},
  threats: [id, name, category, range, direction, level, hitscan,
            major, body_blocker],
  local_goals: [id, name, category, range, direction, level, ...],
  goal: {kind, range}|null,
  macro: {hand, sweep, turn_bias, previous_action}
}
```

All ranges, directions, levels, threat classes, and arithmetic are computed by
code from currently rendered labels.  Jev receives no raw map, omniscient
object list, coordinates, or inaccessible history.

One batched request contains these independent questions:

* `route: Choice(advance, turn_left, turn_right, use, retreat, unknown)` — a
  semantic hint for the next safe sweep boundary.  A high-probability turn is
  stored as `preferred_turn`; `use` sets a one-pulse flag; `retreat` starts the
  bounded escape only when no immediate visible threat already owns control.
* `dodge: Choice(left, right, back, none)` — sets the side of future combat
  strafing only.  It never changes target selection or legality.
* `open: Noul` — probability that a visible interaction or blocked face merits
  one USE pulse.  Values at least **0.72** set the same bounded pulse flag.

Choice answers are accepted when their selected-option probability is at least
**0.57** for `route` and **0.54** for `dodge`; an explicit confidence below
0.10, malformed value, timeout, or runtime failure is rejected.  A response is
used only if it is no more than 55 tics old and is checked against the newest
observation.  Model calls run through `ctx.spawn`; a pending or failed call
falls back to the current sweep/escape/combat controller, not `noop`.  The
request cadence is about 0.63 s with threats and 1.49 s in quiet traversal,
well below the System-1 budget but not so frequent that an option-biased model
owns the route.

The deterministic safety order is: terminal stop; close projectile dodge;
close/hitscan/major/body-block threat aim/fire/retreat; visible interaction
approach and USE; collision USE probe then escape; periodic USE; committed
sweep turn/forward.  A collision is a measured lack of displacement for at
least four movement ticks.  A blocked face receives at most one USE probe per
140 tics.  A sweep segment is 32, 48, or 72 forward ticks depending on the
current bounded policy; a 90-degree turn uses the measured turn rate.  No new
cell for 115 tics triggers a backoff/turn/forward novelty break.  These are
fallbacks even if every Jev answer is garbage or every Astra answer is late.
