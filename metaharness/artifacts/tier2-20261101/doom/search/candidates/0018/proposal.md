# Candidate 0018 — Landmark-ledger expedition with bounded action leases

## Parent and hypothesis

Parent: `candidates/0015/`; comparator: the stronger `0011` motor.  The traces show that the parent's local fixes improve secondary exploration but not the actual task.  In `0011` seed 1012, the player reached a corridor around `(752,-304)` and then took 1,179 `turn_left` and 498 `turn_right` actions; health fell from 63 to 6 while a Shotgun and armor bonuses remained visible.  In `0015` seed 1003, a 210-action `backward_right` run held the player at `(1135,16)` while health fell 40→3 in view of a LostSoul, Zombieman, and DoomImp.  Conversely, the long 0011 seed-1008 run reached 15 kills/19 items and health 100 but spent the remaining episode revisiting the same `(979,-272)`-type corridor, ending at the deadline.  Thus the issue is not a missing aim lease alone: the controller has no durable, public-objective memory and repeatedly lets a current visible pickup/heading dominate navigation.

**Hypothesis:** a persistent ledger of *only previously rendered static landmarks* (keys, weapons, health/armor, and interaction-like labels), combined with a receding-horizon waypoint lease, will convert view observations into recoverable capability/door progress without permitting a single model answer or target bearing to trap the player.  The controller should pursue a high-value landmark for bounded translation, abandon it after measured non-progress, and return to a finite survey/escape program.  A close threat or recent damage always preempts waypoint pursuit.  This is a structural change from 0015's currently-visible goal plus sweep: navigation is now landmark memory + waypoint leases, while survey is a fallback rather than the main route.

## What changed

* **Code owns a landmark ledger.** A label enters the ledger only when it is currently rendered and has a world position.  Static labels are refreshed when seen; a key/weapon/health/armor landmark can remain a target after it leaves the view.  Labels are not treated as omniscient map data, and monster/projectile positions are never persisted as goals.  Priority is exit/interaction, then keys, capability weapons, survival pickups, and finally ammo.
* **Code owns receding-horizon navigation.** A remembered landmark is approached using short turn leases (at most five tics) and longer translation leases.  The target bearing is recomputed only when a lease expires.  A target that fails to get closer is abandoned with a cooldown.  This directly prevents the turn-only oscillations in the audited traces.
* **Code owns an exploratory ray ledger.** It records visited coarse locations and observed/failed heading bins.  Collision or 92 tics without a new location launches a finite backward–turn–diagonal–forward escape program, selecting the least failed local ray.  It does not assume a fixed map grid or use hidden sectors.
* **Combat is an emergency layer, not a route planner.** Close projectiles, major threats, hitscanners, recent damage, and body blockers preempt goals.  Aim correction is capped at three turn actions before a four-tic orbit lease; firing uses only legal available non-splash fallback weapons and movement.  USE is bounded to visible interactions or collision faces.
* **Astra writes a bounded mission card**, not buttons or a route.  It may change only `priority`, `mode`, `side`, `caution`, and `use`; malformed, late, or prose output leaves the current card unchanged.  The card is requested in the background at most twice.

The design deliberately does not claim that a remembered item proves a door is open or that a missing exit label means no exit exists.  It tests whether durable capability landmarks and safer route commitment can make the existing public labels useful enough to reach a required lock/exit.

## Tier-2 System-1 interface

Jev receives one compact strict-JSON state on every asynchronous request:

```text
{
  task, rules,
  vital: {health: critical|wounded|healthy,
          armor: none|present, damage_recent: bool,
          ammo: empty|limited|available},
  motion: {phase: escape|navigate|scan|survey,
           blocked: bool, stuck: bool, novel_cell: bool,
           visited_cells: bounded_int, local_rays_seen: bounded_int},
  threats: [{id,name,category,range,direction,hitscan,major,blocker}],
  landmarks_visible: [{id,name,category,range,direction,...}],
  target: {id,name,category,range,direction,...}|null,
  policy, last_action
}
```

Code computes distances, bearings, buckets, resource legality, and motion facts. Jev never performs arithmetic, invents an actor, or selects a raw button. One request batches three independent judgments:

1. `lease` (`Choice`): `scan`, `push`, `landmark`, `fight`, `disengage`, or `interact`.  A high-confidence `disengage` starts the finite escape only when no immediate emergency contradicts it; `landmark` permits waypoint steering; `interact`/`use` can request a one-tic USE pulse; the remaining values alter only the next high-level fallback.
2. `side` (`Choice`): `left`, `right`, or `neutral`.  Code uses it only as an orbit/escape side; it cannot override projectile safety or a legal-action check.
3. `use` (`Noul`): whether the named visible interaction or blocked-face evidence justifies one USE pulse.  It is ignored in an immediate fight and rate-limited.

The option order is reversed on alternating calls to reduce first-option bias. A `Choice` is accepted only if its selected probability is at least 0.54 (0.50 for side), confidence, when present, is at least 0.08, and the answer is no older than 70 Doom tics. `Noul` requires probability 0.70. Invalid values, missing probabilities, timeouts, runtime errors, and stale answers are discarded. Jev is dispatched about every 20 tics when threats are visible and every 48 quiet tics, with a 0.85-second timeout based on the measured sub-second latency. The controller continues its current legal action program while a request is pending.

## Astra revision trigger and fallback

Astra is launched at the first usable state and, if still needed, after roughly 6,200 tics; no more than two calls are outstanding over the episode.  Its prompt includes current derived facts, recent events, observed-landmark summaries, and the current card, plus explicit instructions not to use hidden geometry or seed-specific knowledge.  A response is applied only at a safe future decision boundary after strict enum validation.  A timeout, arbitrary stand-in text, cancellation, or unknown field keeps the current card.  The hypothesis is specifically supported if Astra's cards correlate with fewer failed landmark pursuits or better survival, rather than merely more calls.

The deterministic fallback priority is:

1. terminal/death stop;
2. close-projectile strafe;
3. immediate threat combat/retreat with bounded aim and legal weapon checks;
4. a committed escape or navigation lease;
5. a remembered/visible high-value landmark and bounded USE at contact;
6. a Jev USE request or collision-face probe;
7. finite survey and ray-ledger escape.

Every posted action is checked against `game.action_spec.names`, and every loop iteration posts a fresh legal action after processing an observation. Pending Jev/Astra calls are canceled on return; no background controller loop remains.

## Confirmation or refutation

Confirmation requires a material increase in exit success or a trace-level precursor: the ledger should show key/weapon/interaction landmarks being approached, fewer repeated target IDs, bounded turn streaks, and longer survival in the 0011/0015 failure signatures. An exit is credited only if the terminal observation reports it. If the candidate merely collects more items, repeats old landmark coordinates, or still dies stationary in a threat corridor, the hypothesis is refuted; if survival improves but exits remain zero, the next bottleneck is likely exit/door recognition rather than local control.
