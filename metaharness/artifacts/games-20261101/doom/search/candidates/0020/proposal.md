# Candidate 0020 — height-aware landmark surveys and verified portal edges

## Parent, comparator, and falsifiable hypothesis

Parent: `candidates/0016/`. Comparator: the eligible leader `candidates/0011/`.
I retained the parent’s asynchronous real-time discipline and its public-only
observation projection, but I did not retain its skill ledger as the navigation
primitive. The new controller is a landmark-survey navigator: it remembers
where useful rendered objects were seen, deliberately samples a bounded
panorama, and chooses the next departure from the resulting view evidence.

**Hypothesis:** the remaining route failure is not only local heading
revisitation or input lag. The controllers are losing the map’s *capability and
height structure*. They either discard a useful key/weapon because it is on a
different floor, or chase it as though its world-coordinate bearing were a
walkable straight path. A visible key/weapon/exit is a route landmark, not just
a one-frame pickup. A height-aware landmark memory plus a short panorama at
safe locations should expose the stair/lift/portal approach that a 2-D sweep
misses. Once a measured obstruction is found, a single settled `USE` followed
by a same-heading crossing and an edge result should be remembered as a portal
experiment, rather than becoming another repeating wall probe.

This is a testable structural change, not a map or seed route. The code uses
only currently rendered labels, their supplied world positions, player
kinematics, and effects observed after its own actions. It never assumes that a
vertical landmark is directly reachable, that a displacement proves a door, or
that the absence of an exit label means the exit is absent.

## Trace evidence motivating the test

The parent’s worst short run (`0016/trace_seed1001.jsonl`) ended at tic 2828
with health **4**, only 5 kills and 3 items, while a `RedCard` was still
rendered at approximately `(856,-320,112)` and the player was at
`(871,-532,20)`. Medikit and clip landmarks were also rendered on elevated
layers. The parent produced 83 skill outcomes and only five interaction pulses,
but its pickup rule rejects a target when its height difference is too large;
there is no durable instruction to look for a route to the elevated card.
The controller consequently spent its remaining margin fighting and moving
without establishing the capability dependency.

The same pattern is visible in `0016/trace_seed1013.jsonl`: the final state at
tic 1857 had the player at about `(-252,-37,0)`, health **18**, while a
`BlueCard` at `(-608,-32,80)` and a `Medikit` remained visible. It had made
secondary progress (5 kills and 10 items), but never converted the height-
separated card into a route objective or exit completion. The parent’s
input/effect alignment fixes were real, but they did not answer which floor or
landmark should be sought next.

The leader’s failure is complementary. In
`0011/trace_seed1012.jsonl`, the run ended at tic 2742 at approximately
`(752,-304,-64)` with health **6**. The action counts included **1179
`turn_left`**, **498 `turn_right`**, only 85 backward actions, and 17 logged
escapes. A close `Zombieman`, barrels, armor and a shotgun were repeatedly in
view. The heading frontier reduced some geometric revisitation but did not
make a deliberate view survey or preserve an objective for a landmark on
another layer. The long `0016` MAP01 run also visited several z layers and
ended at health 3 after 42 interaction pulses without an exit; more USE and
more cells alone were not enough.

These observations support testing a capability/height-aware landmark policy;
they do not prove that every card is the exit route. The implementation keeps
that distinction explicit.

## What changed

### 1. Persistent rendered-landmark memory

Every currently rendered key, interaction object, weapon, health, armor, or
ammo object is stored by its public object id with its last world position,
category, z layer, age, closest observed distance, and whether a close
same-object disappearance supports a local pickup inference. A label that
flickers out is not immediately treated as collected. A key is marked acquired
only after the bounded overlap/disappearance evidence, and terminal state is
the only completion proof.

Candidate goals include recent landmarks for a short grace period, so the
controller can continue toward a card after it passes behind a wall. The
height difference is retained in the goal. Same-floor objects are approached
normally; a different-floor card is treated as a **horizontal relay objective**
that triggers a resurvey if progress stalls, rather than being silently
filtered out or asserted to be reachable in a straight line. Keys and explicit
interaction objects outrank routine loot; health/armor become goals only when
there is a concrete survival need.

### 2. Bounded panorama surveys

When there is no urgent threat, recent damage, or critical health, the motor can
run an eight-segment panorama. Each segment is a short measured turn, and the
controller records the visible landmark names, absolute positions, layer, and
heading bin. The scan has a fixed maximum and is aborted as soon as a dangerous
encounter appears; it cannot turn forever. At the end, an explicit key/exit or
capability bearing is preferred. Otherwise the code chooses a departure using
three measured signals:

* useful landmark evidence observed in that view;
* whether the projected destination cell and local view signature are already
  overused; and
* a remembered failed portal face.

This differs from the earlier heading frontier: a heading is rewarded because
a controlled view survey found a useful landmark or a new *view signature*,
not merely because its heading counter is small. It also differs from the
refuted global frontier candidates: it does not pretend that an unvisited
coordinate is a known graph node, and it does not ask Jev to invent a route.

### 3. Verified portal/USE experiments

A sustained no-displacement face is assigned a local face key consisting of
measured cell, heading bin, and current layer. The controller settles, pulses
USE once, waits for the asynchronous effect, then commits to a same-heading
crossing. It records `portal_pass` only after displacement or a layer change;
otherwise it records `portal_blocked`, gives that face a long cooldown, and
surveys an alternate view. A successful crossing becomes a transition cue and
causes a longer travel lease. This is not an exit claim. The face ledger is
what prevents the 0016-style repeated interaction experiment from spending
health on the same unverified wall.

### 4. Survival-aware committed motor

The deterministic priority is: incoming named projectile dodge; an already
committed finite escape; safe panorama segment; immediate close/hitscan/major
combat with moving fire; stationary watchdog; the controlled portal test;
explicit landmark approach; and bounded travel/USE. A turn lease prevents
one-tic aim reversals, while a stationary watchdog starts a finite diagonal
backoff/turn/departure. Weapons are selected only from observed ownership and
ammo; splash weapons are never added by model suggestion. The motor posts a
legal action for every fresh state, including while both model calls are
pending.

## Astra’s role

Astra is launched asynchronously with a compact map/skill/position snapshot,
recent landmark table, layer transitions, blocked-face outcomes, and counter
changes. It may author a short abstract program such as
`["seek_keys", "survey", "probe"]`, plus `mission`, `scan`, and `combat`
policy fields. The program contains no buttons, coordinates, hidden actors, or
seed route. Code accepts only the enumerated tokens and uses them as durable
priority hints; an absent, stale, malformed, or arbitrary-text answer leaves
the current policy unchanged. At most two calls are launched, with a bounded
output request and a 190-second timeout so the slow layer cannot become the
actuator.

The intended division is that Astra decides whether the evidence calls for a
capability search, exit-biased survey, recovery, or another controlled probe;
code determines whether the named landmark still exists, whether the player is
safe, and how to execute it. The first answer is not required for survival or
for the initial travel lease.

## Jev state, questions, and answer bridge

Each request is strict JSON and contains no raw map or inaccessible history:

```text
{
  task,
  rules,
  mission,
  policy: {mission, scan, combat},
  player: {health: critical|wounded|healthy,
           armor: none|present, layer, damage_recent},
  motion: {mode, blocked, stationary_tics, new_cell_recent,
           visited_cells, view_count, layer_transition_recent},
  threats: [{id, name, category, range, direction, height,
             hitscan, major, body_blocker}],
  landmark_options: [{option: "lm:<id>", id, name, category,
                      range, height, kind}],
  current_goal: {id, name, height}|null,
  portal: {phase, known_faces},
  recent_evidence: [bounded derived events]
}
```

Code computes distances, bearings, height buckets, threat classes, movement
stalls, and candidate ids. Jev never performs arithmetic or generates a target.
One request batches these independent typed judgments:

| Question | Options/threshold | Executable consequence |
|---|---|---|
| `objective: Choice` | `keep`, `scan`, `probe`, plus only supplied `lm:<id>` options; selected probability at least **0.61** | A valid landmark answer becomes an expiring id hint; `scan` requests one safe panorama; `probe` requests a bounded USE opportunity. It cannot create a target. |
| `stance: Choice` | `left`, `right`, `back`, `none`; selected probability at least **0.56** | Changes only a future combat strafe side; `back` starts the finite separation lease. It cannot override projectile, ammo, or health safety. |
| `use: Noul` | yes probability at least **0.74** | Sets a USE flag, which code consumes only at a supplied interaction or measured blocked face. |

The bridge rejects an unknown value, a selected probability below its
threshold, explicit confidence below 0.10, a response older than 52 Doom tics,
or an id that is no longer a current/recent candidate. Option order alternates
between requests. Timeout, `RuntimeError`, cancellation, malformed stand-in
answers, or missing fields preserve the deterministic policy and are logged;
`ModelUnavailable` is allowed to propagate as required by the runtime
contract. The normal request cadence is about 22 tics near threats and 52 in
quiet travel, with a 1.35-second timeout. A response can influence only a
future safe boundary.

## Tier-2 fallback and lifecycle

`run_episode` observes only fresh tics, polls and launches inference in
background futures, posts one legal action, and calls `wait_for_tick`. It
stops on terminal/death, `ctx.done`, or a normal budget/deadline exception.
The finalizer cancels model futures and records a bounded summary in episode
memory. Observation parsing, Astra JSON extraction, action legality, and
stand-in model garbage are defensive, so pre-flight arbitrary outputs cannot
escape as harness errors.

## Confirmation and refutation

The hypothesis is confirmed only if, against `0016`/`0011`, traces show:

1. vertical keys and capability landmarks becoming durable objectives rather
   than being discarded, followed by measured z-layer changes, acquired-key
   evidence, or more safe landmark approaches;
2. panorama samples producing distinct view signatures and fewer long
   turn-only runs or repeated same-face USE pulses;
3. portal results separating a real passage from a blocked wall; and, most
   importantly, a material increase in `completed_by_exit` or a demonstrably
   closer exit sequence without sacrificing survival.

It is refuted if scans mostly expose no actionable information, vertical
landmark homing repeatedly burns health without a layer transition, the face
ledger still cycles, or the candidate merely raises kills/items while exit
success remains zero. A clean low-stale trace or a larger visited-cell count
alone is not confirmation; the primary test is reaching and activating the
exit alive.
