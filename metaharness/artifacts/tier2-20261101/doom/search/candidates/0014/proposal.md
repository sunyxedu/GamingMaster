# Candidate 0014 — contact-following navigator with moving combat leases

## Parent and hypothesis

Parent: `candidates/0009/` (frontier graph with breadcrumb backtracking). The
best archived refinement, `0011`, is also the primary comparator because it
raised the mean to **10.958** while retaining fresh action posting.

The parent lineage proved that the real-time plumbing and visible-label
bookkeeping are useful, but not that the chosen search rule reaches an exit.
The concrete failure is visible in the selected traces:

* In `0011/trace_seed1008.jsonl`, the controller reached **15 kills and 19
  items**, yet ended at tic **31,505** with no terminal exit. It issued **290
  macro escapes**, **152 macro turns**, and **791 USE pulses**. The route still
  revisited the same broad regions; no rendered label was ever categorized as
  Exit, Switch, Door, Teleporter, or Key. More sweeping and more USE did not
  create a route/interaction event.
* In the worst MAP01 episode, `0011/trace_seed1012.jsonl`, health was **84** at
  tic 1943 near `(688,-112)` with Zombiemen, an Imp projectile, barrels,
  shotgun and clip visible. It fell to **48** by tic 2318, **27** by tic 2406,
  and **6** by tic 2669. The action counts were **1,179 `turn_left`** and only
  **67 moving-fire actions**. Thus the safety layer detected threats but spent
  long exposed intervals turning rather than translating around them.
* The frontier parent’s worst `0009/trace_seed1005.jsonl` ended at health 2
  at `(1232,-573)` with two Demons still visible. Its graph recorded only five
  frontier probes and three USE probes before the player became stuck. The
  later heading-frontier refinement improved secondary progress, but `heading_seen`
  was usually zero at its early macro turns and did not establish an exit.

**Hypothesis:** the missing structural primitive is not another heading score or
more model votes. It is a persistent *contact-following route hypothesis*.
A Doom level is a connected planar environment with doors and switches that can
be tested when a movement face is reached. A finite left- or right-boundary
follower should keep making progress through open corridors and expose
side-connected rooms, while a collision-triggered USE pulse tests unlabeled
barriers. It should switch sides only after a measured repeated contact or a
long no-new-space interval, rather than turning on a fixed sweep timer. In
parallel, combat needs a finite moving lease: at most two aim tics followed by
strafe/fire or backward translation, so the player does not die while waiting
for a semantic answer.

This is a falsifiable route change, not a parameter adjustment. It deliberately
trades optional pickup chasing for interactions, keys and survival; the primary
experiment is exit completion.

## What changed

1. **Contact-following replaces the frontier/sweep executor.** Code holds
   forward until measured displacement stops. At a fresh contact it sends at most
   one USE pulse, then executes a finite backward–90-degree-corner–forward
   macro on the selected wall side. A repeated `(cell, heading, side)` contact
   schedules a finite 180-degree escape and flips the side. The code records
   only observed cells, transitions and contact signatures; it does not use a
   map, seed, or training-route coordinate.
2. **Stagnation is a route-hypothesis event.** No new observed cell for a bounded
   interval triggers one backward–180-degree–forward recovery and a side change,
   not an endless local turn. Astra can choose among `wall_left`, `wall_right`,
   `alternate`, and a finite `open_sweep` fallback after reviewing measured
   evidence. Its other choices are bounded combat and door-use policies.
3. **Interaction is contact-first.** Rendered interactions and keys receive
   short, code-computed approaches. Optional health/ammo/bonus labels are
   ignored unless survival/capability thresholds make them nearby goals. At an
   unlabeled blocked face, the contact rule tries USE before changing heading;
   sparse periodic USE is only a safety net.
4. **Combat is a moving lease.** Emergency projectile dodges and code-ranked
   threats outrank models. A visible target may cause no more than two turn tics
   consecutively. The following burst strafes while firing or backs away when
   health, recent damage, or melee range requires it. Weapon switches are legal
   and ammo-checked; rockets/BFG are not selected merely because they are owned.
5. **The model division is changed.** Jev no longer votes on a raw route button
   or merely orders sweep headings. It chooses between finite macro intents
   (`continue`, `left_wall`, `right_wall`, `use_contact`, `reverse`, `retreat`,
   `uncertain`) and can select a currently visible target. Code validates all
   answers and maps them to policy flags or finite scripts. Astra is a route
   supervisor: it reviews unique-cell/contact/repetition evidence and can
   re-author the navigator family, combat posture and contact-USE frequency.
   A late or malformed answer leaves the finite executor running.

## Confirmation or refutation

Confirmation requires at least one `completed_by_exit` episode and a trace in
which the exit follows distinct contact/cell transitions or a successful
contact USE. Supporting evidence would be fewer repeated contact signatures,
more unique cells per unit time, lower long turn streaks, and health above the
0011 worst-case trajectory. A rise in kills/items alone is not confirmation.

The hypothesis is refuted if both wall sides still cycle the same observed
contacts without discovering an exit, if the contact USE policy increases
fatalities without opening/transiting anything, or if moving combat still
shows long turn-only streaks during damage. If the navigator improves coverage
but exit completion remains zero, the next bottleneck is likely an unobserved
exit activation/door convention rather than local search.

## Tier-2 Jev state schema

Each asynchronous Jev request is rebuilt from the newest observation and is
strict JSON:

```json
{
  "task": "Reach and activate the map exit alive",
  "visibility": "rendered labels only",
  "map": "MAPxx",
  "skill": 3,
  "health": "critical|wounded|healthy",
  "armor": "none|present",
  "ammo": "empty|low|available",
  "damage_recent": false,
  "navigator": {
    "policy": "wall_right|wall_left|alternate|open_sweep",
    "wall_side": "left|right",
    "phase": "forward_contact_search|corner|reverse_contact|...",
    "contact": false,
    "blocked": false,
    "stagnating": false,
    "unique_cells": 12,
    "contacts": 8,
    "side_switches": 1
  },
  "threats": [{"id":"...", "name":"...", "category":"Monster",
                "range":"near", "side":"left", "level":"same",
                "hitscan":true, "major":false, "body_blocker":false}],
  "visible_goals": [{"id":"...", "category":"Key", "range":"near",
                      "side":"center", "level":"same"}],
  "recent": [{"tic": 100, "cell":[0,0,0], "heading":2,
               "moved":"yes", "health":90, "kills":1,
               "items":0, "action":"forward"}],
  "policy": {"combat":"protect_health", "doors":"use_on_contact"}
}
```

Code computes all distances, bearings, ranges, contact/repetition facts, threat
scores, action legality, and numeric progress. Jev is not asked to infer a map,
count objects, calculate bearings, or generate an action string.

## Jev questions, thresholds, freshness, and fallback

One request batches four independent typed judgments:

* `macro_a` and `macro_b` are Choices over the seven macro intents above. They
  contain the same options in opposite orders to reduce first-option bias. A
  route macro is accepted only when both agree, the chosen probability is at
  least **0.57**, confidence is not explicitly below **0.07**, and the answer
  is no older than **34 tics in danger / 92 tics in quiet travel**. `left_wall`
  and `right_wall` update only the code-owned wall policy; `use_contact` sets
  one bounded USE request; `reverse` requests one finite recovery; `retreat`
  is accepted only with recent damage or wounded health. Disagreement means
  no macro change.
* `target` is a Choice over the currently rendered monster IDs plus `none`,
  accepted at **0.54**. Code rechecks that the ID is still visible and still
  urgent; it cannot create a target.
* `open` is a Noul. A yes threshold of **0.72** only requests USE when code
  observes a contact or visible interaction, so a bad semantic answer is
  harmless in open space.

Requests run via `ctx.spawn` with a **0.72 s** decision timeout, at most one
outstanding request, roughly every 0.4 s during danger and 1.37 s in quiet
space. While pending, failed, stale, or malformed, code posts the current
finite combat/contact action. The previous answer is not reused after its
freshness window or an Astra policy-version change. If Jev is unavailable, the
same deterministic fallback remains executable.

## Astra route review and safety ownership

Astra is called in the background at most twice. It receives the compact Jev
state, the last observed cell/heading transitions, contact repetition counts,
unique-cell count, and the current policy. It must return only:

```json
{"navigator":"wall_left|wall_right|alternate|open_sweep",
 "combat":"protect_health|clear_blockers|avoid_optional",
 "doors":"use_on_contact|often|sparse",
 "reason":"brief evidence"}
```

The answer never issues a button or a coordinate route. Valid fields update the
policy version and invalidate old Jev advice; invalid/prose/late output leaves
it unchanged. Code owns terminal detection, exact bookkeeping, movement,
weapon/ammo checks, projectile dodges, damage retreats, collision/contact
USE, interaction approaches, and all action legality. The episode starts with
`noop`, posts a legal deterministic action for every processed observation,
waits for the next engine tick, and cancels only its own pending futures on
terminal/deadline.
