# Candidate 0005 — event-driven motor with model-selected jobs

## Parents and diagnosis

Parents were `0001-tier2_astra_only` and `0002-tier2_idle`; I also inspected the
best existing candidate (`0001`) and the Jev-only baseline.  The Astra parent’s
score came from leaving the partner to work while a very occasional action
happened to help: on seeds 0, 1, 4, and 7 it reached 120, but its trace has one
posted action and roughly 399 stale ticks.  Its failure is especially concrete on
seed 2: the System-2 call was cancelled at about 66.95 seconds, the controlled
chef posted no action, and the episode ended at score 0 with 198 collision ticks
and `pot_idle_time=800`.  The randomized/ring failures show the same pattern;
seed 5 ended with score 0, 195 collision ticks, and only a one-ingredient idle
pot.

The idle parent is robust because it never blocks or collides with the scripted
chef, but its 40-point episodes are just the partner’s two deliveries.  It cannot
help when the partner needs the controlled chef to fetch an ingredient, start a
pot, fetch a dish, or clear a choke point.  Conversely, the Jev-only trace shows
why asking Jev to emit a raw direction is not a useful bridge: despite a fresh
answer approximately every 0.16 seconds, ring failures accumulated 188–198
collision ticks and no deliveries.  Jev was being asked to choose among six
motor symbols without a legal interaction target or path representation.

## Hypothesis

The dominant failure is not lack of reasoning about recipes; it is an absent
reliable motor and bookkeeping layer.  A controlled chef that continuously posts
legal shortest-path actions, explicitly starts full pots, and yields around the
partner should retain the idle baseline when the partner can finish, while
adding deliveries on layouts where the partner stalls.  Astra should therefore
choose only a rolling job preference, and Jev should recognize among code-built
legal jobs rather than invent directions.  A slow answer must never pause the
motor loop.

## What changed

This harness is a new controller rather than a copy of either parent:

* It reads the terrain and serialized state at every tick, derives dispenser,
  pot, serving-window, and counter positions, and reconstructs each pot’s
  `empty`, `partial`, `full_idle`, `cooking`, or `ready` phase from active
  recipes and soup fields.
* It builds a production pipeline: deliver held soup; stage a dish or pick up
  ready soup; fill a compatible pot; fetch a missing ingredient; explicitly
  start a full idle pot; and fetch/stage dishes during cooking.  Recipe lengths
  and ingredient types come from `all_orders`, not a hard-coded onion-only
  action script.
* For every medium-level job it enumerates adjacent interaction poses and uses
  BFS over walkable cells.  It turns the final pose into the required facing
  direction and only then emits `interact`.  The current partner cell is a soft
  obstacle; collision/stuck streaks trigger a short escape/replan instead of
  repeatedly pushing into the same choke point.
* One background Astra call receives a compact state and exact candidate IDs.  A
  safely parsed `preferred_task` or `preferred_kind` can bias the next event
  boundary, but malformed or late prose is ignored.  The call is never awaited
  by the real-time loop.
* Jev is called only at state changes or a 12-tick heartbeat, after a safe
  deterministic action has already been posted.  Its `Choice` is over at most
  eight code-generated job IDs and its `Noul` only asks whether a short yield is
  warranted by an explicit partner conflict.  Values are checked against the
  current candidate set and a confidence/probability gate; all failures retain
  the deterministic fallback.

## Evidence that would confirm or refute it

Confirmation would be a substantial reduction from the parents’ 100% stale
fraction and ring collision counts, with at least the idle parent’s deliveries
on seeds where the partner can work and additional deliveries on seeds where it
cannot.  In traces I would look for `start_cooking`, `pickup_ready`, and
`soup_delivery` events interleaved with low collision/stuck counts, and for Jev
answers changing a legal job rather than a raw direction.  The hypothesis is
refuted if the controller posts fresh movement but still leaves full idle pots,
if it repeatedly turns toward an occupied interaction pose, or if score remains
at the idle baseline while pot phases advance; that would indicate an incorrect
serialized-pot interpretation or a partner-specific handoff protocol is needed.

## Tier-2 Jev interface

The state sent to Jev is a strict JSON object with:

* `tick`, `layout`, and recent events;
* `self` and `partner` positions, facings, held-object names, and partner’s last
  action;
* one compact entry per pot: position, phase, current ingredient names, and
  recipe-derived missing ingredients;
* active `orders`; and
* `candidates`, each containing an exact `id`, `kind`, and a plain-language
  description such as “put the held onion into pot (2, 0)” or “fetch a dish from
  dispenser (1, 3)”.

The `job` Choice asks Jev to pick one listed candidate and explicitly forbids
inventing directions or IDs.  The `yield` Noul asks only whether the partner is
carrying soup/dish or directly causes an approach conflict.  A job answer is
accepted only when its value is a current ID and its returned probability is at
least 0.42 (stand-in answers with no distribution are still safely constrained
by the ID check).  A yield answer needs probability at least 0.72 and is used
only for an empty-handed chef when the partner holds a high-priority object.
Late, malformed, low-confidence, unavailable, or budget-exhausted answers are
ignored/disable further Jev calls; they never escape the episode loop.

The fallback policy is deterministic: task priority is delivery, ready-soup
pickup, pot start, filling/building a compatible recipe, dish staging, ingredient
fetch, and finally safe waiting.  BFS and facing validation decide the primitive
action.  A missing or rejected model answer therefore repeats a legal current
route rather than posting malformed input.  Astra is reintroduced only as a
non-blocking strategic hint; if later traces show its preferred jobs are causing
more collisions or pot idle time, the next revision should remove that hint or
re-query it only after verified event boundaries.
