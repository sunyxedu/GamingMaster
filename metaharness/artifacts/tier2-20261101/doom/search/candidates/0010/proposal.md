# Candidate 0010 — keyed frontier search for unlabeled exits

## Parent and hypothesis

Parent: `candidates/0008/`, the best archived candidate.  I also used the
failure evidence in `0006` and `0007`, but this is a new navigation structure,
not a parameter revision of their sweep controllers.

**Hypothesis:** the remaining primary failure is an interaction/observation
mismatch, not insufficient coverage.  All archived `0008` traces repeatedly
rendered keys, weapons, monsters and pickups, but no `Exit`, `Switch`, `Door`
or `Teleporter` label.  For example, the seed-1008 endpoint at tic 31,503 was
at `(1002.5,-272.0,-128)` with 7 kills and 12 items, and had no interaction
label; seed-1012 ended at `(894.9,-265.6,-64)` with a Clip and dead monster,
again with no interaction label.  `0008` nevertheless performed 821 USE
pulses in seed 1008 and swept 281 cells, then revisited the same broad regions
until the deadline.  Thus an objective that approaches a rendered exit cannot
work in this observation projection.  The exit must be treated as an unseen
walk-over/sector/button condition, and search must deliberately test every
measured barrier after acquiring newly visible keys.

The falsifiable prediction is that a **key-milestoned, finite frontier graph**
will produce more distinct reachable branches and more successful USE-at-wall
opportunities than the parent's square sweep, without turning every visible
pickup into a long-range goal.  It should improve exit attempts even when no
exit label appears.  It is deliberately map-agnostic: no map IDs, coordinates,
seed routes, WAD data, automap, sectors or all-object state are used.

## Structural change

`0008` committed a forward/90-degree sweep and used collisions mainly to start
short escapes.  `0010` instead maintains a graph whose nodes are cells the
player actually visited.  Each node has eight world-heading faces with an
explicit state: `unknown`, `testing`, `open`, or `blocked`.  The controller:

1. selects one untested face and turns to it using measured angle sign/rate;
   it advances for a bounded window, then pulses USE only when that face is
   measured blocked (Astra may opt into a pulse at every branch);
2. marks the face open only after measured displacement/new-cell evidence;
3. after a no-displacement window, retries after USE has had time to open a
   door, then records a blocked face and moves to a different branch;
4. when a node is exhausted, BFSes only through previously measured edges to
   the nearest node with an untested face; it never steers to an invented map
   point;
5. resets blocked-face eligibility after a key milestone, because a wall-like
   face may have been a locked door; interaction labels, when present, outrank
   keys and receive a bounded contact USE pulse.

A visible key is the only non-emergency navigation goal.  It is approached
only while rendered, same-level and within a bounded radius, and is abandoned
when its distance does not improve.  Ordinary ammo/health/decoration labels
cannot pull the player into the old 0008 coverage loop.  A key label that is
contacted and disappears, or is followed by a count increment, advances a
capability epoch and reopens measured barriers.  This differs from `0007`'s
short recovery branches: its heading attempts were a collision escape policy,
whereas this candidate persists per-node face status, searches measured edges
back to an untested node, and explicitly invalidates blocked-face conclusions
after a capability milestone.

Combat remains code-owned: close/hitscan/major/body-block threats and nearby
projectiles interrupt graph search; code computes bearings, selects an owned
weapon with ammo, avoids close splash, and fires while translating.  It does
not ask Jev to calculate distances or generate buttons.

## System-1 state and questions

One asynchronous Jev request receives a strict JSON state containing:

- `task` and `rules`: exit first, only rendered labels are visible, and code
  owns graph arithmetic, collision, aiming, USE timing and legal actions;
- policy version and small Astra policy;
- health/armor/ammo buckets, recent damage, current key epoch and names of
  contacted key labels, and whether an interaction label is currently visible;
- current measured cell/facing and motion phase (`key_goal`, `probe`, `route`,
  or `frontier`), stalled/new-cell flags, visited-cell and measured-edge
  buckets;
- the eight supplied branch candidates with semantic heading names and exact
  status/try buckets, not raw coordinates;
- up to five currently rendered threats, three projectiles, four keys and
  three interaction labels, each with code-derived range/side/level and
  declared threat flags.

The batch asks four independent typed questions:

| ID | Type | Bridge |
|---|---|---|
| `branch_a`, `branch_b` | `Choice` over supplied `d0`–`d7` faces plus `none` | The option order is reversed between the two questions.  Only agreement, winning probability at least **0.56**, non-low confidence, current node and currently untested face are accepted as a one-node preference.  Otherwise deterministic least-tried ordering wins. |
| `threat` | `Choice` over current visible monster IDs plus `none` | A high-probability current ID can bias target selection for the next observation window.  Code rechecks visibility and still overrides it for immediate danger. |
| `use_barrier` | `Noul` | A yes at **0.72** only sets a pending pulse inside a current measured probe/retry; it cannot cause arbitrary USE spam or claim an unseen switch. |

Responses older than 52 tics or from an old Astra policy are discarded.  The
branch answer is also discarded if the node changed, the face became tested,
or a key/teleport transition occurred.  Missing probabilities, malformed
values, low confidence, timeout and runtime failure leave the code policy
unchanged.  Requests are launched about every 18 tics in danger and 34 tics in
quiet search, with a 1.45-second timeout based on the archived Jev p50/max
latencies.  While pending, the graph executor posts a fresh deterministic
movement, probe, combat or retreat action each observed tic.

## Astra role and revision condition

Astra is a slow reviewer, not an actuator.  It is called at most twice: after
the graph has accumulated evidence and later after a long search interval or
many blocked faces.  Its prompt includes only public map/IWAD metadata,
current labels, keys, health/resources, measured cell/edge counts, blocked/open
face counts and a short history.  It may return only:

```json
{"order":"least_tried|clockwise|counterclockwise",
 "probe":"barrier_and_junction|every_branch",
 "key_radius":"near|wide", "combat":"blockers|avoid",
 "side":"left|right", "reason":"brief observed evidence"}
```

A valid field revision changes only search tie-breaking, the bounded key radius,
combat optionality, and side preference.  It clears stale Jev branch/target
answers but never changes emergency rules or invents a route.  The important
revision trigger is accumulated evidence (blocked faces, repeated nodes, key
milestones), not one late local decision.

## Late-decision fallback and safety

The first control pass posts a legal finite action before launching new model
work.  If Jev or Astra is late, the current finite probe/route continues only
until its measured horizon; then code picks another branch, retries a barrier,
backtracks on measured edges, or uses a moving fallback.  Terminal/death state
stops immediately.  Every action is checked against `action_spec.names`.
USE and weapon selection are pulses with cooldowns.  Invalid Astra text is
parsed defensively; invalid typed Jev answers are ignored.  Futures are
cancelled on return and no custom thread survives `run_episode`.

## Confirmation or refutation

Confirmation would show, relative to `0008`: more unique cells and measured
edges per 1,000 tics, bounded rather than repeated square loops, explicit
`probe_blocked`/`use_probe` events at changing faces, blocked faces reopened
after key epochs, and at least one `completed_by_exit` (the decisive result).
Health should remain nearer the idle floor because optional pickups are no
longer navigation targets.

The hypothesis is refuted if the graph still cycles through a small set of
nodes, if barrier probes never create movement after USE despite key epochs, if
key-only goals miss reachable exits, or if exit success remains zero while the
measured search is genuinely exhaustive.  More kills/items without exit
completion would not confirm it.
