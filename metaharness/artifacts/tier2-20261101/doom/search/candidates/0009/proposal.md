# Candidate 0009 — frontier graph search with breadcrumb backtracking

## Parents and hypothesis

Parents: `candidates/0007/` and `candidates/0008/`.

The parents establish two complementary facts.  Candidate 0008's continuously
refreshed deterministic motor is the right real-time plumbing: it had only
**0.589% stale ticks**, reached **281 observed cells** in its best seed-1008
trace, and collected 12 items while staying healthy.  Candidate 0007 supplied
the more important search idea: a collision should create a finite branch and a
checkpoint rather than another model-selected button.  Its weakness was
execution rather than the idea: its best trace held stale actions for **5143 of
5586 ticks (89.6%)**, so its finite recovery policy was often not refreshed by
the controller.

The concrete failure in 0008 is visible in its trace, not just its score.  On
seed 1008 it generated **250 escapes and 250 macro turns**, but positions sampled
through the 31,499-tic deadline repeatedly returned to the same approximate
corridors, including around `(1000,-272)`.  On seed 1012, the controller had
**74 escapes** and reached the same general starting-side corridor before dying
at health 1.  Its square sweep changed direction, but it had no memory of which
local heading had already failed or which previously observed branch was worth
revisiting.  The 0007 trace has the dual problem: its checkpoint queues were
finite, but long held macros made the action freshness metric poor and left the
motor exposed while Jev was deciding.  Neither parent ever completed an exit;
no selected trace rendered an Exit, Door, Switch, or Teleporter label, so the
controller must search unseen geometry rather than only chase visible objects.

**Hypothesis:** the missing structural mechanism is an online frontier graph.
At every measured coarse position, record the headings that were actually
tried and whether they produced displacement.  On a collision, try the least-
tried side/reverse heading with a bounded turn-and-forward probe; when all
local branches are exhausted, steer to an earlier observed breadcrumb whose
node still has an untried heading.  This should widen coverage and eventually
reach a key/door/exit route where a periodic sweep merely cycles, while keeping
0008's fresh action posting and 0007's finite USE/recovery behavior.  The graph
is built only from public pose, movement, labels, and observed collisions; it
contains no map, coordinate, seed, or training-route data.

The experiment is deliberately aimed at exit completion rather than another
increment of optional kills/items.  It also reduces far-pickup steering: only a
rendered interaction, key, missing weapon, or urgent nearby health/armor can
become a bounded goal.

## What changed

1. **Observed-edge memory replaces the square sweep.**  A 96-unit position bin
   plus a vertical bin is a local node.  A heading is marked attempted only
   when code actually turns/probes it or observes a collision.  Entering a new
   node marks the traversed edge and suppresses an immediate reverse.  At a
   segment boundary the controller chooses an untried side/reverse/diagonal
   heading in a deterministic hand-biased order.
2. **Finite frontier probes and backtracking.**  A probe consists of a measured
   turn to an 8-way heading and at most 22–30 forward tics.  No displacement
   marks that local heading failed.  A newly blocked face receives at most one
   USE pulse per node/heading cooldown, then another heading is tried.  If all
   headings at the current node are exhausted, the controller searches its
   observed node path backward and steers to the newest earlier node with an
   untried branch.  If no such node exists it performs one bounded reverse
   escape, never an unbounded turn loop.
3. **Fresh motor ownership.**  `ctx.act` is posted for every new game
   observation, including while Jev or Astra futures are pending.  The motor
   therefore retains 0008's low-stale behavior instead of 0007's held-action
   behavior.  Model answers can affect only branch order, target semantics,
   and bounded policy; they cannot issue a raw action or freeze the player.
4. **Exit-first visible interaction.**  A rendered Exit/door/switch/teleporter
   is approached with code-computed bearing and a one-tic USE pulse.  Keys and
   capability pickups have short range and progress expiry.  Sparse periodic
   USE and collision-face USE are retained for doors whose label is not yet
   rendered.
5. **Conservative combat safety.**  Code handles close projectiles, major or
   hitscan monsters, body blockers, weapon/ammo legality, strafing fire, and
   breadcrumb retreat.  Distant routine monsters do not steal frontier search.
   Low health plus recent damage selects retreat before optional combat.
6. **Astra policy is bounded.**  Up to two background reviews may change only
   `hand`, `combat`, `probe`, and `resources`.  Invalid/prose output is ignored.
   A policy revision clears only the short-lived Jev vote, not graph evidence or
   safety state.  The route remains executable if Astra times out.

## Evidence that would confirm or refute the hypothesis

Confirmation is a nonzero `completed_by_exit` rate, ideally with traces showing
an exit/key/door interaction after a sequence of distinct frontier nodes.  The
intermediate signature is: increasing unique nodes and successful directed
edges, fewer repeated positions after a collision, backtracks that arrive at a
node with a new branch, and low stale ticks comparable to 0008.  Survival
should be better than the progress-heavy low-health endings of 0007/0008.  More
kills or items without a completed exit would not confirm the hypothesis.

It is refuted if the frontier graph still cycles through the same nodes despite
all local headings being marked, if breadcrumb steering repeatedly fails to
reach earlier nodes, or if sparse USE probes cause more deaths without a new
interaction.  A lower stale percentage alone is not evidence of success; the
primary test is exit completion and the trace-level widening of observed
space.

## Jev state and typed questions (tier 2)

Every request is rebuilt from the latest observation and contains this compact
JSON schema:

```json
{
  "task": "survive and activate the exit",
  "visibility": "rendered labels only; no hidden map",
  "health": "critical|wounded|healthy",
  "armor": "none|light|present",
  "ammo": "empty|low|available",
  "damage_recent": true,
  "player": {"node": [x,y,z], "heading_sector": 0},
  "motor": {
    "phase": "travel|turn|probe|backtrack",
    "action": "forward",
    "blocked": false,
    "stalled_tics": 0,
    "node_visits": 1,
    "route_depth": 4,
    "frontier_nodes": 3,
    "collisions": 2
  },
  "threats": [{"id":"...", "name":"...", "category":"Monster",
               "range":"near", "side":"left", "level":"same",
               "active":true, "hitscan":true, "major":false}],
  "visible_goals": [{"id":"...", "category":"Door", "range":"near"}],
  "policy": {"hand":"right", "combat":"cautious",
             "probe":"normal", "resources":"capability"},
  "vote": "expired"
}
```

Code computes distances, side/range buckets, node occupancy, collision state,
ammo legality, and the candidate IDs.  Jev is never asked to calculate, infer a
hidden wall, or generate a button.

One batched request contains:

- `mode_a` and `mode_b`: reversed `Choice` questions over `explore`, `fight`,
  `evade`, `interact`, `retreat`, and `hold`.  The code adopts a mode only when
  both agree, the selected probability is at least **0.56**, and the answer is
  fresh; it expires after 54 Doom tics.
- `branch`: `left|right|reverse|straight`, accepted at **0.54**.  It only
  reorders the next finite frontier probe and is cleared after that boundary.
- `target`: a Choice over currently rendered monster IDs plus `none`, accepted
  at **0.54**.  It is advisory; code still verifies that the ID is visible and
  retains emergency threat ranking.
- `use`: a `Noul` that sets one bounded USE request only at a visible
  interaction or measured blocked face.
- `retreat`: a `Noul` accepted at **0.72**.  It is used only with recent damage,
  critical health, or a close threat and cannot override emergency code rules.

A Jev future has a **0.95 s** timeout and is tagged with its observation tic and
Astra policy version.  Answers older than 58 tics, from a superseded policy, or
with malformed/low-probability typed values are discarded.  At most one future
is outstanding; pacing is about 0.60 s in danger and 1.49 s in quiet travel.
While waiting, the current finite route/combat action is posted every tick.
Timeout, ordinary failure, disagreement, or arbitrary pre-flight model output
falls back to the code motor; model-unavailable and episode/budget exceptions
follow the harness contract.

## Astra revision and fallback

Astra sees the same derived state plus the last ten collision/new-node events.
It is asked for only the four enumerated policy fields above, not a route or
raw controls.  The first review starts after the initial search is underway;
there is a long wall-budget guard and at most two reviews.  A malformed or late
answer leaves the current hand, cautious combat, bounded probe, and
capability-bounded resource policy unchanged.

The deterministic priority is:

1. terminal/death stop;
2. close projectile dodge and low-health/recent-damage breadcrumb retreat;
3. close/major/hitscan/body-block threat aiming, strafing fire, or legal weapon
   fallback;
4. rendered exit/door/switch/teleporter/key interaction and one-tic USE;
5. measured collision-face USE once, then a least-tried frontier probe;
6. backtrack to an observed earlier frontier node;
7. sparse periodic USE and ordinary forward travel until the next finite branch.

The initial action is the advertised `noop`.  All later processed observations
receive a legal `ctx.act`, even when inference is pending.  `run_episode`
cancels only its own futures on terminal/deadline and stores a small summary in
`ctx.memory`; it leaves no worker loop behind.
