# Candidate 0014 proposal — portfolio-safe routes with asynchronous arbitration

## Parent and diagnosis

Parent: `candidates/0009/` (with the validated-cycle evidence from `0008`).

0009 established the important safety boundary: exact body transitions plus a
validated Hamiltonian ordering produced 0 stale ticks and 0 collisions on all
six training episodes.  Its mean was 124.333, but it left 12--22 cells empty
at the 4000-tick limit.  The worst trace is concrete: seed 5 finished at score
119 with 22 free cells, while the body remained ordered and live.  It also
shows the throughput bottleneck rather than a timing failure: 1,504 search
starts, 1,034 `not_shorter_than_cycle` discards, but only 35 installed routes.
At ticks 3970--3999 the same food `[2, 1]` caused a new search every few ticks
and every result was discarded as `not_shorter_than_cycle`; the controller
continued safely along one fixed tour until the time limit.  Thus the remaining
problem is the fixed tour's reachable shortcut set, not model latency or
legality.

The failed 0011 rescue is a useful negative result: relaxing the order proof
near the limit produced four self-collisions.  This candidate does not relax
that proof.  0010's cycle portfolio was safe but selected among tours one step
at a time and scored below 0009; this candidate uses the portfolio as a route
search space while retaining 0009's no-regression fallback.

## Hypothesis

A food route that is safe under *any one of a portfolio of independently
validated Hamiltonian tours* can be a better throughput move than a route
restricted to the one tour selected at tick 0.  The useful structural change is
to let code enumerate and simulate those routes, while moving model work out of
the clock path:

1. code proves each transition, body order, growth rule, route prefix, and
   post-food tour compatibility;
2. a background worker searches several compatible tours and returns only routes
   no longer than the current tour's exact greedy fallback (or an equal-length
   route with strictly more post-food tour choices);
3. Astra supplies a low-frequency objective (`SPEED`, `BALANCED`, or `FLEX`),
   not an action; and
4. Jev arbitrates between the already-proved route candidates using two
   reordered typed Choices.  A disagreement, low probability, timeout, stale
   prefix, or malformed answer leaves the code-selected route in control.

This should increase foods per 4000 ticks while preserving the 0009 safety
record.  It is deliberately not a request for Jev to find paths or count
cells.

## What changed in `harness.py`

* The 0009 fixed-cycle controller is replaced by a validated tour portfolio.
  It includes the ordinary row/column-spine tours and checked two-row seam
  tours.  Every generated candidate is rejected unless it contains every board
  cell exactly once and every consecutive pair, including the closing pair, is
  adjacent.
* The foreground retains a complete head-first order check.  `_transition`
  treats a non-food tail move and a food growth move differently.  A route is
  accepted only if every simulated intermediate body and the post-growth body
  remain in strict order on its selected tour.  Switching tours is allowed only
  at an observed body that independently satisfies the new tour; no unchecked
  shortcut or relaxed endgame rescue exists.
* A short worker searches `(body, heading)` states for the visible food on up to
  24 compatible tours.  It uses exact transitions and an A*-style bounded
  search, compares against the *active tour's actual greedy route*, and returns
  at most five safe alternatives.  The worker is time-bounded and the action
  loop never waits for it.  A route is attached to its origin body, heading,
  food and tour id; the foreground re-simulates it and accepts only an exact
  prefix of the current observation.
* The controller installs the best code route immediately when it arrives, so a
  slow Jev answer cannot create a stale tick.  If multiple alternatives survive,
  Jev can replace that route only when both reordered answers agree, the chosen
  probability is at least 0.52 in both answers, and the selected suffix still
  matches the current body/heading/food.  All alternatives are already
  collision-safe; Jev can change throughput/posture, never legality.
* Searches are throttled to one every four ticks for an unchanged food rather
  than repeatedly launching the same late search.  This keeps the worker
  available for a new food and makes a late endgame a clean deterministic
  fallback instead of a search-discard storm.
* Astra is started at most once through `ctx.spawn`, only after the board is
  substantially occupied (55%), or after a late unusual state (tick 1800 with
  at least 35% occupancy).  Its one-token supervisory prompt chooses a named
  route objective.  `FLEX` may select a route at most one step longer than the
  fastest candidate when it leaves more compatible tours; `SPEED` and
  `BALANCED` prefer the shortest route.  Invalid/arbitrary text is ignored.
  Astra never blocks a tick and never emits an action.

There is no seed table, food prediction, or private game state.  The tour
portfolio depends only on the observed dimensions; all route choices depend on
the current public body and food.

## System-1 state schema, questions, thresholds, and bridge

Each Jev request is strict JSON with this shape:

```json
{
  "controller_policy": "BALANCED",
  "observation": {
    "tick": 123,
    "length": 57,
    "free_cells_bucket": "large",
    "food_visible": true,
    "current_heading": "RIGHT"
  },
  "candidates": [
    {
      "label": "R0",
      "steps_bucket": "small",
      "beats_fallback": true,
      "post_tour_choices_bucket": "moderate",
      "post_open_area_bucket": "large"
    }
  ]
}
```

The omitted coordinates and action strings are intentional: code owns exact
arithmetic and Jev sees only derived facts.  Candidate labels refer to route
objects held by code.  Each description states whether the route strictly
beats the fallback, a semantic step bucket, and post-food flexibility/area.
There are two independent `Choice` questions in one request:

* `route_forward`: prefer fewer steps to the visible food, using flexibility
  only as a tie break unless Astra selected `FLEX`;
* `route_reversed`: make the same selection with the option order reversed.

The bridge requires identical valid labels and a probability of at least 0.52
for that label in both answers.  The request timeout is 0.55 seconds; while it
is pending, code continues the validated route/cycle fallback.  It then
revalidates the route suffix against latest `snake`, heading and food.  A
failed call, malformed value, option disagreement, low probability, changed
food, or non-prefix route is ignored.  The deterministic selection is always
executable, so Jev latency cannot produce stale actions.  The controller makes
at most one Jev request per food target and never waits for the result.

## Astra revision trigger

Astra is not asked to review ordinary route discards: those are expected
objective comparisons.  A future supervisor revision is justified only by a
repeated trace pattern such as (a) many portfolio-safe routes but no food
progress, (b) an unexpected loss of tour compatibility after a verified move,
or (c) an unusual board shape where no tour can be entered.  Such a revision
should change the named objective/thresholds or the Jev facts, not bypass the
exact transition/order checks.  Evidence that would confirm this candidate is
score above 124.333, fewer late `not_shorter_than_cycle` stalls, 6/6 live
completions, 0 stale ticks, and 0 collisions.  It would be refuted by any
collision from an accepted route, a recurring route-prefix error, a materially
lower score despite the same safety, or outstanding worker/model tasks after
an episode.
