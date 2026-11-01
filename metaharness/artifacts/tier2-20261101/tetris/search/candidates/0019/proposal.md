# Candidate 0019 proposal

## Parents and diagnosis

Parents: `candidates/0017/` and `candidates/0014/`.

Both parents establish the same essential robustness mechanism: a cheap,
code-owned guardian placement is executed immediately while a bounded queue
search runs in the background, and every movement is revalidated from the
latest observation.  This is why they finish all 200 pieces with no illegal
actions and essentially no stale ticks.  `0017` adds a gated depth-two queue
refinement and is the current best at 63,216.67 mean score; `0014` supplies the
clean guardian/refinement architecture on which that gain depends.

The remaining failure is scoring rather than survival.  In the worst `0017`
training episode (seed 4), the final score is 57,630 with 77 lines, one hole,
and no back-to-back bonus, despite 200 pieces and zero stale ticks.  Its trace
is almost entirely ordinary `hard_drop` locks: the final stretch at ticks
851--865 scores only 860 points for three ordinary placements, and the final
observation still has `back_to_back: false`.  Across the six `0017` traces,
line-clear points dominate but only one episode earns a substantial B2B bonus;
full T-spins are sparse.  The existing rank does not use `combo` or
`back_to_back` and treats a one-line clear almost like four independent
one-line clears.  It can therefore choose a safe single that breaks a combo or
B2B chain when a legal difficult-clear or a prepared T-spin is available.

## Hypothesis

Keep the parent timing/safety architecture, but make the code-owned evaluator
**score-state aware and explicitly recognize prepared difficult clears**.  A
candidate T placement is marked only when the simulated locked board has the
T-spin corner pattern and the target orientation requires a rotation; a
T-spin-like candidate without a rotation is not rewarded.  Tetrises and
rotated, corner-supported T clears receive a restrained bonus, especially when
they preserve an observed combo or back-to-back chain.  Existing hole, height,
and surface penalties remain dominant, and the bonus is applied only to the
already safe shallow/deep shortlist.  The executor additionally prefers a
route whose final maneuver is the target rotation for a marked T placement,
so the heuristic does not ask the engine for a T-spin while translating after
the rotation.

This tests whether the missing score is caused by the rank ignoring the
engine's stateful combo/B2B economics, rather than by insufficient search
breadth.  It is a crossover because it retains `0014`'s guardian and verified
clock control and `0017`'s gated queue beam, while adding a new, local scoring
bridge from observed `combo`/`back_to_back` state to exact candidate semantics.
It does not force an opener, preserve a well blindly, or ask either model to
calculate score.

## Changes in `harness.py`

* Preserve the parent SRS geometry, board simulation, hold checks, guardian,
  background depth-two refinement, freshness tags, urgent hard-drop fallback,
  and finite worker cleanup.
* During placement simulation, compute T's four diagonal corner occupancy and
  the two orientation-dependent front corners before line removal.  A plan is
  a `tspin_hint` only when it has at least three occupied corners and its
  target orientation differs from the spawn/current orientation.  The hint is
  descriptive and conservative; the engine remains authoritative.
* Add a small gated difficult-clear utility to the existing rank: Tetris, and
  rotated corner-supported T line clears, get a larger value than ordinary
  singles; a line clear while `combo` is active is preferred, and a
  difficult clear while `back_to_back` is true is preferred.  No-line T-spin
  setup receives only a small bonus.  New-hole/new-height and recovery gates
  still reject dangerous candidates, and the deep correction remains
  restrained.
* For a marked T-spin candidate, BFS first seeks a legal path ending in the
  target rotation; if unavailable it falls back to the parent's ordinary
  verified route.  Thus a late worker/model result cannot post a speculative
  rotation or stale hard drop.
* Add semantic `tspin_hint`, difficult-clear, combo-continuation, and B2B
  preservation buckets to Jev's role descriptions.  Jev still selects only
  among code-generated role representatives, never coordinates or actions.

## System-1 state and bridge

The strict JSON state keeps the parent's compact schema and adds only derived
facts:

```text
{
  "ruleset": "guideline-marathon-v1",
  "policy": "BALANCED | RECOVER | WELL",
  "piece_instance": integer,
  "board": {"height": bucket, "holes": bucket, "surface": bucket,
            "lines": integer, "level": integer,
            "combo": "active | inactive", "back_to_back": boolean},
  "active_piece": {"type": string, "orientation": integer,
                   "anchor_x": integer},
  "hold": string | null, "can_hold": boolean,
  "next_queue": [string], "fallback_role": "safe",
  "roles": {"safe": description, "clear": description,
            "future": description, "recovery": description},
  "candidate_count": integer
}
```

Descriptions say whether each representative is an ordinary placement,
combo continuation, B2B-preserving difficult clear, or conservative T-spin
hint, along with the existing projected surface buckets.  Two independent
`Choice` questions (`pick_a`, `pick_b`) use reversed option order.  A result is
accepted only when both labels agree, each selected-label probability is at
least 0.45, the piece identity is still current, and hold compatibility is
still true.  The code rechecks the route and lock urgency before every action.
The 0.60 s Jev timeout is asynchronous and never delays a tick.

Astra remains an asynchronous risk-mode reviewer.  Its compact prompt can
change `BALANCED`, `RECOVER`, or `WELL` for future code weights, but it never
chooses a difficult-clear label or a primitive action.  Malformed text,
timeouts, stale answers, and disagreement leave the deterministic plan in
force.

## Fallback and real-time safety

At every tick there is an executable guardian or emergency plan.  A pending
search or Jev response cannot block the clock.  Results tagged to another
piece or hold state are discarded.  If a marked T-spin route cannot be found,
the ordinary SRS BFS route is used; near lock, `hard_drop` takes priority.
Hold is sent at most once and only when `can_hold` is true.  All model output
is treated as labels and all posted actions are checked against
`action_spec`.  Search futures are bounded/cancelled on return and no model
future is joined by the game loop.

## Confirmation and refutation

Confirmation would retain 200 pieces, zero illegal/stale actions, and improve
mean score over 0017 through more doubles/Tetrises, line-clearing T-spins,
combo points, or B2B bonus, especially on seed-4-like tails.  Traces should
show `tspin_hint` or difficult-clear plans selected without a rise in holes or
height, and the engine's observed `last_clear.kind` should validate the
geometry hints.

The hypothesis is refuted if the new rank causes top-outs, holes, illegal
actions, or a score/line drop while the guardian remains unchanged; if hints
are frequent but the engine reports ordinary clears; or if B2B/combo metrics
remain unchanged and the mean matches the parent.  If lines improve but
B2B/T-spin points do not, the change only repaired ordinary multi-line utility,
not difficult-clear recognition.  If difficult-clear bonuses create holes,
the gate/coefficient is too permissive and should be removed before widening
search.
