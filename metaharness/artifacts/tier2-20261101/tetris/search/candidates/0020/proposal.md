# Candidate 0020 proposal

## Parents and diagnosis

Parent: `candidates/0017/`, with the guardian/refinement executor retained as
real-time plumbing. I also inspected `0015` and the worst `0017` traces.

The previous candidates solved survival: 0017 averages 200 pieces, zero illegal
actions, and zero stale ticks. Its remaining failure is that the evaluator sees
only a small fraction of the scoring opportunities. The score is dominated by
ordinary singles and safe drop points: mean line-clear points are 51,200, while
mean back-to-back points are only 333.33. In the worst 0017 seed 4 trace there
are 63 ordinary single clears, only four ordinary doubles, two no-line full
T-spins, and no sustained difficult-clear chain; it finishes at 57,630 despite
200 pieces and only one hole. Seed 5 similarly has 59 singles, seven doubles,
and zero B2B points. The line-weight experiment in 0015 confirms that making a
larger clear numerically attractive is not enough: it scored lower while still
producing no B2B bonus.

The traces also expose a more specific controller mismatch. The search scores a
candidate from its settled board, but the game awards a large additional value
when the *last input before lock was a rotation* and the T reaches a spin
configuration. The code has SRS geometry and routes rotations, yet it has no
candidate feature for ``t_spin``/``t_spin_mini`` and no preference for a route
whose final action is a rotation. Thus it can accidentally make T-spins, but it
cannot deliberately choose a safe scored T-spin over an otherwise similar
single. The traces contain repeated rotation-then-drop T placements labelled by
the engine as `t_spin_mini` or `t_spin`; those outcomes are absent from the
ranking state.

## Hypothesis

A **verified spin-aware placement evaluator** will raise score and B2B without
sacrificing the proven guardian. For every legal T placement, code will inspect
the settled board at the exact final SRS pose and enumerate a short route whose
last maneuver is a rotation. It will classify the resulting corner occupancy
into a conservative `full`, `mini`, or `none` prediction, and reward only
line-clearing full/mini T-spins that pass a height/hole safety envelope. A
no-line T-spin gets a small setup credit, not a large offensive reward. This is
not a request for Jev to recognize or calculate a spin; exact geometry and
scoring features stay in code.

The key test is whether the missing score component is caused by a missing
feature, rather than by insufficient line weighting or queue depth. The
controller must never force a spin: if the predicted spin placement adds a hole,
exceeds the current height envelope, or has no verified rotation route, the
normal 0017 placement remains preferred. A difficult clear is especially
valuable while `back_to_back` is active, but a spin that clears no lines cannot
by itself preserve B2B, so the evaluator gives it only a setup bonus.

## What changed

Starting from 0017's clock-safe guardian, bounded queue refinement, verified
SRS routing, and model plumbing, this harness adds:

* exact code-side spin features to every simulated T placement, using the
  post-placement board and the four diagonal cells around the T center;
* a route-aware `spin_route` marker: the final target pose must be reachable by
  a BFS route ending in `rotate_cw` or `rotate_ccw`, rather than treating a
  coincidental orientation as a T-spin;
* a bounded `spin_value` bonus for safe line-clearing full/mini spins, a very
  small setup credit for zero-line spins, and an additional B2B bonus only when
  an already-active chain gets a predicted difficult clear; and
* a conservative role description so Jev can select a code-generated
  spin-safe alternative without emitting coordinates or actions.

The live executor remains unchanged in its safety contract: it revalidates the
current piece, hold permission, pose, and lock urgency on every tick. Model and
worker lateness still falls back to the guardian. Spin classification is only a
ranking signal; the game observation after lock is authoritative and is logged
for diagnosis.

## Confirmation and refutation

Confirmation is a mean score above 0017's 63,216.67, with the gain in
`line_clear_points`, `combo_points`, or `back_to_back_bonus`, and no increase in
stale ticks, illegal actions, holes, or top-outs. The decisive trace evidence
would be more engine events with line-clearing `t_spin`/`t_spin_mini` and
consecutive difficult clears, especially in the old seed-4/seed-5 failure
pattern. More no-line spins alone would not confirm the hypothesis.

It is refuted if the spin-aware ranks do not change chosen placements, if they
increase holes/height or cause top-outs, or if they produce more spins but lower
score because they break combos and B2B. If score improves only through
ordinary line clears while spin counts remain unchanged, the missing feature
was not the bottleneck; the next experiment should target queue-level setup
planning rather than increase spin bonuses.

## Tier-2 System-1 state and bridge

The state is strict JSON and contains only derived facts needed for a role
judgment:

```json
{
  "ruleset": "guideline-marathon-v1",
  "policy": "BALANCED | RECOVER | WELL",
  "piece_instance": 0,
  "board": {
    "height": "low | moderate | high | critical",
    "holes": "hole-free | few shallow holes | several holes | many or deep holes",
    "surface": "flat | uneven | rough",
    "combo": "active | inactive",
    "back_to_back": true,
    "last_clear": "single | double | triple | tetris | t_spin | mini | none"
  },
  "roles": {
    "safe": "code-generated legal placement summary",
    "clear": "code-generated safe clear summary",
    "spin": "code-generated verified spin-route summary",
    "recovery": "code-generated hole/height recovery summary"
  }
}
```

The request batches two independent `Choice` questions with reversed option
order. Jev is told to choose a role, never arithmetic, a coordinate, a route,
or an action. The `spin` role is offered only when its candidate has a verified
rotation-ending route and is within the safety envelope. Both answers must
agree and each selected-label probability must be at least 0.45; otherwise the
code-ranked plan remains active. A result tagged to another piece, stale hold
permission, or incompatible active type is discarded.

The role bridge is deliberately asymmetric: code still chooses the exact plan,
and Jev can only select among `safe`, `clear`, `spin`, `future`, and
`recovery`. Spin preference is also bounded in code: line-clearing spins outrank
ordinary plans only when they do not increase holes and are not more than five
rows above the current maximum; no-line spins receive a small setup credit.

While Jev, Astra, or the refinement future is pending, the current guardian is
executed immediately. At grounded urgency the controller hard-drops rather than
waiting. Astra remains an asynchronous risk-mode reviewer and may alter only
future evaluator style; malformed model text is ignored. Search futures are
finite and cancelled/bounded on return, and all model output is treated as
untrusted.
