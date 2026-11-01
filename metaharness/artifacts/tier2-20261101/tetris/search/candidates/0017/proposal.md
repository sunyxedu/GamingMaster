# Candidate 0017 proposal

## Parents and diagnosis

Parents: `candidates/0014/` and `candidates/0006/`.

`0014` is the best evaluated harness (mean **60,758.33**).  Its important
mechanism is not the model policy: the guardian placement, background search,
SRS-aware one-action router, and freshness checks keep the real-time game under
control.  It placed all 200 pieces in every episode, had zero illegal actions,
and only four stale ticks in the worst episode.  The trace makes this concrete:
for example, in seed 5 the controller posts `left`/`rotate_cw`/`hard_drop`
sequences while `search_ready` arrives later, rather than repeating an old drop.
Its code-owned fallback is therefore the safety and survival mechanism.

`0006` supplied the other useful mechanism: a bounded queue beam.  It also
completed all 200 pieces with no holes and averaged **78.67 lines**, compared
with `0014`'s **77.50** lines.  Its three-piece continuation was a good answer
to the one-ply failure described in its proposal: a placement that looks safe
now can make the next two pieces awkward.  However, `0006` waited on more
expensive planning in the foreground, resulting in **41.7% stale ticks**, and
its scoring policy produced no back-to-back bonus and almost no full T-spins.
`0014` fixed the timing problem and scored much higher largely through more
valuable clear/T-spin outcomes (its mean back-to-back bonus was only **200**,
though, so there is still headroom).

The concrete failure to address is a tradeoff rather than top-out.  `0014`
finishes safely, but its final scores vary from 56,942 to 64,956 while its
lines stay in a narrow 75--79 range.  Its accepted Jev role changes are rare
(two role events in seed 0 and two in seed 4, none in the other four traces),
so Jev/Astra are not demonstrated sources of the gain.  Meanwhile `0006`'s
three-piece lookahead improved line consistency but did not itself find the
high-value T-spin/difficult-clear opportunities that made `0014` strong.

## Hypothesis

A **small, gated queue beam can improve `0014` without surrendering its
real-time guardian**.  The foreground controller should remain exactly the
cheap immediate guardian and verified executor.  Only the background
refinement should import the useful part of `0006`'s longer horizon: expand a
few already-good one-preview placements through the next two visible pieces,
then use that value as a restrained tie-breaker rather than replacing the
proven immediate score wholesale.

The prediction is that this will avoid a locally attractive surface that
traps the next preview, recover some of `0006`'s line quality, and retain
`0014`'s occasional T-spin/clear-rich choices because immediate line and
surface terms remain dominant.  The prediction is deliberately modest: this
is not a new T-spin planner and it does not force a well or an opener.

## What changed in `harness.py`

* Retain `0014`'s code-owned board features, SRS geometry, hold bookkeeping,
  guardian search, background execution, BFS routing, urgent hard-drop rule,
  and freshness/compatibility checks.
* Keep the one-preview value for every legal current/hold source as the base
  rank.  On the best eight shallow plans, retain a diverse current/hold and
  orientation focus set; expand only the first four focus plans.
* For a candidate that does not increase the current hole count and is not more
  than five rows above the current maximum, run a bounded depth-two continuation
  with widths 6 then 3.  The continuation is finite ordinary Python work in
  the existing `ctx.spawn` search worker.  It is never awaited by the game
  loop, and a late result is discarded by the existing `(pieces, active.type)`
  freshness tag.
* Add only `0.20 * (deep_value - shallow_value)` to the existing rank.  This
  makes the beam a refinement/tie-breaker, not a wholesale change to the
  successful `0014` evaluator.  The refinement is disabled for gravity below
  the worker's safe fast-gravity threshold (`gravity_s_per_cell < 0.12`),
  where the one-preview result and guardian are safer.
* Keep the guardian and deep plan separate.  At a piece boundary a harmless
  fresh `noop` is posted, the immediate coarse plan is installed, and the
  deeper worker starts.  A worker delay therefore causes steering toward a
  legal guardian, never an idle wait or stale hard drop.
* Keep Astra as an asynchronous risk-mode author (`BALANCED`, `RECOVER`, or
  `WELL`) and Jev as a semantic role recognizer.  Neither model emits a
  coordinate or primitive action, and model failures leave the code plan in
  force.  This preserves `0014`'s zero-failure contract behavior while making
  the search change independently testable.

## System-1 state, questions, and bridge

Every Jev request is strict JSON and is rebuilt from the current observation
and current refinement shortlist:

```text
{
  "ruleset": "guideline-marathon-v1",
  "policy": "BALANCED | RECOVER | WELL",
  "piece_instance": integer,
  "board": {
    "height": "low | moderate | high | critical",
    "holes": "hole-free | few shallow holes | several holes | many or deep holes",
    "surface": "flat | uneven | rough",
    "lines": integer,
    "level": integer,
    "combo": "active | inactive",
    "back_to_back": boolean
  },
  "active_piece": {"type": string, "orientation": integer, "anchor_x": integer},
  "hold": string | null,
  "can_hold": boolean,
  "next_queue": [string],
  "fallback_role": "safe",
  "roles": {
    "safe": description,
    "clear": description,
    "future": description,
    "recovery": description
  },
  "candidate_count": integer
}
```

Code constructs four role representatives from already simulated legal
placements:

* `safe` is the normal rank leader;
* `clear` is the best non-obviously-dangerous line-clear alternative;
* `future` favors the strongest visible-queue surface; and
* `recovery` minimizes holes, hole depth, and height.

The role descriptions mention clear type, projected height, holes, surface,
well depth, hold use, and orientation, but Jev is told not to count cells,
compare raw numbers, calculate score, invent coordinates, or produce an input.
The new deep score is reflected only in which legal plan represents a role; it
is not sent as raw arithmetic for Jev to interpret.

One `ctx.decide` request batches two independent `Choice` questions, `pick_a`
and `pick_b`, with criteria in opposite orders.  A result is usable only when
both values are valid role labels and agree, and each selected-label
probability is at least **0.45**.  The answer is also required to belong to the
same piece and same hold-compatible snapshot.  Otherwise the code-ranked safe
plan remains active.  The request uses a **0.60 s** timeout: it is launched
asynchronously after an action has already been posted, so even a timeout
cannot stall the game.  The existing action bridge rechecks the latest active
piece, orientation, x, hold permission, movement result, and lock urgency
before using it.

Astra receives compact derived facts after the opening and on a later danger
transition.  It returns one leading mode token only; the token changes future
code weights and role descriptions, never the current primitive action.  A
malformed, late, cancelled, or unavailable advisory is ignored (persistent
`ModelUnavailable` is allowed to propagate as required by the contract).

## Fallback and real-time safety

The controller always has an executable plan.  While the deep search or Jev
is pending, the current piece follows the coarse guardian through the verified
SRS BFS router.  If a result is late, incompatible, malformed, or no route is
available, it is discarded or replaced with a current-piece emergency plan.
Near lock, `hard_drop` wins over waiting for a model or worker.  Hold is sent
at most once and only when `can_hold` is true.  Every posted action is checked
against `action_spec`; model output is never parsed as an action.  Search
futures are finite and are bounded/cancelled on return, while model futures
are left for the evaluator's normal cancellation behavior.

## Evidence that would confirm or refute the hypothesis

Confirmation would be:

* all training episodes still reach 200 pieces with zero illegal actions and
  stale ticks near `0014` (not `0006`);
* mean lines rise toward or above `0006`'s 78.67 without the hole/height tail
  increasing; and
* score improves over `0014`, especially on seeds where the refinement chooses
  a different first placement, through more doubles/triples/Tetrises,
  difficult clears, combo points, or back-to-back bonus.

The traces should show `search_ready` plans arriving while a guardian is already
being executed, followed by occasional deep-plan changes that remain same-piece
valid.  A seed-level improvement concentrated in `0014`'s lower-line episodes
would support the queue-trap hypothesis; a higher accepted-role rate is useful
but is not required, since code owns exact arithmetic and legality.

It is refuted if the deeper correction recreates `0008`/`0009`-style traps,
causes stale hard drops or illegal actions, reduces survival below 200 pieces,
or lowers mean score/lines while the shallow guardian remains unchanged.  If
scores and lines are indistinguishable from `0014` and `deep_value` rarely
changes the chosen plan, that is still evidence for the guardian architecture
but refutes this beam's usefulness; a later iteration should remove the extra
worker cost rather than increase the beam.  If score improves but holes rise,
the safety gate or correction coefficient is too permissive.  If lines rise
without any difficult-clear or B2B gain, the beam is only a line-quality
improvement and should not be credited with discovering T-spin strategy.
