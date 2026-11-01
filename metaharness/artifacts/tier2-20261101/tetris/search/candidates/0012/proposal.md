# Candidate 0012 — safety-envelope Jev bridge

## Parent and diagnosis

Parent: `candidates/0006/`. I used `candidates/0007/` as the strongest
comparison because it is the best evaluated controller (50,094 mean score), and
read its low-scoring traces as well as the zero-stale but lower-scoring
`candidates/0011/` traces.

The executor and search are not the main remaining failure.  0007 reaches 200
pieces in every episode with zero illegal actions and 78.33 mean lines.  Its
problem is that the Jev bridge can replace the code-ranked placement with a
semantically plausible but numerically dominated option.  In
`trace_seed3.jsonl`, the accepted `jev_candidate` events at pieces 17, 23, and
25 are respectively `C4`, `C5`, and `C1`.  Replaying the public observation
through 0007's own code scorer gives:

* piece 17: fallback `C0` has rank about `+100`, one line, and one hole;
  accepted `C4` has rank about `-120`, no line, and two holes;
* piece 23: fallback `C0` has rank about `+150`, one line, and zero holes;
  accepted `C5` has rank about `-39`, no line, and one hole;
* piece 25: fallback `C0` has rank about `+153`, one line, and zero holes;
  accepted `C1` is a hold placement with rank about `-94` and two holes.

The labels and acceptances are in the trace; the ranks and projected features
are deterministic recomputations from the same public board and queue.  Similar
behavior appears in seed 4: at piece 23, `C4` replaces a `+131` single-clear
fallback with a roughly `+2` no-clear placement, and at piece 27 `C6` replaces a
`+158` single-clear fallback with a roughly `-17` no-clear placement.  The
controller can survive these choices, but they are not useful model overrides.

Candidate 0011 moved search off the clock and achieved essentially zero stale
ticks, but its mean fell to 48,097.  Thus simply improving scheduling does not
repair the quality loss from target selection.  I retain 0007's known-good
executor and one-ply code search, and change only the model/action bridge.

## Hypothesis

**Jev should recognize among a small code-certified safety envelope, not rank
an unrestricted list of placements.** The unrestricted nine-option Choice lets
prose such as “well” or “flat surface” outweigh an exact line-clear and hole
calculation.  If code first removes candidates that are worse on irreversible
safety facts or far below the code winner, then Jev can still resolve genuinely
close surface/hold tradeoffs without being able to choose a dominated
hole-creating target.  A second identical safety check immediately before
adopting the answer should preserve this property under malformed, stale, or
stand-in responses.

This is a structural change to the System-1 interface and answer-to-action
bridge, rather than another placement-weight tweak.  The game remains
authoritative for SRS legality; the harness never turns model output into raw
keypresses.

## What changed in `harness.py`

* It keeps 0007's code-owned board conversion, SRS-shaped placement enumeration,
  current/hold branches, one-preview lookahead, hole/depth/height/bumpiness/well
  features, and verified one-action-per-tick executor.
* Each generated plan records the pre-placement hole count, hole depth, and
  maximum height.  The full executor shortlist is not sent to Jev.
* `_model_shortlist` always retains the code winner `C0`, then retains at most
  four alternatives only when they are no worse than `C0` on holes and hole
  depth, have no more than two additional maximum-height cells or 14 additional
  aggregate-height cells, and are close in code utility.  The normal utility
  margin is 24 (16 when sacrificing a line clear).  A margin of 72 is allowed
  only for a hole-rescue alternative: `C0` creates a new hole while the
  alternative returns to the pre-placement hole/depth envelope.
* `_certified_override` repeats the envelope check after Jev responds.  A valid
  answer can therefore change the target only within the same safety budget;
  an option that was not in the certified subset, creates a deeper/worse hole,
  or is too far below the code winner is ignored.
* Jev still receives two independent `Choice` questions in one request, with
  the candidate order reversed.  Both choices must agree and each selected
  label must have probability at least 0.30.  The request timeout is 0.55 s.
  The request is launched asynchronously after the first deterministic action,
  and its `(pieces, active.type)` identity is checked before use.
* The code winner remains the fallback while Jev is pending, times out, fails,
  disagrees, or returns arbitrary stand-in objects.  Hold permission, active
  type, orientation, x, and piece count are revalidated by the executor.  A
  blocked movement abandons the target rather than repeating it indefinitely.
* One asynchronous Astra review is retained for the long-horizon division of
  labor.  It returns only `BALANCED`, `RECOVER`, or `WELL`; a valid token changes
  future code search weights, while malformed/late output leaves the current
  mode unchanged.  Astra never supplies an action and is not needed for the
  first placement.

## System-1 state and questions

Every Jev request is strict JSON and contains:

```text
{
  ruleset,
  policy: BALANCED | RECOVER | WELL,
  piece_instance,
  board: {
    height: low | moderate | high | critical,
    holes: hole-free | few shallow holes | several holes | many or deep holes,
    surface: flat | uneven | rough,
    lines, level, combo: active | inactive, back_to_back
  },
  active_piece: {type, orientation, anchor_x},
  hold, can_hold, next_queue,
  fallback: "C0",
  candidates: {"C0": description, ...}
}
```

`candidates` contains only the certified subset.  Each description is
code-derived and names current versus hold, orientation, target anchor, clear
kind, projected height/hole/surface buckets, and well depth.  Exact arithmetic,
collision, SRS kicks, and input sequences stay in code.

`pick_a` asks for the safest useful certified candidate in the supplied order.
It prefers a clear only when the description remains low and hole-safe;
otherwise it prefers the flatter candidate.  `pick_b` asks the same independent
judgment with the option order reversed.  Jev is explicitly told not to count,
compare raw numbers, invent a destination, or return a primitive action.

The answer is accepted only when:

1. both answer values are the same existing `C` label;
2. both selected-label probabilities are at least 0.30; and
3. the selected plan passes `_certified_override` against `C0`.

A completed answer for an old piece, a changed hold state, a failed legality
check, timeout, ordinary model error, malformed response, or disagreement is
simply a rejected advisory.  It cannot post an action itself.  The deterministic
`C0` plan continues through the same verified executor.

## Evidence that would confirm or refute the hypothesis

Confirmation would be a mean above 0007's 50,094 (or comparable score with
fewer holes), with seed-3-like early hole cascades absent.  Traces should show
that accepted non-`C0` labels satisfy the envelope, while the previously
observed dominated choices are rejected or never appear as Jev options.  The
zero-illegal-action and 200-piece behavior should remain intact; stale ticks may
remain near 0007's level because this candidate intentionally preserves its
score-winning synchronous planner timing.

The hypothesis is refuted if the envelope still accepts placements that create
new/deeper holes, if suppressing the old overrides lowers lines and score across
seeds, or if Jev's close safe choices are consistently worse than `C0`.  A score
improvement with few accepted overrides would support the code-certified bridge
and indicate that future candidates should reduce System-1 calls rather than
return to unrestricted candidate ranking.
