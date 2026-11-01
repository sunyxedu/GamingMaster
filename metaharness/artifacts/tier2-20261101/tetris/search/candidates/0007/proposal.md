# Candidate 0007 — pipelined placement search with a clock-owning executor

## Parents and diagnosis

Parents: `candidates/0003/` and `candidates/0004/`.  The comparison best is
0004 (mean 337.333), while 0003 is the more robust direct-placement bridge.

Both parents establish that a named destination is a much better model/action
bridge than asking Jev for `left`, `rotate_cw`, or `hard_drop`:

* 0003 improved the original Jev-only score from 202 to 335 and reached 14--16
  pieces instead of about 11.  Its seed-3 trace nevertheless repeats stale
  `hard_drop` at ticks 9--10, 15--17, and 23--28 while a Jev answer is pending;
  it finishes at 306 with 14 pieces, 25 holes, and no line clears.
* 0004 adds hold, rotations, SRS-shaped search, and a preview lookahead.  Its
  successful seed-1 episode reached 474 and cleared one line, demonstrating that
  the search and placement representation can work.  But the mean is only
  337.333, with 41.1% stale ticks, one Jev timeout, and 0.83 illegal actions per
  episode.  In seed 0, after the first placement at tick 7, ticks 8--11 repeat
  stale `hard_drop`; later no-op stretches occur at ticks 12--15 and 20--26.
  Seed 2 also has a stale `hold` at ticks 30--31 and two illegal actions.  The
  six Astra calls informed 0% of actions and were cancelled at game over, so
  Astra was not the source of that score.

The exact cause of every stale span is not observable from the public trace, but
its control symptom is clear: a model/future boundary can leave the world
reapplying the previous irreversible action while the harness is changing
plans.  A late answer must therefore be advisory, not a prerequisite for the
next input.  The prior candidates also expose a second risk: code can enumerate
a good target, but an answer for an old piece or a hold replacement must not be
executed against the new active piece.

## Hypothesis

**If the code posts a verified deterministic placement action before launching
or harvesting a model request, and keeps executing that placement one tick at a
time while the request is pending, then the placement/search gains of 0004 and
the verification/fallback robustness of 0003 will survive late or stale model
answers.**  A short exact search should also make the deterministic fallback
better than repeating the spawn-column hard drop: it enumerates all SRS
orientations, current/hold sources, line removal, holes, height, bumpiness,
well and queue-preservation features, then uses one known preview as a bounded
lookahead.  Astra remains a slow policy author rather than an input source.

This is deliberately not a training-seed opener or a parameter-only change.
The structural change is that the clock and low-level executor are independent
of model completion; Jev can change a target only after the same-piece and
hold/type checks pass.

## What changed in `harness.py`

### Code-owned search and bookkeeping

* The public top-to-bottom board is converted to bottom-origin rows.  Spawn
  shapes use the observed Guideline coordinates for I/J/L/O/S/T/Z and the four
  SRS orientations.  The standard JLSTZ and I kick tables are included as the
  ruleset reference; the installed game remains authoritative for the actual
  kick and collision result.
* Every current and legal hold placement is simulated at each orientation and
  anchor.  The simulation removes complete rows and computes aggregate/max
  height, holes and hole depth, bumpiness, row transitions, wells, landing
  height, and line count.  A code-owned weighted value emphasizes no new holes,
  low top-out height, and clears.  It uses the next visible piece for a bounded
  one-ply lookahead; the search intentionally stops there so planning itself
  does not recreate a real-time stale span.
* The top nine candidates retain the best code result, both current/hold
  sources when available, and orientation diversity.  The deterministic
  fallback is always candidate `C0`, the code-ranked first candidate.
* The executor re-reads `active.type`, orientation, anchor x, `can_hold`, and
  the placed-piece count after every tick.  It emits at most one rotation or
  one horizontal move per tick, then one `hard_drop`.  A hold is sent at most
  once for an instance.  If a target becomes incompatible, it discards the
  target and recomputes the current code fallback rather than using stale
  geometry.  All posted actions come from `action_spec`.

### Pipelining and division of labor

At a new piece the harness first posts a safe `noop` before doing the
roughly-one-tick search.  This prevents the just-locked `hard_drop` from being
reapplied while the fallback is computed.  It then posts the first verified
placement input before starting the Jev future.  It never calls `future.result()`
unless the future is already done.  At most one Jev future is in flight; if it
belongs to an old piece it is consumed and discarded, while the current piece's
request remains queued.  Every loop posts an action and waits with
`ctx.wait_for_tick()`, so a pending model cannot make the world wait or leave a
previous hard drop as the intentional control policy.

One Astra request is started asynchronously after the first action (when
System-2 is available).  It is asked only for `BALANCED`, `RECOVER`, or `WELL`.
A valid leading token changes the evaluator weights at a later piece; malformed,
late, cancelled, or timed-out text leaves `BALANCED` in force.  No Astra text is
used as a keypress or seed-specific sequence, and at most one System-2 request
is made per episode.

## Jev state schema and questions

Each request is strict JSON with this shape:

```text
{
  ruleset,
  policy: "BALANCED" | "RECOVER" | "WELL",
  piece_instance,
  board: {
    height: low|moderate|high|critical,
    holes: hole-free|few shallow holes|several holes|many or deep holes,
    surface: flat|uneven|rough,
    lines, level, combo: active|inactive, back_to_back
  },
  active_piece: {type, orientation, anchor_x},
  hold, can_hold, next_queue,
  fallback: "C0",
  candidates: {"C0": description, ..., "C8": description}
}
```

Each candidate description is code-derived and names whether it uses hold, its
orientation and target anchor, clear kind, projected stack-height bucket,
hole bucket, surface bucket, and well depth.  Jev is not asked to count cells,
perform arithmetic, invent SRS kicks, or generate an action sequence.

There are two independent `Choice` questions in one request:

1. `pick_a` chooses the candidate that best balances a clear with a low,
   flat, hole-free result.
2. `pick_b` asks the same judgment with the candidate insertion order reversed,
   reducing first-option bias.

A result is accepted only if both values are existing labels, equal each other,
and each selected label has probability at least **0.30**.  The confidence field
is not treated as a probability.  This is a moderate gate because every option
has already passed code legality/search; an ambiguous answer is less harmful
than an unverified primitive action, but the deterministic C0 fallback is always
available.

A Jev answer is also tagged internally by `(pieces, active.type)`.  It is used
only while that same piece is falling.  A hold answer is accepted only while
`can_hold` is true; after a hold, the active type must match the chosen held
piece.  Timeout, ordinary model failure, malformed stand-in answers, disagreement,
low probability, an old piece, or a no-longer-compatible plan all select the
current code fallback and continue the tick loop.  `ModelUnavailable` is not
converted into an action error.

## Confirmation or refutation

The hypothesis is confirmed if training episodes show a material increase over
337.333 in score/lines and pieces survived, while stale ticks fall below the
0004 41.1% and illegal actions remain zero or near zero.  Traces should show
that a late Jev answer changes a still-valid named destination, while movement
and hard drop continue even when no answer has arrived.  A nonzero line rate
across more than the isolated 0004 seed-1 result would support the search
portion of the hypothesis.

It is refuted if the new executor still produces long repeated-action spans,
if old answers cause hold/rotation geometry errors, or if it tops out near the
same 13--16 pieces with zero lines despite fresh per-tick fallback actions.  If
C0 consistently beats accepted Jev choices, the model interface is not adding
value and a later candidate should reduce Jev to a reviewer or improve the
candidate descriptions rather than return to raw actions.  If the score improves
but Astra still informs no actions, that confirms the fast search/executor but
refutes the claim that slow policy review is useful at this episode horizon.
