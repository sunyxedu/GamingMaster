# Candidate 0004 — verified placement search with asynchronous ranking

## Parents and diagnosis

Parent: `candidates/0001-tier2_astra_only/`. The comparison baseline is
`candidates/0000-tier2_jev_only/`. There is no `lessons.md` in the archive.

The traces show that the failure is structural, not a bad timeout or a slightly
wrong prompt:

* In the Jev-only baseline, the first request sees the complete raw observation
  and asks “Which action is best right now?”  Its response is `hard_drop`; the
  recorded probabilities give `hard_drop: 0.54`.  That is a reasonable answer
  to an action question whose description advertises immediate drop points, but
  it does not specify a destination.  The resulting seed-0 trace applies
  `hard_drop` at ticks 4–6, 7–11, and so on.  At tick 13 the board already has
  21 holes, aggregate height 65, and top-out height 20; tick 14 tops out with
  score 202 and no line clear.  Across the baseline, 73% of ticks were stale
  and every episode ended after about 11 pieces with zero lines.
* Astra-only is much worse for real time.  In its seed-0 trace, ticks 1 through
  1396 repeatedly apply `noop` while the first Astra call is in flight.  That
  call takes 69.82 seconds, then the returned action is `hard_drop` at tick
  1397.  The episode ends at tick 1403 with score 68.  Seed 4 has the same
  pattern: 59.72 seconds of `noop`, then repeated hard drops and score 86.
  Its 100% stale-tick rate is direct evidence that slow reasoning cannot be the
  clock driver.

The old harnesses both ask a model to synthesize a low-level action from a raw
board.  Neither computes legal destinations, evaluates holes, or makes a
rotation/translation sequence.  The new hypothesis is that the model should
not be asked to invent mechanics at all: exact search can turn the board into a
small set of named placements, and Jev can recognize the safest of those while
code keeps posting actions every tick.  Astra is moved out of the input path
and used only as a slow, high-level policy reviewer.

## Hypothesis and changed structure

**Hypothesis:** A deterministic one-piece SRS-shaped search with a one-piece
lookahead, plus a verified low-level executor, will clear lines and survive
longer than either raw-action model.  Jev should rank seven precomputed
current/hold placements rather than choose among `left`, `rotate`, and
`hard_drop`; this removes the arithmetic and generation failure modes described
in the Jev contract.  A conservative code fallback means that a late or
uncertain answer costs at most a few input opportunities, not the episode.

`harness.py` changes the division of labor as follows:

* **Code:** converts the public top-to-bottom board to bottom-origin cells;
  generates the standard SRS orientation shapes and records the Guideline kick
  tables; enumerates legal hard-drop destinations for the current piece and a
  legal hold alternative; removes complete rows; calculates holes, hole depth,
  aggregate/max height, bumpiness, wells, transitions, landing height and
  top-out margin; scores each result; and looks one known queue piece ahead.
  It also translates a chosen `(orientation, anchor x)` into one rotation or
  horizontal action per tick and then a hard drop.  The engine remains
  authoritative for the actual SRS kick and legality, and the executor
  re-reads the active orientation/x before every input.
* **Astra:** one asynchronous call at episode start supplies only a validated
  high-level token: `BALANCED`, `RECOVER`, or `WELL`.  It is given compact public
  board metrics, hold and queue, never a seed-specific script, and can never
  block the game clock.  Its token changes code's risk weights and is included
  in the next Jev state.  Garbage Astra text is ignored.  The default policy
  remains `BALANCED` if the answer is late or malformed.
* **Jev:** one background request per falling piece receives the current policy,
  symbolic board buckets, hold/queue facts and seven candidate descriptions.
  It chooses a label, not a keypress and not a number.  Code asks two
  independent Choice questions in the same request with reversed option order;
  only a same-label, sufficiently concentrated agreement can replace the
  code-ranked fallback.  A response is accepted only while the same piece
  instance is still falling.
* **Real-time bridge:** the loop posts one valid action and then waits for the
  next tick.  While Jev is pending it sends a maximum of two safe no-ops at the
  top of a piece, then executes the verified fallback.  There is no synchronous
  Astra call and no busy wait.  If a Jev call times out, returns arbitrary text,
  disagrees, or arrives after a lock, the fallback continues.  Hold is issued
  at most once and is revalidated against `can_hold`.

This is a structural change from “model chooses an action” to “code searches
and executes placements, Jev selects among code's alternatives, Astra chooses
a risk regime.”  It should also eliminate the parent’s long stretches in
which the world only sees the initial no-op.

## Jev state and questions

Each request is strict JSON with this schema (all values are concrete, not
references to hidden state):

```text
{
  ruleset,
  policy: "BALANCED" | "RECOVER" | "WELL",
  piece_instance,
  board: {
    height: "low" | "moderate" | "high" | "critical",
    holes: "hole-free" | "few shallow holes" | "several holes" | "many or deep holes",
    surface: "flat" | "uneven",
    lines, level, combo: "active" | "inactive", back_to_back
  },
  active_piece, hold, can_hold, next_queue,
  fallback: "C0",
  candidates: {"C0": description, ..., "C6": description}
}
```

A candidate description says whether it uses hold, orientation and target
column, clear kind, resulting stack-height bucket, hole bucket, surface
roughness and well depth.  These are facts computed by code; Jev is explicitly
told not to count or calculate.  The two Choice questions are:

1. `pick_a`: select the candidate that best balances a clear with a low,
   flat, hole-free stack.
2. `pick_b`: make the same judgment with the exact same candidates in reverse
   insertion order, so first-option bias is detectable.

A choice is accepted only if both values are an existing `Ck`, they agree, and
each selected label has probability at least 0.36.  The threshold is lower
than a lethal free-form action threshold because every candidate was already
validated by code.  Otherwise `C0`, the code-ranked fallback, is used.  A
stand-in model returning arbitrary text or first options therefore cannot
crash the harness or introduce an unverified action.

## Evidence that would confirm or refute the hypothesis

Confirmation would be a material increase in lines cleared, score and pieces
survived on seeds 0–5, with stale ticks substantially below 73%, and traces
showing candidate labels and rotation/translation actions before locks.  In
particular, the final board should no longer consist of a single central stack
with zero lines as in the Jev baseline.

The hypothesis would be refuted if the candidate still tops out near eleven
pieces despite legal destination execution, if code's orientation/anchor
convention produces repeated illegal actions or visibly wrong placements, or
if the two Jev questions disagree so often that the deterministic fallback is
no better than hard-dropping the spawn column.  A high score with very few
Jev-selected placements would indicate that the code search is doing the work
and that the candidate should later remove or reduce Jev rather than ask it
about raw actions.  A late but useful Astra policy update would confirm the
intended slower division; an update that arrives only after game-over is
harmless and should not be counted as a control dependency.
