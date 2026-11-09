# Candidate 0013 — hole-safe shortlist refinement

## Parent and diagnosis

Parent: `candidates/0007/`. I keep its synchronous bounded placement search, verified one-input executor, pipelined Jev reviewer, and one asynchronous Astra policy call. The parent already solved the catastrophic timing problem: every training episode reached 200 pieces with zero illegal actions. Its remaining weakness is placement quality in the two lowest-score episodes, not model plumbing.

The worst parent traces show a repeatable ranking failure:

* In `trace_seed4.jsonl`, the board has zero holes through piece 36. At tick 301, piece 37 locks an L for one line and the observation reports `holes: 1`; the preceding piece-36 state had `holes: 0`. The same trace later reports tick 334, piece 41, a one-line J lock changing the board from hole-free to `holes: 1`. The run still survives all 200 pieces, but ends at score **47,386**, 78 lines, and one hole.
* In `trace_seed3.jsonl`, tick 136 (piece 17) reports an O lock with no line and the first hole, and tick 149 (piece 18) reports an L lock with the hole count rising to 2. That episode ends at **48,974**, only 76 lines, with two holes and aggregate height 42, despite zero illegal actions and all 200 pieces placed.
* This is not merely a stale-action failure: the parent’s score report has `illegal_actions: 0` in both episodes, and the parent’s clock-owning executor reaches the piece cap. The parent’s weighted score gives a large reward to a line clear, so a legal projected clear that adds a buried cell can outrank a clean no-clear placement.

## Hypothesis

**When a legal placement that does not increase the hole count exists, forbidding hole-increasing placements from the one-piece shortlist will preserve enough line-clearing ability while preventing the irreversible buried-cell mistakes responsible for the parent’s low-score tail.**

This is a deliberately narrow change to the parent’s code-owned ranking: it does not change the clock, candidate geometry, Jev interface, Astra schedule, or input executor. It tests whether the parent is too willing to trade a hole for a single line, rather than testing another latency or search-depth hypothesis.

## What changed

After the parent computes and sorts all current/hold placements using its existing line, height, hole-depth, bumpiness, and one-preview value, 0013 applies a **hole firewall**:

1. Compute the current board hole count from the same exact feature function.
2. If at least one legal candidate projects `after.holes <= before.holes`, discard every candidate with `after.holes > before.holes`.
3. If every legal candidate would increase holes, retain the parent’s original full set; the controller is not allowed to pretend a safe placement exists.
4. Build the same nine-entry, source/orientation-diverse shortlist from the surviving set. The existing C0 code-ranked candidate remains the deterministic fallback.

Thus a line clear may still be selected, including a difficult clear, provided it does not add a hole. Existing holes can be reduced or maintained. If no non-worsening placement exists, the parent’s original risk tradeoff is used rather than causing an execution failure.

## Confirmation or refutation

The hypothesis is supported if 0013 keeps all episodes at the 200-piece cap, has zero illegal actions, and reduces the final hole/height tail in seeds like 3 and 4 while maintaining or improving mean lines and score relative to 0007’s 50,094 mean. Traces should show the early seed-4 line-clear candidate replaced by a projected hole-free alternative, and no hole-increasing target should appear in a shortlist when a non-worsening plan exists.

It is refuted if the firewall routinely forces tall or unusable placements, lowers mean score/lines materially, or fails to reduce holes. It is also refuted if holes still appear while a non-worsening candidate was available, which would indicate a search/geometry or hold-state mismatch rather than the parent’s weighting. A lower score with unchanged holes would show that the parent’s line-clear tradeoff, not buried-cell creation, was the dominant issue.

## System-1 bridge and tier-2 timing

The Jev state remains the parent’s compact, derived JSON schema:

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

Candidate descriptions are code-derived and name hold/current source, SRS orientation, target anchor, clear kind, projected height bucket, hole bucket, surface bucket, and well depth. Because the firewall runs before this state is created, Jev normally compares only non-hole-increasing candidates; it never has to infer whether a raw board placement creates a hole.

One `ctx.decide` call batches two independent `Choice` questions:

* `pick_a`: choose the safest legal candidate that balances a clear with a low, flat, hole-free result.
* `pick_b`: repeat the same judgment with the option insertion order reversed to reduce first-option bias.

An answer is accepted only if both values are valid candidate labels, agree exactly, and each selected label has probability at least **0.30**. Confidence is not treated as a probability. A timeout, malformed stand-in answer, disagreement, low probability, old piece, or incompatible hold answer leaves the current code-selected plan in force. Actions are never generated by Jev.

At each new piece, the harness posts a safe `noop`, computes the bounded search, and re-reads the game observation. It then posts at most one verified rotation, translation, hold, or hard drop per tick. A Jev future is launched only after deterministic control has been posted; its answer is tagged by `(pieces, active.type)` and is discarded if stale. The fallback is C0, and a blocked movement causes a verified hard-drop fallback rather than repeating a blocked input. Astra is still started once in the background after the first action and may change only the BALANCED/RECOVER/WELL evaluator policy; its slow text is never a keypress or a prerequisite for a tick.
