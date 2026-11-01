# Candidate 0018 — soft leaf-last coverage with landing-aligned Jev control

**Mode:** rewrite. **Parent:** `candidates/0015/`. **Best-score reference:** `candidates/0016/`.

## Hypothesis

The plateau is not primarily a missing local danger label: both 0015 and the best 0016 controller reach five or six scored clears, but still lose all reported lives in roughly 200–240 seconds. The bottom-corner leaves 15 and 20 each have only one graph exit, so making them ordinary route goals can leave a vulnerable return hop; however, a hard rule that forces both to the very end may also delay coverage and expose Q*bert to pursuing hazards longer. Candidate 0018 tests a **soft leaf-last preference**: while ordinary cubes remain, ordinary targets win ties and entering a corner has a modest cost, but a safer or materially more useful corner route stays legal.

The controller chooses the nearest unfinished target anew from the public graph on each standing observation (with the Astra-selected sweep as a deterministic tie-breaker) and computes each candidate's remaining graph distance to that target. In a live 0018 development trace, Jev repeatedly changed the route at ticks **379/399/419/439/459**, alternating between the corner approach 9→14 and return 14→9 while the same target (node 15) remained unfinished. This reproduced the broader archive warning that advisor activity is not proof of improvement: a confident local choice can undermine a deterministic coverage route. Candidate 0018 therefore tests two related structural safeguards: prefer ordinary coverage before one-exit leaves without imposing an unsafe detour, and let Jev break ties only when its action strictly reduces the current target-distance potential. The tie order uses only the fixed 21-node graph and observed unfinished cubes; it is not seed-specific.

The local trace identifies a second failure that gives the route invariant a concrete rationale: in `candidates/0016/trace_seed0.jsonl`, tick **1584** is at node **14** with two cubes unfinished and selects **RIGHT** toward one-exit leaf node **20**; the hop lands at **1593**, followed by a life-loss recovery at **1603**. This is not proof the corner caused the collision, but it is an avoidable first-order exposure with a falsifiable edge-level prediction. The archive also records later empty-hazard deaths (for example 0016 seed 0 at ticks **2109/2932/3062**) so the leaf policy is not presented as a universal collision solution.

A live preliminary 0018 seed-0 trace with the first, hard-gated version did defer the corners, but still lost its first life at tick **338** while escaping the final corner sweep. That short run cannot establish causality, but it is evidence against treating “hard leaf-last” as an established safety rule; this candidate uses a soft preference rather than the hard gate. The explicit target-potential check addresses the other observed failure directly: Jev should not create a high-confidence route reversal merely because it judges a locally different edge attractive.

The remaining deaths are still not all explained by corner traps. 0016 seed 0 later loses at ticks **2109, 2932, and 3062**, and 0016 seed 5 at **2109, 2492, and 2933**; several late observations have `hazards=[]`. 0015 seed 2 likewise loses at **1161** with an empty hazard list. This is why 0018 does not claim to decode the missing hazard from purple-ball timer 36 or make broad timer-based waits. It changes route topology around the first identified bottleneck, retains only short rendered-object risk, and gives Jev a more useful interval to reply by asking about the next departure while the current hop animation provides time.

## Structural changes

### Code: phase demand plus a leaf-last routing invariant

`BoardClock` learns the destination colour from a local, score-bearing palette change and tracks cube needs from the visible 21-color vector. This preserves the verified destination-phase signal while avoiding a known decoder artifact: during the global clear animation, `needed_landings` can alternate between all-zero and all-one even as the score tally is still being awarded. A large simultaneous palette transition and its score tally confirm/rebase a board; a uniform target palette with no clear opens one bounded second-pass probe after 24 ticks. The probe is not itself counted as completion.

`CoveragePlanner` is a receding-horizon graph controller. It ranks immediately legal edges by public short-memory danger, useful landing, distance to the nearest unfinished target, revisits and escape count. Ordinary unfinished cubes win target ties; nodes **15 and 20** receive a small additional cost while ordinary demand remains, reflecting their one-exit geometry without banning them. The adaptor's current `safe_moves` remains the hard legality filter.

### Jev: speculate on the *next* move, not the current one

The previous corridor advisers asked about a current choice and often completed after it was no longer actionable; 0016's worst trace had **86/100** corridor replies stale and only **8.3%** Jev→action freshness. Here, when a movement edge is posted, code immediately constructs the legal candidate list at its **predicted landing node** and submits that as a speculative request. The hop lasts several game ticks, so a typical ~152 ms Jev answer can arrive before Q*bert is ready to choose again. On the observed landing, code recomputes all legal candidates from the fresh observation and only considers an answer if node, board, life, and policy versions match and the selected edge remains admissible and close to the deterministic best.

The strict JSON state contains: observed node and motion; forecast landing; board epoch, inferred scored clears and unfinished count; the soft one-exit-corner preference; a compact hazard list with kind, visibility, proximity and motion; and per-direction candidate records with destination, unfinished/transit, hazard bucket, escape bucket, target-progress bucket and code rank. It excludes raw RAM, raw coordinates, palette numbers and score arithmetic.

One batched request asks:

1. `choice_a`: choose among the supplied next-hop directions by safety, escape and useful coverage;
2. `choice_b`: the same Choice with reversed option order to reduce position bias;
3. `veto_proposed`: a `Noul` asking only whether the proposed next move has *supported immediate* lethal contact, not whether a distant/possible threat exists.

Code accepts a Jev-selected direction only if both Choices agree, its selected probability is at least **0.58**, the answer is no older than **14 ticks**, the forecast landing is the current standing node, and the life/board/policy versions match. A non-fallback Jev choice must also strictly reduce the freshly recomputed nearest-unfinished-target distance and stay within **0.35 risk / 1.0 cost** of the deterministic fallback; this monotone-progress gate is intended to prevent the observed 9↔14 oscillation. The Noul veto is acted on only at **0.86**, and only for a lower-risk alternative that also advances that target. Otherwise Jev is advisory and deterministic code controls. Every accepted edge is rechecked against current `safe_moves`; Jev never supplies FIRE, NOOP, node arithmetic or a new action.

The fallback does not wait for a model: finish the active one-hop motor transaction, neutralize during motion/recovery/flash, then take the freshly ranked safe direction. A missing/garbage stand-in answer, timeout, disagreement, stale landing, or model exception simply returns to the code-ranked choice.

### Astra: author a durable tie-order, not a posture audit

Astra gets a compact progress/threat snapshot initially and after a life loss if the first request is still in flight. It selects one of `balanced`, `spine`, or `zigzag` as a durable tie-breaking order among equally safe reachable unfinished cubes. Code owns the non-leaf target preference, short-horizon threat assessment, and the actual route; Astra cannot choose directions or turn the soft corner preference into a ban. Calls are asynchronous, at most two, with 100-second timeouts and strict JSON parsing. An invalid or prose-only answer leaves the existing policy unchanged.

## Confirmation and refutation

The narrow hypothesis is supported if traces show ordinary cubes generally prioritized while corner entries remain possible when current risk/route facts justify them, preserve the parent's early scored-clear rate, and improve mean score or lives/clears without repeating the 9↔14 adviser cycle. Event traces should identify when the one-exit cost changed the selected route. The pipeline sub-hypothesis is supported if more Jev answers are accepted on their exact forecast landing than 0016's stale replies, without increasing unsafe overrides.

It is refuted if the soft leaf preference still delays clear tallies or lowers score without reducing corner-related exposure, if target-potential gating does not eliminate repeated Jev reversals, or if the gains disappear across seeds. If empty-hazard deaths dominate after the route rule is exercised, the result points toward a different missing safety mechanism (validated disc escape or verified hazard timing), not a reason to invent a timer-decoded enemy location. Astra's benefit remains unproven; its chosen mode and any resulting tie-order changes are logged for future matched ablation.