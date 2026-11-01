# 0010 — Shape-local contact learning, not another global safety margin

Parent: `0008` (explore). Working route-search/motor plumbing also taken from evaluated best `0006`. No unfinished sibling candidate was inspected.

## Hypothesis

The remaining positive-margin deaths are **structured model errors**, not a shortage of generic clearance. A color-derived `turtle` label does not establish that all 32 pixels of a stretched carrier are safe. Further, a failed final UP is not necessarily a failure of carrier support, and the median need not remain a sanctuary at higher levels. A controller that separates these causes and learns constraints in **shape-local coordinates** should preserve lives without progressively eliminating every route.

Concrete archive evidence:

* Best candidate 0006, worst seed 109: ticks 2532 and 3814 die on row 10 at frog-left offset **25** from shape **163**, NUSIZ 7, despite complete-home route margins **1.25**. Ticks 5768 and 7262 repeat the failure at offset **24** on shapes 170 and 163. These observations are not generic timing evidence: the same small part of two forms is implicated repeatedly.
* Across 0006's six training traces, these shapes die at offsets 21–26. Shape 184 at the same nominal size has many surviving observations at offsets 24–26. The public `kind=turtle` annotation hides a useful distinction.
* Parent 0008, seed 109, issues final UP at x137 for the bay-144 route at ticks 665 and 1037 and x41 for the bay-48 route at 1491; each fails immediately despite a reported margin of 1. The parent’s trace shows the frog being carried between route certification and the final press. Best 0006’s 94 successful home entries have press offsets between center-minus-five and center-plus-one. Keep that empirically supported window and recheck the live state before pressing; do not trust a stale route certificate.
* Best 0006, seed103, dies on median row5 at ticks 9841,11113,11870; seed113 at11412 and seed127 at10101. An unconditional safe median is not a permanent invariant.

This is a new contact-identification experiment, not a repeat of copy expansion, global margin inflation, semantic leases, or clock tuning.

## Implementation

1. Retain six-frame receding-horizon routes to the median or an open home, replica expansion, neutral-rearmed motor control, and durable route goals from 0006. Add an explicit live-state guard before every directional press and treat the supported home-entry window as a precondition, not merely a forecast score.
2. Give stretched shape163/170 only a 24-pixel usable tail rather than all32 pixels. With the existing carrier inset this excludes frog-left offsets above18. The rule depends on the public form, not the seed or absolute location; unfamiliar shapes are not silently relabelled.
3. Maintain a cold-start episode-local **contact book**. An observed landing/lateral death quarantines a small band around the last shape-relative frog position. The ban follows that row/form/copy layout as the object moves. It does not expand all obstacles or poison unrelated carriers.
4. Separate immediate home-entry, median-idle/lateral, river-transition, and carrier-landing failures before learning, using the action actually posted rather than the old route’s intended action. A median contradiction revokes unlimited **NOOP dwelling** but still permits short lateral repositioning or immediate departure; otherwise the escape policy itself can create no-route deaths. Immediate home-entry failure is routed to entry-window review, not taught as water.
5. Re-observe after route search and independently recheck the first hop. Model choices never override this current-state guard.
6. Record contact contradictions and model-caused control changes explicitly. Continue until game end or the context deadline.

## System-1 schema and bridge

One compact request contains game/rule context, Astra policy, current road/river segment, complete candidate route goals with code-derived shortest/detour, widest/narrower, comfortable/tight and current-goal categories, timer urgency, the latest localized contact evidence, and explicit unknown home-crocodile status.

Batched independent questions:

* `goal`, `reverse`: identical durable-goal Choices with reversed order and UNKNOWN. Both must agree with winning probability >=0.65. Same life/home/level epoch, segment, policy version and age<=90 ticks are required. A goal lasts at most180 ticks and only biases a currently recomputed near-equivalent route.
* `review`: Noul for explicit prediction-failure evidence; >=0.8 may schedule an eligible Astra review.
* `diagnosis`, `diagnosis_reverse`: Choices over contact/entry/bank/unknown, asking which observed failure the evidence supports. Both must agree at >=0.75, refer to the current contact-book revision, and pass a deterministic evidence gate. Contact prefers safer routes; entry narrows the accepted entry window to its interior; bank suppresses median dwell. A stale historical diagnosis can remain meaningful across a respawn, but never across a changed evidence revision.

Missing probabilities, garbage, uncertainty, disagreement, transient failure or late goal answers leave code's executable policy in force. The motor uses the leading current geometry route; if it has no route, or the live guard rejects every legal candidate, it posts NOOP until a fresh state yields a certified move. At most one S1 call is outstanding; timeout3s is above the measured0.18–0.2s latency. All arithmetic and contact association remain in code.

## Astra and interface revision

Astra receives recent localized contradictions and route summaries asynchronously, with a 120s timeout and at least60s between requests. It selects bounded JSON policy enums, including contact/entry/bank focus; it cannot generate executable code or unsafe geometry. The prompt explicitly describes the fixed Jev state/questions, thresholds, and conservative answer-to-action bridge. Malformed stand-in text is ignored. Revise focus after repeated contradictions, not after every hop; a policy from a different level is rejected.

## Confirm / refute

Confirm if shape163/170 right-end deaths disappear, homes per life rise beyond0006, and non-head routes remain reachable. Refute if deaths persist inside the retained tail, local bans multiply without predictive benefit, or median/home heuristics cause stalls. Report positive-margin deaths by shape-relative contact and distinguish entry failures from landing failures. Stale-tick reduction alone is not evidence of success.

## Validation

Initial complete harness and proposal were written before trials. Syntax and stand-in-contract replay checks pass on all six archived public observation streams, including arbitrary Astra text and first-option/malformed-confidence Jev stand-ins. A live Jev-only trial on training seed109 with the interval-type fix reached score1468, 17 homes, level4 and ended `game_done` with no model or harness errors. After correcting action attribution and changing the bank rule from “block lateral motion” to “block only median NOOP dwelling,” a second Jev-only seed109 trial reached score1574, 19 homes, level4, four deaths, and ended `game_done` with zero errors. These are one-seed smoke results, not a leaderboard claim; the full evaluation must test all training seeds and Astra-enabled operation. No held-out evidence or game implementation is used.
