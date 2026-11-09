
### Candidate 0003 — public instruction codebook

- Baseline seed 12 spent two model calls on malformed color clues without advancing turn 2; seed 14 ended with 23 cards still in the deck. Separate transport parsing/latency failures from Hanabi strategy quality.
- A truthful clue can carry a preagreed instruction about an untouched slot. The sender validates the visible target card; the receiver decodes only the public clue attribute, turn and hand length. Rank signals must never be sent unless their encoded play is safe.
- Exact joint own-hand marginals need no Monte Carlo: a subset DP over at most five labelled slots and 25 identities computes without-replacement weights efficiently. It matched exhaustive enumeration on 100 small random examples; policy-dependent selection bias remains outside this literal belief model.
- First dev episode completed at 19/25, 74 actions, four Astra calls and no episode error despite two caught 90-second timeouts. Repeated compulsory discard instructions left the giver's initial hand untouched until turn 40. A discard recommendation should yield to a verified scoring opportunity in either seat.
- Per-seat sessions are mandatory, even when one Python object controls the team. Test malformed replies and timeout fallback explicitly: a passing pre-flight may exercise zero adviser calls and therefore miss those branches.
- A turn-based harness can gain substantial wall-clock headroom by reserving Astra for spaced ambiguous decisions; deterministic safe plays and legal discards let a complete episode finish before the deadline.
- Exact finite-population hand marginals are practical with subset DP over five slots and catch duplicate/critical-card mistakes that independent per-slot probabilities miss.
- Public clue promises must be validated from the giver's visible target hand and reconstructed from public touched IDs; private face information must never enter the shared protocol state.
- Timeout and malformed Astra replies are routine, not exceptional: catch them and fall back to a prevalidated legal action, while keeping one session per absolute seat.

### Candidate 0001 — executable clue protocol and bounded strategic advice
- Baseline score was limited by episode completion: seed 12 ended with 24 cards left and three lives; seed 14 with 23 cards and three lives. Improve action throughput before tuning risk appetite.
- The generic parser dropped color arguments: seed 12 posted `REVEAL_COLOR 1` then `REVEAL_COLOR`, both rejected at turn 2. Select complete dictionaries from `legal_actions` instead of reconstructing commands.
- Shared model sessions across alternating seats leak private views. Keep sessions seat-local and restrict any shared convention tracker to information reconstructible from public actions.
- A subset dynamic program gives exact multiplicity-weighted own-hand marginals without enumerating all 25^5 assignments; 100 small randomized cases matched exhaustive enumeration.
- Pre-flight may make zero model calls because its budgets differ; explicitly test malformed responses, out-of-range choices, and model exceptions separately.
- Dev request-to-return time can greatly exceed reported backend latency (one call: 510 seconds elapsed versus 24.8 seconds reported). Final 0001 uses at most three adviser calls, a 120-second outer wait on ctx-managed model futures, and autonomous completion after a timeout. Both dev attempts were interrupted before completion; no completed dev score is claimed.

Candidate 0003 follow-up: a sixth dev adviser request remained queued for over eight minutes despite a 90-second backend timeout. The final harness adds a 90-second `Future.result` caller wait through `ctx.spawn`, disables further advice on failure, and finishes automatically; the evaluator cancels the outstanding model call at episode end. Backend timeout and total caller wall time are distinct budgets. This guard passed pre-flight and explicit queue-timeout fallback tests; no third dev evaluation was run.

### Candidate 0005 — predictable discards and preventive saves

- Parent 0001's public chop prediction disagreed with its private-information discard on 28 of 81 archived discards. Seed 12 lost B5 at turn 8 and the last G4 at turn 31 on mismatched slots, despite four clue tokens. Recipient prediction must use the recipient's actual declared rule.
- A public-only discard commitment can be audited by erasing both hands' faces: predicted slots must remain identical. Hypothetical post-clue commitments must also match recomputation from the recipient's literal masks and public promises. Both audits pass on the archived observations.
- Value a save by the actual before/after discard it induces. Touching a critical card does not help if it was never threatened, or if the clue redirects discard to another equally costly card. Negative information can redirect chop even when the endangered card is untouched.
- First dev game finished 21/25 with all lives and no unique-card losses; saving last copies alone still stranded a yellow sequence. At turn 52 a safe B5 outranked protecting the predicted playable Y1 discard. Spare copies do not make an immediately accessible prerequisite dispensable near the end.
- Preflight made zero model calls. Separate fake-context checks cover garbage, invalid choices, model exceptions, outer queue timeout, and resuming only the acting seat's session. A caught caller timeout can coexist with zero failed backend calls in eval.json; read the harness events too.
- The playable-chop refinement raised the same dev deal from 21 to 24 (68 actions, all lives, no unique-card losses, no episode error). Yellow completed; the remaining G5 was excluded by final-round timing. This is paired dev evidence only, not a training/held-out score claim. Both permitted dev evaluations are now used.

### Candidate 0006 — predictable public discards and critical-save promises

- Leader 0003's weakest trace loses W5 at turn 31 and the final G3 at turn 37 while finishing with three lives. Once episodes complete reliably, separate lost physical ceilings from misplays and final-round access losses.
- A partner-discard prediction based on public beliefs can disagree with the recipient's private marginal choice. Candidate 0006 makes uncertain fallback discards public and predictable; privately proven-dead overrides remain safe. A save can then protect the exact threatened slot.
- A rank clue touching that scheduled slot can convey a public critical-identity constraint, provided the sender verifies it and the receiver reconstructs it only from public counts, masks and touched IDs. Face-redaction invariance checks passed on 72 archived observations.
- The first dev run scored 23 with all lives, seven save clues and no live last-copy losses. Two held fives missed the final countdown, showing that preserving the physical ceiling does not ensure sufficient time/access to realize it. This is one dev seed, not a training result.
- Model queue timeout remains distinct from backend timeout: an 80-second outer wait expired, then automatic play completed with no episode error. Fake-context tests must cover this branch because preflight made zero model calls.

### Candidate 0004 — adaptive public packets and discard-loss correction

- Fixed slot arithmetic can waste a truthful clue on a card that is already impossible to execute; filtering message targets through public masks gives a smaller but safer codebook.
- A color clue can serve as a persistent, face-blind noncritical certificate keyed by stable card ID; reapply physical-count checks after every public discard/play so the certificate expires when its assumptions do.
- Replay audits should erase every visible face from decoder input and compare sender/recipient results; candidate 0004 matched 30 dev messages and executed 23 guaranteed plays without a misplay.
- Dead-card probability already contributes zero expected discard loss; adding a separate dead bonus double-counts the benefit and can select an uncertain last copy. Removing that bonus was a principled correction but did not improve the repeated dev deal.
- Two dev runs on one seed scored 22 with all lives and no errors; this confirms robustness of the transport and protocol, not generalization or a score increase.
- Preflight and synthetic contexts must exercise queued model timeouts and malformed choices because the evaluator dry run commonly answers no model call.

### Candidate 0009 — discard signals and full-token play packets

- A discard index can carry a preagreed safe-play promise while restoring a clue. Reserve a public neutral slot, and forbid private dead-card overrides on mapped symbols unless the encoded partner play is true; otherwise an innocent discard can induce a misplay.
- Channel availability is part of the strategy. The first dev run sent only one discard signal and scored 19 because tokens repeatedly reached eight, where discard is illegal. A complementary full-token rank-play/color-literal mode raised that same deal to 21 and produced six discard signals plus ten full-token play clues.
- All six final-dev discard signals were followed by successful plays, but one discarded the last G4 to signal G2. Public chop saves protect only the neutral slot; signaling creates additional discard threats. Lower clue pressure alone does not prevent lost ceilings.
- Audit the decoder with both hands' faces erased, then replay its persistent state from public events. The final implementation passed 139 archived observations and reconstructed all 17 promises in its final dev trace; all inferred domains contained the later-observed actual faces.
- Final dev completed 21/25 with all lives and no error, below parent 0006's archived 23-point dev result. Preserve that negative evidence: the channel works but its net scoring gain is unproven. Both allowed dev evaluations are used; model queue timeout fallback and separate seat sessions passed fake-context tests.


### Candidate 0007 — posterior rollout search over the leader protocol

- A simulator can be validated against archived before/after observations without reading engine source: 278 nonterminal transitions matched, including masks, legal actions and countdowns; separate fixtures cover immediate terminal cases.
- Sample own hands from the joint literal posterior, then shuffle only the remaining unseen multiset. Every simulated actor must receive a freshly redacted view; a root determinization is not permission to choose future actions from full hands.
- The first rollout dev episode scored 19 with all lives and still lost the last W3 and R3. Search over a weak continuation policy and a literal-only posterior is not automatically stronger than the parent.
- Comparing the maximum of many 24-sample means caused small-margin action changes. The final refinement uses up to 96 common worlds, all protocol-valid options, and a paired uncertainty margin before overriding the parent.
- Model queue timeout remains separate from backend timeout; the bounded adviser failed safely and the game finished. Synthetic tests exercise this and garbage responses because preflight may answer no model calls.

### Candidate 0008 — discard-index communication

- A discard index can carry a preagreed play promise while restoring a clue and drawing. The sender validates the partner's visible card; the receiver reconstructs the promise from public masks/counts, slot order and the accepted discard event. The discarded card's hidden face is never needed for decoding.
- Measure transmitted messages, not just legal codebook coverage. The first dev version offered valid signals in archive audits but sent zero in play: two neutral alternatives plus a public near-cost filter removed most symbols, and signals were scored below equivalent paid clues. Its 22-point result did not test the intended benefit.
- Separating alphabet coverage from private action valuation made the mechanism usable: keep one lowest-cost public discard neutral, reserve other indices, and price each signal using the sender's own discard belief. The second dev run sent ten signals, all followed by successful plays; this deliberately trades away some ordinary discard freedom.
- Free play signals do not automatically save total clue turns. Paid play clues fell from 13 to seven, but literal/save clues rose from 11 to 16, so total clues only fell from 24 to 23. Both dev versions scored 22; the revised version lost no live last copies but ended at R5/Y4/G3/W5/B5. Physical preservation and executable tempo remain distinct objectives.
- Face-erasure checks covered 268 parent observations and 168 offered signals; stateful replay of the final dev's 65 actions preserved all public promises. Ten actual signals also exercised persistence and shifting slots: one promise waited three turns and still resolved safely. Explicit fake contexts covered malformed replies, model/queue failures, and separate seat sessions because preflight answered no model calls.

Candidate 0007 follow-up: the final uncertainty-controlled search scored **22/25** on the second permitted dev episode, 70 actions, all lives, no error, 311 seconds. It recommended seven deviations from the parent and lost no lower-rank last copy, but discarded W5 and Y5. This paired 19-to-22 improvement supports the refinement on this deal only; sampling-error estimates do not measure literal-posterior or continuation-policy bias. Both dev evaluations are used.

### Candidate 0011 — turn-access weighting

- Safe communication and complete lives can still finish below the physical ceiling when prerequisite chains are exposed too late; score resource preservation and turn access separately.
- A finite final-round horizon is available in every observation. Discounting clues with no consumable target near that horizon is a structural test distinct from adding another clue convention.
- Keep public protocol state face-erased and seat-local even when adding horizon features; only stacks, deck size, countdown and public masks are shared.
- Model output must remain advisory: deterministic candidate ranking and legal-menu selection make arbitrary preflight replies and caught timeouts harmless.

### Candidate 0012 — inverse clue-choice constraints

- Making the clue encoder deterministic turns unchosen higher-valued commands into usable negative evidence: either their encoded slot is not better, or their clue attribute is absent. The encoder must depend only on the recipient hand and public board so a receiver can reproduce this reasoning.
- Branching on the chosen slot identity and propagating those implications gives sound support sets without enumerating full hands. Unioning branches drops correlations; safe support-based plays remain justified, but posterior probabilities are only exact conditional on the relaxed unary supports.
- Face-erasure audits passed on 819 archived observations; 400 small exhaustive fixtures retained all 1,237 encoder-compatible worlds. More than two thousand identity exclusions occurred outside the commanded slot, confirming that the mechanism carries more than a one-slot promise.
- First dev completed 24/25 with all lives and no live last-copy losses. Fixed-history ablation found one additional certain play: Y2 at turn 62 was only 50% playable under literal masks plus ordinary rank certificates. This is mechanistic evidence, not a counterfactual score or generalization claim.
- Preserving cards and improving inference still does not guarantee perfect play: G5 remained held after G4 used its seat's final action. Queue-timeout fallback also remains necessary; the sole Astra request timed out at the caller, after which automatic play finished without an episode error.


### Candidate 0010 — exact paired rollouts over the current leader

- An exact subset-partition inclusion-exclusion formula made without-replacement hand marginals roughly four times faster on an initial-state benchmark. It matched the parent's matching DP on 500 randomized inputs and preserved all 258 archived candidate menus; full exact continuation beliefs were practical without an independent-card approximation.
- A simulated public pre-action snapshot must own its discard list. Appending a discard into an aliased list silently changes the counts used to decode that very discard signal. Immutable-snapshot and 254 archived-transition checks caught and excluded this failure mode.
- Sample own hands conditional on both literal masks and persistent public promises; carry only that public promise state between simulated seats. Face-erasure, conservation, replay and 1,254 sampled continuation steps confirmed truthful messages and correct certain plays.
- The dev episode improved the parent's archived 22 to 24, completing both yellow and green chains, with 63 actions, all lives and no error. Search changed five preparation-clue decisions. This supports testing terminal-score comparisons over the stronger continuation policy, but remains evidence from only one dev deal.
- R5 was still discarded. The preceding save clue had estimated gain 0.458 but paired standard error 0.376 over 24 samples, so the confidence margin rejected it. More focused sampling around critical saves is a plausible follow-up; lowering the margin after seeing the true outcome would confuse variance with evidence. Dominant-safe-play search skipping can also miss preventive saves.
- Preflight's low budget skipped search/advice, so explicit fake contexts and observation audits were necessary. The sole dev adviser request timed out at the caller, later completed, and was ignored; backend success/latency metrics alone do not show whether an answer informed an action. One permitted dev evaluation was used.

### Candidate 0015 — risk-gated discard communication
- Leader 0010's paired means still discarded R5 at seed 14 turn 53: a 0.417 critical posterior was hidden by a 0.5 dead posterior and sampling noise.
- Free discard signals can erase a unique card while delivering a valid partner play; gate the signal itself while retaining ordinary discard fallback.
- A sender can protect the recipient's predicted public chop with a truthful rank save when no safe recipient play exists; the sender never needs the recipient's private faces.
- Preflight does not answer Astra calls, so malformed-output and timeout branches remain separate from the contract smoke test.

### Candidate 0014 — endgame execution scheduler
- The leader's residual losses point to turn access: preserving a card is insufficient when its prerequisite chain is not scheduled before the final draw.
- A phase change at `final_turns_remaining` or very small deck size is a distinct test from changing clue scores; it prioritizes certain plays and only then actionable clues.
- Endgame rules must still use current slot indices and seat-local literal beliefs; public horizon facts alone cannot justify hidden-card plays.
- Keep the parent rollout and timeout fallbacks intact so arbitrary model output and queued calls remain harmless during preflight.

### Candidate 0013 — late-horizon risk guard
- Candidate 0010's seed-14 trace shows a concrete terminal failure: at turn 66, deck size 3, a sampled rollout selected a 50% PLAY B3; the action misplayed, while lives were otherwise intact.
- Paired terminal search is useful early, but late marginal plays are especially sensitive to posterior/model mismatch because only a few draws remain; a legal discard or clue can preserve the final round.
- Add horizon guards from public `deck_size`/`final_turns_remaining` and literal posterior `p_play`; keep certain plays automatic and require a higher success probability for speculative plays near exhaustion.
- Do not treat preflight's arbitrary model text as strategy evidence: deterministic candidate ranking and legal fallback must survive zero model calls.

### Candidate 0016 — late live-five discard guard

- Candidate 0013's horizon guard addressed marginal late plays but left ordinary discard scoring untouched; seed 14 still discarded R5 at turn 53 with eleven cards before the draw and `p_critical=0.417`.
- Rank 5 deserves a separate posterior: each live copy is unique, so discarding it permanently removes that color's final point even when the slot is not fully identified.
- The 0016 change is deliberately narrow: only deck size <=12 and live-five posterior >=0.15 are penalized; early communication, masked rollouts, and legal fallbacks remain inherited from 0013.
- Candidate traces show that preserving physical cards and scheduling their prerequisites are separate failure modes; a discard guard should be evaluated alongside final-round access, not as a replacement for it.

### Candidate 0017 — actionable-clue scheduler correction

- Parent 0014's short-horizon scheduler promotes every clue, even when it creates no safe slot or save. Replay at seed-14 turns 63 and 65 finds certainly dead DISCARD 4 (value 1.702) displaced by unproductive clues (-0.55 and -0.63).
- A clue's action type is insufficient to justify endgame priority: require a new executable partner play, no existing certain partner move, and a remaining recipient turn. Other clues still compete by the inherited scalar and rollout values.
- Copying a candidate dictionary and adding a temporary priority key breaks equality-based deduplication. Compare action dictionaries when removing promoted entries from the remainder.
- This explores access scheduling on parent 0014; candidate 0016 separately explores likely-five discard risk on the leader. Local replay proves ranking changes, not counterfactual score gains.

### Candidate 0018 — exact public-factor support, conservative use

- A deterministic clue encoder can be audited as a public factor: exhaustive small-hand reconstruction matched all retained supports, while face-erased inputs preserved the decoder output.
- Full weighted conditioning on the encoder's choices is risky because the harness does not model every strategic selection effect; the first dev run fell from the parent's 24 to 21 after a modest posterior shift changed a discard.
- A safer use is all-or-nothing joint support closure followed by the existing multiplicity prior. This can prove a card critical without treating policy-conditioned relative frequencies as calibrated probabilities.
- Historical factors must retain the original hand length and slot rotation after earlier cards leave the hand; using the current length silently changes which identity the old clue addressed.
- Bounded enumeration must return no estimate on node/time exhaustion and fall back to the parent's approximation; preflight and explicit audits exercised this path.
- The two dev runs both scored 21, so exact public inference is a correctness and hypothesis test rather than evidence of a score improvement on this deal.
