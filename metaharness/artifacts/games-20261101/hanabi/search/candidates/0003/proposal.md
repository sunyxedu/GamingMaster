parent: none

# Hypothesis

The naive baseline is bottlenecked by expensive, unreliable action-by-action linguistic communication. A mechanically decoded, preagreed instruction channel can turn a truthful clue into a precise immediate action, including a card that the clue does not touch. This should preserve hidden information while making safe plays and low-loss discards executable without another model call. This is a fresh structural design, not a refinement of the baseline prompt.

# Evidence from the archive

In `0000-tier1_astra_naive/trace_seed12.jsonl`, step 3 sends `REVEAL_COLOR 1` and step 4 sends `REVEAL_COLOR`; both observations report `illegal or malformed action; state unchanged` and remain at turn 2. The episode ends at score 9 after 31 posted actions, 33 model calls and 1800 seconds; one call times out at 300 seconds. Seed 14's last step has score 9, deck_size 23, three lives, and valid turn 29: it has not exhausted its opportunities to score. The baseline code also reuses a single session across both observers, contrary to the seat-isolation contract. These results cannot establish that cautious Hanabi itself is capped near ten points.

# New mechanism

Both seats preagree on a public rotating codebook. A rank clue commands PLAY at `(rank - 1 + clue_turn // 2) % hand_length`. A color clue recommends DISCARD at `(color_ordinal + clue_turn // 2) % hand_length`, with colors ordered R,Y,G,W,B. The recipient decodes the previous public history event, not any private sender memory. Rank messages are offered only when that precise visible recipient card is currently playable. Color messages are scored by the actual visible discard loss; own scoring plays and verified play clues for the partner override them. Rotation avoids permanently inaccessible slots when some attributes are absent. All clues must still be truthful and selected from legal_actions.

Literal deductions remain independently useful. A subset dynamic program computes exact joint own-hand marginals without replacement from card multiplicities, positive and negative clue masks, public played/discarded cards and the acting seat's visible partner hand. Public missing prerequisites define dead cards and reachable ceilings. Probabilities are labelled literal-mask beliefs, not exact beliefs conditional on the signaling policy. No unseen faces or private plans are shared. Only public history encodes instructions; each seat has its own model session.

Code executes instructions and guaranteed plays, scores safe signaling opportunities and discard loss, and handles the final countdown. Astra arbitrates at most six ambiguous strategic choices per episode, from a prevalidated list, with 90-second timeouts and a deterministic fallback for garbage output or failure. Automatic actions continue after the advisory allowance. There is no open-loop cross-seat plan, no engine access, no seed input, and no model-generated command parsing.

# Predictions and falsification

Confirming evidence: natural deck exhaustion or perfect completion, zero instruction-induced misplays, much fewer model calls, no rejected commands, and scores above the baseline. Refuting evidence: low legal-symbol coverage strands playable cards; repeated color discard messages burn too much deck without progress; last-copy losses occur when zero-token turns force blind discards; or advisory calls add latency without improving choices. Log each chosen action and whether it came from a decoded message, literal deduction, or Astra to make these mechanisms distinguishable.

# Validation

Implementation was written before experimentation. Pre-flight passed, and the two permitted dev evaluations test the initial protocol and its targeted refinement on the declared dev seed. No training or held-out evaluation is run by this proposer.

## First dev result and targeted refinement

Dev seed 201 (`dev/20261029-232134`) completed naturally at **19/25**, 74 actions, four System-2 calls, 473 seconds, three lives and no episode error. Two 90-second adviser timeouts were caught; automatic play continued. The trace confirms an instruction-role bottleneck: seat 0 gave repeated color/discard messages while its own hand remained untouched until turn 40. The rank codebook itself caused no misplays.

The final decoder therefore treats a color instruction as a recommendation that yields to an own scoring play **or a verified rank/play signal to the partner**. This is a public priority rule, not information sharing. It allows the recipient to return initiative without needing to guess or discard first. Rank commands still execute immediately. Discard recommendations expire if overridden; no private sender plan is transferred or silently retained. The final-round risk rule can also supersede a discard when its expected score gain is positive.

Local validation passed: pre-flight before and after the refinement (both natural completion; 22-point smoke-test scores), 100 random three-slot exhaustive checks of the belief DP, legal/safe rank-message checks on archived observations, public-only encode/decode consistency, and two-seat garbage-response/timeout recovery with separate initial sessions. The pre-flight made zero model calls, so the explicit fake-context tests cover the otherwise untested error paths.

## Service-queue finding and final runtime guard

The second dev trace (`dev/20261029-233240`) confirmed the role-handoff improvement: after the first color/discard recommendation, the recipient sent a safe rank clue back and the original giver played on turn 4. It reached 17 points with eight cards left and all three lives, then its sixth advisory request remained pending for more than eight minutes despite `timeout_s=90`. Reported backend latency does not include all service queueing, so a backend timeout alone cannot guarantee episode completion.

The final harness calls Astra through `ctx.spawn` and waits on its future for at most 90 real seconds. Any advisory failure disables further advisory calls for the episode. The automatic policy then finishes; at most one outstanding model call may remain, which the evaluator cancels when the episode ends as specified by the contract. There is no custom background simulation or unbounded worker loop. This is a runtime fix motivated by the measured queue delay, independent of card strategy. The final guard passed pre-flight and explicit completed-future, malformed-output, model-timeout and queued-future-timeout tests. No third dev run was launched; both permitted dev runs preceded this guard.
