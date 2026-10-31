parent: 0001

# Hypothesis

The remaining score loss is partly a coordination failure in discard prediction. A sender cannot reliably save a threatened prerequisite if its prediction uses public beliefs but the receiver selects a discard using private card exclusions. Make the discard policy an explicit public commitment, then evaluate saves by whether they actually move that commitment to a lower-loss visible card. This tests a different mechanism from leader 0003's rotating instruction channel.

# Archive evidence

Leader 0003's worst training episode (`trace_seed14.jsonl`) finishes at 20 with three lives. At turn 37, with zero clues, it discards card 25, the last G3, while green is at 1. Its logged estimate is `own discard E_loss=0.605, p_dead=0.349`. This permanently removes three potential green points. At turn 31 it also discards the unique W5 (`E_loss=0.542, p_dead=0.286`). These are real lost resources, not model latency or transport failures.

Parent 0001 finishes the same seed at 18 with three lives and four clues. Its turn-39 discard removes card 28, the last R3, at red height 0 and one clue. The code computes `predicted` from partner public beliefs, but computes its own actual discard from private beliefs and allows every slot. Thus save urgency is based on a different decision rule from the recipient's. Its final Y2 draw also arrives too late to finish the yellow chain, showing that saving alone cannot solve every missing point.

Replaying all four archived parent episodes found **28 disagreements among 81 discards** between the public predictor and the actual discard. In seed 12, turn 8 predicts slot 1 but discards slot 4 (unique B5), and turn 31 predicts slot 2 but discards slot 4 (last G4). Both occur with four clues. This is direct evidence for the proposed coordination failure, rather than attributing every prerequisite loss to it.

# Changes

Retain the parent's exact finite-copy hand inference, direct color-focus play promises, rank save clues, legal dictionaries, and bounded seat-isolated Astra adviser. Replace unconstrained private discard selection with a public discard commitment based on public remaining counts, literal/protocol domains, protection and slot order. Private beliefs still inform plays and whether to clue instead of discard; they cannot silently change which card a discard removes.

For every truthful protocol-valid clue, calculate the recipient's public discard before and after the clue. The sender may use its legitimately visible target faces to value the consequences. Give urgent saves priority when they redirect a threatened unique prerequisite to a less costly discard and the recipient has no publicly guaranteed play. Protection is useful only to the extent that it changes the subsequent action. Public commitments are recomputed after every event and identified by stable card IDs; no private target faces enter shared memory.

# Confirmation and refutation

Confirm if predicted and executed discard indices agree, saves reliably redirect endangered last copies, critical prerequisite losses decrease, and completed-game scores improve without increasing misplays. Refute if public-only discard choices lose more value than accurate saves recover, protection congests hands, or clues run out before a save can occur. Log before/after discard predictions and visible loss improvements so later proposals can distinguish these mechanisms. No training seeds, hidden deck, or engine internals are policy inputs.

# Validation

Implementation and proposal written before experimentation. Preflight, focused protocol tests, and at most two allowed dev evaluations will be recorded below. No full training or held-out evaluation will be run by this proposer.

## First dev result and targeted refinement

The first dev episode (`dev/20261030-002446`, seed 201) finished normally at **21/25**, 69 actions, three lives, one System-2 request and no episode error. The request hit the caught outer caller timeout; automatic play completed in 121 seconds total. No unique reachable card was discarded. Logged saves at turns 34, 36, 38, 60 and 64 redirected the public chop or enabled its play. This supports the mechanism but a single dev deal is not a measured improvement over the parent's training mean.

The remaining loss revealed an overly narrow urgency definition. At turn 52, own B5 takes priority over saving visible Y1/card 37 on the partner's exactly predicted chop. The partner discards that Y1 on turn 53, and the next accessible Y1 is not played until turn 67. The final fireworks are R5/Y1/G5/W5/B5. A rank-1 clue at turn 52 would protect the playable chop and move the next discard to a lower-loss card; it also removes the rank-1 possibility from older yellow cards that obstruct the color-focus protocol.

The final refinement therefore treats a currently playable threatened chop as urgent even when spare copies exist. It still requires a clue to enable that card's play or move the public discard to a no-more-costly card. If a certain own play would make that exact duplicate obsolete, the rescue receives no extra urgency. This is an extension of the public commitment hypothesis, not a new communication convention.

## Final validation and result

The second and final permitted dev episode (`dev/20261030-002831`, seed 201) completed at **24/25**, 68 actions, three lives, two clues, one System-2 request, 121 seconds, and no episode error. Again the caller timeout was caught and the automatic policy completed the game. No unique card was discarded. The playable-chop rescue sends a rank-1 clue at turn 48 in this trajectory, and yellow reaches 5. Final fireworks are R5/Y5/G4/W5/B5: G3 is played on the final draw, and the two remaining actions clue then play G4, leaving no turn for G5.

This paired dev improvement from 21 to 24 supports retaining the refinement. It does not establish an improvement over 0001 or 0003 on training or unseen deals. The principal remaining risks are the lost benefit of private discard exclusions, protection congestion, and insufficient turns for late dependencies. These are deliberately exposed tradeoffs of the public commitment experiment.

Final preflight passed with natural game completion (21-point stand-in smoke test, 66 actions, zero model calls). Six focused tests passed: public discard invariance when all hand faces are erased; binding candidate discards and truthful color promises; exact agreement between predicted and recipient-recomputed post-clue discard slots; malformed/out-of-range replies, model exceptions, queue timeout, and separate seat sessions; a synthetic threatened last-R2 save; and the first dev trace's playable-chop rescue regression. The invariant checks cover archived observations from both strong candidates. The harness imports only standard-library modules and resets promises and per-seat sessions on every episode. Both allowed dev evaluations are used; no full evaluation was launched.
