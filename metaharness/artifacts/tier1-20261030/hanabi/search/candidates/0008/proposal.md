parent: 0001

# Hypothesis

Prior policies spend clue tokens to communicate but discard without using the
choice of discard slot as a communication channel. A publicly agreed discard
codebook can restore a token, draw a card, and enable a guaranteed partner play
in one action. This should reduce clue pressure and reach prerequisites sooner,
without adding speculative plays. This is an action-choice communication
experiment, not another critical-save or rotating clue-codebook refinement.

# Archive evidence

Leader 0003's weakest training trace, seed 14, ends at 20 with all three lives.
At turn 31 it discards W5/card 1; at turn 37 it discards the last G3/card 25 with
zero clues before the action. Its logged loss estimates are respectively
`own discard E_loss=0.542, p_dead=0.286` and
`own discard E_loss=0.605, p_dead=0.349`. The first G3 was discarded at turn 26.
These are permanent lost ceilings, rather than model or parser failures.
Near the end it spends turns 63, 65 and 67 on color/discard messages; the last
draw comes on turn 70, leaving only Y3 and Y4 playable in the final round.
Thus communication also consumes opportunities to expose and execute chains.

Parent 0001 scores 21.25 mean and keeps much more literal information than the
leader's immediate instruction scheme. Its weakest game loses the last R3 at
turn 39 and ends at 18 with three lives and four clues. Candidates 0005 and 0006
already tested stronger saves/public discard commitments but scored 19.75 and
20.25. This proposal tests using action indices for information, rather than
assuming more saving will improve the same failure traces.

# Changes from parent

Keep exact without-replacement hand beliefs, literal masks, color focus play
promises, rank literal/save clues, stable IDs, bounded Astra calls, robust legal
action selection, and separate seat sessions. Add a public discard codebook:

- Compute both hands' public-only beliefs, without subtracting visible hands.
- Keep the best public discard alternative neutral. Reserve the remaining
  slots as message symbols; use private discard cost when choosing a message.
- Map reserved slots to partner slots which are publicly possibly playable,
  rotating by public turn number. If the partner already has a publicly certain
  play, or no subsequent turn, disable the channel and allow ordinary discards.
- A reserved discard is permitted only when its mapped partner card is actually
  playable in the sender's permitted view. Its value includes the enabled play.
  A neutral slot always remains available when discarding is legal.
- After an accepted discard, reconstruct the promise from the public event and
  a face-erased pre-action snapshot. Store only the possible identities playable
  at that time. Recompute beliefs and action indices on the recipient's turn.

The policy deliberately sacrifices freedom over discard slots to communicate.
The lowest-cost public neutral alternative guarantees an ordinary fallback.
This is not an information leak: every receiver deduction is reproducible from
public actions, public masks/counts, and the preagreed protocol. Private beliefs
only select among valid messages; no private face enters the public tracker.

# Confirmation and refutation

Confirm if discard signals enable safe plays with fewer paid play clues, reduce
zero-clue stalls, and improve completed-game scores. Refute if channel coverage
is too sparse, reserving slots increases last-copy losses, or additional deck
consumption strands held chains sooner. Log each transmitted/received signal,
its target stable ID, discard loss, and resulting public promise. A signal that
misplays refutes protocol correctness regardless of score.

# Validation

Harness and proposal were written before experiments. Preflight, face-erasure
audits, sender/receiver consistency tests, and up to two permitted dev evaluations
will be recorded below. No full training or held-out evaluation is run here.

## First dev result and channel-coverage refinement

The first version preserved two neutral discards and restricted message symbols
to slots within 0.35 of the lowest public cost. Preflight passed (natural ending,
22 points, 67 actions; a contract smoke test, not a performance estimate).
An observation-only audit of 268 parent turns verified face-erasure invariance,
38 safe offered discard signals, and 1,307 matching clue decodings. Explicit
fake contexts covered garbage/out-of-range replies, model and queue failures,
and separate seat sessions.

The first dev episode, `dev/20261030-004131`, finished at **22/25**, 65 actions,
all three lives, three System-2 requests, 264 seconds, and no episode error.
One request reached the caught outer timeout. Crucially it transmitted **zero
discard signals**, so the result does not support the communication hypothesis.
At turn 7 a valid signal scored 2.113 versus 2.260 for a paid rank clue; after
turn 25 the public near-equivalence filter frequently removed every signal slot.
The game ended with G3 held and both G4s discarded (turns 4 and 51).

The final version separates encoding coverage from private action valuation:
one lowest-cost public discard is neutral, all other slots are symbols, and the
sender's private loss estimate prices whether to send one. A newly enabled play
gets the same 2.5 marginal value used by the existing clue scorer. Every symbol
still requires an actually playable target. This tests the original mechanism
with a usable alphabet, while explicitly exposing the cost of restricting
ordinary discards. No additional communication convention is introduced.

## Final validation and second dev result

The final preflight passed with natural completion, 63 actions, eight decoded
discard signals, and no errors (23-point smoke-test score). The
expanded observation audit offered **168** safe signals across 268 archived
turns, with the same 1,307 matching clue decodings. Face erasure left every
codebook unchanged. Model garbage, invalid choices, exceptions, queue timeouts,
and separate initial/resumed seat sessions all passed explicit fake contexts.

The second and final allowed dev run, `dev/20261030-004932`, completed at
**22/25**, 65 actions, three lives, three System-2 calls, 203 seconds, and no
episode error. It sent **ten discard signals**, at turns 3, 6, 9, 30, 32, 36,
42, 58, 61 and 62. All ten target cards were subsequently played successfully;
nine on the next turn, and the turn-61 promise on turn 64 after another signal
took priority. There were no misplays and **zero live last-copy discards**.
A stateful replay of all 65 actions independently reconstructed every promise
from face-erased snapshots and public events, verified every action belonged to
the validated candidate menu, and checked all visible promised cards against
their public domains.

Paid play clues fell from 13 to seven, but literal/save clues rose from 11 to
16: total clues only fell from 24 to 23. Final stacks were R5/Y4/G3/W5/B5, versus
R5/Y5/G2/W5/B5 in the first run. The signal channel works and preserves useful
cards on this deal, but the unchanged score **does not establish a performance
improvement**. Excess literal/filler clues and late dependency access remain
plausible limits. The neutral-slot restriction may also worsen private discard
choices on other deals. This candidate exposes that tradeoff to training
evaluation; it does not claim the dev seed predicts the training mean.

No decision-policy changes followed this run. Both permitted dev evaluations
are used, no full evaluation was launched, and all policy imports are from the
standard library. The written files and observation-only checks are ready for
the evaluator.
