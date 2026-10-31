parent: 0008

## Hypothesis

The current leader preserves cards and communicates safely, but still strands
reachable cards because it values a safe local play or clue more than the number
of turns available to consume the resulting chain. The failure is a **time/access
bottleneck**, rather than another inference error: the deck can be exhausted with
all lives intact while a prerequisite is only exposed on the last turn. A small
lookahead based on the exact remaining turn horizon should trade a little local
discard value for earlier prerequisite access and improve terminal score.

## Evidence from the parent

Candidate 0008 averaged 22.75, with all four training episodes completing safely
and no life loss. Its final observations nevertheless left incomplete fireworks
(for example seed 11 ended W3; seed 12 ended B3) while clue and life resources
remained. The parent proposal records a second-dev episode ending at 21 with
the physical ceiling already 21: last B4 was discarded at turn 22 and last G4
at turn 43. Those are access/order failures after the transport and discard
protocol worked. The same report notes that six free discard signals succeeded,
so protocol correctness alone did not solve late dependency access.

## Change from 0008

The exact copy-weighted hand marginals, stable-ID promises, public face-erased
decoder, legal-action validation, and seat-isolated Astra sessions are retained.
The action scorer now computes an `access` value for each candidate identity: the
number of prerequisite links that can still be executed before the final-round
counter (or a conservative deck-based horizon before it starts). Certain plays
that unlock visible partner prerequisites receive this value; discard signals
receive only a smaller bonus when their target can still be consumed. Clues with
no newly safe target are penalized once the horizon is short. Final-round clues
are accepted only when their newly enabled play has at least two turns left.
The model is advisory only when deterministic candidates are close; malformed,
slow, or unavailable replies fall back to the validated top candidate.

## Confirmation or refutation

The hypothesis is supported if the harness completes more colors before the deck
empties, reduces incomplete prerequisite chains in the final two turns, and keeps
the parent’s zero-misplay behavior. It is refuted if access weighting causes
valuable critical cards to be discarded, lowers clue quality while ample turns
remain, or fails to increase score despite earlier prerequisite plays. Traces
should log final stacks, remaining deck/horizon, and any access-weighted choice;
no hidden deck order or seed is used.

No training or held-out evaluation was run by the proposer. Preflight is the
required validation; this candidate is intended for the evaluator’s training
seeds.
