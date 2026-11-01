# Candidate 0003: release each joystick hop

## Parent and hypothesis

Parent: `candidates/0000-tier2_jev_only/` (the current best, mean score 6.0).
I keep its Jev-only, one-choice-per-decision architecture and change only the
answer-to-action bridge.

The hypothesis is that the parent is not executing Jev's action as one Frogger
hop.  In this real-time interface, a tick without a fresh post repeats the last
posted action.  The parent posts a directional answer and then waits for the
next Jev answer, so Jev's roughly 0.15 s latency holds UP/LEFT/RIGHT for about
9 frames.  The study explicitly warns that games 1--4 need a NOOP debounce.

The worst parent episode gives direct evidence.  In
`trace_seed103.jsonl`, tick 13 is a fresh `UP`, and ticks 14--21 are all
`{"action":"UP","fresh":false}` before another fresh `UP` at tick 22.
The episode has 3026 ticks but only 323 Jev calls (2704 stale ticks); its
observations show score 2 with one death at tick 1262, score 5 with four
deaths and no reserves at tick 2840, and it ends at score 5 with
`last_action: "UP"`.  Across the parent evaluation every episode has four
deaths and zero frogs home, despite 5--7 points.  This is consistent with a
latched joystick driving repeated blind hops rather than with Jev merely
choosing too few actions.

## Targeted change

`harness.py` still sends the complete current observation to one Jev Choice
question and asks again after each result.  For a directional result it now:

1. posts that direction;
2. waits for exactly one live game tick, so the hop is actually applied; and
3. immediately posts the game's NOOP action before starting the next Jev call.

NOOP answers are posted normally and are not followed by an extra wait.  Thus
Jev's latency can still produce stale ticks, but those ticks repeat NOOP rather
than a dangerous directional press.  No fixed route, seed rule, decoder, or
new model call is added.

A result is accepted only if its `.value` is one of `action_spec.names`; a
missing or malformed stand-in answer maps to NOOP.  Timeout and ordinary model
failure leave the last released action in place and wait one tick before
retrying.  Budget/model-unavailable exceptions retain the contract's normal
propagation behavior.

## Jev interface and control policy

The state is the exact JSON object returned by `game.observe()` (including
raw RAM, decoded frog/score/life fields, timer, mode/difficulty, frame
counters, and the last action).  The sole question is:

- `act`: a `Choice` over `game.action_spec.names`, using the adaptor's action
  descriptions.  It asks for the single next-frame action, and explains that
  a directional action is released after one frame.

There are no numeric thresholds or inferred probabilities in this targeted
variant.  The typed choice is used only when its value is legal; every other
answer, a late failed call, or an unavailable answer falls back to NOOP.  The
freshness rule is deliberate: an answer is never reused as a directional
command.  The only action held across Jev latency is NOOP.  Jev's confidence
is not used because the parent did not gate it; the safe consequence of
uncertainty is supplied structurally by release/debounce.

## Astra policy

This candidate deliberately makes zero Astra calls so the debounce hypothesis
is isolated; there is no return-to-Astra event in this experiment.  If release
is confirmed by the trace but score remains at the parent floor, the next
iteration should have Astra redesign the Jev state/question around hazards and
position.  If directional stale ticks remain, the action bridge—not Astra's
interface—still needs correction first.

## Confirmation or refutation

The hypothesis is supported if this candidate reduces deaths and reaches at
least one home, or materially raises score, while the trace shows directional
posts followed by one applied frame and then NOOP.  It is refuted if the
candidate still ends with repeated directional stale ticks, or if deaths and
zero-home outcomes remain essentially unchanged despite the release pattern.
A score decrease with fewer deaths would suggest that the parent needed a
better state/question or more timely decisions, not that debouncing was the
wrong semantic fix.
