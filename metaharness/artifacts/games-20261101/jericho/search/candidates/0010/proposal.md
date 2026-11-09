parent: 0005

## Hypothesis

0005 demonstrated that retaining a short route across positive score changes can
save expensive Astra calls, but its traces expose a second failure: when a route
reaches a zero reward room, Astra can alternate actions while the textual state
does not change. In the enchanter seed 2 trace, `west`, `look`, `west`, `look`,
and `east` repeatedly return the same Long Road prose and score. The existing
command/response pair guard misses this because the commands alternate. A
continuation session also makes that loop more expensive as hidden context grows.

## What changed vs. the parent

0010 retains 0005's reward-preserving short routes, but starts every planning
call with a fresh Astra session and a bounded packet containing the opening text,
recent exact transcript, failed actions, score, and provisional beliefs. Code
maintains a normalized response-plus-score state counter in addition to an exact
command pair counter. Three visits to a state, or two identical command/state
pairs, immediately cancel the suffix and force a new hypothesis. The prompt
explicitly asks the new call to break repeated no-reward movement/`look` cycles
with an untried salient clue, object, or exit. Parser failures, hazards,
clarification/menu text, score losses, and terminal observations still cut a
route. Output decoding and all model/action failures remain total for arbitrary
pre-flight model text.

## Evidence that would confirm or refute it

Confirmation would show fewer alternating movement loops, more distinct
command/state signatures, and higher worst-seed native score while preserving
0005's actions-per-call advantage. Traces should show a cycle-triggered replan
with the exact repeated prose in its evidence packet. Refutation would be lower
scores because the broader state guard cuts a useful maze traversal, or repeated
cycles after replanning despite fresh contexts; that would indicate the model
needs richer object/map extraction rather than a recurrence guard.
