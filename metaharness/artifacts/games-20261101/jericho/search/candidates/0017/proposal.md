parent: 0011

## Hypothesis

The leader's main remaining failure is a model-side self-loop after a
prerequisite has been disproved. In the 0011 Enchanter seed-2 trace, the
parser says that `rezrov` is not committed to memory, yet Astra keeps returning
plans such as `['rezrov gate', 'open gate', 'east']` (the same route is emitted
several times near 1494--1701 seconds). The controller then spends another
slow call instead of changing the question. Library seed 11 shows the same
shape late in the episode: after `enter door`, Astra successively proposes
`go through wooden door`, `west`, and `go through door` before finally finding
the book at step 37. A repeated-plan circuit breaker should preserve time for
an observed alternative or stop cleanly.

## What changed vs. the parent

0017 retains 0011's repaired `command` transport, fresh bounded evidence
packets, directed frontier, total output decoder, 75-second call cap, and
parser/hazard/clarification interrupts. It records the normalized command
sequence returned by each Astra call and counts identical plans only when no
action occurred between calls. The third identical answer activates a
two-turn, code-only cooldown: try an exit already extracted from the current
observation, then at most one `look` or `inventory`, and return if there is no
observed frontier. An attempted edge also grants one local fallback chance.
This tests plan repetition directly rather than changing the route planner or
adding title-specific commands.

## Evidence that would confirm or refute it

Confirmation would show `plan_repeat_guard` events around blocked puzzles,
fewer late model calls and repeated commands, and equal or higher final native
score on episodes like Enchanter seed 2 without more parser failures or deaths.
The local fallback should produce a new observed exit when one exists.

Refutation would be a score drop because the same command sequence is often a
legitimate route after a state change, or because the frontier is empty at the
moment a fresh Astra call would have discovered an unusual puzzle verb. Traces
would show guards followed by useful missed actions, or no guards despite the
same-plan loops; then this threshold should be removed or made state-aware.
