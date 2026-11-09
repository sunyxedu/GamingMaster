parent: 0000

## Hypothesis

The zero score is caused primarily by an unverified control loop rather than lack of
TextWorld knowledge. The parent gives Astra one fresh call per turn, but it accepts a
repeated answer forever. The traces show `drop key` repeated nine times on seed 100,
`examine table` thirteen times on seed 103, and `examine shelf` twenty-five times on
seed 108 while score and room state stayed unchanged. With roughly 50 seconds per call,
this burns the 1800 second deadline before exploration or prerequisite completion.

## What changed vs. the parent

This harness asks Astra for a short 1–6 command plan with the full current admissible
list and a compact, observation-backed action history. It resolves every planned command
against the *current* exact names, executes one command, and checks `ctx.progress()` plus
the public observation signature. A rejection, no-effect action, stale command, or
terminal event immediately invalidates the remaining plan and forces replanning with the
evidence. A bounded fallback chooses a non-destructive unseen admissible interaction when
the model returns garbage or its budget is exhausted. Destructive actions are deferred
unless the literal objective supports them. Model timeout/runtime failures are caught so
an episode can continue or end cleanly.

## Evidence that would confirm or refute it

Confirmation would be nonzero scores on seeds where the parent loops, fewer repeated
actions, and traces showing replans after rejected/no-effect commands while preserving
the exact action strings. Refutation would be zero scores despite diverse, state-changing
actions and a plan that reaches relevant rooms, indicating that the remaining problem is
objective interpretation or prerequisite planning rather than stale-loop control.
