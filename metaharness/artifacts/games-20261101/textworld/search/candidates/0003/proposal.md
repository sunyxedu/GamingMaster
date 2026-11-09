parent: 0000

## Hypothesis

The incumbent's zero score is caused by stale action translation and an overly
long conversational loop. The traces show Astra answering `go north` while the
harness submits `drop key`, answering `insert spork into dresser` while it submits
`drop spork`, and answering `open cabinet` while it submits `examine table`. Once
the wrong command is sent, the next answer is conditioned on the same state and the
episode burns its 1,800-second deadline. Candidate 0001 adds verified short plans
and candidate 0002 compiles multi-step programs; this proposal tests a different
control boundary: every model answer is a one-step decision over a numbered,
current command list, with no session history or plan that can become stale.

## What changed vs. the parent

The harness renders a compact public observation and an indexed list of the exact
commands available on this turn. Astra must return only `{"index": N}`. Parsing
accepts a strict integer/JSON index and never mines a command from prose. The index
is resolved immediately against the same list, then one action is submitted. A
fresh call is made after every action, so movement, opening a container, and taking
an item cannot invalidate queued commands. `ctx.progress()` and a public-state
signature record illegal or no-effect actions as evidence in the next prompt; a
state-conditioned tried set discourages repeating a failed command. Model errors
and garbage output are caught, and a bounded non-destructive fallback keeps
preflight and low-budget episodes alive. Destructive actions are deferred unless
the objective explicitly names the relevant transformation and item.

## Evidence that would confirm or refute it

Confirmation would show actions equal to the selected current indices, no parser
rejections caused by stale names, and higher scores on the incumbent's looping
seeds (especially the gateway/cabinet and spork/dresser quests). The traces should
show a new decision after each state change and no repeated action in an unchanged
state. Refutation would be a zero score despite correctly bound, state-changing
commands, indicating that the main limitation is Astra's quest planning or the
one-call-per-action latency rather than action translation. Unseen seeds are
needed to test whether the fallback and index protocol generalize beyond the
training objectives.
