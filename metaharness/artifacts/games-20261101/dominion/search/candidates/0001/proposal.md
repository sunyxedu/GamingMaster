parent: 0000

Hypothesis: the leader loses mostly because it spends one 45–130 second Astra call on every
prompt. Seven of eight episodes reached the deadline. In seed 1201, Astra chose four cards to
trash with Chapel at decisions 5 and 11, but those selected IDs were lost before execution; the
turn-7 trash prompt also passed despite four Copper choices. One early turn therefore left the
deck unthinned, while repeated inference consumed most of the clock. The harness needs a fast local
executor with a small number of strategic model calls, while still giving Astra a chance to change
the plan when the public race or observed state invalidates it.

What changed versus the parent: Astra now receives a compact snapshot and returns a validated JSON
strategy (mode, acquisition priority, and support-card targets), at most once every six turns and
again on a detected stall or Province-race transition. All action, trash, discard, reaction, gain,
and buy prompts are then answered locally from current legal options. The executor caps engine
pieces at planned targets and ranks purchases using the current supply, VP race, deck counts, and
the Astra priority, and rebuilds occurrence IDs on every prompt. It checks
`ctx.progress()` and the observation after every action; rejected or unchanged actions trigger a
fresh review and deterministic recovery. Model failures are logged and never escape the episode.

Evidence that would confirm the hypothesis: training episodes should finish before the deadline
with substantially fewer Astra calls, while maintaining or improving VP margin, especially on
Chapel/engine and curse-race boards. A trace should show safe multi-card Chapel trash choices,
completed games, and a strategy review when the Province race changes. Refutation would be games
whose margins fall below the parent despite the extra time, or repeated stalls caused by the local
policy; that would indicate the strategic plan needs more frequent Astra intervention rather than
just a cheaper executor.
