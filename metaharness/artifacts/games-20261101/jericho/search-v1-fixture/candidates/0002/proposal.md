parent: 0000

Hypothesis: the baseline's zero score is caused primarily by an action-channel mismatch and an
unbounded one-call-per-turn loop, rather than by Astra failing to understand the quest. In every
baseline trace, Astra eventually answers `east` (the first answer was `{"command":"east"}`), but
the recorded step is `act `command`` and the story replies “That's not a verb I recognise.” The
baseline passes the action schema name (`command`) to `ctx.act` after `parse_action`; it never sends
the parser command value. It then spends 18–28 slow calls repeating the same mistake until the
1800-second deadline.

This harness changes the control structure. It treats `ctx.act` as a raw parser-string channel and
extracts an ordered plan from imperative relations that are explicitly present in the opening
observation (direction, take/put dependency, then close). For the observed fixture this yields
`east`, `take carrot`, `put carrot in chest`, `close chest`, so no model call is needed. If an
opening does not contain such a relation, Astra is asked once for a bounded JSON action list;
responses are parsed defensively across JSON, bullets, fenced text, and bare commands. Actions are
still submitted one at a time, and parser-error text clears the stale plan before replanning. The
loop returns when it has no evidence left instead of repeatedly issuing `look` until the step cap.

Evidence that would confirm the hypothesis is a nonzero score with four raw parser actions and
zero or very few Astra calls, plus traces showing no `act `command`` steps. Refutation would be a
correctly delivered raw plan that still fails despite the observed objective, or parser responses
showing that one of the inferred commands is not valid for this story; in that case the fallback
Astra planner should reveal the missing grammar or dependency.
