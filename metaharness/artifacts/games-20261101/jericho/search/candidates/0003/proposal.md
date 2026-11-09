parent: 0000

## Hypothesis

The near-zero baseline score is primarily an action-binding failure, not evidence
that Astra cannot solve these stories. In `trace_seed0.jsonl`, Astra returned
`{"command":"look"}`, `look`, and `east`, yet every step recorded
`action: "command"` and the story replied “That's not a verb I recognise.”
The baseline's generic `parse_action` recognizes the sole action *name* and
throws away the parser command *value*. Correctly forwarding the value should
recover ordinary Jericho play.

## What changed from the parent

This harness parses JSON objects/lists, tagged lines, and bare commands into the
actual parser string and submits that string directly. It asks Astra for a
compact 1–4 command plan, executes one command at a time, and discards the
remaining plan after parser errors, clarification-like text, score changes,
terminal text, or repeated unchanged outcomes. It retains a compact recent
transcript and failed-command list so Astra can diagnose grammar and revise its
beliefs. Model failures and malformed stand-in replies fall back safely to
`look` without escaping the episode.

## Evidence that would confirm or refute it

Confirmation would be traces whose `last_action` contains parser commands such
as `look`/movement rather than the literal `command`, with fewer “not a verb”
responses and positive score on at least the simple training stories. If actions
are forwarded correctly but scores remain at the baseline, or if batched plans
produce many stale/unsafe commands, the hypothesis is incomplete: the next
iteration should focus on story-specific dependency memory and more conservative
single-step replanning.
