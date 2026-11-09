parent: 0003

## Hypothesis

The remaining ceiling in 0003 is stale hidden conversation state rather than
parser transport. Its traces show a single Astra session growing from about
14,000 to more than 50,000 input tokens. On the library seed it begins with
the useful clue “We keep it locked away, in the Rare Books Room”, then spends
later calls issuing isolated `west`, `east`, `north`, and repeated questions.
On Enchanter it eventually repeats `e` several times and revisits the same
`n` route. The model receives a compact recent transcript in the new prompt,
but session continuation also retains every prior prompt and prediction. That
unbounded context can make calls slower and bias Astra toward its old route,
which is especially costly with a roughly 40–60 second call latency.

## What changed vs. the parent

Candidate 0004 keeps the repaired raw-command transport and bounded response
checked batches, but starts **a fresh Astra session on every decision**. Each
call receives a self-contained evidence packet: the exact latest observation,
the opening text, the last twelve command/response records, score events,
failed commands, a short reason for replanning, and the remaining budget.
The packet is capped and stored in `ctx.memory`; no hidden conversation is
allowed to grow. Positive score changes do not automatically discard a valid
suffix, while parser failures, clarification/menu text, hazards, empty
responses, negative score, terminal text, and exact command/response loops do.
The decoder remains total for JSON, fenced output, wrappers, bare commands,
and arbitrary pre-flight text. Model, action, and observation errors are
caught so deadline cancellation is a normal end.

## Evidence that would confirm or refute it

Confirmation would show smaller, stable model input packets, fewer repeated
direction/question loops, more useful commands per fresh call, and improved
final native score at similar or lower call count. A library trace should
replan from the Rare Books clue instead of replaying the stale route; an
Enchanter trace should avoid repeated `e`/`n` actions after unchanged text.
The hypothesis is refuted if fresh sessions still produce the same route
cycles, or if the loss of conversational continuity makes Astra forget useful
clues and lowers score. The key comparison is final native score and parser
failure/repetition counts, with latency and actions as diagnostics; transport
correctness alone is already established by 0003.
