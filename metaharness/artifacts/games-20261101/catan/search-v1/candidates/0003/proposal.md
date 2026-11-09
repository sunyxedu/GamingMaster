parent: 0000

## Hypothesis

The leader is failing primarily because every decision is delegated to a slow,
stateful language model whose answer is treated as a fresh action even after
the board has changed.  The traces show this directly: seed 105 posts
`build_road 11 29` fifteen times; seed 101 posts `move_robber 12` repeatedly;
seed 106 repeats `build_settlement 49`.  These loops consume the 40-call budget
and the 1,800-second deadline while producing no progress.  Catanatron already
does the hard legality work and exposes a complete, revision-local action list,
so a legality-first controller should convert the same weak bot matchup into
steady progress.  The test is intentionally structural: zero model calls,
fresh action IDs on every observation, and deterministic board/resource
bookkeeping.

## What changed vs. the parent

The parent rendered a very large observation and asked Astra for one textual
action on every step.  This harness never asks Astra to repeat forced choices.
It selects only an ID from the current `legal_actions`, ranks initial
settlements by dice production/resource diversity (favoring wheat and ore),
ranks initial roads by access to productive vertices, chooses cities and
settlements before roads during normal play, and chooses robber destinations by
opponent production, buildings, and public hand size.  Roll, discard, and end
turn are handled as phase actions.  A previous action is remembered solely to
avoid selecting the same still-listed ID when an alternative of that type is
available.  All parsing is defensive for JSON/dict observations and all
adapter/model exceptions are contained so garbage stand-in output cannot fail
pre-flight.

## Evidence that would confirm or refute it

Confirmation would be a large reduction in repeated identical actions and
deadline truncations, with more than two public/true VP on weak seeds and a
substantial fall in action latency and System-2 calls.  Refutation would be a
controller that legally advances but still stalls at the initial two points,
or loses score because the fixed priority (especially city/settlement before
road) ignores tactical races; traces would then identify which legal action
class needs a new feature or a narrowly scoped Astra call.
