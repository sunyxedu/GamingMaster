parent: 0000

## Hypothesis

The leader fails mainly because its control granularity is mismatched to the game and the
latency budget. It pays one Astra call (about 50 s median, up to 178 s) for every primitive
step. The traces show the consequence: seed 101 spent its only call in a long initial answer and
made zero actions; most other seeds stopped at 20–32 actions and a few boxes on goals even though
the action cap was 2,000. The model also occasionally pushes a box off a goal (seed 107, step 12),
which is a symptom of no static dead-square guard or transition checking.

## What changed vs. the parent

This harness first builds a static reverse-push useful-cell mask and runs a bounded A* search over
push states. The search flood-fills player reachability, expands only legal pushes, prunes static
dead destinations, and uses a matching lower bound. A complete solution is executed locally with
no model calls. If bounded search does not finish, Astra receives the full JSON plus the useful
mask and returns up to eight primitive actions in an explicit `ACTIONS:` line. The harness parses
only known action names, validates every action against the latest observation and dead-cell mask,
and replans after each push. A safe legal fallback keeps garbage or timeout responses from
crashing pre-flight or burning the episode on malformed actions.

The structural test is whether push-state computation and short action batches increase useful
actions and solved levels per 1,800-second episode, while the static guard prevents irreversible
off-goal deadlocks. The search is bounded so hard levels still reach Astra rather than consuming
the whole episode in computation.

## Evidence that would confirm or refute it

Confirmation would be fewer System-2 calls per episode, more actions before the deadline, and a
higher solved/boxes-on-goals score, especially on the leader's zero-action and low-score seeds.
Trace evidence should show model responses containing compact plans and no rejected pushes. If
the score remains flat while calls fall, the bottleneck is search completeness or goal-order
reasoning rather than latency. If scores fall on simple levels, the bounded search's canonical
reachability/deadlock pruning is too aggressive and should be relaxed.
