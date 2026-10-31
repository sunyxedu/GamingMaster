parent: 0000

## Hypothesis

The leader's main loss is the cost of asking Astra for every primitive action, but a macro
controller alone still spends model calls on levels whose push structure is mechanically
searchable.  The traces make this concrete: seed 103 spent the entire deadline in its first
call with zero actions, seed 102 made only seven actions after eight calls, and seed 100 used
32 calls for 32 actions.  A bounded solver with a reverse-push-distance matching heuristic
should solve a useful subset without any model latency and leave Astra for genuinely ambiguous
levels.

## What changed vs. the parent

This harness starts with a local push-state A* search.  It computes, for every goal, the exact
static reverse-pull distances through the board geometry and uses a bitmask minimum matching
as the lower bound.  Player reachability is flood-filled for each candidate push; static
unreachable destinations, non-goal corners, and small 2x2 freezes are pruned.  A found solution
is executed directly and checked against a locally predicted transition.

When the bounded search times out, Astra is asked for a compact `ACTIONS:` continuation.  The
controller validates each direction against the current JSON, rejects blocked or statically
deadlocking pushes, clears the remainder after every push, and replans.  Timeout, budget,
backend, and malformed-output paths all fall back to a safe legal movement so pre-flight garbage
cannot escape as an error.

## Evidence that would confirm or refute it

Confirmation would show solved or higher boxes-on-goals scores with materially fewer Astra calls,
especially on seeds 101--103 where the leader made zero to seven actions.  Traces should contain
local-search actions before the first model request and reverse-distance-guided pushes after
fallback requests.  If search consumes time without finding plans, or its freeze pruning removes
valid routes, the weak seeds will still end with low boxes-on-goals and the traces will show
search-to-Astra handoffs without progress; that would refute the stronger heuristic hypothesis.
