parent: 0000

## Hypothesis

The leader's 0.75 score is limited by a control-loop failure rather than a
lack of puzzle knowledge. It spends one slow Astra call for every primitive
turn and gives Astra a very large, mostly unchanged JSON transcript. On seeds
103 and 107 the trace shows 27–34 actions, repeated `noop` and reversals, and
then the 1800 s deadline. A planner that can commit a short sequence, inspect
the resulting board, and discard a stale sequence should both reduce latency
and recover from blocked pushes.

## What changed vs. the parent

Candidate 0001 asks Astra for 1–8 actions in a strict `ACTIONS:` line. The
harness renders every non-empty coordinate and stack member, active rules,
agent state, and terminal fields in a compact deterministic summary. It
executes each primitive separately, calls `ctx.progress()` after every step,
and immediately replans when the state is unchanged for two directional turns,
an action is rejected, or active rules change. Model timeouts and malformed
stand-in replies are caught; a missing action falls back to `noop`. A session
ID preserves Astra's context while the current state and verification evidence
are sent on each replan.

## Evidence that would confirm or refute it

Confirmation would be fewer Astra calls and shorter wall time, with seed 103 or
107 reaching `success=true` instead of timing out, while retaining or improving
the 0.75 training mean. Traces should show plans being cut at genuine stalls or
rule changes rather than blindly replayed. Refutation would be a similar score
with no latency gain, or a lower score caused by Astra's multi-action prefixes
becoming stale before the per-step checks trigger; in that case the batch size
or replanning trigger needs tightening.
