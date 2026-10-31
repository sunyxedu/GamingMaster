parent: 0000

## Hypothesis

The leader fails mainly because it pays Astra latency for every primitive move. The trace for seed 103 spent the whole 1800-second episode in the first call and posted zero actions; seed 102 posted only seven actions after eight calls and ended at -0.57. Even its successful seed 100 used 32 calls over 1531 seconds. A compact, verified macro plan should turn each expensive call into dozens of actions and leave enough wall time to reach later pushes.

## What changed vs. the parent

The parent asks for exactly one action and blindly posts it. This harness asks Astra for up to 60 directions in a single `ACTIONS:` line, executes them one at a time, and retains the session for replanning. Before each action, local code predicts the Sokoban transition, rejects blocked moves and pushes into reverse-pull dead squares or frozen 2x2 blocks, then compares the returned board/player/box state with that prediction. A mismatch clears the remainder and prompts Astra with the failure reason. Static reverse-push analysis, visited-state bookkeeping, and a cheap safe fallback keep garbage or timed-out model responses from crashing or wasting an entire call.

## Evidence that would confirm or refute it

Confirmation would be more actions per System-2 call, fewer deadline-ended episodes, and higher boxes-on-goals/solved counts on the weak seeds while keeping zero harness errors. If plans are mostly rejected, or Astra's longer sequences push boxes into dynamic corridor deadlocks despite the static checks, the hypothesis is refuted; traces should show whether the limiting factor is plan quality rather than call granularity.
