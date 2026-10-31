parent: 0000

## Hypothesis

The leader failed because it asked Astra for a complete action on every step while exposing a very large observation; the first malformed or stale parameter trapped play in setup, and the 48–182 second calls exhausted the 1,800 second episode. A purely scripted tactical policy may fix legality but can still follow the wrong economic route on unseen boards. The stronger structural test is to separate strategic planning from execution: ask Astra for a compact goal, then compile that goal into fresh legal actions and replan only when `ctx.progress()` shows stagnation or rejection.

## What changed vs. the parent

This harness uses one short JSON plan call initially and evidence-conditioned replans thereafter. Astra never supplies an action ID. The local executor scores randomized board sites by dice pips and resource diversity, handles forced discard/robber/setup decisions, and chooses only IDs from the current `legal_actions`. It checks progress after every action; rejected actions refresh the observation, while sustained no-change evidence triggers a new plan. Arbitrary Astra output, timeouts, and runtime failures fall back to the checked local policy.

## Evidence that would confirm/refute it

Confirmation would be normal-turn play with near-zero illegal actions, substantially fewer Astra calls than the leader, and higher true VP on boards where a fixed build order stalls. Refutation would be persistent low VP despite legal execution, or evidence that compact goal plans do not improve over the local policy because the tactical compiler chooses roads, trades, or development cards poorly.
