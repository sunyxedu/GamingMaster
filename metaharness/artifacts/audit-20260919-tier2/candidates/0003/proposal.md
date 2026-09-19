# 0003 — Reusable semantic bindings during cold start

Parent: `0002-previous_astra_winner`. This is one new candidate, with no performance claim before evaluation.

## Hypothesis

The parent's main observed limitation is unproductive cold-start time, not failure to survive or insufficient search depth. Jev is much better at classifying a stable rule interpretation than choosing an action from raw observations. A small, confidence-gated semantic bridge should turn those interpretations into useful actions within a few ticks, while Astra compiles a general reactive policy in the background. Every subsequent tick should recompute the action from current numbers rather than reuse an old action answer.

This tests an architectural change: use Jev to bind reusable meanings early, not just to vote on transient actions. It does not assume Astra will become faster.

## Archive evidence

All three parent episodes reach 120 ticks, with zero stale ticks. Their scores are 7, 8, and 11. The worst episode is seed 0.

* In `../0002-previous_astra_winner/trace_seed0.jsonl`, line 6, the first Jev action answer is `a4` (`keep`), with confidence **0.39** and probability **0.51**. In the SAME response, the safety-mask interpretation is `unsafe_true`, with confidence **0.99** and probability **1.0**. This is direct evidence for separating stable semantic classification from action selection. The other seeds repeat this pattern.
* Seed 0 line 166 installs the first policy at **frame 54**, about **27.008 seconds** into a 60-second episode. Line 368 reports `"compiled_frames":66,"bootstrap_frames":54` and `"search_frames":0`. Seed 2 does not install its policy until frame 61.
* Seed 0's score increases at ticks **3, 62, 78, 94, 104, 111, 117**. Thus the first reward is followed by 58 rewardless ticks before the next reward. Seeds 1 and 2 earn their first rewards only at ticks **64** and **63**, respectively. These traces associate reward progress with reactive-policy availability; they do not establish a causal score estimate for the proposed bridge.
* Seed 0 line 367 installs the world model at **62.156 seconds**, with `"episode_active":false`. Its expensive model never acts. Seeds 1 and 2 instead return `unsupported`; seed 1's explanation includes: “Food-induced growth and collision timing for a vacating tail are unspecified.” Every parent episode has zero search frames.
* The Jev-only baseline's seed 0 first answer chooses `keep` with confidence 0.57; it dies at tick 6. The Astra-only baseline takes 6.595 seconds for its first answer and dies at tick 6 with all ticks stale. Neither synchronous baseline is an appropriate cold-start controller.

The manifest specifies wall time and episode-local memory. The late compiled world cannot help the next seed under that protocol.

## New control procedure

1. Post an action immediately, then start one Astra compilation and a Jev discovery batch asynchronously. Retain the parent's generic discovery of Boolean maps indexed by action names. Add shape discovery for short numeric vectors in observation dictionaries. No key spelling determines a field's meaning.
2. In one typed call, Jev independently classifies each candidate safety map and selects an existing signed objective-error vector, or `none`. Absolute positions, dimensions, velocities and inventories explicitly do not qualify. Both chosen-option probability (at least 0.80) and confidence (at least 0.70) must pass.
3. If a residual is identified, a second batch asks one narrow question per action: its constant additive change to that residual during a permitted execution with the objective unchanged. Options are unit changes in the discovered dimensions, plus `none`. The question explicitly distinguishes residual change from position change. State-dependent/default effects must abstain.
4. Code evaluates the sum of absolute residual components after each bound action, using the CURRENT observation. Only actions explicitly certified by a learned safety mask can enter this provisional controller. This is a restricted, optional generic error-reduction bridge, not an assumed geometry or transition model. Games without matching observable semantics bypass it.
5. Compare predicted residual changes with successive observations. Two consecutive mismatches for an action revoke its effect binding. Skip predictions that reach zero, because the objective may then change. Timing checks omit ambiguous intervals. This check is conservative: moving objectives can invalidate an otherwise useful binding.
6. Astra returns a small expression policy: ordered features, per-action facts, safety, one or two utility modes, and a compact context for Jev. The expressions derive all domain semantics from the actual rules and observations. Validate the policy on recent observations before installation. The compiled policy then takes control; the semantic bridge remains the fallback and a logged comparison.
7. Jev classifies the compiled strategy using Astra's context and concrete mode criteria, including a no-match option. A mode answer requires probability >=0.70 and confidence >=0.60, the same policy revision, identical context and allowed-action set, and age <=4 ticks. Without an explicit context, the entire computed fact table is the freshness key. Utilities and legality are always recomputed live. Direct bootstrap action answers require exact observation equality and age <=2 ticks.
8. Return to Astra only for a rejected or broken compiled policy, up to the remaining four-call episode budget. Remove the unconditional second world-model compiler and all search. This isolates effort on live reactive coverage; no cross-episode cache is used.

The main loop posts an action before scheduling inference, then waits for the next tick. Pending decisions never block the wall-clock controller. All spawned futures are joined in `finally`, even after an error or episode termination. Inference exceptions propagate, including `ModelUnavailable`; local expression-validation errors can request a repair.

## Measurement and falsification

The evaluator should compare mean score first, then stale ticks and System-2 calls under the same manifest. More specifically:

* Confirm the mechanism if `effects_bound` occurs well before `policy_installed`, `control` records actual `jev_binding` interventions during that interval, and pre-compilation reward improves. Look at binding audits, not just binding acceptance.
* Favor the candidate only if earlier reward persists into a better mean final score without an offsetting increase in early deaths. The parent already survives perfectly on these three seeds.
* Refute the semantic hypothesis if Jev frequently abstains, confuses coordinate signs, or receives repeated `binding_revoked` events. Refute the throughput hypothesis if the discovery batch is itself too slow, or if earlier aggressive progress produces longer bodies and more deaths before Astra becomes available.
* A higher final score without bridge interventions would not confirm the proposed mechanism; the smaller Astra prompt could be the explanation. This candidate also removes the unused world pass, so lower S2 use alone is not evidence for better control.

`control` logs the posted action, source, shield/direct/binding alternatives, compiled default, selected mode, and applicable intervention labels. Shadow binding differences while Astra is controlling are not counted as Jev binding interventions. `coverage` aggregates source frames and control changes. Model results received only during final draining are never logged as installed control.

## Limits and validation

The optional unit-residual bridge applies only when public semantics justify it. It does not cover arbitrary actions, moving targets, unequal coordinate costs, or unknown field meanings. Such cases retain shielded fallback until the fully general expression compiler responds. Greedy error reduction does not establish future safety. Astra's context must also capture the facts necessary for a reusable mode judgment.

Local checks used mocked interfaces only: Python syntax; compatibility with the parent's archived policy response; renamed-field shape discovery; both confidence gates; semantic bridge and compiled control; and joining every future exactly once during normal exit, S1 unavailability, S2 unavailability, and simultaneous inference failures. No model backends, game implementations, held-out games, or performance experiments were run.
