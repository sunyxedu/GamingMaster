# Candidate 0004 — planner card plus pipelined pulse controller

**Parent:** `candidates/0002-tier2_idle/` (the required rewrite parent). I also used the best available comparator, `0000-tier2_jev_only`, because the idle parent supplies no positive-control behavior.

## Hypothesis

The failure is not simply that Frogger needs more model calls. The Jev-only controller asks for an action from a huge, mostly uninterpreted observation, then blocks the real-time loop for about 0.15–0.16 s. In the worst trace (`trace_seed103.jsonl`) the first several answers are all `UP` with probabilities near 0.51, and the answers arrive at ticks 0, 12, 21, 32, 42, …; the frog remains at the start candidate until it earns one point, then repeatedly dies. Across the six traces Jev selected `UP` overwhelmingly (294/323 calls in seed 103 and 343/363 in seed 113), while the baseline reports roughly 89.6% stale ticks and zero homes. The model is therefore acting as a slow, biased joystick holder rather than as a hazard judge.

My testable hypothesis is: **separating real-time execution from judgment, and making every directional answer a one-frame pulse that must survive two independent typed checks, will preserve the baseline's useful forward hops while reducing deaths caused by held/stale input. A short Astra route card should give Jev a subgoal and a reason to wait, rather than asking Jev to discover the whole strategy from raw RAM on every call.**

## What changed

* Astra is launched asynchronously at reset with the complete first observation, including raw RAM, the adaptor hypotheses, and recent evidence. It is asked for a short, seed-independent planner card (`SUBGOAL`, `WAIT_RULE`, `FORWARD_RULE`, `LATERAL_RULE`, `REPLAN_IF`), not for a frame-by-frame action sequence. It can revise the card after a recorded death, at most three calls total, using the existing session when available.
* The main thread never waits for Astra or Jev. It owns the game clock, posts an explicit `NOOP` before the first wait, and posts one legal action after every tick. Thus a late model answer leaves an executable action in place instead of leaving the adaptor to repeat an old joystick input.
* Jev receives a compact normalized state: score/deaths/reserves/timer/level, broad frog-zone and row/x candidates, named RAM hypotheses, adaptor-provided semantic candidates if present, selected RAM changes since the previous observation, a short history, the current planner card, and the pulse/debounce status. Full RAM is retained for Astra but is not allowed to drown the immediate question in every request.
* One Jev request batches four typed judgments: two independently ordered choices of the next one-frame action, a three-level immediate-risk `Score`, and an immediate posture/subgoal `Choice`. Code, not Jev, maps answers to strings, checks legality, checks freshness, and performs cooldown bookkeeping.
* A directional answer is accepted only when both action choices agree and each gives at least 0.40 probability to that action. Ambiguous directions become `NOOP`; a high-confidence lateral correction can be used only when the semantic subgoal agrees and the risk is not `DANGEROUS`. This specifically counters the documented first-option bias without requiring Jev to calculate geometry.
* After a direction is posted, code posts `NOOP` for a three-frame debounce interval. Answers older than 18 ticks or judged on a previous life are discarded. Model failures and garbage stand-in answers are logged and become `NOOP`; budget/deadline exceptions are allowed to end the episode normally.

## Evidence that would confirm or refute it

Confirmation would be a substantial reduction from 0.896 stale fraction, fewer deaths before the first home, and a score above the Jev-only mean of 6 while action traces show isolated directional pulses rather than long `UP` runs. A useful intermediate signal is forward score increasing without the four-death/zero-home pattern. Refutation would be zero or near-zero forward hops because the agreement gate is too conservative, repeated deaths despite isolated pulses, or a score at/below idle. If action pulses are safe but progress stops, the next revision should relax the probability gate or improve the RAM/phase decoder rather than add Astra prose.

## Jev state schema

Each request is strict JSON with this shape:

```text
{
  tick, frame, mode, difficulty,
  score, score_delta, deaths, reserve_frogs, timer_s, level,
  terminal, terminal_reason,
  frog: {x, row, state},
  zone: START_OR_RESPAWN | LOW_ROAD_OR_FIRST_HOP | ROAD_OR_MEDIAN |
        RIVER_OR_HOME_APPROACH | UNKNOWN_ACTIVE_ROW | TERMINAL,
  ram_fields: adaptor hypotheses,
  candidate_fields: any adaptor-provided road/river/home/hazard candidates,
  ram_changed_since_previous: ["address:old->new", ...],
  planner_card: short Astra text,
  recent_history: last eight {tick, score, deaths, frog_row, frog_x,
                    timer_s, action_posted_for_previous_tick, zone},
  last_action, action_pulse_rule
}
```

The state explicitly says that row/object labels are hypotheses and that a direction is a pulse. Jev is not asked to decode BCD, count frames, predict a numeric score, or invent an action mapping.

## Questions, thresholds, and bridge

* `action_primary`: `Choice` over `UP, LEFT, RIGHT, DOWN, NOOP`, asking for the next single-frame action under the current subgoal.
* `action_check`: the same judgment with reversed option order (`NOOP, DOWN, RIGHT, LEFT, UP`). Agreement is required for a normal directional pulse. This makes option-order bias visible in one request.
* `risk`: `Score` with `CLEAR`, `UNCERTAIN`, `DANGEROUS`, grounded in the current named fields and temporal changes. Code treats its returned fractional level index as 0/1/2; it is a gate, not arithmetic.
* `subgoal`: `Choice` over `ADVANCE, ALIGN_LEFT, ALIGN_RIGHT, WAIT, RETREAT, ESCAPE`. It supplies a semantic fallback for a high-confidence cheap lateral correction, never a hidden multi-hop plan.

A matching `NOOP` is always executable. A matching direction needs probability at least 0.40 in both action answers, no active debounce, and (for `UP`) must not be marked dangerous unless the subgoal is escape/retreat. A disagreement or stale answer defaults to `NOOP`; a lateral disagreement is admitted only with subgoal agreement and probability at least 0.58 in at least one action answer. `ctx.act` is still called every live tick, so the fallback is fresh even while Jev (about 0.15 s) or Astra (tens of seconds) is in flight.

This intentionally gives Astra planning/adaptation and Jev recognition/choice, while code owns timing, exact action names, debounce, freshness, and score/life transitions. The idle candidate's proven behavior—never risking malformed actions and allowing the emulator to advance—is retained as the final fallback, but it is no longer the only policy.
