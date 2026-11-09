# 0003 — Neutral-rearmed Jev actions

## Parent and isolated hypothesis

Parent: `0000-tier2_jev_only` (also the current best candidate).

**Hypothesis:** the universal one-point result is primarily an actuator/debounce
failure, not evidence that Jev cannot select successive hops. Posting another
`UP` while `UP` remains held does not create another edge in standard Frogger.
Keeping the parent's architecture and replacing its held-action bridge with
one-frame direction pulses separated by a centered frame should restore
repeated movement and score beyond the initial point.

## Trace evidence

All six parent episodes tie for worst score (1), with no homes and four reported
deaths. The following are excerpts/field projections from actual parent traces:

* `trace_seed103.jsonl:17`: the first answer selects `UP`, probability `0.68`,
  latency `0.22903180122375488` seconds.
* `trace_seed103.jsonl:19`: `{"type":"tick","tick":14,"action":"UP",
  "fresh":true,"score":0.0}`. Line 20 is already stale `UP`, tick 15, score 1.
* `trace_seed103.jsonl:31`: the next answer again selects `UP` (probability
  `0.71`); line 32's next request reports frog row 0 and score 1.
* `trace_seed103.jsonl:43`: the subsequent request reports frog row 0, state 8,
  reserve 3, score 1. Line 133 reports the respawn on row 255, reserve 3, still
  score 1. Later requests at ticks 2043, 4058 and 6014 are also respawned on
  row 255, with score 1 and progressively fewer reserves.
* Across that entire 7,924-tick episode: 13 initial `NOOP` ticks followed by
  **7,911 consecutive `UP` ticks**, with no release at all. Seed 101 likewise
  has 12 initial neutral ticks then 7,907 consecutive `UP` ticks; seed 127
  has 12 initial neutral ticks then 7,940 consecutive `UP` ticks.

The public action contract explicitly requires `NOOP` between intentional hops.
This is a concrete mechanical mismatch independent of numeric hazard reasoning.
The parent median inference latency is about 0.2 seconds (roughly 12 frames),
so a neutral frame need not cost another model round trip.

`lessons.md` is absent from this archive; no recorded refutation was available.

## Exactly what changes

Keep the parent's synchronous Jev-only control loop, full raw observation,
original game-description action question, original option order/descriptions,
and one decision request per iteration. No Astra, planner, trajectory search,
new heuristic action selector, observation compression, or probability gate.

The sole behavioral change is the answer-to-action bridge:

1. Post the selected legal action.
2. For a direction, wait until the next game tick consumes it.
3. Post `NOOP`, and wait until one centered tick has been consumed.
4. Ask Jev again while the joystick is centered.

A malformed/absent action value maps to `NOOP` so stand-in model output cannot
crash the harness. Log `hop_pulse` after a consumed press/release pair. Those
logs are diagnostic only. No special treatment of any seed or configuration.
One-frame pulses also prevent accidental held repeats in Speedy Frogger.

## System-1 interface and control contract

* **State:** exactly `game.observe()`, unchanged from the parent, including the
  named frog/road/river/home fields, bookkeeping, metadata and raw RAM.
* **Question:** one `Choice`, ID `act`: the complete `game.describe()` followed
  by “Which action is best right now?” Options and descriptions come unchanged
  from `game.action_spec` (`NOOP`, `UP`, `RIGHT`, `LEFT`, `DOWN`).
* **Answer bridge:** a legal `.value` determines the next pulse; invalid values
  become neutral. A fresh Jev answer is required for each directional pulse.
* **Thresholds:** none, as in the parent. Do not change the parent's confidence
  policy in the same experiment. No calculations are newly delegated to Jev.
* **Freshness:** no new observation-age rejection, matching the parent. The
  initial decision can be latency-stale; this experiment fixes rearming only.
  The centered joystick, rather than a held direction, persists during inference.
* **Fallback:** `NOOP` while waiting for inference. On a transient timeout or
  runtime failure, wait for a tick and retry, still centered. Budget exhaustion
  and unavailable backends retain the parent's termination behavior.
* **Scheduling:** synchronous requests, no worker threads or pending loops;
  `ctx.wait_for_tick()` provides the press and release barriers. No assumptions
  that an inference round trip equals one emulator frame.
* **System-2:** zero calls, preserving the designated parent's architecture.
  Whether Astra should revise the question interface is deliberately left to
  later iterations after the actuator is known to work.

The automatic neutral release is not a new model decision. Consequently the
reported Jev-to-posted-action fraction can be approximately half the parent's
100%, without reducing Jev's control of intentional directional moves.

## Confirmation and refutation

Confirm the mechanism if traces show consumed `direction → NOOP → direction`
sequences, additional forward-score changes, and progression beyond road row 0
or repeated successful new-life departures. Confirm the performance hypothesis
if training scores exceed the parent's mean 1, with no episode errors.

Refute the mechanical claim if isolated pulses fail to produce further hops
despite successful neutral consumption. Refute its sufficiency if rearming
works but the score remains at or below 1. In that case distinguish poor/stale
hazard selection from missed input sampling using the new action tick traces;
do not silently add a planner to this candidate.

## Validation

Implementation was written before running live trials. Contract-style checks
passed for syntax, legal pulse ordering, malformed stand-in answers, and
transient `TimeoutError`/`RuntimeError` recovery. A three-minute live try on
training seed 103 then produced:

* score `22.0`, versus the parent's `1.0` on seed 103;
* `2793` ticks, `255` Jev calls, zero Astra calls;
* `2481/2793` stale ticks (the neutral release and model latency are visible);
* `ended_by=game_done`, with exit code 0 and no harness error.

This is a single try rather than the required six-episode evaluation, but it
confirms the actuator part of the hypothesis: the neutral-rearmed bridge can
produce substantially more score before game over. The final evaluation must
still establish whether Jev's raw-observation choices generalize across all
training seeds and whether the stale-tick tie-break worsens enough to offset
that gain.
