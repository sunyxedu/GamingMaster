# Candidate 0003 proposal

## Parent and targeted hypothesis

Parent: `candidates/0000-tier2_jev_only/`.  It makes one Jev `Choice` over the
six literal actions, using the complete observation, and otherwise lets the
last action repeat while the synchronous request is pending.

The parent’s worst concrete evidence is seed 1: score 105, 11 invaders, 425
ticks, and all three lives lost.  The trace shows 247 stale ticks versus 178
fresh decisions.  At frame 0 Jev chose `fire`; it repeatedly chose `fire` while
the cannon remained at x=90.  Late in that run, calls at frames 1656--1696,
with score 105 and one life remaining, selected `right_fire`/`fire` while the
player was around x=62; the terminal observation at frame 1697 shows lives=0,
player x=90, and no decoded bomb.  Seed 2 is similar (110 points, 11 kills,
417 ticks, 249 stale ticks), while seed 5 demonstrates that the same loop can
survive longer but still ends with zero lives and only 17 kills.  The parent’s
raw requests are about 2,760 input tokens, and the measured p50 Jev latency is
about 0.14--0.16 seconds, over two game ticks.  Thus the model is both asked to
calculate geometry from a large state and unable to control every tick.

**Hypothesis:** keeping Jev as a fast recognizer but moving exact shot-lock,
target alignment, and emergency movement into code will improve score and
survival.  A small derived state should make the model’s decision about danger
and target columns more reliable than direct action generation, while the
code bridge can use a fresh observation to produce a legal combined action.
This is intentionally one architectural change: symbolic tactical bridge and
state compression; no Astra is introduced.

## What changed

* The full observation is reduced to named facts: living columns, lowest living
  target per column, player x, laser-active lock, and a coarse bomb-pressure
  bucket with side labels.
* Jev answers two independent typed choices in one request: immediate dodge
  direction and, when safe, a preferred target column.
* Code converts those judgments to `left`, `right`, `left_fire`,
  `right_fire`, or `fire`, checking the latest observation and legal action set.
  It never fires a second shot while `player_shot.active` is true.
* If Jev times out or has a normal call failure, a deterministic fallback
  re-reads the live state, dodges a close bomb, fires only when the laser is
  inactive, and sweeps away from boundaries.  On an immediate backend failure
  it waits for a game tick before retrying instead of spinning.  Model failures
  are therefore not fatal and do not leave an accidental indefinitely repeated
  action as the intended policy.
* Malformed model values, malformed observations, and unknown action names are
  handled by the fallback.  Budget/model-unavailable exceptions retain the
  contract’s normal propagation semantics.

The implementation still posts one action per completed Jev request, so this
candidate does not claim to eliminate real-time stale ticks.  It isolates
whether better state/action structure matters before adding concurrency or
System-2 planning.

## System-1 contract

The JSON state sent to Jev contains:

* `frame`, `score`, `lives`, `wave`, `waves_cleared`, and `invaders_killed`;
* `player_x`, `shot_active`, and `shot_x`;
* `bomb_pressure` (`high` when a bomb is in the near-cannon y bucket, otherwise
  `low`) and `bomb_sides` (`left`, `center`, or `right`);
* `nearest_threat` with only its column, row, and `upper`/`low` y bucket;
* `living_columns` and `lowest_by_column` with the current target x/row/y;
* a tiny `previous` record containing the previous action, score, and lives.

The request has two independent `Choice` questions. `danger` chooses
`SAFE_LEFT`, `SAFE_RIGHT`, or `HOLD`. `priority` chooses the same emergency
labels, `HOLD`, or `COL0`--`COL5`. The question descriptions explicitly say
not to count or calculate. The danger answer takes precedence only for an
urgent dodge. Otherwise the priority column is translated by code to a
movement/fire action; an absent or stale column falls back to the nearest
living column computed in code.

The bridge checks `player_shot.active` before firing. A missing/invalid answer,
exception, or timeout uses the conservative fallback. A Jev response is tied to
the request-time state, so the bridge re-reads the current observation before
acting: a current bomb emergency overrides a stale target choice, and the chosen
column is validated against current living columns (otherwise code chooses the
nearest current living column). No action is posted if the episode ended during
the decision. The request timeout is 0.22 seconds (above the measured warm
p90 and below a full second); bomb pressure is summarized at y >= 145, and the
code treats a bomb at y >= 140 and within 16 x-pixels as an urgent collision
candidate. A Jev dodge is accepted only for such a current emergency, with
left/right boundary checks at x=25/135; otherwise the target choice is used.
When aligned within 7 x-pixels of the selected living target, code fires; while
misaligned it moves without firing. The model-confidence threshold is not a
numerical probability: only exact typed labels are accepted, and unknown or
undecided values route to the code fallback. The harness does not treat the
answer’s confidence field as calibrated action probability.

## Confirmation or refutation

Confirmation would be a higher mean score, more invaders killed, and longer
survival on the same seeds, especially seeds 1--4, without fatal episodes. It
would also be useful if the traces show fewer long runs of `fire` while the
laser is active and fewer deaths while a close bomb is present. Refutation
would be unchanged or lower scores despite comparable stale fractions, or
systematic target-column movement that creates more bomb collisions. If
refuted, the next iteration should isolate latency/pipelining or add Astra
adaptation rather than further changing the typed schema.
