# Candidate 0003: route-aware Jev bridge

## Parent and targeted change

Parent: `candidates/0000-tier2_jev_only/`.

I keep the parent's architecture: one synchronous Jev `Choice` request per
control decision, no Astra call, and the same real-time fallback loop. The one
targeted change is the **state-to-action bridge**. Instead of passing the full
observation to Jev and allowing it to select any of the 18 raw actions, code
now supplies a compact tactical state and translates Jev's tactic into a
study-derived route action. Geometry, start/restart timing, and imminent
hazard jumps are deterministic; Jev recognizes whether the tactical situation
supports the route or an escape.

## Hypothesis

The parent's zero score is primarily an interface-grounding failure, not a
lack of Jev calls. Jev was asked to infer both the ladder route and the action
semantics from a large state containing all ten ladders and 128 RAM bytes. It
then converged on a locally plausible but strategically useless `right`/`up`
loop, with no typed hazard interrupt and no directional jump bridge.

The traces make the failure concrete:

* In `trace_seed2.jsonl`, the request at tick 37 has Mario at `x=80,y=176`
  and Jev answers `up`; requests at ticks 42, 47, 51, 55, 60, 64, 66, 69,
  71, 73, 76, 78, and 80 continue to answer `up` while Mario remains on the
  bottom band. At tick 180 the run enters `transition` with no reward; the
  later lives are also lost without a rivet removed.
* In `trace_seed0.jsonl`, from ticks 167 through 220 a hazard moves from
  approximately `(32,157)` across the bottom approach to `(113,185)`, but the
  answers remain `up` on every listed decision. The transition begins at tick
  222 while Mario is still at `x=104,y=178`.
* The aggregate parent result is six game-over episodes, score `0.0`,
  `rivets_removed=0`, `screens_completed=0`, despite 1,907 successful Jev
  calls. The 58% stale-tick rate is secondary: the controller never posts a
  jump when its own trace visibly contains a closing hazard.

The bridge therefore tests whether grounding Jev in named tactical choices,
while giving code a bounded route recovery and a hard escape guard, is enough
to make the same fast model produce upward progress and nonzero reward.

## What changed

1. The observation sent to Jev is filtered to stage, semantic player motion,
   route target/alignment, hazard relation/urgency, and progress counters. Raw
   RAM and irrelevant ladder detail are not sent to the model.
2. Code follows the fixed ladder rows from the study, preferring the
   alternating far-side route. Four aligned `UP` decisions are followed by a
   short `UP+direction` alignment retry; after another bounded retry, a second
   ladder on the same row is tried when one exists. This prevents the parent's
   unbounded stationary `UP` loop.
3. Hazards are classified in code using same-band and closing-motion tests.
   A `soon` or `immediate` hazard forces an early directional FIRE action;
   Jev can agree with or refine that jump, but cannot suppress the hard guard.
4. Idle start and post-death transitions are explicit. FIRE is held for at
   most three initial decisions, and a spare life is restarted only after the
   transition has lasted 45 emulator ticks.
5. Rivet state is represented as a bottom-to-top serpentine route with a
   conservative row target, so a successful ramp run has a non-random next
   mode rather than falling back to arbitrary raw actions.

## System-1 state schema and question

Each request is strict JSON with these fields:

```text
{
  game, stage,
  player: {
    horizontal_position, vertical_band, motion, jump_active
  },
  route: {
    command, target_band, target_side, alignment,
    alternate_ladder_available
  },
  hazard: {
    present, urgency, row_relation, closing,
    safe_jump_direction, visible_count, items[]
  },
  progress: {
    lives_remaining, screens_completed,
    rivets_removed, bonus_points_remaining
  },
  last_posted_action
}
```

All coordinate comparisons and rivet-byte interpretation happen in code;
Jev sees named buckets such as `same_band`, `soon`, `on_target`, and
`approach_target`. `items` contains only semantic relative position and
motion, not raw coordinates.

The single `Choice("tactic")` has stage-specific options:

* idle: `start` or `wait`;
* transition: `restart` or `wait`;
* active route: `follow_route` or `climb_target`, `jump_left`,
  `jump_right`, and `hold_safe`.

The first option is the computed nominal tactic so the executable stand-in
model remains safe. Option descriptions explicitly say that code owns exact
joystick mapping and that FIRE is only for a closing hazard or a known gap.
The answer is never used to perform arithmetic or invent an action name.

## Thresholds and fallback policy

* A visible hazard is a candidate when it is within 26 vertical pixels of the
  player and within 62 horizontal pixels while moving toward the player (or
  descending into the player's band). `soon` is at most 62 horizontal pixels;
  `immediate` is at most 44. The jump side is selected from landing room, so a
  Mario at the right edge jumps left rather than further into the wall.
* An imminent/soon hazard is a hard code-level jump guard unless a jump is
  already active. A Jev jump answer is accepted at probability `>=0.66` for a
  non-immediate watch hazard; otherwise the deterministic route action wins.
  This prevents a low-confidence FIRE from disrupting a ladder approach.
* On a ladder target, the bridge uses four normal `UP` decisions, four
  directional-UP retry decisions, and four more `UP` decisions. It then tries
  the alternate same-row ladder, if available, and finally alternates short
  directional-UP nudges. A genuine upward y change resets the retry counter.
* On a Jev timeout or ordinary failed Jev call, code posts the current
  deterministic fallback immediately and waits for one tick; the previous
  action is never allowed to be replaced by malformed model text. Budget,
  episode-over, and model-unavailable exceptions retain the contract's normal
  propagation behavior.
* With no System-1 backend, the same deterministic bridge runs at one action
  per tick instead of idling. Malformed answer values, missing probabilities,
  or unknown options also use the fallback.

## What would confirm or refute the hypothesis

Confirmation would be any positive reward, an observed transition from the
bottom band to a higher ladder band before the first life loss, and directional
FIRE actions around the trace's previously ignored closing hazards. Stronger
confirmation is a completed ramp screen or fewer repeated `UP` decisions at a
stationary `y=176`.

The hypothesis is refuted if the candidate still finishes all training
episodes at score zero with Mario stationary on the bottom band, or if the
new bridge posts no jumps despite the semantic hazard state being `soon` or
`immediate`. If it climbs reliably but dies at the first unusual route or on
the rivet screen, that would instead identify a later route/hazard calibration
problem rather than vindicate the parent's unconstrained action interface.
