# 0020 — event-driven formation-relative patrol

Parent: **0016**.  This is a structural rewrite of its deterministic sweep, retaining only the empirically useful combined-FIRE fallback.  I did not modify any sibling candidate.

## Hypothesis

The best controller is scoring well because continuous combined FIRE and lateral motion work, but its lateral policy is a metronome rather than a real-time policy.  In 0016 every trace is dominated by repeated roughly **39–40 tick** left/right legs, independent of the number of surviving invaders, formation height, shield state, and lives.  That is a poor match for the documented mechanic that the remaining formation accelerates as it is cleared.  The controller's temporal-threat reviewer never reached control: lessons report **0 `down_in_shot_field`, 0 promotions, 0 strong dodges, and 0 changed actions**.  A new projectile inference is therefore not a good next experiment.

I test the alternative that late-wave deaths are caused partly by a fixed patrol phase and fixed screen corridor repeatedly exposing the cannon to the same enemy-fire rhythm.  A formation-relative, event-jittered patrol should keep the firing coverage that produced 0016's score while shortening its lateral horizon as the formation becomes sparse/low, adapting its corridor to the currently visible formation, and reinitializing a brief recovery maneuver after a life loss.  The phase jitter is derived from observed score/formation events, not from training seeds, so it breaks a fixed timing alias without hard-coding a seed.

## Archive evidence

- 0016 led at **405.625** but averaged only **25.12 kills**, **665.25 survival ticks**, **0 waves**, and zero lives remaining in every episode. Its code has no alive-count/formation-height patrol phase; it keeps the broad sweep while the wave speeds up.
- In 0016 seed 3, the controller still posts the same long sweep while the score rises to **610** and the shield masks become `[0, 0, 0]` at frame **1988**; it loses a life at frame **2584** and all lives by **2730**. In seed 1 it reaches score **145** by frame **1000**, then receives no useful projectile evidence and ends at frame **1687** with all lives lost. These observations do not prove the bomb cause, but they show that the current timing policy is not adapting before failure.
- 0015 likewise logged many shield escapes but `jev_changed_actions=0` and still lost all lives; 0017's model-selected column intercepts produced **21 zero-score-delta outcomes** and reduced the mean to 270.625. I therefore do not add another target lease or shield-only threat label.

## What changed

### Code-owned controller

1. Keep `right_fire`/`left_fire` on every valid tick. Repeated FIRE is a deliberate control: the ROM enforces the one-laser lock and this is the supported score mechanism.
2. Decode the current alive invader x/y values into `remaining`, `formation_min/max`, and `bottom_y`. The corridor follows those observations with a minimum width, rather than assuming the opening `[38,116]` forever.
3. Use three pressure phases: `OPENING`, `PRESSURE`, and `CRITICAL`. The maximum lateral leg is long in opening, shorter when the formation is sparse/low, and shortest in critical pressure or with one life left. This is a game-time horizon change, not a model-calculated coordinate action.
4. Recompute a leg deadline after score/formation/life events and add a small event-derived phase jitter. Reversals remain bounded by the observed corridor and current x, so a malformed/abbreviated observation cannot create a zero-width or illegal route.
5. On an observed life decrement, clear no death lane and do not learn from the respawn x. Instead, run a short alternating recovery direction and then resume the formation-relative patrol. Explicit decoded bombs get a small code-owned projected collision check; missing bombs never imply safety.

### Astra

One asynchronous, one-token call selects `CAUTIOUS`, `BALANCED`, or `AGGRESSIVE`. It authors only the risk posture: cautious shortens all horizons and adds an edge margin; aggressive preserves coverage; balanced is the default. The call starts immediately and never blocks the actuator. Invalid, verbose, timed-out, or unavailable output leaves the deterministic balanced controller unchanged. This is intentionally one call, since 0016's evidence-review Astra never made an effective action and System-2 is slow.

### Jev

Jev is a phase recognizer, not a geometry calculator or target planner. One request carries two independent reversed `Choice` questions over `SHORT`, `WIDE`, and `CENTER`. The state is the strict JSON schema `space_invaders_patrol_phase_v1`:

- `observation`: frame/score-event/lives/wave, qualitative remaining-invader bucket, `OPENING`/`PRESSURE`/`CRITICAL` formation phase, bottom-threat bucket, player lane, current corridor width and heading, shot state, visible-threat bucket, shield-cover bucket, and `RESPAWN`/`NORMAL`;
- `plan`: Astra posture, current hint, and the rule that every accepted phase still keeps combined FIRE;
- `criteria`: explicit meanings of the three choices.

`SHORT` means shorten the current lateral leg under pressure/recovery/near threat; `WIDE` is allowed only in a genuinely safe opening; `CENTER` is a temporary recovery steer. The answer only affects a 48-frame hint and is rechecked against current code-owned bomb risk, corridor, lives and wave; it never directly supplies a joystick string. The pair must agree, each selected probability must be at least **0.64**, and non-None confidence must be at least **0.22**. Answers older than **32 emulator frames**, from another life/wave, malformed, or disagreeing are discarded.

## Late-answer and fallback policy

The action is always available without a model answer: current x is clamped to the formation-relative corridor, a phase-specific leg deadline chooses the patrol heading, explicit near bombs can override it, and the action bridge maps the result to one of the six legal names. A stale/failed Jev answer expires; a stale/failed Astra answer leaves `BALANCED`. There is at most one outstanding call of each type. Jev uses a 1.0 s timeout and runs in a mailbox; Astra uses a 55 s timeout in a mailbox. The loop waits on the game tick and catches only normal `BudgetExceeded`/`EpisodeOver` termination exceptions, so garbage model output cannot end an episode with a harness error.

## Falsification and diagnostics

The hypothesis is supported if the new controller has longer survival, fewer life losses before the same kill count, or reaches a first-wave clear while retaining comparable score; especially useful evidence is a change from fixed ~40-tick legs to shorter phase-appropriate legs before late deaths. It is refuted if phase changes never occur because observations are unusable, if shorter formation-relative legs reduce kills without survival improvement, or if deaths occur at the same survival frames despite the changed patrol schedule. Logs distinguish `formation_phase`, `patrol_reversal`, `life_transition`, `visible_bomb_escape`, `jev_phase`, `astra_posture`, and `patrol_finished` statistics, including score, lives, phase, model accepts and reversals.
