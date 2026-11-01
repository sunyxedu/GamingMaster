# Candidate 0015 — filtered receding-horizon interception

## Parent and targeted hypothesis

Parent: `candidates/0014/`. I keep its event-contract architecture,
interval motor, asynchronous code-owned action loop, and advisory Jev/Astra
bridge. This candidate makes one low-level change: replace the parent's
instantaneous per-frame velocity with a same-sign two-sample RAM estimate and
refresh the catch interval as a filtered receding horizon during descent.

**Hypothesis:** 0014 mistakes the Atari RAM decoder's normal subpixel cadence
for new physical velocity. Components alternate adjacent values (for example
`dy=1,2,1,2`, and later `dx=-1,-2,-1,-2`). Its 0.75 component threshold then
recomputes a reflected landing that jumps between distant intervals, driving
motor reversals. A same-sign adjacent mean should represent the physical
velocity; refreshing with that stable estimate each frame should track the
moving ball without the parent's flip storm or a stale long-horizon landing.
Sign changes are retained immediately as collision clues.

## Trace evidence and probes

0014 was already fresh and error-free (`0.000296` stale fraction), so this is
not a stale-tick or model-availability change. In the selected worst traces:

* Seed 3 loses at frame 7064 after `return_contract` landing values alternate
  `35.5, 20.67, 35.0, 21.33, ...` from frames 7034–7063. Actions include
  LEFT at 7035–7042, RIGHT at 7048–7053, and NOOP at 7059–7064.
* The same seed loses at frame 8128 after left-edge contract values alternate
  around `14.7` and `18.5`/`17.5`, with LEFT/RIGHT/LEFT runs.
* Seed 1 shows the same pattern at frames 3326–3356, alternating landing
  values near `111.5` and `96.3` before the loss at 3357.

A guard-only probe that stopped resetting a valid interval at `y<94` still
scored 388 on seed 3 and reproduced the identical sequence. A diagnostic of
the parent recorded frames 150–180 as `dx=1`, `dy=1,2,1,2` with a valid
descending ball and a contract velocity following each alternation. A later
diagnostic showed raw `dx=-1,-2` while the paddle was held at 87 for a
predicted interval 82–98, but the actual ball arrived left of that interval.
A first filtered-and-latched trial removed the flips but left a stale interval
and scored 265; the final change keeps the filter but refreshes the interval
every descending frame. These probes isolate the instantaneous estimate and
stale receding horizon as the target, not the model layer.

## Code change and safety boundary

The code caches one filtered velocity result per frame because both `_flight`
and `_make_return_contract` query it. When the previous observation is valid
and each component has the same sign, it uses the adjacent mean for `dx` and
`dy`; a sign change uses the current sample without smoothing. After the ball
is below the brick-band guard and descending, `_make_return_contract` is
called each frame with that filtered velocity. Its interval remains a full
paddle-overlap set, not a point; motor leases and braking remain unchanged.

The interval refresh is not enabled in the collision guard: if no valid
contract exists below `y<94`, the old wide home fallback remains. Upward `dy`,
brick hits, side bounces, paddle bounces, serves, life losses, and wall
transitions still reset the contract. The filtered result is cached only for
the current frame and all model, FIRE, legality, deadline, and episode
handling is unchanged. No model answer emits a raw action.

## Jev state, questions, thresholds, and fallback

The state sent at event boundaries is unchanged from 0014 and is strict JSON:

```text
{
  frame, event, lives, score, serve_required,
  phase, urgency, speed, landing_lane, catch_interval,
  paddle_motion, eta, fallback_action,
  rally_contract, astra_plan, left_edge, right_edge,
  left_count, right_count, bricks_remaining, wall_index,
  contract_epoch, rally_version
}
```

`phase` is `serve`, `ascending`, `collision_zone`, `descending`, or
`uncertain_descent`; `urgency` is `serve`, `early`, `near`, or `imminent`.
Code owns arithmetic, reflection, counts, legality, and action selection. The
fallback is FIRE when `serve_required` and lives remain, otherwise the
interval motor. Jev is never on the 60-Hz critical path.

Each boundary request batches two reordered `Choice` questions. Both must
agree on `PRESERVE`, `LEFT`, `RIGHT`, or `NO_CHANGE`, with selected-option
probability at least `0.55`. LEFT/RIGHT only shifts a plentiful-life early
home anchor; it cannot alter the catch interval. Stale-version, post-life/wall,
malformed, late, failed, low-life, or open-edge advice is rejected. The motor
continues while Jev runs. Astra remains a gated one-shot short-token prior;
timeout or unrecognized prose is ignored.

## Confirmation or refutation

Confirmation would be fewer alternating landing targets and lease reversals in
the same descending flights, fewer losses at comparable frames, and a mean
above 0014's `411.5` without more stale ticks or errors. The strongest signal
is clearing the first wall in 0014 seeds 1 and 3, each of which finished with
eight bricks remaining.

Refutation is unchanged or earlier loss, filtered targets still oscillating,
or losses while the paddle is inside the refreshed overlap interval. That
would implicate collision timing, sign-event detection, or the actuator model
and justify a separate transition-search experiment rather than more
model-interface work.
