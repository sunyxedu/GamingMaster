# Candidate 0015 — bounded moving aim in the 0011 motor

## Parent and targeted hypothesis

Parent: `candidates/0011/`. I keep its heading-frontier exploration, finite collision/no-novelty escapes, visible-label combat and USE handling, asynchronous Jev bridge, and bounded Astra policy. The one behavioral change is in the combat aim correction: after six consecutive turn-only aim decisions, the controller takes a four-tic lateral/retreat movement lease before turning again.

**Hypothesis:** 0011's worst episodes are losing health while the aim controller oscillates between `turn_left` and `turn_right` against a close hitscanner. The controller turns until the target is within a narrow angular tolerance, but every observation recomputes the target and error; near the crosshair, small target motion can reverse the required turn. This creates long stationary exposure rather than a navigation failure. A short moving lease should preserve the parent's ability to reacquire a target while reducing damage during aim correction.

## Trace evidence

The parent trace for seed 1012 is direct evidence of this failure mode. Its action counts were **1,179 `turn_left` and 498 `turn_right`** versus 788 `forward`; the longest pure-turn run covered tics **2420–2587 (168 tics)**, while health fell from **27 to 9** and the player moved only about 50 world units. Near the end, tics **2645–2742 (99 observations)** alternated turn actions at almost exactly `(752,-304)`; visible Zombiemen were as close as **37–73 units**. The episode ended with health **6**, only 2 kills/3 items, score **3.213**, and no exit.

The same signature appears on parent seed 1011: a **172-tic** turn run from tic 200 to 371 reduced health 91→81, a **127-tic** run from 511 to 637 reduced it 73→50, and the final 87-tic run faced a ShotgunGuy about 274 units away while position changed by only about 31 units; health finished at 9. This is not fixed by adding more heading memory: 0011 already improved local exploration over 0008, but its combat still permits unbounded turn-only control. Candidate 0013 changed escape priority rather than the aim loop and still documented a 227-tic turn streak, so this candidate tests a different, narrower intervention.

## What changed

In `combat_action` only, the old unbounded branch

```text
if target_error is outside tolerance: return turn_action(...)
```

now counts consecutive off-angle aim decisions. At six, it chooses a legal `backward_left/right` action when the target is close or damage is recent, otherwise a legal `strafe_left/right`, flips the dodge side, and holds that movement for four calls. The lease is canceled when the target is no longer an immediate blocker or becomes aligned. Weapon switching, firing tolerance, target ranking, emergency projectile strafing, route goals, collision escapes, heading-frontier selection, USE cadence, Jev questions, and Astra policy are unchanged. No map, seed, WAD, or hidden geometry is used.

The movement lease is code-owned and does not wait for a model. It is therefore safe while Jev or Astra is pending and posts the same legal action on every processed engine state. The controller still uses the parent's final legality check.

## Confirmation or refutation

Confirmation requires traces to show substantially fewer long pure-turn streaks during close visible threats, more displacement during aim correction, and higher terminal health or longer survival on seeds such as 1012/1011. The strongest confirmation is any increase in exit success without a rise in stale ticks; improved survival with zero exits would still support the local combat hypothesis but identify exit discovery as the next bottleneck.

The hypothesis is refuted if the four-tic leases do not reduce stationary turn streaks, cause repeated wall/collision loops, materially reduce kills without improving health, or lower aggregate score/exit success. More secondary kills/items alone are not confirmation. The change should not be credited for an exit unless the trace shows an actual terminal exit.

## Tier-2 Jev interface and fallback policy

The Jev state is unchanged from 0011 and contains only derived, current facts:

```text
{
  task: "Reach and activate the exit alive; optional kills/items are secondary.",
  rules: "visible labels only; code owns bearings, collision, ammo legality and the 35 Hz motor",
  health: critical|wounded|healthy,
  armor: none|present,
  damage_recent: boolean,
  motion: {
    phase: escape|turning|sweep,
    blocked: boolean,
    novelty: new_cell_recently|revisiting,
    visited_cells: bounded integer,
    same_cell_visits: bounded integer
  },
  threats: [id, name, category, range, direction, level,
            hitscan, major, body_blocker],
  local_goals: [id, name, category, range, direction, level, ...],
  goal: {kind, range}|null,
  macro: {hand, sweep, turn_bias, previous_action}
}
```

One asynchronous `ctx.decide` request batches three independent typed questions:

* `route`: `Choice` among `advance`, `turn_left`, `turn_right`, `use`, `retreat`, and `unknown`; a selected turn sets the next macro bias, `use` requests a pulse, and `retreat` can start the parent's finite escape.
* `dodge`: `Choice` among `left`, `right`, `back`, and `none`; `left/right` only biases the future fire/movement side. The new moving-aim lease remains deterministic and does not ask Jev to select its button.
* `open`: `Noul` asking whether the named visible interaction/blocked-face facts justify a USE pulse.

A Choice is accepted only when its selected probability is at least **0.57** for `route` or **0.54** for `dodge`; a confidence below **0.10**, malformed answer, failed/late call, or answer older than **55 Doom tics** is ignored. `open` requires probability at least **0.72**. While a request is pending or rejected, the fallback is the deterministic priority stack: close-projectile strafe, immediate legal combat (including the bounded aim lease), an already committed escape, visible goal steering, blocked-face USE, periodic USE, and heading-frontier sweep. All actions are rechecked against `game.action_spec.names`.

Astra remains a background author of only `hand`, `sweep`, and `combat`, with malformed or late JSON leaving the current policy intact. Model futures are canceled on terminal/deadline, and garbage stand-in model text is parsed defensively. The moving-aim change is deliberately below both model layers so its effect can be separated from Jev/Astra influence.
