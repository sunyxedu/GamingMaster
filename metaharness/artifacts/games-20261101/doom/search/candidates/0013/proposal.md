# Candidate 0013 — honor committed escape macros before routine combat

## Parent and one targeted hypothesis

Parent: `candidates/0011/`. I keep its heading-frontier exploration, finite collision/no-novelty/retreat macros, visible-label combat and USE rules, asynchronous Jev bridge, and bounded Astra policy. The only behavioral change is the priority of an already-started escape sequence.

**Hypothesis:** The worst episodes are not merely failing to choose a new heading; they are starting a finite escape and then abandoning it as soon as a routine monster is visible. That leaves the player turning in place in the line of fire. If a committed escape is allowed to finish (while the existing close-projectile emergency still overrides it), the player will create separation/leave the trap, preserve health, and give the heading-frontier search a chance to continue toward the exit.

This is a safety/execution fix, not a new map assumption or a new model interface.

## Trace evidence

The parent’s worst episode, `candidates/0011/trace_seed1012.jsonl`, makes the priority inversion concrete:

* At tic **1950** the harness records `macro_escape` with `reason="jev_retreat"`, but the subsequent action stream is immediately dominated by `turn_left`/`turn_right`; from ticks 1900 onward it contains **401 `turn_left` and 310 `turn_right`** actions, but only 78 `forward` actions.
* The player is then exposed in the same arena: at tic 2404 it is at about `(762,-207)` with health **42** and Zombiemen/barrels visible; at tic 2454 it is at `(764,-267)` with health **27**; at tic 2604 it is at `(752,-280)` with health **9**. The terminal observation is still at `(752,-304)`, health **6**, damage taken **117**, with `last_action="turn_right"`, and `exit_success=0`.
* The parent’s `action()` checks `immediate` combat before `self.escape`, so a visible hitscanner can replace the escape’s backward/turn/forward deque on every observation. The trace’s repeated retreat event followed by aim turns is the expected signature of that bug, rather than evidence that the frontier heading was ineffective.

The best parent episode, `trace_seed1008.jsonl`, demonstrates why this matters even when the motor survives: it reached score **18.428** with substantial kills/items and full health/armor, but still ended by the deadline with `exit_success=0`. The change does not claim to solve exit discovery; it tests whether losing survival in local combat is preventing the existing exploration motor from getting a completion opportunity. It leaves the parent’s exit/interaction behavior unchanged so that result is interpretable.

The archive also rules out simply adding more frontier bookkeeping: candidates 0009 and 0010 underperformed 0008, while 0011’s local heading memory helped. Thus this candidate does not add another graph or another model call.

## What changed

In `action()` the order is now:

1. If a currently rendered projectile is within the parent’s 190-unit emergency radius, strafe as before.
2. If `self.escape` contains a finite collision/no-novelty/semantic-retreat macro, dequeue its next legal action and let it complete.
3. Otherwise compute the current threat and run the parent’s weapon/aim/combat logic.

The old post-combat escape check is removed; the escape branch that follows a newly detected collision remains unchanged. Every tick still posts a legal deterministic action. No labels, coordinates, seed-specific route, omniscient geometry, weapon rule, question, cadence, or Astra prompt is changed.

## Confirmation or refutation

Confirmation requires a measurable reduction in the parent’s death trap: traces should show escape sequences actually containing their backward and forward portions, fewer long turn-only runs after `macro_escape`, more displacement after a retreat, and higher terminal health/survival on episodes like seed 1012. The strongest confirmation is any increase in `exit_success` or longer exploration before death without increasing stale ticks.

The hypothesis is refuted if escape macros still terminate at the same cells, if survival falls because combat is ignored too long, or if the aggregate exit rate remains zero while the change only adds harmless movement and no reduction in turn traps. More kills/items without less turn-in-place exposure would not confirm it. If survival improves but the best episodes still time out with no exit, the next bottleneck is exit discovery/activation rather than this execution inversion.

## Tier-2 Jev interface and fallback policy

The Jev interface is intentionally unchanged from 0011. Code sends a small derived JSON object, rebuilt from the newest observation:

```text
{
  task: "reach and activate exit alive",
  rules: "visible labels only; code owns bearings, collision, ammo legality and 35 Hz motor",
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

One asynchronous `ctx.decide` batches three independent questions:

* `route`: a `Choice` of `advance`, `turn_left`, `turn_right`, `use`, `retreat`, or `unknown`. A sufficiently confident `retreat` starts the finite escape only when no current immediate-threat rule has already taken control; once started, this candidate’s code now honors it.
* `dodge`: a `Choice` of `left`, `right`, `back`, or `none`, used only to bias future combat strafing.
* `open`: a `Noul` asking whether the named visible facts justify one USE pulse.

A Choice is accepted only when the selected option probability is at least **0.57** for `route` or **0.54** for `dodge`; explicit confidence below **0.10**, malformed values, timeout/runtime failure, or an answer older than **55 Doom tics** is ignored. `open` requires probability at least **0.72**. Jev never emits a raw button or route. While a request is pending or rejected, code continues the current finite escape, projectile strafe, deterministic combat, USE pulse, or heading-frontier sweep. The only emergency priority above an escape is the parent’s close-projectile strafe; all legality is rechecked before `ctx.act`.

Astra remains a background, bounded policy author at the inherited cadence. Its accepted fields are only `hand`, `sweep`, and `combat`; malformed or late JSON leaves the current policy unchanged. Astra cannot cancel an active escape, choose a hidden route, or issue a button. System-2 and System-1 futures are cancelled on terminal/deadline, and model garbage cannot escape the parser or stop the episode.
