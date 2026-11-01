# Candidate 0011 — heading-frontier refinement of committed macro exploration

## Parent and one targeted hypothesis

Parent: `candidates/0008/`. I keep its committed sweep/escape motor, visible-label combat and interaction rules, asynchronous Jev bridge, and bounded Astra macro policy. The only behavioral change is how a macro turn is selected: the parent already records `heading_visits`, but it uses that counter only for question ordering. This candidate uses it in the motor to choose a locally under-observed outgoing heading at the current measured cell.

**Hypothesis:** the parent's remaining failure is a deterministic geometric loop, not a lack of model freshness. A fixed approximately-90-degree sweep and alternating collision escape can revisit the same corridor indefinitely. Choosing the least-observed local heading should break those loops and expose new corridors or the exit without adding map knowledge or model latency.

## Trace evidence

The best parent episode, `candidates/0008/trace_seed1008.jsonl`, reached 281 cells and 7 kills/12 items but still ended at the deadline with no exit. It spent **13,628 forward** and **15,586 turning** ticks. Its harness events show the same local cycle repeatedly: around tic 18,084 it escaped at `(1198.4,-270.5)`, around tic 18,220 it was back at `(1062.2,-267.5)`, and around tic 18,298 it was back at `(1197.1,-267.5)`. Later collision pairs recur at approximately `(596,80)` and `(657,65)` (for example tics 24,525–25,212 and again through 26,429), while health stayed 96. More time and more USE pulses did not turn this loop into an exit.

The parent's worst cited episode, `trace_seed1012.jsonl`, logged **74 macro escapes** and **40 macro turns**. At tic 6,276 it was already escaping from a repeated collision; at tic 7,060 a Jev retreat escape began with health 4. The final observations alternate one-to-four-tic `turn_left`/`turn_right` runs at the unchanged position `(894.9,-265.6)` and end at health 1, with 1,807 left and 1,875 right turns overall. The parent's own final Astra state reported `same_cell_visits: 12`, yet the motor still chose the old turn pattern. These are direct evidence that the existing collision/novelty watchdog detects recurrence but does not choose a new local branch.

## What changed

`frontier_turn_plan` is the single new control mechanism. At each sweep boundary and each bounded collision/no-new-cell/semantic-retreat escape it:

1. computes the current 64-unit visited cell and current observed heading;
2. evaluates candidate turns of roughly ±90 degrees first, then ±45, ±135 and 180 degrees;
3. reads the code-maintained `heading_visits[(cell, heading_bin)]` count for each resulting heading;
4. chooses the least-observed heading, using the current hand or an accepted Jev turn only as a tie-breaker; and
5. executes the same finite motion macro as the parent (`backward` then a bounded turn then `forward`).

No WAD/map data, seed-specific route, hidden object list, or new model call is introduced. Visible interaction handling, periodic USE, combat safety, weapon legality, Astra's `hand/sweep/combat` policy, and all Jev questions are otherwise unchanged. The model still cannot issue a raw button; code rechecks every resulting action against the action spec.

## Confirmation or refutation

Confirmation requires more than normalized kills/items: compared with 0008, traces should show fewer repeated position/heading cycles, fewer escape pairs returning to the same cells, and either a material rise in `exit_success` or a visibly longer frontier before death. Health should not fall below the parent's worst episodes merely because the turn choice changed. A result with the same exit rate but fewer loops would support the motor diagnosis but show that exit recognition/activation is the next bottleneck.

The hypothesis is refuted if heading-frontier turns still cycle the same cells, if they create substantially more in-place turning or stale ticks, or if they reduce survival without increasing exit completion. More kills/items alone do not confirm it; the primary test is reaching and activating the map exit.

## Tier-2 Jev interface and fallback

The state sent to Jev is the parent's compact, derived JSON schema:

```text
{
  task,
  rules,
  health: critical|wounded|healthy,
  armor: none|present,
  damage_recent: boolean,
  motion: {phase, blocked, novelty, visited_cells, same_cell_visits},
  threats: [id, name, category, range, direction, level,
            hitscan, major, body_blocker],
  local_goals: [id, name, category, range, direction, level, ...],
  goal: {kind, range}|null,
  macro: {hand, sweep, turn_bias, previous_action}
}
```

Code derives ranges, directions, threat classes, blocked status, cell counts and damage recency from the currently rendered observation. Jev receives no coordinates, raw map, omniscient actors or arithmetic task. One asynchronous request batches three independent typed judgments:

* `route: Choice` over `advance`, `turn_left`, `turn_right`, `use`, `retreat`, `unknown`. An accepted turn is only a tie-breaker for the heading frontier; `use` sets a bounded USE pulse and `retreat` starts the finite escape only when no immediate threat owns control.
* `dodge: Choice` over `left`, `right`, `back`, `none`. It biases only future combat strafing and cannot override target, ammo or collision safety.
* `open: Noul`. A value at least **0.72** requests one bounded USE pulse.

A Choice is accepted only with selected-option probability at least **0.57** for `route` or **0.54** for `dodge`; explicit confidence below **0.10**, malformed values, timeout, runtime failure, or age over **55 Doom tics** rejects it. An accepted answer is rechecked against the newest observation and legal actions. Requests use the parent's asynchronous cadence (about 0.63 seconds with threats and 1.49 seconds when quiet) and a 1.35-second timeout. While pending or rejected, the deterministic frontier/sweep, combat, USE and escape controller acts on every processed tick; it never falls back to a blocking model call or an unsafe `noop`.

Astra remains a background bounded-policy author, called at the parent's initial/review cadence. Its only executable fields are `hand`, `sweep`, and `combat`; arbitrary text, late output, unknown values and cancelled calls leave the existing policy intact. Astra does not select a heading, route, target, or button. Terminal/deadline handling and future cancellation are unchanged from the parent.
