# Overcooked adaptor

This adaptor connects the Meta-Harness to the local, open-source
`HumanCompatibleAI/overcooked_ai` symbolic MDP at commit
`739950a079cdaed5a44fcc662efc40244c205d06`. It does not launch the commercial
Overcooked game, a GUI, a server, or an account. The installed evaluator Python
imports the upstream `OvercookedGridworld` and `OvercookedEnv` directly, with
`SDL_VIDEODRIVER=dummy` for headless pygame imports.

## Interface and clock

`make_game()` returns a tier-2 `RealtimeGame`. The upstream environment is a
synchronous transition oracle, so the adaptor exposes one real MDP transition
on each `tick(action)` at `tick_s = 1/6` seconds. `ticks` is the authoritative
`OvercookedState.timestep`; the harness runner owns the wall-clock ticker. The
adaptor itself starts no process or thread. A fresh valid action replaces the
current mailbox action; `None` repeats it, and malformed input is converted to
`stay` and recorded in the next observation rather than raising.

The controlled chef uses one of `north`, `south`, `east`, `west`, `stay`, and
`interact`. Directions are single-tile primitives. `interact` acts on the tile
immediately in front of the chef. The partner is scripted: the standard one
three-onion-order scenarios use the upstream `GreedyHumanModel`; mixed and
forced-coordination scenarios use a deterministic feature/path planner. The
forced layouts use an explicit shared-counter handoff protocol and are not fed
to
`GreedyHumanModel`, whose upstream source documents that it is not suitable for
unilateral completion there.

## Observations

Observations are strict JSON-safe dictionaries under 200 kB. They include the
layout terrain, scenario/seed, controlled-agent index, tick and deadline,
`state.to_dict()` (players, held objects, counter/pot soups, recipes and
`timestep`), sparse team score, per-agent sparse/shaped transition rewards,
recent and cumulative event counts, actions, and the partner's current task.
No screenshot or binary perception is needed: the selected interface is
symbolic.

## Seeds and scenarios

The fixed split in `game.py` is:

- train/development: `0` cramped_room, `1` asymmetric_advantages, `2`
  coordination_ring;
- held out: `100` counter_circuit_o_1order validation, `101` forced_coordination,
  `102` forced_coordination with controlled-agent index reversed, `103`
  counter_circuit, `104` cramped_room_tomato, `105` asymmetric_advantages_tomato
  with reversed control and a randomized valid start, `106`
  forced_coordination_tomato with a randomized valid start, and `107`
  counter_circuit_o_1order with a randomized valid start.

Seeded selection controls the layout, controlled-agent index, valid starting
positions/orientations for randomized cases, and deterministic partner
style/tie-breaking. All starts are made through the upstream MDP's valid
walkable-position set. The upstream recipe/pot state is authoritative; no
commercial-game timers, burning, tips, combos, or star thresholds are added.

## Score and diagnostics

`score` is exactly the cumulative sparse team reward: the sum of
`OvercookedEnv.step`'s `sparse_r_by_agent`, or equivalently the terminal
`ep_sparse_r`. It counts accepted soup deliveries using active recipe values
and bonus orders; shaped rewards are never included. `score_breakdown()` and
`info()` additionally report sparse reward by agent, deliveries, episode
length, event counts, collision/stuck ticks, pot idle time, and handoff
attempt/success/failure diagnostics.

There are no credentials or external services. The package dependencies and
pinned upstream source are listed in `requirements.txt`; optional headless
settings are comments in `env.example`. Runtime-generated files, if later
added, belong under `runtime/` (the current implementation needs none).
