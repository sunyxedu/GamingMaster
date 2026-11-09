# Flappy Bird adaptor

This adaptor implements the study-selected **local Gymnasium interface** using
`flappy-bird-gymnasium==0.4.0`. It is a headless symbolic environment: no
browser, display, account, network connection, server, or credentials are
needed. Pygame is used only as the package dependency and the adaptor sets
Dummy SDL defaults before importing it, so it runs on the offline Linux host.

## Interface and timing

`make_game(**kwargs)` returns a tier-2 `RealtimeGame` named
`Flappy Bird (flappy-bird-gymnasium)`. The runner owns the clock and calls
`tick(action)` once per frame. `tick_s` is `1/30` seconds, matching the
package's advertised 30 FPS cadence. `tick()` never sleeps and advances exactly
one Gymnasium simulation frame; `ticks` is the number of frames applied.

The action vocabulary is:

- `IDLE`: no flap impulse for this frame.
- `FLAP`: one upward flap impulse for this frame.

The action spec's `noop` is `IDLE`. Bare case-insensitive strings are the normal
grammar; integer aliases `0=IDLE`, `1=FLAP`, and `{"action": "FLAP"}` are also
accepted. A valid action becomes the held action. A missing action repeats the
held action (initially `IDLE`), as does a malformed action; malformed input is
counted and reported in `last_action_error` rather than raising from `tick()`.

## Game and observations

The default engine configuration is the package's fixed 288x512 world:

- Bird is 34x24 pixels at a fixed x near 20% of the screen.
- Gravity adds 1 px/frame², capped at 10 px/frame downward.
- `FLAP` sets vertical velocity to -9 px/frame.
- Pipes move left 4 px/frame, are 52 px wide, and have a fixed 100 px gap.
- A pipe or ground collision ends the episode; touching y<0 incurs the package's
  top-edge shaping penalty but is not itself terminal.

`observe()` returns strict, compact JSON. `features` preserves the package's
12-value unnormalised feature vector in `feature_names` order:

```text
last_pipe_x, last_pipe_top_y, last_pipe_bottom_y,
next_pipe_x, next_pipe_top_y, next_pipe_bottom_y,
next_next_pipe_x, next_next_pipe_top_y, next_next_pipe_bottom_y,
bird_y, bird_vertical_velocity, bird_rotation
```

The `pipes`, `target_pipe`, `bird`, `target_geometry`, `score`, `frame`, action,
and termination fields are named views and bookkeeping around that vector. For
a pipe still beyond the right edge, the upstream package deliberately reports
screen width and gap boundaries 0/screen height; this is not replaced with a
private pipe peek. Observations also include `valid_trial` and an infrastructure
reason if the local engine fails.

## Seeds and scenarios

`reset(seed)` creates a fresh Gymnasium environment and calls
`env.reset(seed=int(seed))`. The package's seeded NumPy RNG selects the vertical
offset from its fixed set for the initial and recycled pipe pairs. Seed does not
select a hand-authored level and does not alter the default screen, physics, or
spacing. Training seeds are `0..7`; held-out seeds are `101..108`. Both groups
have eight seeds, use the same scenario family, and are disjoint.

The default `EVAL_KWARGS` is `{"tick_s": 0.03333333333333333}`. Constructor
options `screen_size`, `pipe_gap`, `bird_color`, `pipe_color`, `background`, and
`max_ticks` are explicit robustness/time-limit experiments only; they are not
used by the main benchmark and are not changed by seed. `max_ticks` is disabled
by default so ordinary termination comes from the selected game engine.

## Score and diagnostics

`score` is the environment's direct score: the number of pipe pairs passed.
That is the primary score and is exposed also as `pipe_pairs_passed` and
`primary_score`. The package's shaped reward is intentionally diagnostic only:
it normally gives `+0.1` per surviving frame, `+1` on a pipe passage, and a death
penalty. `score_breakdown()` and `info()` record final/max score, pipe passages,
alive frames, cumulative shaped reward, survival shaping, pipe reward, death
penalty, flap count, termination cause, seed, and validity. Policies should be
ranked by pipe pairs passed, not by survival shaping.

The adaptor owns no background thread or external resource. `close()` is
idempotent and closes the Gymnasium/Pygame environment; `reset()` also closes
any prior local environment before creating a fresh one.

## Limits and provenance

This is the maintained community Gymnasium clone selected in the study, not an
official API for the withdrawn commercial Flappy Bird application. Its code and
assets are supplied by the MIT-licensed PyPI dependency; the adaptor does not
copy or distribute proprietary game assets. The wrapper reports the clone's
physics and feature-vector semantics, which may differ from any commercial
binary. The dependency is pinned in `requirements.txt`.
