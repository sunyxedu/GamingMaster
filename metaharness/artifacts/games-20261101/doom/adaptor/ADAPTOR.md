# Doom adaptor

## Connection and lifecycle

This is a tier-2 adaptor for classic single-player Doom. It uses the official
`vizdoom==1.3.2` Python binding directly, with `DoomGame` configured for
headless software rendering and `Mode.ASYNC_PLAYER`. No display, GPU, server,
account, or online service is used.

One adaptor instance owns one `DoomGame`. `reset(seed)` stops and closes the
previous native episode, selects one map and skill, sets ViZDoom's native RNG
seed, initializes a fresh episode, and starts one controller thread. That
thread is the only caller that advances the native game. It repeatedly submits
the current action and calls `advance_action(1)`, so the world continues at
Doom's 35-tic/s cadence while the model is deliberating. `tick(action)` only
updates the action mailbox and returns; `ticks` counts accepted action posts.
`close()` stops and joins the controller and closes the native game, and is
idempotent.

`VIZDOOM_WAD_PATH` may point at a legally obtained `doom.wad`, `doom2.wad`, or
compatible Freedoom IWAD. When it is unset, ViZDoom's own bundled
`freedoom2.wad` is copied into this adaptor's `runtime/` directory as an
explicit no-account fallback. The observation and evaluator-only `info()`
identify the family and whether the bundled fallback was used. A supplied
IWAD whose filename is ambiguous must be paired with
`VIZDOOM_IWAD_FAMILY=doom1|doom2|freedoom1|freedoom2`; map families are never
silently mixed.

## Scenarios and seeds

Each episode is one map, not an implicit campaign. The mapping is:

- train seeds `1001..1008, 1011..1018`: early maps E1M1-E1M4 or MAP01-MAP04,
  alternating skills 2 and 3;
- held-out seeds `2001..2008, 2011..2018`: a mixed early/unseen set through
  E1M8 or MAP08 at skill 4 and new ViZDoom RNG seeds;
- `VIZDOOM_SKILL=5` is a separate stress evaluation override, not part of the
  train/held-out split;
- an explicit `VIZDOOM_MAP` fixes the map for every reset and is checked against
  the selected IWAD family.

Doom 1 families allow E1M1-E4M9; Doom II families allow MAP01-MAP32. Original
Doom and Freedoom results remain separately labeled. The declared `SEEDS` are
non-empty, disjoint, and contain 16 seeds per group. Custom integer seeds use
the held-out map mix at skill 3 unless environment overrides are supplied.

## Actions

The action vocabulary is native Doom buttons translated using
`get_available_buttons()` rather than relying on hard-coded vector positions.
Movement/fire actions are held until a replacement arrives. `use`, weapon
selection, and the weapon pulse actions are cleared after one engine tic.
Legal fixed names are exposed in `action_spec`; in addition, the parser accepts
`move <forward> <turn>` with each value `-1`, `0`, or `1`, and `select <1-7>`.
Unknown or malformed actions are ignored and appear as `action_error` in the
next observation rather than raising from `tick()`.

The initial and missed-tick action is `noop`; after the first accepted action,
the last sustained action remains held. `tick_s` is exactly `1/35` seconds.

## Observations

Observations are strict, compact JSON dictionaries. They contain the IWAD
family, filename, and source, map, skill, seed, tic/elapsed time, engine and
controller tick counts, health, armor, death/ground/attack readiness, selected
weapon and ammo, all AMMO0-AMMO9 and WEAPON0-WEAPON9 values, KILLCOUNT/ITEMCOUNT/
SECRETCOUNT/DAMAGE_TAKEN, player position/angle/velocity, latest and total
ViZDoom reward, terminal state/reason, and visible `state.labels` entries with
object name/category, label value, pixel bounding box, world position, and
velocity. Labels are view-visible only. Screen pixels, whole-map automap,
sector information, and omniscient object lists are intentionally not exposed;
the primary interface is symbolic and stays well below 200 kB.

## Score and terminal metrics

A non-death, non-timeout ViZDoom terminal episode is recorded as
`completed_by_exit`. Exit success contributes 1000 points and therefore
strictly dominates optional exploration. Secondary score components are bounded
kill/item/secret count progress, final health/armor, remaining ammo, a small
time tie-breaker, and the native total reward. Kill/item/secret counts are
reported in `info()` and in the observation. ViZDoom exposes current counts but
not map totals through this API, so exact percentages and map totals are
reported as `null`; the bounded count references are documented rather than
pretending to be completion percentages. `score_breakdown()` exposes every
numeric component, including `vizdoom_reward`.

`info()` records the IWAD family, filename/source, map, skill, seed, scenario
group, completed-by-exit/death/timeout, elapsed tics/seconds, native total reward,
progress counts, final health/armor/ammo, and trial validity. Native engine or
connection failures set `valid_trial=False` and `invalid_reason` so the
orchestrator can retry the seed. Bad play, death, timeout, and malformed
actions are not infrastructure failures.

## Limits

This adaptor does not provide an exact route planner and it does not chain
maps; a campaign harness must save each map's terminal metrics before resetting
to the next map. Exact commercial map behavior requires the corresponding
lawfully obtained IWAD. Bundled Freedoom is a faithful offline engine
integration but its maps and assets are not identical to commercial Doom.
`VIZDOOM_MODE=PLAYER` is rejected because its synchronous stepping would violate
the tier-2 real-time contract.
