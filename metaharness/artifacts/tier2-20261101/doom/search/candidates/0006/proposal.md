# Candidate 0006 — bounded moving macros, semantic supervision

## Parent and hypothesis

Parent: `candidates/0001-tier2_astra_only/`. I also retained the strongest
execution evidence from `0003` and the diagnostic traces of `0004`/`0005`.
This is a structural rewrite, not a latency or threshold tweak.

**Hypothesis:** the dominant remaining failure is an ownership mismatch at the
motor boundary. A raw Astra action stays held for tens of seconds while Astra
thinks; a raw Jev choice is refreshed often but can repeatedly select `forward`
or an in-place turn without testing whether it moved the player. The
contact-cartography rewrite improved survival in some episodes, but its combat
servo can still let one visible actor own the controller: on seed 1006 it
posted `turn_left`/`turn_right` for 30,952 of 31,499 ticks, and on seed 1007
it alternated those turns until health reached 1. Thus the new controller
should make every model answer semantic and short-lived, while code owns
movement, aiming, collision tests, USE pulses, weapon legality, and emergency
retreat. A finite moving macro must either make measurable progress or be
abandoned; no label may freeze the player in a turn-in-place loop.

The testable prediction is that this separation will preserve the idle
baseline's survival floor better than model-owned combat while increasing
measured cell coverage, item/weapon/key progress, and (most importantly) the
chance of reaching a visible exit and pulsing USE on it.

## Evidence from the archive

- The Astra parent (`0001`) posted no fresh-action replacement for its long
  inference gaps. On seed 1001, `noop` was held for 4,533 ticks and the only
  other action was `fire_forward` for 541 ticks; it ended at health 1 without
  an exit. On seed 1004, repeated `use`/`forward` holds consumed 22,932 ticks
  and ended at a fixed-looking location with health 6. A slow planner cannot
  be the real-time actuator.
- The best candidate (`0003`) fixed one important parent defect: its
  position-based watchdog changed position after `stuck_recovery` events.
  But seed 1002 still spent 15,102 ticks on `forward` and 14,204 on
  `forward_left`, reached only 11 items, killed 0 monsters, and hit the
  deadline. Seed 1008 ended at health 3 after 984 `forward` and 544
  `forward_right` ticks. Breaking a fixed coordinate loop is not the same as
  selecting and activating the exit.
- The 0004 trace makes the motor problem concrete. On seed 1006 the summary
  repeatedly reported `steer_pickup` at the initial position
  `[224,1184,-21]`; the final action distribution was 15,491 `turn_left` and
  15,461 `turn_right`, with 16 items but no kills or exit. On seed 1007 the
  controller used 83 `turn_right` and 82 `turn_left` actions before death at
  health 1. Freshness alone (0004/0005 had about 0.5% stale ticks) did not
  make those actions useful.
- The 0005 trace shows the opposite over-correction. Seed 1001 had only two
  stale ticks, but its executor still posted 161 `turn_left` and 127
  `turn_right` actions, only 54 `forward` actions, and died at health 1 with
  no kills/items. The semantic bridge was fresh, but the motor policy was
  allowed to oscillate while steering toward an actor/landmark.
- The idle trace is a useful safety bound: it survives at health 100 with one
  position and no progress. The proposed controller therefore never asks
  Jev/Astra to authorize emergency evasion, ammo legality, or collision
  recovery; these are deterministic and can fall back toward movement/idle.
- All archived candidates had zero exit successes. Optional item/kill gains
  are not enough: a visible exit/door/switch must be treated as a primary
  measured landmark and receive a bounded USE pulse at contact.

## Structural change

### Code owns a finite motor/search loop

`MovingMacroController` normalizes the public observation and keeps only
observed data: visible labels, exact position/angle, health/ammo deltas,
coarse visited cells, measured adjacency, breadcrumbs, attempted headings,
and blocked faces. It does not enable all-object buffers or infer hidden map
geometry.

1. **Moving macro, not raw button:** every navigation choice is a short
   steering/movement macro. A large angular error gets at most three turn
   ticks, and after two correction requests the next translation/fire-strafe
   macro is held for about seven tics before another turn can be requested. A
   smaller error uses forward/diagonal movement. `forward_left`/`forward_right`
   and fire-strafe actions keep translation occurring while the view is
   corrected. The controller recalibrates the sign/rate of turning from
   observed angle deltas and never lets a target own an unbounded
   turn-in-place loop.
2. **Measured route search:** an exploration goal is a finite radial probe
   toward one of eight headings. Ten ticks of movement with less than 9 units
   of displacement marks that face blocked, records one bounded USE test, and
   advances to another heading. Successfully traversed cells form the only
   backtrack graph. If a landmark route stops improving for 26 ticks or 145
   ticks elapse, it is abandoned rather than replayed indefinitely.
3. **Exit-first observed landmarks:** visible Exit/Switch/Door/Teleporter
   objects, keys, useful weapons and urgently needed health/armor are ranked
   from the current rendered labels. Different-level or stale labels are not
   treated as direct goals. When an interactable is within measured contact
   range and roughly ahead, code pulses `use` once and waits for the engine's
   terminal/movement evidence. A blocked face receives at most one USE pulse
   per cooldown window.
4. **Safety and combat are deterministic:** code ranks current live monsters
   by category, proximity, hitscan/major/body-block evidence, switches to an
   owned weapon with available ammo, aims from world bearings, and fires while
   strafing. It avoids close rocket/BFG splash. Recent damage/projectiles,
   critical health, and close threats override model intent and choose a moving
   dodge or measured breadcrumb retreat.
5. **Models supervise, they do not execute:** Jev supplies an expiring
   semantic intent/target/side. Astra supplies a durable bounded phase policy.
   Both outputs are validated against enumerated options and rechecked against
   the newest observation. Invalid or late answers preserve the current
   deterministic controller.

## System-1 state schema and questions

Each asynchronous Jev request receives one strict JSON object, built from
code-derived facts rather than the full raw state:

- `task` and `rules`: exit is primary; only rendered labels are visible; code
  owns distances, aiming, collision, weapon legality and route memory;
- `plan`: Astra's bounded phase/priority/combat/side/interaction policy;
- semantic `health`, `armor`, `ammo`, selected `weapon_slot`, and
  `recent_damage` buckets;
- up to four `visible_threats`, each with stable current `id`, name/category,
  range bucket (`contact`/`near`/`medium`/`far`), side, same/different level,
  active evidence, hitscan and body-block flags;
- up to five useful currently visible landmarks with the same derived fields;
- `motion`: blocked-face evidence, measured cell/edge counts, stalled status,
  and current goal kind;
- current semantic intent and an explicit statement that Jev chooses intent,
  not a button.

The request batches five independent typed judgments:

| ID | Type and role | Bridge |
|---|---|---|
| `intent_a`, `intent_b` | `Choice` among `fight`, `retreat`, `interact`, `advance`, `scan`, `hold`, with the option order reversed between the two | Accept a high-probability agreement for 1.65 seconds. If only the low-risk `advance/scan/hold` answer is accepted, use it for 1 second; otherwise retain the deterministic mode. |
| `target` | `Choice` over the supplied current threat IDs plus `none` | Store only an ID still visible in the newest observation; it cannot create a hidden target. |
| `side` | `Choice` among `left`, `right`, `forward` | Changes the dodge/sweep tie-breaker only; code still handles collision and danger. |
| `use_now` | `Noul` | Its >=0.76 yes can make a blocked-face probe immediately eligible, but the code's face/12–18 tic cooldown and legal `use` name remain authoritative. |

Choice values are accepted only when the returned option probability is at
least 0.52 and non-null confidence is at least 0.10. The two intent choices
must agree for a consequential `fight`, `retreat`, or `interact`; a lone
low-risk answer is allowed only for `advance`, `scan`, or `hold`. Noul uses
`.value` as the yes probability and requires 0.76. Results older than about
2.2 wall-clock seconds or from an older Astra plan are discarded. Current
visibility, health and action legality are rechecked before use. A Jev
request is normally launched every 0.48 seconds in danger and 1.05 seconds in
quiet exploration, with one outstanding request; the game continues under the
current macro while it is pending. Model failure, malformed stand-in answers,
or low confidence leave code in control.

## System-2 planning and revision

Astra is launched asynchronously once early, and at most once more after a
long measured stall or damage crisis. Its prompt includes map/IWAD/skill,
current health/ammo, current visible labels, measured cells/edges/failed
faces, breadcrumbs and the current bounded policy. It must return only:

```json
{
  "phase": "sweep|approach|clear|interact|retreat",
  "priority": "exit|key|weapon|recover|explore",
  "combat": "clear_near|bypass|clear_all",
  "side": "left|right",
  "interaction": "use_on_contact|use_on_block",
  "focus_id": null,
  "reason": "short observed evidence"
}
```

Only enumerated values and a currently supplied visible `focus_id` are
applied. Applying a plan invalidates the old Jev intent and route goal, but
never changes the code emergency layer. Astra therefore tests durable
division of labour—choosing when to sweep, approach, clear, interact or
retreat—rather than reproducing the parent's unsafe raw-action answer.

## Late answers and contract safety

The initial posted action is the advertised noop; the first fresh observation
immediately selects a legal moving/combat macro. If a Jev/Astra answer is
late, the current macro remains held only for its finite executor horizon and
is re-evaluated from each tick. If the horizon expires, the code's next
measured steering, evasion, collision, or sweep action takes over. All action
strings are checked against `action_spec.names`. `TimeoutError`, ordinary
model/runtime failures, cancellation, malformed JSON and malformed typed
answers are logged and converted to safe fallback behavior. Budget, episode,
and model-unavailable exceptions follow the harness contract. Background
work consists only of bounded inference futures; there is no immortal worker
loop.

## Confirmation or refutation

**Confirmation** would be visible as fewer long same-cell turn streaks, a
nontrivial increase in measured cells/edges after a blocked face, more
successful movement after USE probes, weapon/key/exit landmark approaches,
and fire that produces kills without the 0005 health collapse. The decisive
confirmation is any `completed_by_exit`, especially on a seed where 0003 or
0005 reached many items but no terminal exit.

**Refutation** would be a controller that still alternates turns at one cell,
repeatedly fires at harmless distant actors while health falls, or records
many failed radial probes without discovering/activating a route. It would
also refute the hypothesis if the finite movement safety layer preserves
health but remains an idle-like one-cell policy, or if Astra/ J ev answers are
never adopted because the semantic interface is too conservative. A lower
stale-tick fraction by itself is not evidence of success.
