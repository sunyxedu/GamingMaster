# Candidate 0005: bounded central fire with asynchronous supervision

## Parents and diagnosis

Parents: `candidates/0000-tier2_jev_only/` and `candidates/0002-tier2_idle/`.
There is no `lessons.md` in this archive, so there were no additional refuted
lessons to incorporate.

The Jev parent earns real points, but its mechanism is not a stable policy. It
asks one broad question over the complete raw observation and then repeats the
selected action until the next slow answer. In the worst seed, the trace begins
with 27 NOOP ticks while the first Jev response is pending; later fresh choices
are almost uniformly `UPRIGHT`. The seed-10 action summary is 1,740 `UPRIGHT`
ticks, 56 `NOOP` ticks, score 80, and four lost ships after 1,796 ticks. Seed 4
has the same terminal pattern: 1,741 `UPRIGHT` ticks, score 110, and four lost
ships after 1,816 ticks. The model-call payloads show why: at the initial state
Jev assigns the largest probability to `UPRIGHT` (for example 0.27--0.35 in
successive calls) despite the state containing four large rocks and no need for
thrust. This is a delayed, open-loop action latch disguised as real-time
control.

The idle parent supplies the robustness mechanism. It never spends a model
latency holding a dangerous action and survives substantially longer: 3,512 to
4,282 ticks in the six recorded episodes, with scores 110--140. Its score is
not sufficient by itself, but it establishes that preserving ships and avoiding
unnecessary thrust is valuable in this cartridge. The Jev parent also supplies
evidence that occasional firing can be highly valuable: seed 6 alternates
`UPFIRE` and `UPRIGHT` late in the trace and reaches 1,080, but this is not
repeatable across the other seeds.

## Hypothesis

The flat/low results are caused primarily by the control bridge, not by Jev's
inability to recognize rocks. A controller that makes every native-frame
choice locally, prefers central stationary firing, and exposes only a small
semantic choice to Jev should retain the occasional scoring benefit while
approaching idle's survival robustness. In particular, thrust should be absent
from the default policy: it is an irreversible momentum change and the parent
traces provide no evidence that holding it is useful. The local controller must
also keep firing and rotation as short pulses, because FIRE is level-triggered
and a delayed answer must not hold either FIRE or a turn indefinitely.

The experiment is successful if it raises score above the Jev-only mean without
returning to the idle parent's short, four-life game overs. It is refuted if a
stationary/pulsed controller scores no better than idle, or if the derived
collision fallback loses ships faster than the Jev-only controller despite
posting actions on every tick.

## What changed

This is a structural crossover, not a prompt or timeout tweak.

* `LocalPolicy` owns the frame clock. It derives compact facts from each
  observation: player presence and heading, wrap-aware relative geometry,
  velocity-known flags, approximate time-to-contact bands, danger class, target
  priority, and life/respawn discontinuities. Raw RAM is deliberately kept out
  of Jev's decision state.
* The local action bridge is always executable while models are pending. With no
  target it performs a left/right sweep and firing pulses. With a target it
  turns toward a precomputed heading and fires only when its local release
  cooldown is zero. It never uses UP in its normal posture. At derived
  emergency danger it turns away immediately; it will use DOWN only when Jev
  has independently approved a current emergency and the defense cooldown is
  clear. A missing player, life change, stale answer, malformed answer, or
  model failure falls back to NOOP/short steering rather than raising.
* Jev runs asynchronously, with one outstanding request every 24 frames and a
  two-second timeout. It does not generate action names or arithmetic. Its
  answer changes only the local posture and target slot. Responses older than
  48 observed frames or from a different life are discarded.
* Astra is launched once, in the background, after 480 observed frames if the
  System-2 budget is available. It receives only an aggregate score/life and
  danger summary and returns an allow-listed JSON policy. That policy is copied
  into the state sent to later Jev calls; Astra cannot post an action, issue
  hyperspace directly, or block the local tick loop. The delay allows the
  idle-like local controller to collect evidence before strategic revision.

## Jev contract

The JSON state has these top-level fields:

* `rules`: verified action semantics, including level-triggered FIRE, release
  between shots, NOOP inertia, heading-code direction, hyperspace DOWN, and
  unknown enemy coverage.
* `observation`: frame, score, lives, player presence, heading, coarse position
  and speed bands, asteroid size counts, danger class, nearest-distance band,
  nearest-TTC band, and motion-discontinuity flag.
* `candidate_targets`: at most four code-ranked rocks, each with an id, size,
  bearing, danger, distance band, and velocity-known flag. A selected source
  slot is accepted only with size continuity and a short freshness window; all
  slots are cleared at life changes.
* `current_policy`, `last_action`, `fire_release_frames_remaining`, and a
  bounded defense state.

Each request batches five independent judgments:

1. `risk_a` and `risk_b` are the same four-way Choice (`clear`, `caution`,
   `emergency`, `unknown`) with reversed option orders. The controller trusts a
   model danger class only when the two answers agree. This directly addresses
   Jev's documented first-option bias; code still retains the deterministic
   geometry danger fallback.
2. `target` is a Choice over the four precomputed candidate ids plus `none`.
   The answer selects a current slot, never a numeric trajectory.
3. `posture` is a Choice among `focus`, `sweep_left`, `sweep_right`,
   `steady_fire`, `evade`, and `recover`. It names a local mode, not a
   sequence or duration.
4. `hyperspace` is a Noul. It is usable only when its yes probability is at
   least 0.78, the two risk answers agree on `emergency`, the player is
   present, the answer is fresh, and at least 240 frames have passed since the
   last defense pulse.

A posture answer is accepted only when its selected-option probability is at
least 0.18; a target requires at least 0.14. These are low-stakes semantic
choices, while defense uses the stricter gate above. An answer that fails a
probability, freshness, life, or agreement check is ignored. No answer can
cause thrust or an unbounded held action.

## Fallback and timing

The first action and every later tick are computed locally. While a Jev or
Astra future is running, `wait_for_tick()` advances exactly one emulator frame
and the next `ctx.act()` posts the local result. If a model times out, returns
arbitrary stand-in text, returns an unexpected object shape, or fails with a
recoverable runtime error, the future is dropped and the local policy continues.
System budget/deadline and model-unavailable exceptions follow the harness
contract. The module also checks `game.done` and `ctx.done` before issuing more
work, and does not create background loops besides the bounded inference
futures.

This should produce far fewer stale ticks than the synchronous Jev parent: an
action is posted after each observed tick even when the latest semantic answer
is old. The primary expected gain is survival from stationary, pulse-limited
control; the secondary gain is score from code-ranked aiming and Jev's target
and posture recognition. Astra's job is intentionally narrow: when a run
survives long enough for its slow answer to matter, it can switch between
steady fire and sweep/target priorities, while Jev re-evaluates those
preconditions at frame rate.
