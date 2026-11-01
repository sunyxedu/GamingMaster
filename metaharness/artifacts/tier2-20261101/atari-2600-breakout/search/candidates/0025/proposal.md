# Candidate 0025 — lag-aware terminal motor

## Parent and diagnosis

Direct parent: `candidates/0022/`. I keep its asynchronous, code-owned
60-Hz loop, filtered reflected landing interval, one-flight shaping bookkeeping,
measured motor fit, impact lock, and advisory Astra/Jev bridge. This is one
low-level refinement; models still cannot post a paddle action or block the
emulator.

The parent is already strong (mean 763.333), but its remaining failures are
concentrated in the last native frames before a miss. The trace evidence does
not support changing the Jev questions or adding more Astra authority:

* In parent seed 1, the miss at frame **7221** had a lock at frame 7211,
  physical interval about `[38.86,50.86]`, paddle x=47, and measured velocity
  +0.26. By frame 7216 the paddle had moved to x=33 with velocity -3.74; the
  posted requests were `LEFT, LEFT, NOOP, RIGHT, RIGHT, RIGHT`, and the ball
  was already at y=182--195. The ball was lost at 7221. This is a queued/input
  response failure, not a stale tick (every listed tick was fresh).
* The same seed's frame **7786** miss starts with target `[112,124]`; while
  the impact lock was adaptive, x moved 121 -> 125 -> 130 -> 136 despite
  repeated `LEFT`, then only reached x=124 at the final sample. The observed
  velocity peaked at +4.54 and the loss followed at 7786. The parent’s
  adaptive rule decides from the current point and a two-frame lookahead, so it
  reacts after the already-requested input has taken effect.
* Parent seed 3 gives the complementary failure: at frame **14803**, the
  target was roughly `[37,49]`. x was 56 at ETA 3.9 with +2.19 velocity, then
  48 at ETA 1.7 with -2.09 velocity; requests changed `LEFT, NOOP, RIGHT,
  RIGHT, RIGHT` while the terminal target continued to move with the RAM
  quantization. The ball was lost even though the recorded path passes near
  the nominal interval. The parent’s miss log contains similar fresh
  `LEFT/RIGHT` reversals at frames 11888, 29273, and 37630.

These traces justify a careful actuator change: at impact, the next request is
not an instantaneous velocity command. The parent already measured an
approximately `v_next = drag*v + gain*u` response and tracked a queued input,
but its terminal lock used only `px + v*look` to choose a brake. That mixes
collision prediction with an unmodeled one-frame command delay and can select a
correction whose own carry exits the interval.

## Hypothesis and one change

**Hypothesis:** losses that occur after the paddle enters or nearly enters the
catch interval are caused by choosing the terminal request from the present
velocity rather than by rolling the measured actuator one or two queued/sticky
steps through the *fixed* catch interval. A lag-aware request selector should
reduce terminal overshoot and late reversals without changing the successful
early interception policy.

The only behavioral change is the terminal motor primitive. Once the parent
creates an impact lock, 0025 evaluates `NOOP`, `LEFT`, and `RIGHT` by a tiny
local rollout using the live fitted drag/gain, the observed velocity, the
previous queued/effective input, and both likely sticky branches. It scores
final interval error, peak interval excursion, terminal speed, and a small
center/tie penalty. The same selector chooses the lock’s initial brake, so the
first correction is also delay-aware. The existing two-frame switch hysteresis,
fixed physical interval, lock expiry, and dy-reversal resolution remain. The
early `legacy` motor, shaping, cleanup rays, model schedules, and all geometry
outside this terminal primitive are unchanged.

This is not clone-state search: it advances no emulator state, does not use a
seed, and models only paddle motion over at most eight native frames. The
controller still posts one legal action before waiting for the next tick.

Evidence that would confirm the hypothesis:

* miss windows have fewer late direction reversals and lower absolute paddle
  speed at the final descending samples;
* the lag-aware rollout’s chosen action is measurably different from the
  parent’s point/lookahead choice mainly when a queued input is carrying the
  paddle, not on settled catches;
* parent-like intervals are reached and held more often, increasing second-wall
  completion and reducing lives lost without increasing stale ticks.

Evidence that would refute it:

* the new selector still loses with a settled paddle inside the fixed interval,
  or systematically chooses a correction that leaves the interval;
* misses remain at the same rate while terminal speed and reversal counts fall,
  implying the landing/collision mask or reflected ball estimate is wrong;
* early-flight survival or wall progress falls broadly, indicating that the
  terminal-only change is being invoked too early or its rollout is too
  conservative.

## Jev interface and Astra role

The parent’s model interface is retained. Each asynchronous Jev request is a
strict JSON state of derived, named facts: frame/event, lives, score, wall and
remaining-brick buckets, flight phase/urgency/speed, landing lane and catch
width, shape eligibility and candidate side, edge occupancy, terminal-risk
bucket, current return profile, impact-lock phase, paddle-motion bucket,
contact/trial outcome, endpoint evidence, contact id, and plan revision. Code
computes all geometry, counts, reflections, timing, legal actions, and motor
rollouts; Jev never performs arithmetic or chooses a frame action.

Two independent reordered `Choice` questions select among `DAMP`, `TRACK`,
`SHAPE`, and `NO_CHANGE`. Both must agree, have selected-option probability at
least **0.60**, and match the current life/contact/plan epoch and freshness
window (120 native frames). Safety gates coerce shaping to the deterministic
catch fallback when lives, speed, edge occupancy, or geometry are unsuitable.
Timeouts, malformed stand-in answers, missing probabilities, conflicts, and
stale answers are ignored. The frame loop continues with the code policy.

Astra remains an asynchronous coarse profile adviser at wall/stall boundaries,
with a short token contract (`DAMP`, `TRACK`, or `SHAPE`). Its answer is only a
prior for a future return and cannot alter the terminal motor, interval, action
mapping, or timing. The parent’s event-triggered revision schedule is retained
because the trace evidence implicates actuator execution rather than strategic
model latency.

## Fallback and real-time contract

`run_episode` uses the parent’s native-frame loop: observe, update local
bookkeeping, compute a legal fallback, call `ctx.act(action)`, then wait for the
runner tick. FIRE is repeated while `serve_required` is true; a missing ball is
not an episode reset. The model futures are polled without waiting. If a model
call times out, returns arbitrary text, returns missing fields, or is cancelled,
the deterministic reflected interceptor and terminal rollout continue. Only
normal `BudgetExceeded`/`EpisodeOver` termination is caught in the action loop;
no harness-owned worker loop remains after `ctx.done`.

The candidate uses only the standard library and `metaharness.*`, keeps all
state episode-local, and preserves the parent’s strict JSON model states and
legal action names. The selector’s diagnostics (`impact_motor_choice`) expose
its chosen action, rollout horizon, branch scores, paddle state, and fixed
interval so the evaluation can distinguish this hypothesis from model-call
count or stale-tick effects.
