# Candidate 0013 — delay the far edge of hazard jumps

## Parent and hypothesis

Parent: `candidates/0009/`.

The parent is the strongest controller so far: mean score 250 across the six
training seeds, zero stale ticks, zero failed model calls, and no blocking
Astra. Its worst episodes show a repeated early-jump pattern on the ramp:

* In `trace_seed1.jsonl`, the parent starts jumps at ticks 149, 363, and 418
  with horizontal distances 50, 58, and 57 pixels respectively. The first
  and second are labeled `urgency: "soon"`; the latter has vertical delta
  `30.5`. The run ultimately scores only 100.
* The same trace's useful crossings are the later near/immediate events: tick
  199 has distance 16 and vertical delta 3.5 and earns the 100-point jump;
  tick 449 is distance 13 and immediate, but arrives after the earlier jumps
  have disturbed the route. Seed 3 shows the analogous sequence: an early
  distance-44 jump at tick 149, a useful distance-24 jump at tick 198, then a
  final transition at tick 480 with score 200.

The parent treats every closing obstacle out to 66 pixels as actionable. The
hypothesis is that the far end of that `soon` bucket spends the committed
8-quantum jump before Mario is in the crossing window; delaying only that far
edge lets the existing route continue while retaining the parent’s successful
near crossing.

> Start a hazard jump only when the code-computed horizontal distance is at
> most 45 pixels. Keep the parent’s vertical band, side/direction rule, jump
> duration, cooldown, route, restart state machine, and Jev interface intact.

This is a single timing gate, not a new route or model policy. It specifically
isolates whether the parent’s poor seeds are caused by premature hazard
interrupts.

## What changed

`harness.py` is the 0009 controller. In `hazard_action`, after the existing
same-walkway/closing test, a jump is suppressed when the normalized hazard
`distance` is greater than 45. The parent allowed all `soon` hazards through
(up to 66); all other action and state logic is unchanged. A later observation
can trigger the same committed directional jump once the obstacle enters the
45-pixel crossing window.

The public-seed smoke traces support the timing hypothesis. With this one gate,
seed 1 changed from the parent’s 100 to 300: the tuned trace starts at
43/22-pixel crossings on the first life, 44/23 on the second, and 45/22 on the
last, rather than committing at the parent’s 50/58/57-pixel warnings. Seeds 0,
4, and 5 retained 300. Seed 3 retained 200. Seed 2 fell from the parent’s 300
to 200, so this is an explicit tradeoff rather than a claim of universal
improvement; the evaluation tests whether the mean gain on the worst timing
stream outweighs that loss.

Confirmation is a training/evaluation trace where a distance-50-plus warning
produces no jump, a later <=45 crossing still earns the jump reward, and the
controller survives one additional hazard cycle or reaches a higher route
level. A higher mean score with zero stale ticks and no new errors is the
primary confirmation. Refutation is a consistent loss of the parent’s near
jump reward, deaths before the hazard reaches 45 pixels, or a lower mean across
seeds because the delay is too conservative. If so, the next experiment should
condition the threshold on measured hazard motion/edge position rather than
reintroducing slow model calls.

## Jev interface and fallback

The state sent to each Jev request is strict JSON and contains only derived
facts:

* `game` and `stage`;
* `player_band` (left/middle/right);
* `route.ramp_level`, `route.rivet_row`, the four raw-derived
  `route.rivet_rows`, and `route.jump_committed`;
* one normalized hazard with `present`, side, motion, named distance bucket,
  `closing`, and urgency;
* `last_action`, `lives_remaining`, and `screens_completed`.

The state does not expose raw RAM or ask Jev to calculate geometry. The one
request batches two independent questions:

1. `side`: a `Choice` over `left`, `right`, and `wait`, asking which side is
   already supported by the current rivet mask and serpentine route. A
   left/right answer is used only during the rivet stage and only when that
   side is still a legal remaining target.
2. `danger`: a `Noul` asking whether the normalized facts show an immediate
   same-row closing hazard. This is diagnostic; the deterministic hazard and
   jump guards always take precedence.

Requests are spawned at least 12 ticks apart and use a 0.75-second timeout.
A future is collected only when complete; malformed, failed, late, or
irrelevant answers leave the deterministic action unchanged. There is no Astra
call, and the actor never waits for Jev. The distance threshold is computed in
code and is not delegated to Jev.

The fallback runs on every tick: one FIRE from idle; NOOP during transitions,
with sparse FIRE restart edges only when lives remain; deterministic ramp
routing; a committed directional jump only after the normalized hazard passes
the <=45 distance guard; mask-driven rivet routing and gap jumps; and NOOP for
terminal/last-life transition states. Thus slow/garbage model output cannot
stall or alter this timing experiment.
