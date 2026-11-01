# Candidate 0017 — temporally latched interval control

## Parents and hypothesis

This is a crossover of `candidates/0014/` and `candidates/0006/`.

`0014` is the strongest parent (mean 411.5): its score comes from an
asynchronous, code-owned reflected landing interval and a leased motor that
posts an action every native frame. It correctly keeps Astra/Jev off the
critical path. `0006` supplies the useful missing mechanism: recent
collision-aware displacement history, direction reversal invalidation, and
measured paddle motion rather than trusting one quantized `dx,dy` sample.

**Hypothesis:** the 0014 interval motor is mostly correct, but it treats each
RAM velocity sample and each score/side event as authorization to rebuild a
landing interval. The traces show the remaining misses at the end of otherwise
long rallies: immediately before losses the motor often posts `NOOP` while its
latched target jumps or is repeatedly rebuilt (for example seed 0 at frame
4777, and the final losses in seeds 1, 2, 3, 4 and 5). A one-frame quantization
or a brick/side collision can therefore make a late predicted landing move
outside the catch set after there is no travel time left. Smoothing motion over
recent clean frames, explicitly invalidating on real vertical reversals, and
freezing the interval in the final contact window should preserve the parent’s
survival while removing this late-target failure.

The test is deliberately structural rather than a parameter tweak: a
`FlightTracker` separates raw RAM samples from a stable flight epoch; the
interval is made once from that estimate, is allowed to widen but not chase
small late changes, and a final contact lock chooses a deterministic
catch-preserving motor action. The strategic model contract remains advisory.

## Concrete archive evidence

- `0014` scored 436, 388, 418, 388, 421, 418, with only 0–8 stale ticks per
  episode, so real-time scheduling is not the bottleneck.
- In its loss windows, seed 0 frame 4777 and seed 5 frame 3469 repeatedly
  rebuild intervals immediately after a brick event; final losses in seeds 1,
  2, 3, 4 and 5 contain a changing interval near impact, while the posted
  actions are predominantly `NOOP` until a replacement FIRE. This is evidence
  for a target/flight-estimate problem, not model latency.
- `0006` demonstrated that averaging recent displacement and resetting history
  at direction changes is a safer interpretation of quantized one-frame RAM
  motion than trusting a single sample, although its point servo scored only
  200.8. Its temporal estimator is therefore crossed into 0014’s better
  interval/motor architecture, not copied as its point controller.
- Jev was accepted 718 times in `0014` but changed only about 1% of actions;
  Astra timed out in all six episodes and changed none. More strategic prose
  is not the tested lever. Models remain behind a fresh, versioned contract
  and cannot steer a near return.

## Harness changes

1. `FlightTracker` keeps short, elapsed-frame weighted `dx/dy` displacement
   histories, preserves direction through zero-delta quantization, and clears
   the relevant history on a score, a true sign reversal, a life/serve, or a
   large observation gap. A velocity is accepted only after corroborating
   samples. Paddle velocity is also smoothed from observed displacement.
2. The reflected paddle-overlap interval is latched after the ball is below
   the brick band. Small velocity noise does not rebuild it. A brick hit or
   side bounce starts a short observation quarantine; a new interval is
   authorized only after a clean, corroborated flight estimate. Once eta is
   within the contact-lock horizon, the current interval and its action lease
   are frozen; only a measured miss/serve resets it.
3. The 0014 leased interval motor and one-pulse measured brake are retained,
   but its final action is selected from the stable interval and observed
   paddle drift. NOOP is not assumed to brake; a counter-pulse is allowed only
   before the lock and only when measured drift would leave the interval.
4. FIRE remains a bounded code-owned action whenever `serve_required` is true.
   All model output is parsed defensively; a late, malformed, failed, or
   contradictory answer falls back to the deterministic motor.

## Jev interface and bridge

At event boundaries or after a clean paddle bounce, Jev receives a small strict
JSON state of derived facts, not raw arithmetic:

```text
{
  frame, event, flight_epoch, lives, score, serve_required,
  phase: serve|ascending|quarantine|descending|locked,
  urgency: serve|early|near|imminent|locked,
  speed: normal|fast|unknown, landing_lane,
  catch_interval: wide|narrow|locked,
  estimate: corroborated|settling|unknown,
  paddle_motion: left|right|still, eta,
  fallback_action: interval_motor|FIRE,
  rally_contract, astra_plan, left_edge, right_edge,
  bricks_remaining, wall_index
}
```

One request batches two reordered `Choice` questions (`contract_a` and
`contract_b`) over `PRESERVE`, `LEFT`, `RIGHT`, and `NO_CHANGE`. A matching
answer with selected probability at least 0.55 is accepted only if its
request epoch/version is still current. Code then vetoes LEFT/RIGHT for low
lives, fast or unsettled flight, an open requested edge, quarantine, or a
locked/near return. The accepted contract can shift only the broad home anchor
while the ball is ascending; it can never select a frame action or alter the
latched catch interval. Unknown answer objects and stand-in garbage are
ignored.

A fresh answer is stale when its epoch, rally version, or safety facts no
longer match. While Jev is pending or times out, the interval motor acts every
frame. Calls are spawned asynchronously with a 1.25 s timeout and a boundary
cooldown; no model call blocks `ctx.act`/`wait_for_tick`.

## Astra handoff and revision trigger

Astra gets one compact strategic prompt with the derived phase, side-edge
buckets, and current flight evidence. It may return only `PRESERVE`, `LEFT`, or
`RIGHT`; the token is an untrusted home-anchor prior and is revalidated by the
same code gates and next Jev boundary. It cannot issue an action. If the
interval controller survives longer but leaves the final first-wall bricks
intact, a later Astra revision should design a verified contact-section or
side-tunnel objective. If losses still occur while the locked interval remains
physically catchable, this hypothesis is refuted and the next candidate should
inspect collision timing/geometry rather than add model authority.

## Confirmation and refutation

Confirmation is a score above 411.5 with no increase in stale ticks or errors,
plus fewer `return_contract` rebuilds during the final 10 frames, fewer rapid
lease reversals, and loss windows in which the paddle remains inside the stable
catch interval. Refutation is earlier loss, repeated misses with a stable
corroborated interval, or survival without brick progress; that would show
that the public reflected geometry is insufficient or that scoring requires a
separate verified contact-shaping/tunnel policy.
