# Candidate 0003: release held flaps after one tick

## Parent and hypothesis

Parent: `candidates/0000-tier2_jev_only/` (the current best candidate). I keep
its Jev-only architecture: one raw observation and one `Choice` over the game
actions per decision, with no Astra call and the same timeout fallback.

The targeted hypothesis is that the parent is not failing because Jev cannot
recognize the useful action. It is failing because a one-frame `FLAP` decision
is accidentally held for the whole Jev round trip. A real-time tick repeats
the last posted action, so a slow synchronous decision converts sparse taps
into a burst of flaps and drives the bird through the top edge.

## Trace evidence

The parent trace for seed 0 requests a decision at frame 5; the `FLAP`
response takes 0.169 s, then the fresh action is applied at tick/frame 6 and
repeated through frames 7--9 before the next request (which selects `IDLE`)
can post its action at frame 10. Later, the request at frame 14 selects
`FLAP`, which is again repeated at frames 15--18. The run eventually has 40
flaps in 50 frames, ends in a pipe collision at score 0, and the final bird y
is -71. Seed 5 has the same shape: the first request at frame 6 receives a
`FLAP` answer after 0.215 s, which is applied at frame 7 and held through
frame 11; the episode ends at score 0 with 38 flaps in 50 frames and bird
y=-86. The parent has 80% stale ticks, but the conspicuous failure is the
action held during those stale ticks, not a backend error (all 66 Jev calls
succeeded).

## Change

After a valid Jev `FLAP`, this candidate posts `FLAP`, waits for exactly one
`wait_for_tick()`, then posts the action-spec noop (`IDLE`). The next Jev
request therefore runs with a released, safe action while its answer is
pending. Jev's raw state, question, scheduling, and timeout behavior are
otherwise unchanged. Invalid or missing typed output is mapped to noop so the
pre-flight stand-in cannot create an illegal action.

This is deliberately not a local physics controller, a new state schema, or a
cooldown heuristic: it isolates whether correcting the real-time action
bridge is enough to turn Jev's existing choices into sparse impulses.

## Falsifiable result

The hypothesis is supported if flap count falls substantially, top-edge
shaping penalties disappear or decrease, and the same raw Jev decisions
survive farther or pass a pipe without adding System-2 calls. It is refuted if
one-frame release still produces the same early deaths (especially ground or
pipe collisions while the bird remains low), or if scores remain zero while
flap counts are no longer excessive; that would indicate the raw-state
question/schedule needs redesign rather than the action bridge.

## System-1 contract and fallback

The state sent to Jev is the complete strict JSON returned by
`game.observe()`. The single question is `act`, a `Choice` whose criteria are
`game.action_spec.names` and their descriptions. Jev chooses the immediate
action; code never asks it to do arithmetic or generate an action.

The answer is accepted only when its `.value` is a string matching a legal
name case-insensitively. Missing, malformed, or unknown values select the
spec's noop. A `FLAP` is gated to one game tick by the release sequence above;
`IDLE` is posted normally. During a timeout or ordinary runtime failure, the
harness waits one tick and retries, leaving the already-held action in place.
Since every completed flap is released before the next request, that fallback
cannot extend a flap burst. Budget and model-unavailable exceptions retain the
parent's normal propagation semantics.
