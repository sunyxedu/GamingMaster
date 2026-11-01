# Candidate 0005: timestamp queue with Jev as an advisory

## Parents and diagnosis

This is a crossover of `0000-tier2_jev_only` and `0002-tier2_idle`.
`0000` demonstrated that Jev can recognize a useful lane from an observation, but
its action bridge was fundamentally too slow for this real-time interface.  Its
contract-safe fallback was effectively the last action while Jev was thinking.
For example, in the seed-1002 trace the first Jev response selected `noop` with
probability 0.95, and the next responses also selected `noop` (0.91, 0.91, and
0.84) while the visible chart already contained a long-note head at 668 ms.
The calls took roughly 0.134--0.286 seconds, or 7--16 game ticks.  The same
trace eventually selected ordinary `press` actions, but they were current-time
actions rather than timestamped note events; the episode scored 0.  The other
worst episodes show the same failure: seeds 1003, 1007, and 1008 each made only
5--17 actions, had about 0.18--0.26 second Jev calls, and scored 0.  Across the
best parent, 89.85% of ticks were stale.

`0002` is useful for a different reason: it never makes a risky call or illegal
action and all 12 episodes end cleanly.  It is a robust idle fallback, but its
score is 0 because it never posts a note event.  The new harness keeps that
robustness—if Jev is absent, times out, returns an unusable answer, or becomes
unavailable, code continues with the current queue and eventually posts noop—
while replacing idle/current-action play with the simulator's explicit
lookahead protocol.

## Hypothesis

The large score loss is not primarily a recognition error.  It is the missing
scheduler: the observation gives exact authoritative target timestamps and the
simulator accepts events queued ahead of the virtual clock, whereas the parent
asks a model to choose one action after the target may already be gone.  A
code-owned rolling queue should therefore turn the same visible information
into exact press and release events, including simultaneous chords and both
ends of each long note.  It should score much better than Jev-per-tick while
remaining correct on held-out lane layouts because it does not encode a seed or
lane pattern.

## What changed

* Each observation is parsed defensively.  For every fair-visible pending note,
  code validates the lane and target time, queues a `press`; for every visible
  long note it queues a `release` at `end_ms` while the note is pending or held.
  Events are sorted by timestamp, with a release before a coincident new press
  on the same lane.  Note/part keys prevent resubmitting an event as the fair
  window is rescanned.
* Events are posted as one `events [...]` action, so chords share one timestamp
  and LNs preserve their held-lane state.  A rolling horizon is used rather than
  assuming that the hidden remainder is visible.  The default 1,800 ms horizon
  is refreshed every tick; it stays within the supplied 2,000 ms fair lookahead.
  Because a realtime tick repeats the last posted action, the batch is allowed
  through one tick and then replaced with noop; this prevents a repeated batch
  from reinterpreting future entries as current presses.
* On ticks without newly discovered events, the harness posts the game's noop.
  This is the idle parent's safe fallback and also makes the real-time loop
  explicit/fresh instead of depending on a stale guessed press.  Already queued
  timestamped events remain in the simulator queue.
* Jev is launched once, asynchronously after the initial queue has been armed,
  with a compact typed state.  It chooses `wide` (use the full fair horizon) or
  `near_term` (use a 650 ms conservative horizon).  The answer only changes how
  far code looks ahead; it never invents a lane, performs arithmetic, or sits on
  the timing path.  Low-confidence, invalid, timeout, and model-unavailable
  responses use the deterministic 1,800 ms fallback.  Astra is intentionally
  not called: a slow plan cannot improve a known timestamp schedule during a
  single run.

## System-1 state, questions, thresholds, and fallback

The one Jev state is a strict JSON object with these derived fields:

```json
{
  "clock_ms": 0,
  "visible_pending": 4,
  "visible_held": 0,
  "nearest_target": "near",
  "patterns": ["stream", "chord"],
  "scheduler_horizon": 1800,
  "timestamp_targets_are_explicit": true
}
```

`nearest_target` is a code-computed bucket (`imminent`, `near`, `far`, or
`unknown`), and `patterns` is only a bounded list of observed labels.  Jev is
asked one `Choice` question:

* `wide`: explicit timestamps and pending/held states are enough to queue every
  visible press and long-note tail, so use about 2,000 ms;
* `near_term`: use only the next short slice and rescan frequently because the
  visible state is ambiguous.

The bridge accepts `wide` only when its returned probability for `wide` is at
least 0.45; it uses 2,000 ms then.  A valid `near_term` answer selects 650 ms.
Every other value, missing answer field, probability problem, timeout, or
runtime failure selects 1,800 ms.  These choices cannot make the scheduler
blind: the loop observes every tick and queues a note before it enters the
selected horizon.  The Jev request is launched through `ctx.spawn` after the
first event batch has had one tick to arm.  The main loop polls its Future while
continuing to post noop/events, so the roughly 0.17-second Jev latency produces
no stale ticks.  The worker catches model failures and the main thread treats a
missing result as the idle fallback.  No System-2 call is made.

The action fallback is similarly explicit: invalid/minimal observations produce
no events, the last events are not guessed again, and the next tick receives a
valid noop.  The game itself reports duplicate/impossible events as inert, but
stable keys and lane-count validation avoid sending them in normal play.

## Evidence that would confirm or refute the hypothesis

Confirmation would be a large increase in accuracy/normalized score and a
near-zero miss count on the same training seeds, with event actions visible in
traces and no rise in game errors; the exact-timestamp mechanism should also
transfer to unseen random lane arrangements.  The local trials now confirm
this strongly: all twelve training seeds reached 1,000,000 with zero stale
reported ticks and one advisory Jev request per episode.

Refutation would be high scores only when Jev happens to choose a lane, or
systematic non-Marvelous errors despite exact `start_ms`/`end_ms` event times.
That would indicate that the adaptor applies a hidden timing offset, does not
retain timestamped events across noop ticks, or requires a different LN state
transition.  I would then use the observation's signed judgement errors and
`last_action_error` to adjust the scheduler, not ask Jev to guess individual
presses.
