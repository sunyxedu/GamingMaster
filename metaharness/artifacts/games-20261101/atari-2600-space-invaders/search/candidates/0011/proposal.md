# Candidate 0011 — continue through a stalled route instead of reversing

Parent: `candidates/0006/` (the current best candidate, mean 280.0).

## Hypothesis

The useful part of 0006 is its sustained, firing lateral traverse, but its
stall recovery can turn that traverse back into a short oscillation.  When the
controller has not seen the cannon move for 24 frames, `action()` marks the
route `blocked`; `commit()` then always reverses `self.direction`.  That is
especially dangerous when the object decoder has no bomb in its sampled state:
the controller repeatedly revisits the same hidden bomb lane instead of
getting out of it.

The concrete worst trace is seed 2.  It records route 2 as `blocked` after only
45 pixels of progress, then routes 3, 4, and 5 as `blocked` with **0.0
pixels** each at frames 404, 432, and 460.  They alternate right/left/right,
and route 6 is left when the first unseen loss occurs at frame 508.  The trace
has no decoded bomb at the sampled observations and ends at score 15 after the
third loss (frames 508 and 896 are both logged as `blind: true`).  The decisive
trace evidence is:

```text
program_finished route_id=3 reason=blocked distance=0.0
program_committed id=4 direction=-1 from_x=64.0 x=28.0
program_finished route_id=4 reason=blocked distance=0.0
program_committed id=5 direction=1 from_x=64.0 x=116.0
program_finished route_id=5 reason=blocked distance=0.0
life_loss frame=508 previous_x=64.0 blind=true
```

The same initial pattern occurs in seed 4: routes 3–5 are zero-distance blocked
reversals and the first blind loss is at frame 508; later, routes 30–32 repeat
the zero-distance alternation before the episode ends.  Even a high-scoring
seed 0 shows the same failure mode late: routes 15–18 are zero-distance
blocked reversals before a blind loss at frame 2512, and routes 28–31 repeat
it before the next blind loss at frame 3820.

This is one targeted change to 0006: a route that ends because of the
controller's stalled-motion test no longer immediately reverses.  The next
route continues in the same direction and uses the normal broad spatial band,
subject only to the existing endpoint/too-short guards (so a known movement
edge still turns the cannon away).  The route remains bounded by the existing
expiry, visible-bomb risk check, and legal-coordinate clamp.  Thus the
experiment isolates whether replacing rapid reversal with a one-way escape
sweep improves survival and kill throughput, without changing 0006's Jev
interface, firing policy, latency, or Astra scheduling.

## What would confirm or refute it

Confirmation would be fewer consecutive zero-distance `blocked` routes,
fewer repeated direction changes after a stall, later blind losses, and higher
score/kills on the seed-2/seed-4-like episodes, while retaining 0006's strong
long-traverse episodes.  Refutation would be unchanged zero-distance loops,
more boundary deaths, or a lower mean despite the same fresh-tick rate; in
particular, if the loss happens during a decoder-hidden respawn/death interval
where no joystick action can help, continuing the direction will not improve
that episode.

## System-1 interface and control contract

This candidate preserves 0006's compact typed interface.  Each Jev request is
JSON containing:

- `rules`: one-shot firing, committed spatial traverses, and the rule that an
  empty bomb list is not evidence of safety;
- `plan`: Astra's validated `focus` (`broad`, `front`, or `edge`) and `fire`
  (`stream` or `selective`) policy;
- `evidence`: categorical geometry (`front_distinct`, `edge_viable`,
  `selective_supported`), explicit progress facts (`stalled_motion`,
  `prolonged_no_score`), `low_formation`, visibility warning, and recent
  program names;
- `programs`: code-computed legal programs and their qualitative span.

The questions are one batched request: two reordered `Choice`s for
`broad/front/edge/unknown` (a high-stakes program is accepted only when both
agree and each has probability at least 0.60), a `Choice` for
`stream/selective/unknown` (threshold 0.65), and a `Noul` asking only whether
there is an explicit stalled/no-progress repair condition (threshold 0.80).
Answers are discarded on life, wave, plan-version, or age changes.  The
program choice is advisory: code owns geometry, route lifetime, action
legality, fire lock, and visible bomb interception.  No Jev arithmetic or
coordinate planning is required.

While an answer is pending or malformed, the controller uses its validated
plan fallback (`broad` plus stream firing in the ordinary dense case), and the
last committed route remains active.  A stale/uncertain answer cannot reverse
the joystick.  Synchronous posting is avoided by the persistent background Jev
mailbox; every tick posts a legal fresh action.  Astra remains off the actuator
path and can revise only the next program boundary, exactly as in 0006.
