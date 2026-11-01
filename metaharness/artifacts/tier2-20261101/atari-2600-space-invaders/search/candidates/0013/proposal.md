# Candidate 0013: Bounded Scoreless Traverse Extension

## Parent and Hypothesis

Parent: `candidates/0006/` (mean score 280.0). This is one targeted change to
its route-boundary transition. The tick-local controller, persistent latest-only
Jev worker, four typed questions, firing policy, bomb override, and off-path
Astra policy revision are preserved.

**Hypothesis:** immediately reversing after an unproductive completed traverse
can retrace an unseen bomb lane at a repeatable time. Extending that traverse
once, only as far as an empirically reachable endpoint, changes the return
timing before the first stall or life loss. It may improve the low-score tail
without changing firing throughput or replacing broad sweeps.

This is a hypothesis about reversal timing, not evidence that scorelessness
detects a bomb. It deliberately acts before the blocked-route symptom.
Candidate 0011's same-heading recovery after a stall reduced the mean to
256.25 and left zero-distance blocked loops. Those stalls may already represent
a hit or animation with delayed life reporting; the archive does not establish
that the cannon is controllable throughout them.

## Trace Evidence

The following are excerpts of relevant fields from 0006's worst episode,
`trace_seed2.jsonl` (15 points, 3 decoded kills, 349 ticks):

```text
program_committed id=1 start=144 from_x=37 x=110 direction=1
program_finished route_id=1 frames=144 distance=72 score_gain=0 reason=reached
program_committed id=2 start=288 from_x=109 x=25.8778 direction=-1
program_finished route_id=2 frames=116 distance=45 score_gain=0 reason=blocked
program_committed id=3 start=404 from_x=64 direction=1
program_finished route_id=3 frames=28 distance=0 score_gain=0 reason=blocked
program_committed id=4 start=432 from_x=64 direction=-1
program_finished route_id=4 frames=28 distance=0 score_gain=0 reason=blocked
program_committed id=5 start=460 from_x=64 direction=1
program_finished route_id=5 frames=28 distance=0 score_gain=0 reason=blocked
life_loss frame=508 previous_x=64 blind=true
life_loss frame=896 previous_x=102 blind=true
controller_finished actions=349 routes=17 jev_accepted=58 jev_effective=0
```

The second-lowest episode, seed 5 (90 points), also completes route 1 with
`frames=144 distance=71 score_gain=0 reason=reached`, reverses, blocks after
44 pixels, and logs `life_loss frame=508 previous_x=64 blind=true`. All 15
blocked finishes in each of these two traces occur after that first scoreless
completion. No decoded bomb appears in their sampled observations.

Importantly, the first scoreless completion also occurs in the stronger seeds.
The trigger does not distinguish doomed episodes or prove causation. Seed 0
survives the original return and scores at frames 508, 620, and 840. A useful
result must improve the aggregate, not just alter the weak seeds.

Travel-envelope evidence comes from successful parent motion, not invented
screen bounds: seed 0's route 3 starts at x=37 and moves 82 pixels to x=119;
route 4 returns 82 pixels. Its committed starting positions range from 37 to
119. Later rightward routes repeatedly stop at 119. Candidate 0010's worst
training episode similarly logs expired sweeps at x=119 at frames 720, 1148,
and 1576. Thus the extension uses 39/117, a two-pixel margin inside those
observed extremes. These are canonical-build geometric estimates, not
seed-specific coordinates or a claim that the range is safe from bombs.

The lessons support preserving continuous FIRE and caution against interpreting
empty bombs as safety. This candidate changes neither principle. It also does
not repeat 0011's blocked-route direction repair, infer permanent limits from
an interior stall, or add a seed-specific opening schedule.

## Single Change

At an ordinary route completion, arm an extension only when:

- reason is `reached` or `expired`, not `blocked`;
- observed displacement is at least 35 pixels and duration at least 80 frames;
- score gain is zero;
- the completed route was not itself an extension.

The next commit retains the old heading and targets x=117 when moving right,
or x=39 when moving left, provided that endpoint lies ahead by more than the
parent's arrival tolerance `max(2, speed * 4)`. Otherwise the ordinary parent
commit runs unchanged. The flag is consumed at this boundary. An extension
cannot arm another extension, and life/wave/discontinuity resets clear the
flag. Its expiry, stall recovery, visible-bomb override, and firing rules are
unchanged. Only this extension bypasses the ordinary 14-pixel minimum-route
guard, since a short continuation at an outer lane is precisely the experiment.

For the archived seed-2 boundary at x=109, the counterfactual next command is
`right_fire` toward 117 instead of `left_fire` toward 25.88. This is a local
action difference, not a replayed score estimate. Actual sticky actions and
projectiles after that intervention must be evaluated normally.

Events `scoreless_escape_armed` and `scoreless_escape_applied`, plus the
`scoreless_escape` flag on `program_committed`, distinguish a qualifying
completion from an extension actually executed. The harness never reads
archive files or seeds at runtime.

## Jev Interface

The inherited JSON state has four named sections (no new schema identifier):

- `rules`: committed spatial programs, the one-laser constraint, and unknown
  bomb visibility;
- `plan`: validated Astra focus (`broad/front/edge`) and fire mode
  (`stream/selective`);
- `evidence`: computed geometry permissions, dense/sparse coverage,
  `stalled_motion`, `prolonged_no_score`, low formation, blind-death evidence,
  in-transit status, and recent program names;
- `programs`: code-computed availability and qualitative span for each program.

Every request batches four independent questions:

1. `program`: Choice of `broad/front/edge/unknown`.
2. `crosscheck`: the same Choice with reversed option order.
3. `fire_mode`: Choice of `stream/selective/unknown`.
4. `repair`: Noul for explicitly reported stall or prolonged no-score.

The program Choices must agree, each selected probability must be at least
0.60, and any supplied confidence must be at least 0.30. Missing confidence
does not fabricate certainty; the probability tests still apply. Fire mode
requires probability at least 0.65 with the same confidence gate. Unknown,
missing, malformed, or conflicting answers leave the fallback in charge.

Responses must match life, wave, and plan version and be no more than 48 frames
old on arrival. Accepted semantic advice can remain usable until 120 frames
from its original observation; geometry permissions are rechecked at commit.
Advice chooses the next permitted program and firing mode. Selective firing
is still allowed only for sparse tracked targets, so Jev can affect FIRE
through the parent's existing mode bridge; it cannot issue raw joystick actions.
The repair probability is stored as a boolean at 0.80 but is not consumed by
the parent's action bridge. This known limitation is preserved, not presented
as demonstrated adaptation. The new extension is code-owned and does not rely
on any repair answer.

## Astra, Timing, and Fallback

One persistent Jev worker consumes a latest-state mailbox. Submissions occur
every 24 emulator frames or on a life/wave event. The unchanged 2.0-second
timeout exceeds parent seed 2's measured p50 0.143 seconds and maximum 0.232
seconds; it does not block the per-tick actuator. A result can time out or
become stale while code continues its route. The worker checks `ctx.done`
and has a stop event; the main loop closes it in `finally`.

Astra remains a bounded policy selector, not an interface generator: its
short JSON chooses focus/fire, which become both Jev state and the code
fallback at a later boundary. It runs asynchronously at startup, with at
most one further revision for life loss, wave change, or prolonged no-score
after two completed routes, separated by at least 1500 frames. The timeout
remains 55 seconds. Malformed JSON is ignored. Epoch changes invalidate Jev
advice; Astra replies are checked against the current wave. The parent prompt
already explains the local typed contract and supplies categorical evidence.

The inherited adaptation hypothesis is that repeated failures justify a
different existing program, not new model questions. Changing the model
interface or Astra cadence would confound this refinement, so both are left
unchanged. Parent Astra calls informed no posted actions in the archived
evaluation; no new claim of useful Astra cooperation is made here.

With no usable model answer, the controller uses its validated plan (initially
`broad/stream`), commits a bounded route, checks visible bomb risk, and posts a
legal action each tick. Stream holds FIRE while moving; the ROM enforces the
single active laser. Startup/reacquisition uses `fire`. Budget and episode
exceptions remain normal termination; transient model failures retain fallback.

## Confirmation and Refutation

Support requires both a causal trace signature and outcome improvement:
qualifying completions should generate actual bounded extensions, change the
next heading before a stall, retain firing coverage and tick freshness, and
improve mean score over 280 or materially improve the low tail without losing
the strong episodes. Later blind losses or more scoring before the first loss
would support the timing hypothesis. Wave completion remains the useful goal.

Refute the hypothesis if applied extensions leave early death timing unchanged,
cause longer boundary stalls, or lower aggregate score by damaging productive
returns. Fewer blocked log entries alone is not success: they may describe
already-lost control. No applied extension means the mechanism was not tested.
The scoreless trigger fires in strong episodes too, so regression there is a
central evaluation criterion, not an exception to discard.

## Local Validation

Syntax compilation succeeds with the specified evaluator interpreter. All
8 offline regression tests in `test_harness.py` pass, including three short
synthetic episodes with first-option answers, arbitrary payloads, and a timeout.
They verify the archived boundary's action change, symmetric left extension,
no chained extensions, parent-equivalent ordinary routes, transition resets,
garbage-answer rejection, import restrictions, and worker shutdown.

The `metaharness` package is not importable from that interpreter's isolated
search path in this workspace, so local contract tests use explicit stand-in
modules and synthetic observations. These are control/robustness checks, not
the evaluator's official pre-flight or evidence of a score improvement. No game
implementation source, unfinished sibling directory, held-out result, emulator
episode, or model backend is used during validation.
