# Candidate 0015 — zero spare lives is not game over

## Parent, hypothesis, and isolated change

**Parent:** `candidates/0012/` (generation-4 winner, mean ALE reward 316.67).
**Mode:** refine. Preserve the parent's architecture and all gameplay parameters.

**Hypothesis:** the parent's worst deadline episode forfeits a playable final Mario
because its transition handler treats `lives_remaining == 0` as terminal. The
counter describes spare Marios, not whether the current Mario can play. Use the
actual terminal signals to stop; do not use the spare-life count to veto the
existing restart protocol.

The sole behavioral change is removal of the `if lives <= 0: return "noop"`
guard from `Controller.choose`'s transition branch. The existing first FIRE at
transition age 18, spacing of 28 ticks, maximum four pulses, and neutral release
between pulses are unchanged. A `restart_pulse` event records tick, spare lives,
transition age, pulse number, game state, and life-loss flag. This is diagnostic
logging, not an additional control rule. The earlier terminal branch and the
`ctx.done` / `game.done` actor-loop checks remain in force. No reset action is used.

The calibrated player-edge route, phase thresholds, alignment retry, hazard
selection, RAM-owned jumps, recovery tail, rivet controller, model scheduling,
and one-post-per-tick loop are copied unchanged from 0012. In particular this is
**not** a jump-timing, ladder, or model-prompt experiment.

## Trace evidence and relationship to lessons

The inspected worst episodes are `0012/trace_seed5.jsonl` (200 points, deadline)
and `0012/trace_seed3.jsonl` (200 points, game done). The best seed 2 and the
parent's `eval.json`, proposal, and generation lessons were also checked.

* Seed 5, line 429: at clock **26.4689 s**, `state_boundary` reports
  `old_stage="ramp", stage="transition", old_lives=1, lives=1, phase=2`.
* Seed 5, line 447: at **27.6021 s**, another boundary reports
  `old_stage="transition", stage="transition", old_lives=1, lives=0`.
  The parent forbids every subsequent restart because of the zero count.
* Its final observation at **tick 13499** is still `stage="transition"`,
  `game_state=1`, `lives_remaining=0`, `life_lost_flag=false`,
  `terminated=false`, `truncated=false`, player `(45,176)`, and no hazards.
  Score is still **200**, and the episode ends by the **900 s deadline**, not
  game over. This is a nonterminal wait, not evidence that no Mario remains.
* Seed 3 supplies a positive control for the counter semantics: line 1236, at
  **77.4015 s**, reports `old_stage="transition", stage="ramp", old_lives=1,
  lives=0`; line 1237 immediately chooses `right` with zero lives. At **82.0021 s**
  it reaches phase 1 at `(105,154)`. Only at tick **1311** does the observation
  become `stage="game_over", terminated=true`. Zero spare lives therefore
  demonstrably permits active play in this cartridge and adaptor.

The study brief independently says the initial `lives=2` represents the two
spare Marios after the current one appears. Generation-4 lessons correctly
identify the transition deadline as a reliability gap, but describing the zero
counter as all lives lost obscures this distinction. Earlier candidates also
stalled or returned at the last-life transition. This experiment addresses that
specific semantic error rather than repeating their route/jump changes.

## Expected evidence; falsification

Primary mechanistic confirmation: the first zero-spare nonterminal transition
now produces a logged FIRE pulse on the **same schedule** as positive-spare
transitions; the subsequent observations return to active ramp/rivet play with
`lives_remaining=0` instead of remaining neutral until the deadline. Additional
reward on that final Mario is stronger evidence of score benefit. No action
should differ from the parent before the new eligibility condition is exercised.

The expected effect is narrow: recover lost final-life play, not guarantee a
screen completion. If that life earns no more points, restart liveness can still
be fixed while the score hypothesis remains unsupported. The hypothesis is
refuted if the observed nonterminal zero-spare transition cannot be started by
the unchanged pulse schedule, or if those pulses reset a genuinely finished
episode rather than resume it. Terminal guards must continue to suppress FIRE.

Training-seed trials and contract/regression checks are recorded below after
running them. No held-out seed, game source, private simulator state, or
seed-conditioned policy is used.

## Tier-2 System-1 interface and fallback (inherited unchanged)

Neither model participates in transition handling or ramp control. Their
inherited advisory bridge activates only on the rivet screen, which the parent
never reached in its six evaluated episodes. Preserving this division isolates
the lifecycle test and makes no claim of newly demonstrated model cooperation.

**Jev state:** a strict JSON object containing the game/stage; player horizontal
and vertical bands, motion, and jump-commit flag; route phase, climb-open flag,
rivet row and four raw row bytes; one code-derived current-row hazard's presence,
side, motion, distance bucket, closing flag and urgency; lives, screens and
rivets progress; and last action. Code computes geometry and hazard urgency.

**Batched questions:**

1. `sweep`, a Choice over `left`, `right`, `wait`: select a rivet sweep side;
   `wait` preserves an active climb/jump. A left/right result requires returned
   option probability **>= 0.55** when the probability dictionary is present.
   The parent also accepts a recognized option if that dictionary is absent
   (stand-in compatibility); this is not a claim of measured confidence.
2. `danger`, a Noul on whether the supplied hazard is closing and soon/immediate.
   It is advisory/trace evidence only and never overrides the hard jump guard.

At most one Jev request is outstanding, asynchronously spawned every **14 ticks**
or every **5 ticks** near a watch/soon/immediate hazard, timeout **0.65 s**. The
actor continues independently. A consumed sweep hint can change the next rivet
preference only after rechecking the current row's `remaining_side` result and
that no climb is open. Failed or malformed model outputs leave code control in
place. **Freshness limitation preserved from the parent:** there is no explicit
response-age/version expiry; the present-state legality check is the only sweep
revalidation. This refinement does not silently fix that separate interface issue.

**Astra:** starts once asynchronously upon rivet entry, timeout **80 s**, and
returns only JSON `first_side` left/right. Parsing accepts only those two labels;
garbage is ignored. It cannot alter a ramp route, jump, or lifecycle action.
There is no new runtime interface-revision trigger in this experiment. A future
iteration should consider revision only after actual rivet observations show
repeated unsafe/stale sweep choices; adding it now would confound the restart test.

**Fallback and lateness:** deterministic route and RAM-synchronized jump control
remain active regardless of inference. Idle starts with FIRE, transition posts
sparse FIRE edges even with zero spare lives, terminal observations produce NOOP,
and all other time in transition is neutral. Models are never awaited by the
actor. The loop uses `ctx.wait_for_tick()` and continues to actual game end or
`ctx.done`, not a harness-defined step count.

## Validation results

All six training seeds were run through the supplied `metaharness try` interface,
with default sticky actions and a three-minute cap. **All ended naturally before
the cap**, so no result is a shortened/deadline-censored episode. These are local
trials, not a claim that the official evaluation has run. No Astra/Jev calls
occurred, because these episodes did not reach the rivet screen.

| Seed | Parent archived score | 0015 local score | Ticks | Wall seconds | End |
|---|---:|---:|---:|---:|---|
| 0 | 300 | 300 | 1312 | 87.47 | game_done |
| 1 | 300 | 300 | 493 | 32.87 | game_done |
| 2 | 600 | 600 | 658 | 43.87 | game_done |
| 3 | 200 | 200 | 1311 | 87.40 | game_done |
| 4 | 300 | 300 | 548 | 36.54 | game_done |
| 5 | 200 | **300** | 547 | **36.47** | game_done |

Mean reward is **333.33**, versus the parent's archived **316.67** (+16.67, about
5.26%). All six trials report **zero errors and zero stale ticks**. Screens and
rivets remain zero: the fix recovers the last Mario but does not solve survival
through the upper ramp.

Mechanism is directly confirmed in
`tries/seed5-193939/trace_seed5.jsonl`: `restart_pulse` at controller tick **414**
records `lives=0`, `game_state=1`, `life_lost_flag=false`, transition age **18**.
The pulse applies on game tick **415**, followed immediately by
`state_boundary: transition -> ramp, old_lives=0, lives=0`. The final Mario earns
100 more points and reaches genuine game over at tick 547.

Action-level regression comparison against the parent archive found **identical
complete tick/action/reward/score sequences on seeds 0–4**. Seed 5 also matches
through tick 414; its first difference is exactly the intended game tick 415:
parent `noop`, candidate `fire`. This supports attribution to the single gate
change rather than a shifted opening or incidental route adjustment.

Additional checks passed: Python compilation/import; restart pulses exactly at
transition ages 18/46/74/102 with neutral release; unchanged positive-spare
transition behavior; no restart from terminal observations; identical choices
on archived active observations; and tolerance of garbage Astra/Jev outputs in
synthetic rivet observations. The latter are isolated bridge checks, not evidence
of successful rivet gameplay or a substitute for the evaluator's preflight.

Machine-readable local results are in `validation.json`; all six full trial
traces are under `tries/`. No further behavioral refinement was added after this
confirmation, keeping the experiment a one-change test.
