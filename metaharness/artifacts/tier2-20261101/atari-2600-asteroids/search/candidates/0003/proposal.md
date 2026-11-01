# Candidate 0003 — one-frame FIRE bridge

**Parent:** `candidates/0000-tier2_jev_only/` (Jev only, one `Choice` request per
observation, raw adaptor observation as state).  I kept that architecture: no
Astra call, no extra question, no local geometry controller, and no change to
when Jev is queried.

## Hypothesis

The parent's low scores are primarily an action-translation failure: Jev is
usually selecting the launch-like `UPRIGHT` maneuver, which contains no FIRE,
so the game receives long held thrust/rotation but almost no shots.  A
fire-bearing version of the same maneuver, with a real release boundary, should
raise reward without changing Jev's movement judgment.

This is supported by the worst traces, although it is not proof of causality:

* seed 10 scored only 80.  The trace has 160 `UPRIGHT` choices out of 161
  Jev calls; score rose 0→20 at frame 923, 20→40 at 1335, 40→60 at 1649,
  and 60→80 at 1884, then all four lives were gone by frame 1896 while the
  final action was `UPRIGHT`.
* seed 4 scored 110 with 162 `UPRIGHT` choices out of 165; it lost lives at
  frames 1198, 1638, 1880 and 1892, again ending with `UPRIGHT`.
* The exceptional seed 6 scored 1080.  It selected `UPFIRE` 89 times (and
  alternated `UPFIRE`/`UPRIGHT` near its later score gains), unlike the
  low-score traces.  The parent's aggregate was still only 291.667.

The trace also reports 91% stale ticks, but this candidate intentionally does
not change scheduling or expiry: this isolates the firing/level-semantics
hypothesis rather than combining it with a stale-action fix.

## Change and bridge

The Jev state is unchanged: the complete JSON returned by `game.observe()`,
including frame/ticks, score/lives, player, rocks, missiles, coverage and RAM.
The single question remains a `Choice` over the legal action names and their
published descriptions.  Its only extra instruction tells Jev that the bridge
will pulse FIRE for ordinary movement choices.

After a valid Jev choice, code applies this deterministic mapping:

* `UP`, `RIGHT`, `LEFT`, `UPRIGHT`, and `UPLEFT` become their corresponding
  `UPFIRE`, `RIGHTFIRE`, `LEFTFIRE`, `UPRIGHTFIRE`, or `UPLEFTFIRE` for one
  native tick.
* Existing fire choices are also pulsed (`FIRE`→`NOOP` release,
  `UPFIRE`→`UP`, etc.).
* `NOOP` and `DOWN` retain their parent semantics; defense is not silently
  repeated or fire-augmented.

The harness waits for that one tick before posting the release action.  Thus a
repeated choice is a sequence of FIRE press, non-FIRE release, FIRE press,
not a held FIRE level.  A malformed/garbage model value falls back to the
published NOOP action.  A failed Jev call leaves the already released action in
place and waits one tick before retrying.

## What would confirm or refute it

Confirmation would be a material score increase on the same bounded-launch
training seeds, ideally with more score-changing events per remaining life,
without a corresponding increase in early deaths.  If score stays near 80–110
while lives still end with long `UPRIGHT` spans, firing was not the limiting
factor and the next iteration should test bounded thrust/action expiry.  If
shots and score rise but survival collapses, the hypothesis is only partly
right: offensive fire helps, but the unchanged movement/staleness bridge is
then the bottleneck.

## Jev thresholds and fallback

There are no confidence thresholds in this isolated test: every legal Jev
choice is accepted exactly as in the parent, because adding a confidence gate
would confound the fire experiment.  The typed answer's `.value` is checked
against `action_spec.names`; invalid values use NOOP.  `TimeoutError` and
ordinary `RuntimeError` use the current released action for one tick and retry;
`BudgetExceeded` and `ModelUnavailable` propagate as the normal contract end
conditions.  Defense, legality, game-over, and score accounting remain owned
by the adaptor.
