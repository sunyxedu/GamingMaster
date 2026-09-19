# Meta-Harness implementation and live-model audit — 2026-09-19

This audit covers both `meta-harness-tier1` and `meta-harness-tier2`. The latter
adds real-time games, baselines and tests; their core, model backends and tier-1
implementation were identical at the start (`a672125` / `69a522e`).

## What was already implemented

There is a real code-search outer loop: an Astra coding-agent proposer reads
prior programs, scores and rollouts, writes a new harness, and receives measured
feedback. Runtime adaptation belongs to the generated harness. This matches the
central design of [Meta-Harness](https://arxiv.org/html/2603.28052v1), rather than
providing one fixed, manually written game adapter.

Tier 1 exposes turn-based rule puzzles and an offline ARC-AGI-3 integration.
Tier 2 exposes wall-clock Snake and Lane Runner, with slow Astra planning and
fast typed Jev decisions. Minecraft and Genshin are **not integrated or tested**.
The neural-network tier was outside this audit and was left untouched.

The old archives contain actual `codex:gpt-6-astra` results as well as earlier
explicit Claude stand-ins. Those historical scores have not been overwritten.
They use a different protocol and are not direct comparators for this audit.

## Corrections made during the audit

- Independent episodes are now the default (`--memory-mode episode`). Explicit
  `shared` mode measures ordered adaptation across seeds. Each episode gets a
  fresh simulator and harness module. Memory sharing is recorded in the result.
- Rejected budget requests no longer inflate model-call counts; admission is
  atomic across background workers. Wall deadlines include harness construction,
  and a closed context cannot post late actions into a later episode.
- Model unavailability remains fatal even if a generated harness catches the
  exception. Partial, nonfinite and erroneous evaluations cannot win selection.
  Unjoined background work is reported as an episode error. In-flight requests
  are now logged before waiting, so a deadline cannot erase the request itself.
- The runner stops exposing simulator diagnostics through `game.info()` or its
  archive. The old puzzle implementation's diagnostics included the hidden
  movement map and rule family. The previously selected harnesses did not use
  this interface, but exposing it invalidated the intended observation boundary.
- A search manifest fixes seeds, models, budgets, clock and memory protocol.
  Incompatible resumes and overlapping train/test seeds are rejected. Search
  cannot resume after held-out results have been written into the archive.
  Each evaluation records a harness SHA-256 and model configuration. Repeating
  an evaluation replaces its rollout files rather than concatenating episodes.
- Codex resume explicitly pins Astra. Failed/empty CLI completions are errors;
  CLI timeouts terminate the process group, and Jev retries share one deadline.
  The proposer uses workspace write permissions. Player calls disable external
  tools and reject observed external tool use. Agentic proposal events are
  retained for new runs. Authentication stays in the existing CLI login and
  gitignored `.env`; the Jev key is not passed to Codex subprocesses.

The [Codex non-interactive documentation](https://developers.openai.com/codex/noninteractive)
and [configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
were checked against installed CLI 0.155.1. The
[Jev request contract](https://docs.typesafe.ai/api) was checked against a live
`jev-1.13.0` response. The supplied X discussion could not be retrieved, so no
claims here depend on its contents.

## Experiment protocol

`python -m scripts.metaharness_audit search --tier 2 --root /tmp/mh-tier2-audit`
performs search; `test` freezes the selected candidate before exposing test
results. Substitute tier 1 for the turn-based suite. Both proposer and player
use **Astra, effort high**; tier 2 uses **real Jev**. No latency override or
scripted backend is used in these experiments.

Search compares the original baselines with the frozen prior Astra champion,
then generates new candidates. Tier 2 uses training seeds 0/1/2, 0.5-second
ticks, 120-tick episodes, at most 4 Astra calls and 240 Jev calls per episode.
Tier 1 uses seeds 0/1, two puzzle levels, 50 actions per level and 12 Astra calls.

Final tests use fresh seeds 700/701/702 (tier 1: 700/701), and compare frozen
harnesses under the same final backend settings. Tier 2 also measures shared
memory, transfer to Lane Runner, and removal of each model separately. Tier 1
also tests ARC-AGI-3 `ls20` and `cd82`, with 150 actions and 12 Astra calls.
`ls20` and Lane Runner were used in earlier work; they are transfer regressions,
not previously unknown benchmark families. `cd82` is an additional transfer
check not supplied to this search proposer.

Selection uses training score, action-mailbox staleness, Astra calls and action
efficiency (turn games), or survival duration (real-time games). A fresh mailbox
action does **not** imply a fresh model decision: code can
post an action every tick. Model calls and any controller coverage events must
be examined separately. ARC results count levels completed; they are not an
official ARC benchmark score. Small seed counts characterize these runs, not
statistical evidence of universal game mastery.

The search processes began on commits `02e5173` / `de1ea3a`; hiding privileged
diagnostics and inference-only tool enforcement were added while they ran.
Their loaded runtime remained unchanged. The final comparisons use the corrected
runtime for every candidate. Search and final evaluation should therefore be
compared through their matched controls, not as one continuous learning curve.
Tier-1 validation re-evaluated all candidates with the corrected runtime before
selection. Its turn-game tests used two independent workers; tier-2 timing
comparisons ran serially. The later survival tie-break fix does not change either
frozen selection. Request-start logging was added after observing the ARC
timeout; the earlier archived timeout cannot contain a response that never arrived.

## Limits of the implementation

Generated Python is trusted research code, not an adversarial security sandbox.
Workspace write restrictions do not constitute complete read isolation. The
proposer is instructed to read only its archive; tool transcripts and generated
source can be audited for compliance. Strong benchmark secrecy would require
separate filesystem/container boundaries and an inference service.

Timeout handling is cooperative for Python harness code. The runner closes its
context, ignores late trace writes and uses fresh game instances, but cannot kill
an arbitrary CPU-bound Python thread. CLI process trees are killed on their own
deadlines. A production evaluator should run each untrusted candidate in a
separate worker process/container.

A high Snake score establishes task performance, not generality. A transferable
harness must learn observation semantics, preserve a useful control loop during
Astra latency, validate generated bridges and show that Jev's decisions improve
behavior. The cold/shared and model-removal controls are intended to expose those
distinctions.

## Search and validation results

Tier 2 discovery (three independent seeds; scores are food eaten in at most
120 ticks; calls are totals over all three episodes):

| Harness | Scores | Mean | Jev calls | Astra calls | Mean ticks |
|---|---|---:|---:|---:|---:|
| Jev-only | 0, 0, 0 | 0.00 | 18 | 0 | 6.0 |
| Astra-only | 0, 0, 0 | 0.00 | 0 | 3 | 6.0 |
| Prior Astra winner | 7, 8, 11 | 8.67 | 180 | 6 | 120.0 |
| First new proposal (`0003`) | 2, 2, 0 | 1.33 | 180 | 6 | 120.0 |
| Second new proposal (`0004`, selected) | 13, 12, 14 | 13.00 | 3 | 3 | 119.3 |

The first proposal failed for measurable reasons. Jev reliably identified the
safety mask and goal-error field but did not confidently infer numeric action
effects. Astra also used positional feature references where the generated
executor accepted named references only, triggering another slow compilation.
The next proposer read these failures, supported both reference conventions,
and learned action effects from repeated observed transitions. Jev supplies
semantic interpretation, the empirical controller acts while Astra compiles,
and the generated policy takes over when ready. No game-specific action mapping
was manually supplied by this audit. The seed-0 trace records 67 empirical
control frames and 12 actual Jev-binding interventions. This is stronger evidence
of model use than merely counting API calls. One training episode still died at
tick 118, so the higher score did not imply uniformly better survival.

Tier 1 validation reran every written candidate under the final runtime. The
proposer reached its 600-second CLI deadline after writing `0002` and its
proposal; the timeout is retained as `proposal_error.json`. The evaluator, not
the proposer's final message, establishes the following results:

| Harness | Levels per seed | Mean levels | Total Astra calls | Mean actions |
|---|---|---:|---:|---:|
| Naive Astra | 1, 1 | 1.0 / 2 | 24 | 12 |
| Prior Astra winner (selected) | 2, 2 | 2.0 / 2 | 7 | 36 |
| New compiled-policy challenger (`0002`) | 2, 2 | 2.0 / 2 | 7 | 40 |

The challenger compiles an episode-local policy in a bounded interpreted
language and reuses it across observations. It worked, but did not improve the
training objective: its call count tied the incumbent and it used more actions.
Selection therefore retained the incumbent. Both are tested after freezing;
transfer results cannot retroactively change that selection.

## Frozen test results

Tier 1 uses independent memories and the final validated programs. Puzzle test
seeds are 700/701; each ARC game uses seed 700. ARC has the same cap of 12 Astra
calls and 150 actions for all three controllers, plus a 900-second wall deadline.
The parenthetical timeout results below are **partial observations, not valid
completed evaluations**.

| Task | Naive Astra | Prior champion, selected | Compiled-policy challenger |
|---|---:|---:|---:|
| Rule puzzles, mean of two seeds | 0 / 2 | **2 / 2** | **2 / 2** |
| ARC `ls20`, version `9607627b` | 0 / 7 | **3 / 7** | 1 / 7 before timeout |
| ARC `cd82`, version `fb555c5d` | 0 / 6 | **4 / 6** | 2 / 6 before timeout |

On the independent puzzles, the challenger used five Astra calls total versus
seven for the incumbent, but this did not predict ARC performance. The incumbent
used 104 actions and 866.8 seconds on `ls20`, and 78 actions and 741.8 seconds on
`cd82`, with 12 calls in each. The challenger reached its wall deadline after
8 calls / 55 actions on `ls20` and 10 calls / 34 actions on `cd82`. The naive
baseline used 12 actions on each game. These results support keeping the simpler
checked-planning incumbent: policy compilation did not reliably pay off on the
more difficult transfer tasks.

There were 18 final tier-1 evaluations (16 without errors, two explicitly marked
deadlines), plus four successful discovery baseline episodes. The
[final tier-1 archive](../metaharness/artifacts/audit-20260919-tier1/) and
[discovery archive](../metaharness/artifacts/audit-20260919-tier1-discovery/)
include the actual programs, both proposal and episode timeout records, complete
evaluation summaries, compressed traces, tool-use audits, environment versions
and game-file checksums. The final archive has 72 returned player session IDs;
none of those session rollouts contains an external tool call.

Tier 2 has 48 completed evaluations without errors. See the
[full tier-2 report and archive on its branch](https://github.com/sunyxedu/GamingMaster/blob/meta-harness-tier2/docs/meta-harness-tier2-audit-2026-09-19.md)
for the per-seed numbers and mechanisms.

| Tier-2 condition, mean over 700/701/702 | Jev-only baseline | Prior champion | New 0004 |
|---|---:|---:|---:|
| Snake, independent cold starts | 0.00 | 7.33 | **11.67** |
| Snake, shared memory | 0.00 | 11.33 | **11.67** |
| Lane Runner, cold starts | **2.47** | 1.13 | 1.13 |

Runtime model ablations of 0004 give **11.33 without Astra** and **0 without
Jev**, versus 11.67 for the complete controller. The outer proposer remains
Astra in both controls. Thus Astra-designed code plus Jev semantic calibration
accounts for most of the measured Snake benefit; three seeds and a 0.33-point
difference do not demonstrate substantial additional runtime Astra value. Jev
performs semantic binding in this discovered controller, rather than selecting
every frame's action. Continuous high-level/low-level cooperation remains an
untested claim.

## Assessment against the original goal

The repository now has a working, experimentally audited Meta-Harness outer loop
and real Astra/Jev integrations. Tier 1's incumbent transfers to two actual ARC
games. Tier 2's new Astra-generated candidate improves independent Snake scores
and greatly reduces calls. These are useful results, **not evidence of mastering
arbitrary games**.

The largest remaining gap is the training distribution: single-game search can
optimize around Snake's unusually informative observations. A stronger generality
study should train on several distinct games and observation schemas, preserve
new games for a final blind test, and score cold-start adaptation separately from
amortized play. Lane Runner and `cd82` have now been exposed and should not be
reused as fresh blind tests. Minecraft/Genshin additionally need environment
connections, richer action schemas and visual perception; none was evaluated here.

The next objective should also measure actual inference cost and time to useful
control, not just call count. Keep successful simpler candidates when a more
elaborate compiled policy regresses. Preserve model-removal controls: an optimizer
can obtain a high score mostly through generated code, without demonstrating the
particular model cooperation the research is intended to study.

## Reproduction and verification

```bash
# Discovery with actual Astra as player and proposer.
uv run python -m scripts.metaharness_audit search --tier 1 --iterations 1 --root /tmp/my-tier1-discovery
# Re-evaluate all written programs, including a complete program left by a timed-out proposer.
uv run python -m scripts.metaharness_audit validate --tier 1 --source /tmp/my-tier1-discovery --root /tmp/my-tier1-final
# Freeze selection and evaluate incumbent, newest challenger and baseline.
uv run python -m scripts.metaharness_audit test --tier 1 --jobs 2 --root /tmp/my-tier1-final
# Tier 2 uses serial wall-clock evaluations.
uv run python -m scripts.metaharness_audit search --tier 2 --iterations 2 --root /tmp/my-tier2
uv run python -m scripts.metaharness_audit test --tier 2 --root /tmp/my-tier2
```

Use the CLI's existing account login. The Jev credential remains in the ignored
`.env`; ARC uses the local offline cache. Recorded environment: Python 3.14.5,
Codex CLI 0.155.1, arc-agi 0.9.9, arcengine 0.9.3. Model generations are stochastic;
fixed environment seeds do not guarantee identical model outputs. The stored
programs can be evaluated directly using `python -m metaharness eval`.

Final regression checks: **50 tests pass on tier 2; 35 pass on tier 1**. Ruff's
`F` checks pass for changed runtime, driver and test files; whitespace checks pass.
Scanning all historical code also reports seven pre-existing unused imports or
variables in frozen experiment artifacts and the scripted baseline; those files
were preserved. Exported artifacts were scanned against the
configured credential values before publication. Both branches retain the
historical archives rather than rewriting earlier results.
