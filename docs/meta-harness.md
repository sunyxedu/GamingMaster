# Meta-Harness: designing the glue between foundation models and games

`metaharness/` is the second half of GamingMaster. The trained network in
`model/` covers games where every millisecond matters. Everything else is
handled by *foundation models plus a harness*, and the harness is not written
by hand: an outer loop (the meta-harness) designs it, evaluates it, and keeps
the archive that lets the next iteration do better.

## Three tiers of games

| tier | latency pressure | who plays | who designs the glue | example games |
|------|------------------|-----------|----------------------|---------------|
| 1 | none (world waits) | Astra (System-2) | meta-harness | ARC-AGI-3, hidden-rule puzzles |
| 2 | some (world ticks in 100 ms – 1 s) | Astra thinks, Jev acts | meta-harness | Minecraft, Genshin, snake |
| 3 | extreme (tens of ms, hundreds of decisions) | trained network (`model/`) | humans + RL | StarCraft, Clash Royale, generals.io |

Tier 1 and tier 2 are what this package implements. The models:

* **Model profiles** live in `metaharness/models.toml`: a profile names a
  *backend* (how to reach a model) plus a model id and options, e.g.
  `[system2.astra] backend = "codex", model = "gpt-6-astra"`. Backends know
  nothing about specific models. `--system2 astra`, `--system2 sol`,
  `--system2 claude`, `--proposer …`, `--system1 jev` select profiles;
  `--effort` overrides a profile's reasoning effort.
* **Astra** — OpenAI GPT-6 Astra, reached through the `codex` CLI backend
  (`metaharness/models/codex_cli.py`). Non-interactive `codex exec --json`, thread
  resume so conversation context carries across turns, the CLI's own login
  (no API key in the repo). The integration follows Auto-CTFer's codex
  backend. When codex reports a usage limit the adapter raises
  `ModelUnavailable` and the loop stops cleanly instead of scoring garbage.
* **Jev** — TypeSafe's System-One model (`metaharness/models/jev.py`).
  `POST https://api.typesafe.ai/v1/systemone` with typed questions
  (`Choice` / `Score` / `Noul`) and calibrated probabilities back in
  70–700 ms. Key in `.env` as `TYPESAFE_API_KEY` (never in code).
* **Claude** — the `claude` CLI backend (`metaharness/models/claude_cli.py`);
  profiles `claude` (CLI default model) and `claude-opus`. Used as an explicit
  stand-in when codex has no usage; never substituted silently.

## What a harness is

A harness is one Python file with `build(ctx) -> program` and
`program.run_episode(game)`. It owns the whole control loop of an episode and
gets everything through `ctx` (`metaharness/core/contract.py`):

* `ctx.think(prompt, session_id=…)` — Astra. Slow. Budgeted per episode.
* `ctx.decide(state, questions)` — Jev. Fast. Budgeted per episode.
* `ctx.spawn(fn)` — background thread, so Astra can think while Jev acts.
* `ctx.act(action)` — turn games: step; real-time games: post the current
  action to a mailbox that the game clock reads every tick.
* `ctx.wait_for_tick()`, `ctx.clock`, `ctx.memory` (persists across seeds),
  `ctx.log(...)` (goes into the JSONL trace).

The harness therefore decides exactly the things the tier-2 discussion is
about: how Astra's slow output becomes state + questions for Jev, how Jev's
typed answer becomes a game action, when to go back to Astra, and what to do
when a decision arrives late.

## The game clock and honest latency accounting

Real-time games (`RealtimeGame`) tick on their own clock. A tick with no fresh
action re-applies the last one. The runner (`metaharness/core/runner.py`)
counts *stale* vs *decided* ticks, records every model call's latency, and
reports them next to the score. Two clock modes:

* `wall` — real time; a thread ticks the game every `tick_s`.
* `virtual` — deterministic discrete-event clock for tests and cheap
  experiments. Only the harness thread advances time (model latency is
  charged, real or declared); a background Astra call blocks until the clock
  reaches its wake time, so a 2 s think overlaps with 2 s of Jev calls exactly
  as it would in wall time. A loop that never calls a model still pays a 1 ms
  overhead per `act`, so it cannot freeze the world.

This is what makes “Astra alone on a real-time game” score exactly what it
deserves (100 % stale ticks) and what makes a harness that decides one tick
late visibly worse than one that predicts one tick ahead.

## The outer loop (arXiv 2603.28052 style)

```
runs/<experiment>/
  task.md               game description + contract + budgets (what the proposer is told)
  leaderboard.md        regenerated after every evaluation
  loop.log
  candidates/0000-<seed baseline>/   harness.py  proposal.md  eval.json  trace_seed*.jsonl
  candidates/0001-<seed baseline>/
  candidates/0002/                   (first proposed candidate)
  ...
```

Each iteration: the **proposer** (Astra as a coding agent, working *inside*
the run directory with unrestricted read access to every prior harness,
proposal, score and trace) writes `candidates/NNNN/harness.py` and a
`proposal.md` with its hypothesis. The evaluator runs it on the training
seeds, archives `eval.json` and the traces, regenerates the leaderboard, and
the loop repeats. At the end the best candidate is re-evaluated on held-out
seeds. Selection is lexicographic: mean score, then fewer stale ticks, then
fewer System-2 calls.

Baselines are seeded deliberately weak so the loop has something to beat:
Jev-only, Astra-only (tier 2) and naive Astra (tier 1). The
`metaharness/baselines/tier*_reference.py` files are *test fixtures* that a
scripted proposer emits in offline tests; they are not the deliverable.

## Running

```bash
# offline, no network, seconds: exercises the entire pipeline on a virtual clock
python -m metaharness loop --tier 2 --game snake --system1 heuristic --system2 scripted \
    --proposer scripted --clock virtual --iters 1

# tier 2 for real: Jev acts, Astra thinks and proposes (needs codex login + TYPESAFE_API_KEY)
python -m metaharness loop --tier 2 --game snake --system1 jev --system2 astra --proposer astra --iters 3

# tier 1 on ARC-AGI-3 (pip install -e '.[arc3]'; cached game files + OPERATION_MODE=offline in .env)
python -m metaharness loop --tier 1 --game arc3 --game-kwargs '{"game_id": "vc33"}' \
    --system1 none --system2 astra --proposer astra --iters 3

# evaluate one harness file
python -m metaharness eval --tier 2 --game snake --harness runs/<exp>/candidates/0003/harness.py \
    --system1 jev --system2 astra --seeds 100,101,102
```

Games: `snake` (tier 2), `rulepuzzle` (tier 1, hidden-rule grid puzzles with
key/door, toggle, teleport, mirrored-controls families), `arc3` (tier 1, real
ARC-AGI-3 through `arc_agi.Arcade` offline).

Tests: `uv run pytest tests/metaharness` (26 tests; the live Jev test is
skipped without a key).

## Cross-game held-out evaluation (why it exists)

The very first live runs exposed the classic meta-harness failure: the
proposer scored a perfect 2.0 on the rule puzzles by writing a scripted
puzzle solver that never called Astra, and the tier-2 winner embedded an
exact snake simulator. Both crashed with `KeyError` on a game they had not
seen — and so did the naive baselines that seeded them, which had taught the
proposer to hard-code observation fields.

The defence, as in the paper, is held-out evaluation on something the
search never optimised: `--heldout-game NAME` re-evaluates the best candidate
on a different game with the same interface (`lanerunner` for tier 2, `arc3`
for tier 1), reports it on the leaderboard without using it for selection,
and the task spec now states the generality requirement. Seed baselines are
game-agnostic by construction (`metaharness/baselines/_generic.py`, tested on
every registered game), and the tie-break rewards fewer actions.

## Experiments (2026-09-19)

All four runs used real Jev (`jev-1.13.0`, p50 latency 0.57–0.62 s on a
0.5 s tick) and Claude (`claude -p`) as an explicit stand-in for Astra as
both player and proposer, because the codex account had exhausted its usage
window. Archives with every proposal, harness and eval are under
`metaharness/artifacts/` (traces stay in `runs/`, gitignored).

### Tier 2 — snake (Jev acts, Astra thinks), held-out game lanerunner

Round 2 (`exp2-tier2-snake-heldout-lanerunner`, generality clause on):

| candidate | snake train (3 seeds) | stale ticks | S1 / S2 calls | what it did |
|-----------|----------------------:|------------:|--------------:|-------------|
| Jev-only baseline | 0.0 | 33 % | 15 / 0 | raw observation, one Choice per tick, dies in 6 ticks |
| Astra-only baseline | 0.0 | 81 % | 0 / 9 | one think per decision; the snake runs into the wall while it thinks |
| 0002 | 0.7 | 23 % | 26 / 6 | first generic bridge; still one decision late |
| 0003 | 9.3 | 13 % | 481 / 3 | Astra writes the bridge **as code** (`bridge(obs) → options/utility`), harvested across episodes; Jev arbitrates |
| **0004** | **23.7** | **0.2 %** | 187 / 5 | every tick runs the bridge; Jev only on near-ties; stall detector escalates to Astra with the stalled trajectory |

Held-out for 0004: snake seeds 100–101 → 27.0; **lanerunner (never seen) →
37.05** (the Jev-only baseline gets 0.85 there; the round-1 winner, which
had hard-coded snake, crashes).

Round 1 (`exp-tier2-snake-jev-claude`, before the generality clause) went
0 → 24 → 32 → 32.3 (held-out seeds 35.0, 0 % stale) by diagnosing from the
traces that Jev's answer lands one tick late and pipelining Jev on the
*predicted* next state — but with an exact snake simulator inside the
harness, hence 0 on lanerunner. The proposer also found a real runner bug
(the wall ticker was not phase-aligned with `wait_for_tick`); fixed in
`1e8239b`.

### Tier 1 — hidden-rule puzzles, held-out game ARC-AGI-3

Round 2 (`exp2-tier1-rulepuzzle-heldout-arc3`):

| candidate | rulepuzzle train (levels of 2) | actions / episode | S2 calls | ARC-AGI-3 `vc33` (7 levels, 150 actions, ≤15 calls) |
|-----------|------------------------------:|------------------:|---------:|------------------------------------------------------:|
| naive baseline | 0.0 | 15 (budget death) | 16 | 2.0 / 1.0 |
| 0001 | 2.0 | 30 | 0 | – |
| 0002 | 2.0 | 28 | 0 | – |
| **0003** | 2.0 | **15.5** | 0 | **4.0 / 4.0** (12–14 calls) |

On a second ARC game the proposer never touched (`ls20`, 1 seed): 0003
reaches 1 level, the baseline 0. Caveat: the proposer had shell access and
dry-ran `vc33` itself while designing 0003, so `vc33` is held out from the
*selection* but not from the proposer; `ls20` is the strict number.

What the winning tier-1 harness does: scripted probing to learn the movement
map, explicit colour beliefs (passable / blocked / consumable / toggle /
teleport) that are refuted by evidence, Dijkstra over believed-passable cells
with *curiosity* for rare untouched colours, and Astra only when stuck — with
plans of many actions per call and no re-probing after a level-up.

### Running with real Astra

Same commands with `--system2 astra --proposer astra` once the codex account
has usage again (`codex exec` reports the reset time). Nothing else changes;
the adapter is exercised by the offline tests against recorded codex output.

### Round 3 — real Astra (gpt-6-astra, reasoning effort high) as player and proposer

Same settings as round 2, run once the codex account had usage again
(`astra-tier2-snake-heldout-lanerunner`, `astra-tier1-rulepuzzle-heldout-arc3`).

**Tier 2, snake.** 0 → 2.0 → 18.7 → **29.7** food per episode over three
proposals, 0 % stale ticks throughout, all three winning episodes survive
the full 300 ticks; held-out seeds 23.0. The winner (`0004`) keeps a small
Astra-compiled reactive policy plus a separately compiled, empirically
checked transition model with bounded lookahead, and uses only 2 Astra calls
per evaluation. Transfer to lanerunner is weak (3.9, dies at tick 7–13):
better than Jev-only (0.85) but far below the Claude-round winner (37.05),
whose "Astra writes the bridge as code and re-writes it on stalls" design
adapted to the new game at runtime.

**Tier 1, rule puzzles.** Perfect 2/2 levels from the first proposal on, but
unlike the Claude round the harness keeps Astra in the loop (6–7 calls per
episode: evidence-carrying action batches with guarded execution instead of
a scripted solver). On ARC-AGI-3 `vc33` the winner (`0002`) reaches
**4 / 7 levels on both seeds** (94 and 116 actions, 15 calls); on the
strictly unseen `ls20` it reaches **3 levels** (Claude round: 1, naive
baseline: 0). Astra-high calls take ~40 s, so the loop's default 600 s
harness timeout truncated the first ARC pass at 3–4 levels; the numbers above
use `--harness-timeout 1800`.

Two adapter bugs were only visible with the live account and are fixed in
this round: `codex exec resume` rejects `--color/-s/-C`, and `-C` must be an
absolute path when the subprocess cwd is the run directory.
