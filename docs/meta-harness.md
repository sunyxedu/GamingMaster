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

* **Astra** — OpenAI GPT-6 Astra, reached through the `codex` CLI
  (`metaharness/models/astra.py`). Non-interactive `codex exec --json`, thread
  resume so hidden reasoning state carries across turns, the CLI's own login
  (no API key in the repo). The integration follows Auto-CTFer's codex
  backend. When codex reports a usage limit the adapter raises
  `ModelUnavailable` and the loop stops cleanly instead of scoring garbage.
* **Jev** — TypeSafe's System-One model (`metaharness/models/jev.py`).
  `POST https://api.typesafe.ai/v1/systemone` with typed questions
  (`Choice` / `Score` / `Noul`) and calibrated probabilities back in
  70–700 ms. Key in `.env` as `TYPESAFE_API_KEY` (never in code).
* **Claude** (`claude -p`) is wired as an explicit stand-in for Astra
  (`--system2 claude`, `--proposer claude`). It is never substituted silently.

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
