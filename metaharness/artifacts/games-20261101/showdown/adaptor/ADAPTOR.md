# Pokémon Showdown: offline Gen 9 Random Battle

`game.py` implements Tier 1 `make_game(**kwargs)` / TurnGame, reusing the engine
plumbing preserved in `game_v2.py`. It runs the real Pokémon Showdown simulator,
with freshly generated six-Pokémon teams, entirely offline. No GUI, account,
credentials, local server, or connection to a live game service is needed.

## Installed opponent and measured strength

The fixed opponent is **[Foul Play](https://github.com/pmariglia/foul-play)**,
revision `8f541f80f91f960603673796559f796b10ccf0fb`, using **poke-engine 0.0.48**
compiled with Gen 9 Terastallization support. Foul Play is an established open-source
search bot, substantially stronger here than the old adaptor's heuristic. Showdown
itself supplies only a random example player; that is not used as the opponent.
No trained policy or live service is required.

Each decision uses Foul Play's protocol parser and random-battle inference to
sample **four plausible hidden teams**, then runs Rust Monte Carlo tree search
for **200 ms per sample, one thread**. Search visits are aggregated by their share
of visits in each sample; the highest-ranked legal choice is selected. It searches
attacks, switches, setup, recovery, status and Tera, with public-set inference.
Every scenario uses the same opponent and budget; no weaker option is exposed.

### Hard compute bounds (this revision)

The saved worker already supplied 200 ms to the native engine, but had no external
search watchdog. poke-engine 0.0.48 checks elapsed time only after batches of 1,000
iterations, so that parameter is not a hard wall-clock limit. The reported
150–490-second stalls were not reproduced in this build session; the first three
old-worker steps measured about 0.91 seconds. This revision explicitly contains
native overruns and stalls elsewhere in the decision pipeline:

* Each decision has a **2.0-second budget**, starting before protocol parsing.
  Inference and MCTS run in a fresh Linux fork child. It publishes the cumulative
  policy after each completed sample. The parent stops/reaps the child at the
  deadline and selects from the completed MCTS results. Cleanup allows up to
  0.2 seconds beyond the search deadline. No random/heuristic fallback is used.
* A **3-second Node watchdog** covers worker startup, parsing, inference, native
  search and IPC. If no sample completes, or the worker stalls, the trial is
  marked invalid with an infrastructure error; the evaluator can retry it.
* A **6-second Python transport timeout** bounds an entire reset/step, including
  chained opponent choices. Infrastructure failure also closes the process group.
  Cleanup can take up to six additional seconds on an unresponsive process; it
  is not a successful gameplay step. The injected worker stall returned in 3.08 s.
* Defaults are **120 turns / 180 accepted player decisions**. An additional
  **480 seconds of cumulative engine work** forces a tie at the next completed
  step. This budget counts reset/step work, excludes the agent's thinking time,
  and can overshoot by the final bounded RPC. It keeps engine work below roughly
  nine minutes even for unusually long episodes, leaving time for model calls.

The per-sample target is unchanged from the prior bot, retaining its normal
search strength. The killable child is necessary because a Python thread or signal
handler cannot reliably interrupt a native extension holding the GIL. New referee
telemetry records search visits, bot decision count, deadline hits, total/max bot
compute seconds, and cumulative engine seconds. None is shown in observations.

### Measured latency and opponent strength

**Bounded Foul Play won 103 / 110 games (93.6%), with 7 losses and no ties**
against the exact old heuristic, approximately 95% Wilson interval **87.4–96.9%**.
Seeds were `1000003 + 7919*i`, `i=0..109`, disjoint from all evaluation splits,
with generated teams and the same fixed compute settings. Foul Play played p2;
the preserved heuristic played p1 from its own player stream, using unchanged
ratings and seeded tie-breaking. Its only generalization is finding the enemy by
opposite side ID. The benchmark-only `baseline` RPC is not an exposed action.
This is a small single-orientation strength estimate, not a ladder rating.

One initial game (seed 1799822) exposed an existing Illusion parser bug: `replace`
can omit HP. The bridge now preserves the previous public HP/effects in that case.
That seed was replayed after the fix and completed normally. The 110 results above
include that one replacement, not an extra game. Raw initial results, the failed
trial, and retry are retained in `runtime/opponent-benchmark-bounded.json` and
`runtime/benchmark-illusion-retry.json`; the merged results and summary are
`runtime/opponent-benchmark-final.json` and `runtime/benchmark-summary.json`.
All final games finished normally. The heuristic's best score was +1.11958,
below the maximum +1.25; it won seven games, leaving useful score headroom.

Measured on this Linux host with the evaluator Python, Node 24.18.0 and eight
concurrent benchmark games (some integration checks overlapped):

| Measurement | Bounded result |
| --- | ---: |
| Step calls across strength benchmark | 2,643 |
| Median / p95 / maximum step latency | 0.950 / 1.887 / 2.555 s |
| Mean / maximum full battle duration, including reset | 21.36 / 53.38 s |
| Mean turns / accepted player decisions | 19.09 / 24.03 |
| Maximum bot decision compute | 1.064 s |
| Natural search deadline hits | 0 |

Full random-player games on both dev seeds also completed normally:

| Seed | Turns / decisions | Reset | Whole battle | Median / p95 / max step |
| --- | ---: | ---: | ---: | ---: |
| 869431 | 24 / 29 | 2.017 s | 25.785 s | 0.955 / 1.000 / 1.883 s |
| 895003 | 21 / 27 | 1.886 s | 21.000 s | 0.943 / 1.002 / 1.007 s |

Across all six integration games there were 159 steps, median 0.945 s, p95 1.858 s,
maximum 1.913 s, and battle durations 19.79–25.79 seconds. All six baseline players
lost. A 40–60-decision game at the measured pace has ample room inside 15 minutes;
these random players generally lost sooner. Timings exclude model calls, include
real engine work, and can vary with load. The watchdogs bound pathological cases;
no natural stall was needed to verify them (see fault injection below).

The historical unguarded report (102 wins / 8 losses in 110 games, 92.7%) remains
in `runtime/opponent-benchmark.json` for comparison; it is not substituted for the
bounded measurement. Native MCTS randomness means individual replays differ.

## Setup and dependencies

Run `bash setup.sh` in this directory. Build time may use the network; play time
does not. Setup checks the supplied built Showdown checkout and installs
`requirements.txt` using `uv` into the evaluator's Python 3.12.3 environment:

`/home/hemlix/GamingMaster-mh-tier1/runs/venvs/showdown/bin/python`

It clones Foul Play if absent and checks its pinned revision. Rust/cargo is needed
only to build poke-engine. Defaults are `~/gm-tools/pokemon-showdown`,
`~/gm-tools/foul-play`, and `node`; optional overrides are in `env.example`.
No environment variable is mandatory. `SHOWDOWN_PYTHON` is internal and is set by
`game.py` to `sys.executable`, ensuring the bot uses the evaluator's environment.

Required Showdown revision: `a5df8274e85b0889bf2a9b3422a08b39732374fc`.
The supplied build uses Node 24.18.0; Node >=22.18 is required. Revision mismatches
fail explicitly. Keep both external checkouts and compiled dependencies unchanged;
git HEAD checks do not detect local modifications or stale build products.

Public random-battle inference data is bundled in `data/gen9randombattle.json`,
downloaded during this build from
`https://pkmn.github.io/randbats/data/full/gen9randombattle.json`.
SHA-256: `6a4b6fc8391d334e252001d8a48ec383159a7ccfdcab23830ab6def00aab793c`.
The worker replaces upstream's download-capable loader with a local-file-only
loader; missing data fails rather than making a network request. This is a public
prior over possible sets, not either side's generated team or RNG state.

Showdown is MIT licensed. Foul Play is GPL-3.0; its source and license remain in
the pinned checkout. poke-engine source is at
`https://github.com/pmariglia/poke-engine`. Retain upstream licenses if distributing
these dependencies. All software used here is free software.

## Connection, privacy and lifecycle

Each reset closes the previous process group and starts `bridge.cjs`. Python and
Node exchange JSON lines over pipes. Node creates a first-party `BattleStream`
with `getPlayerStreams`, `gen9randombattle`, and independently seeded random teams.
Only p1 protocol feeds the observation; only p2 protocol and its private request
feed `foulplay_worker.py`. Unused spectator/omniscient streams are drained and
discarded. The referee alone accesses true engine HP to calculate evaluator scores.
The bot receives neither the agent's pending choice nor unrevealed actual sets.
Its next choice is submitted before the adaptor returns the next p1 decision.

`step` waits through resolution and any opponent replacement decisions until p1
has another choice or the battle ends. Pivots and forced replacements can create
multiple decisions in one turn. Invalid actions report `last_error` without
advancing; hidden trapping can cause an engine rejection and updated request.
`noop` waits without advancing and clears the last error. No wall-clock move timer
runs while the evaluated agent thinks. Transport timeout defaults to six seconds and cannot be raised above six.

Both Node and the bot worker belong to the instance's process group. `close()`
closes stdin, waits, terminates/kills if necessary, joins the reader, closes pipes,
and cleans the process group even after a bridge crash. It is idempotent. There
are no search children, servers, or connections left behind. Stderr,
verification reports and benchmarks are under `runtime/`; playing generates no
source files. A worker/parser/search/bridge failure marks `valid_trial=False`
with `invalid_reason`, rather than silently falling back to an easier policy.

## Seeds and determinism

`SEEDS` declares 10 train, 10 held-out, 10 validation and 2 dev seeds, all disjoint
and drawn with the same generation procedure and opponent difficulty. Arbitrary
integer seeds also work. SHA-256 hashes `showdown-adaptor-v1:<domain>:<decimal seed>`;
the first eight bytes become four big-endian uint16 Showdown seed words. Domains
`battle`, `team-p1`, `team-p2`, and `opponent` separate battle RNG, each generated
team, and Python's opponent sampling RNG. Explicit team seeds are necessary because
this Showdown revision otherwise generates nondeterministic teams.

Seeds vary species, leads, levels, moves, abilities, items, Tera types, damage rolls
and ties. Initial observations reproduce; **`DETERMINISTIC=False`** honestly
reflects that poke-engine's native MCTS randomness and wall-clock iteration count
make full trajectories stochastic. No scenario seed or referee score appears in
observations. Teams are never fixed/packed for evaluation.

## Observations and actions

`describe()` and `ActionSpec` explain rules, scoring and the interface to the agent.
Choose a string in `legal_actions`: `move 1` through `move 4`, optionally suffixed
with ` terastallize`, or `switch 1` through `switch 6`. Indices are one-based; team
order changes after switches. Forced switches need a switch, and Revival Blessing
selects a fainted teammate with a switch command. `noop` always waits.

`request` is the private p1 request: exact own HP/status, stats, items, abilities,
moves, PP/disabled flags, Tera availability and known trapping. `battle_log`
contains up to 700 p1-visible lines / 65k characters; `recent_events` contains up
to 300 lines / 30k characters since the last choice. Timestamps are removed.
`opponent_revealed`, `active`, and `public_summary` cache common public events:
rounded enemy HP, revealed moves/items/abilities, Tera, boosts, effects, hazards,
weather and terrain. `move_dex` and `species_dex` supply public reference data for
own moves/team and revealed species. Protocol is authoritative; summaries are not
a complete interpreter of Illusion, Transform, boost transfers or unusual typing.

Observations are strict JSON below 200 kB, with no hidden enemy sets, exact enemy
HP, p2 request, RNG state, seed or referee components. Terminal observations clear
requests/actions and preserve the public result and recent log. Repeated observe
calls are side-effect free; returned data can be mutated without changing state.

## Score and limits

`self_hp_fraction = sum(current_hp/max_hp over six Pokémon)/6`, likewise for the
opponent. `hp_margin = self_hp_fraction - opponent_hp_fraction`. Outcome is +1 for
a win, -1 for a loss, and 0 for a tie/unfinished battle.

**Scalar score = outcome + 0.25 * hp_margin**, range [-1.25, 1.25]. Winning remains
primary. `score_breakdown()` provides outcome, both fractions, margin, survivors,
turns, accepted decisions, bot search visits, timing/deadline telemetry and score. `info()` adds seed mappings,
validity, engine revision, opponent, truncation and termination reason. These are
evaluator-only. Intermediate scores reflect current HP with outcome zero.

Defaults are 120 turns and 180 accepted decisions, plus 480 seconds of cumulative
engine work. Limits force an engine tie, preserve HP margin and mark `turn_limit`,
`decision_limit` or `engine_time_limit` as truncated. Normal
wins use `normal`, natural ties `engine_tie`. Noops/bad actions do not consume
accepted decisions; the evaluator must bound agents that never act.

Foul Play/poke-engine is a separate mechanics model from authoritative Showdown,
so unusual mechanics or new data can differ. Public-set inference is uncertain;
MCTS has a finite budget and is not optimal play. This adapter uses upstream
Foul Play's parsing/sampling and poke-engine search, but integrates it over local
pipes instead of its normal WebSocket client and chooses the highest aggregated
legal visit share rather than upstream's random selection among close candidates.

## Verification

The requested generic validator prints **PASS with no warnings**. Run from the
repository root using the evaluator environment:

```sh
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/showdown/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('runs/showdown/adaptor', expected_tier=1).to_markdown())"
```

`runtime/validation.md` records the result. From this adaptor directory:

```sh
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/showdown/bin/python verify_adaptor.py
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/showdown/bin/python verify_deadlines.py
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/showdown/bin/python benchmark_opponent.py --count 110 --workers 8
```

`verify_adaptor.py` plays six complete games (two each with random, first-available
and type-aware scripted actions). It checks JSON/privacy/size, score arithmetic,
malformed/noop handling, turn/decision limits, repeat reset, first-observation
reproducibility, a deliberately killed bridge, and idempotent cleanup. Per-step
latencies and complete episode results are in `runtime/verification.json`.

`verify_deadlines.py` injects search hangs with and without a completed result,
then suspends a real worker with SIGSTOP. The two child-search checks returned in
0.203 seconds for a 0.2-second test budget. The real worker watchdog invalidated
the trial in 3.08 seconds; all children were reaped. Details are in
`runtime/deadline-verification.json`. These tests exercise the stopping mechanism
without waiting for a naturally occurring multi-minute native stall.

A further complete random-player battle runs under `strace -f -e trace=network`.
The trace contains no network connect, listen or send calls. Its source-hash checks are in `runtime/network-trace-bounded.log` and
`runtime/offline-verification.json`. Runtime data stays under `runtime/`; playing
does not modify adaptor source or dependency manifests.

The Illusion regression is covered by `node verify_protocol.cjs` and the complete
replayed battle. An injected exhausted cumulative budget also produced a scored
tie in the real simulator (`runtime/engine-limit-verification.json`). The largest
observation in the strength benchmark was 26,846 bytes, below the 200 kB limit.
