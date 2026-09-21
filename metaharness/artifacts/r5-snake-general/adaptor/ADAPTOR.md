# Snake adaptor

This is an original, faithful local instance of the study's **classic-walled-v1**
Snake rules, connected directly through the fixed Python `RealtimeGame` interface.
It is not an adaptor to Google Snake or a commercial release. No account, asset
download, display, service, credential, environment variable or extra pip package
is needed. Python 3.11+ and the host `metaharness` package are required. Development
used the existing project virtual environment (Python 3.14.5).

## Setup and use

From this directory, run `./setup.sh`. It uses the current `python`, or the enclosing
project's `.venv/bin/python`, installs `requirements.txt` into that environment,
runs the regression suite, and requires the generic validator to pass.

```sh
export PATH="/Users/hemlix/Documents/GamingMaster/.venv/bin:$PATH"
python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('/Users/hemlix/Documents/GamingMaster/runs/r5-snake/adaptor').to_markdown())"
python run_local.py --seed 3 --output verification/wall-seed-3.json
python run_local.py --replay verification/wall-seed-3.json
```

`make_game()` selects the 10x10 demonstration. Use
`make_game(profile="primary-v1")` for the cross-board suite. `reset(seed)` returns
the initial observation; `observe()` never advances the game. `tick(action)`
applies precisely one movement, without sleeping. `tick_s` is 0.200 in both
primary profiles; `ticks` counts attempted movements including fatal ones.
The game is ready after construction and reset. A reentrant lock makes each
reset, tick and observation atomic; returned snapshots cannot mutate the game.

## Clock ownership and the study's proposed transport

The task's fixed contract supersedes the study's tentative child-process JSONL
transport. Meta-Harness owns the clock and current-action mailbox. Starting an
additional game timer would double-advance gameplay. Consequently there is no
child process, start/act RPC, episode ID, sequence number, deadline, or stale-RPC
filter here. The runner can apply a late model decision on a later tick, as its
contract specifies. Its mailbox retains the last action until replaced; at an
empty initial mailbox it supplies NOOP. Each `tick` resolves one heading only.
Reset between episodes after stopping the old runner; direct calls from an old
runner cannot be distinguished by the fixed `tick(action)` signature.

The actual Meta-Harness wall runner is exercised with a controller sleeping for
1.05 seconds: its independent ticking thread still advances the snake. Calling
`tick` rapidly is useful for deterministic diagnostics but is not real-time
evaluation. `run_local.py` is a separate random-player evaluator using monotonic,
phase-aligned 0.200-second deadlines by default. `--fast` explicitly labels its
output as offline diagnostics. It flags lateness over five intervals as an
infrastructure truncation. The adaptor cannot detect clock stalls or input receipt
times in a third-party runner; those are that runner's responsibility.

## Rules and observations

Cells are `[x,y]` with top-left origin, x right and y down. Body is head first.
Legal actions are `UP`, `RIGHT`, `DOWN`, `LEFT`, `NOOP`. `None` is also accepted as
NOOP for convenience. Names are case sensitive; malformed actions raise without
moving. Reversal is ignored; NOOP continues straight. Growth is one cell per food.
Food is uniform over row-major empty cells. Walls and retained body cells kill;
entering the departing tail on a non-food move is legal. Collision preserves body
and food score, records the attempted destination, and counts the fatal tick.
Final growth commits before victory; a full board has `food=null`. All terminal
states freeze until reset. There is no wrap, starvation, acceleration or opponent.

`describe()` and `action_spec` fully document the player-visible rules. Observations
contain schema/ruleset/scenario versions, profile, dimensions, initial length,
ordered body, direction, food, tick, status, length, raw score, food count,
normalized completion, cadence, requested/applied heading, reversal flag,
attempted destination, and terminal flags/reason. Initial action/destination fields
are null. Snapshots are JSON data and well below 200 kB. They do not expose seed,
RNG state, future food, controller-derived safety masks or paths. No screenshots
are needed because the local game exposes its board directly.

## Seed mapping and suites

Sampling version is `snake-scenarios-v1`. The local RNG hashes ASCII
`snake-scenarios-v1:{seed}:{counter}` with SHA-256, starting counter at 0 and
incrementing after every attempted draw. Interpret the digest as a big-endian
integer; for a bound n, reject values at or above
`2**256 - (2**256 % n)`, then return value modulo n. This avoids modulo bias and
Python RNG version dependence. Seeds are integers, excluding booleans.

The first draw selects a board index (even the singleton demo consumes a draw).
The second selects one of eight transforms. Start offsets are head `(1,0)`,
middle `(0,0)`, tail `(-1,0)` about
`((width-1)//2, (height-1)//2)`. Transforms 4–7 reflect x, then all transforms
rotate clockwise `transform % 4` times by `(x,y) -> (-y,x)`. Direction matches the
head offset. Transform the centered start, keeping the selected board dimensions
fixed, so the rectangle is always 16x12 rather than introducing a fifth 12x16
shape. The eight transforms yield four distinct straight-body headings, with
reflection duplicates. Every supported start has at least two clear forward
moves. Each food draw indexes empty cells ordered by y then x, after committing
the current body. Player choices therefore affect future food locations.

| Profile | Boards | Seconds per tick |
| --- | --- | --- |
| `demo-v1` (default) | 10x10 | 0.200 |
| `primary-v1` | uniform 8x8, 10x10, 12x12, 16x12 | 0.200 |
| `generalization-odd-v1` | uniform 9x9, 11x11, 13x13 | 0.200 |
| `stress-fast-v1` | primary mixture | 0.100 |
| `stress-slow-v1` | primary mixture | 0.300 |

`seeds.json` provides disjoint development and held-out lists. Tests use only
development seeds. Held-out seeds are an evaluation split, not secrets or a
security boundary. Score the odd and speed suites separately. Obstacles are not
implemented; adding them requires a separately versioned generalization profile.
Unknown profiles/configuration keys are rejected rather than silently ignored.

`DETERMINISTIC=True` means identical resets and identical trajectories given seed,
profile, ruleset/scenario version and applied tick-indexed actions. It does not
claim that model arrival jitter repeats. `replay()` returns evaluator-only seed,
requested/applied actions and final state; observations contain none of the seed
metadata. `run_local.py --output ...` saves this plus receipt/application elapsed
times, clock mode, cap and metrics. `--replay` checks the recorded final snapshot.
Meta-Harness separately logs its actual tick actions; adaptor replay records are
kept in memory until the evaluator persists them. No game policy is built into
the environment.

## Scoring, caps and integration limits

Raw score is `food_eaten = length - 3`. `game.score` is the primary cross-board
scalar `food_eaten / (width*height - 3)` (0–1). Observation `score` is the raw game
score; observation `completion` and `info()["score"]` are normalized. `info()` also
provides `raw_score`, `food_eaten`, `won`, ticks and terminal flags. Aggregate mean
completion and the fraction with `won=true`; compare ticks only at equal
completion. There is no idle-survival reward.

The evaluator must cap at `game.recommended_tick_cap = 2*(width*height)**2`.
`run_local.py` does this externally, checking natural death/victory first and then
calling `game.truncate("external_tick_cap")` if still alive. Its `--max-ticks` may
only lower the cap. Other evaluators must do the same after each tick; the stock
Meta-Harness runner does not currently enforce this game-specific cap. On earlier
cancellation call `truncate("external_cancel")`; infrastructure failures use
`truncate("infrastructure_failure")`. These hooks are evaluator operations, not
legal player actions, and preserve score and tick count. `terminated` indicates
natural death/victory, while `truncated` indicates external termination. A natural
terminal state on the cap tick wins precedence.

The current stock runner's restricted `GameView.info()` omits game-specific metrics
from its final report. Evaluators needing win rate/truncation must collect the
underlying `game.info()` or terminal observation, as the included local runner
does. No changes were made to the surrounding harness repository.

## Verification

`test_game.py` checks reversal/noop, growth, legal departing-tail entry, retained
body collision, wall collision and freeze, final-cell victory, seeded replay,
all primary shapes/headings, JSON size and snapshot isolation, invalid inputs,
external truncation/score preservation, and the actual wall runner during a
controller stall. An independent test-only Hamiltonian player also completes a
real seeded 8x8 game from reset to full-board victory; it is never used by the
adaptor. Verification artifacts and measured timing are in `verification/`.
