# TextArena offline Tier 1 adaptor

`game.py` implements `make_game() -> TurnGame` using the real, installed **TextArena
0.7.4** local Python environments. It runs headlessly in the evaluator's Python 3.12.3
venv at `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/textarena`. There are no accounts,
credentials, environment variables, servers, model weights, downloads during play, or GUI.
No hosted TextArena agent is instantiated. The package is MIT licensed; its original
license is in `TEXTARENA-LICENSE`. Dependency licenses remain in the retained wheels.

## Frozen manifest and seed splits

The study proposed example games but supplied no mandatory manifest. This adaptor freezes
one single-player and one two-player game, equally weighted. Other catalogue entries
are deliberately outside this manifest; failures never select substitute games.

| Index | Native registry ID | Constructor settings | Players | Opponent | Score |
|---|---|---|---|---|---|
| 0 | `TicTacToe-v0-raw` | native defaults, empty 3×3 board | 2 | Perfect minimax v1 | win 1, draw 0.5, loss 0 |
| 1 | `Mastermind-v0-raw` | `code_length=4, num_numbers=6, duplicate_numbers=False, max_turns=2` | 1 | none | native fixed numeric reward, bounds [0,1] |

`seed % 2` selects the manifest entry. For TicTacToe, `(seed // 2) & 1` chooses the
learner's seat (player 0 is O, player 1 is X). Native player 0 always moves first.
No opening prefixes are inserted. Seeded tie breaks among equally optimal baseline moves
provide additional empty-board game diversity without reducing opponent strength.

A stream seed is the unsigned big-endian integer represented by the eight-byte BLAKE2b
digest of UTF-8 `f"{seed}:{label}"`. The independent labels are `constructor`,
`environment`, and `baseline`. The environment digest is passed to native reset. Neither
the external episode seed nor these derived seeds appears in player observations. The
external seed is retained only in evaluator-only `info()`.

`SEEDS` declares 8 train seeds (10–17), 8 held-out seeds (110–117), 2 dev seeds (210–211),
and 8 validation seeds (310–317). Every full split contains four games of each type and
both learner seats twice for TicTacToe. Dev is one example of each game. All integer seeds
are supported using this same mapping; only declared groups are used for evaluation.
Mastermind initial prompts can be identical across seeds because its varied secret code
is hidden. A few seeds can coincidentally produce the same native code or optimal opening.

## Observations and actions

The adaptor uses native raw messages, consumes each active player's queue once, and keeps
separate text histories. `observe()` reads a cache and never advances or drains the engine.
At terminal baseline moves, it additionally drains only the learner-addressed pending
queue, because native terminal turns do not rotate. It never exposes an opponent prompt,
secret code, private RNG state, or a dump of `game_state`. The TicTacToe board is a public
model reconstructed from accepted actions; the baseline sees only that public model.

Observation fields are `game_id`, `player_id` (learner), `current_player` (native active
player or last terminal mover), `observation_text`, `public_state`, `legal_actions`,
`move_count` (all native plies), `remaining_attempts`, `done`, `score`, `terminal_reason`,
`history` (last eight actions), and optional `last_error`. TicTacToe board cells are
row-major, `null` if empty, otherwise player ID 0 or 1. Mastermind feedback is in the
native text history. There are no image or binary observations. Returned containers do
not alias the internal board/history.

`action_spec` changes to the selected game after reset and enumerates every native move
string, plus `noop`. TicTacToe uses `[n]`, n=0–8. Mastermind uses `[a b c d]`, four distinct
digits 1–6 (360 possible strings). Only empty cells or previously unguessed codes appear
in `legal_actions`. Noop is always available before termination, and is documented
separately rather than mixed with playable moves. Whitespace is permitted; actions must
be strings no longer than 128 characters. Parameter ranges, rule details and every
observation field are also in `describe()`.

Every learner call consumes one of **32 wrapper attempts**, including noop and invalid
input. Noop leaves the native state unchanged. The **third invalid learner action in an
episode** forfeits with score zero. Illegal or malformed input is never passed to the
native parser, so it cannot crash the episode. Repeated Mastermind guesses are invalid.
After termination, step is inert. Both games are strictly turn based and have no wall-clock
deadline. Local rules models and search using permitted information are allowed.

## Native behavior and scoring

The installed API is `reset(num_players, seed)`, `get_observation() -> (player, messages)`,
`step(action) -> (done, step_info)`, and `close() -> (rewards, game_info)`. The wrapper reads
terminal rewards/reasons from the same state objects that native close returns; close
itself owns no external resources. `close()` is idempotent.

For TicTacToe the native reward is −1/0/+1, normalized by `(r + 1)/2`. The strongest
opponent makes learner wins impossible from the normal empty board. An optimal learner
can earn 0.5 through a draw; this is not remapped to a win.

**Mastermind is a numeric-score game in this release.** Solving earns native reward 1.
At its native turn limit the reward is `(black + 0.5 * white) / 4` for the last guess.
Normalize by `(r - 0)/(1 - 0)`, the fixed declared bounds [0,1]; no observed batch maximum
is used. Wrong but informative guesses can earn partial credit. With four distinct digits
from six, the smallest achievable *legal* failed terminal score is 0.25; the largest is
0.75. Wrapper forfeits and truncations score zero. Literal solved/win is reported separately.

There is an audited upstream off-by-one: `check_turn_limit()` executes before the turn
counter increments. Therefore `max_turns=2` displays **2** in the original prompt but
actually permits **3 distinct guesses**, with a solve taking precedence on the last guess.
`describe()` explicitly corrects this. The adaptor does not patch the native rules. The
short, fixed budget leaves a meaningful score gradient for code inference policies.

In-progress score is zero. A wrapper attempt-cap termination is `wrapper_truncation`,
invalid-input termination is `wrapper_invalid_forfeit`, and a native outcome is
`native_terminal` (or `native_invalid_forfeit` if the engine reports one). An external
evaluator stopping early leaves an unfinished episode with score zero. Infrastructure
exceptions mark `valid_trial=False`, include `invalid_reason`, terminate the instance,
and are eligible for evaluator retry; they are not silently counted as bad play.

`info()` retains all native raw rewards, learner reward, terminal reason, normalized
score, opponent name, seat, total native move count, learner attempt count, invalid count,
termination class, game ID, and weight. `score_breakdown()` provides numeric score, raw
reward, literal win/solve, wrapper truncation, and a game-specific metric. For multiple
episodes use **0.5 × mean(TicTacToe) + 0.5 × mean(Mastermind)**; equal-sized stratified
splits make this the ordinary episode mean. Scores within an episode are not multiplied
by the game weight. Per-game results remain available in the recorded metadata.

## Opponent, reproducibility, installation and tests

The installed `ta.agents` inventory consists of human input and local/hosted LLM clients;
it provides no offline game-specific TicTacToe engine. The custom **Perfect minimax v1**
searches the complete legal game tree to terminal wins/draws/losses, preferring faster
wins and later losses. It is game-theoretically optimal; an external bot cannot be stronger
under these rules. It never chooses a suboptimal move for diversity or seed difficulty.
On 20 balanced-seat native games per policy (seeds 400,402,…,438), it beat first-legal-move
play **20/20 (100%)**, beat a win/block/center/corner/edge policy **1/20 (5%)** with 19 draws,
and drew **20/20** against an independent cached negamax solver, losing zero games in all
three tests. These finite measurements supplement the full-tree search guarantee.

The audited selected constructors contain no randomness. `State.__init__` seeds Python's
global `random`; Mastermind then samples the code. Neither selected game uses NumPy or
runtime external assets. The adaptor seeds before construction, saves/restores Python RNG
state around native construction/reset/step under a lock, and owns a separate baseline
`random.Random`. This preserves the host RNG and makes interleaved adaptor instances
independent. Unrelated threads using global `random` without this lock should not execute
inside these short native calls; separate evaluator worker processes avoid that limitation.

`requirements.txt` and `requirements.lock` pin and hash all **33** dependency wheels.
`runtime/wheelhouse/` contains the complete Linux x86_64 / CPython 3.12 closure (about
12.8 MB), including a locally built chess wheel. This declared dependency is not used
by the selected games. `package-manifest.json` records wheel hashes, release metadata,
and the audited selected TextArena source hashes. No game assets need staging.
Run `./setup.sh` for an entirely offline installation into the exact evaluator venv;
it uses `--no-index --require-hashes` and runs `pip check`. No optional environment variables
are needed, as documented in `env.example`.

Run acceptance checks from this directory with:

```sh
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/textarena/bin/python verify.py
```

`verify.py` blocks socket connection and DNS APIs **before importing the adaptor**. It
completes real native episodes and replays every declared seed with identical actions,
observations and outcomes. It checks both learner seats, independent strong scripts,
random legal actions, native reward boundaries and outcomes, partial scoring, idle
stability, interleaved RNG isolation, invalid/repeated actions, wrapper truncation,
terminal idempotence, and injected infrastructure failure. It writes only
`runtime/acceptance.json` and logs when redirected. Scoring fixtures may inspect hidden
codes solely to verify native outcome boundaries; measured learner policies use only
public observations. `runtime/acceptance.json` records measured per-game scores and
latencies. Runtime execution does not write or modify source files.

The orchestrator smoke test is run from the repository root:

```sh
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/textarena/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('runs/textarena/adaptor', expected_tier=1).to_markdown())"
```

This is a small two-game benchmark, not the entire TextArena catalogue. Its held-out seeds
are finite and repeated play can overfit; train/dev/validation/held-out partitions must
remain separate during harness search. TicTacToe skill saturates at a draw against perfect
play; remaining discrimination comes from Mastermind inference under its short budget.

Final acceptance measurements on all 24 train/validation/held-out scenarios: random legal
play averaged **0.21875** (TicTacToe 0.0, Mastermind 0.4375); an independent perfect
TicTacToe solver plus a visible-feedback candidate-elimination Mastermind policy averaged
**0.53125** (TicTacToe 0.5, Mastermind 0.5625). Thus the competent script remains below the
suite's theoretical 0.75 ceiling imposed by TicTacToe draws. On this machine the largest
observed reset/step call was under 0.8 seconds and the largest observation under 6 kB.
These are fixed-policy measurements, not normalization bounds or difficulty tuning by seed.
