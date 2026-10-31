# TextArena study

Study date: 2026-10-31. Target: Linux x86_64, evaluator Python `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/textarena/bin/python` (verified Python 3.12.3).

## Evidence and limits

TextArena is a collection of text-controlled environments, not one game with a shared ruleset. The official project and distribution are https://github.com/LeonGuertler/TextArena and https://pypi.org/project/textarena/ . The project site is https://www.textarena.ai/ . These are source locators, **not pages successfully retrieved in this session**: requests to PyPI and raw.githubusercontent.com failed with DNS resolution errors. No TextArena distribution was installed in the evaluator venv; `pip list` showed only pip 26.2.2. Searches of the workspace and pip/uv caches found no TextArena source or wheel. Consequently, package signatures, current versions, dependencies, registered game IDs, and exact action grammars below require confirmation against the release obtained during setup. No game-playing code was written.

The supplied notes establish an offline, seeded, fixed mixture of single-player and two-player games, baseline opponents, and win/normalized-score evaluation. They do **not** identify the version, mixture, baseline implementations, weights, or score normalization. These are evaluator configuration requirements, not facts to invent from the upstream catalogue. `interfaces.md` gives a concrete integration specification and freeze checklist; `strategies.md` gives conditional strategy modules.

## Rules, objectives, and controls

Each episode selects one environment and variant. Its opening observation should describe the rules, player role, objective, action format, and often the initial board or puzzle. Later observations contain board updates, feedback, opponent actions, and game messages. Submit a string action using that environment's grammar; many games use bracketed coordinates, columns, guesses, or commands. There is no universal direction/action vocabulary and no guaranteed global legal-action list. Coordinates, player numbering, pass syntax, move limits, and permitted multi-action submissions must be checked per variant.

Examples of relevant families, **conditional on the frozen manifest**:

| Family | Objective and important rules |
| --- | --- |
| Tic-tac-toe / Connect Four | Complete a line before the opponent; placement-only versus gravity changes legal moves. |
| Othello / chess / Nim | Respect flips and forced passes; king safety and draw rules; or heap-removal and normal/misère termination respectively. |
| Sudoku / sliding puzzles | Satisfy all constraints or reach a target arrangement within the environment's action limits. |
| Wordle / Mastermind / number guessing | Infer a secret from feedback before exhausting guesses; the secret is private. |
| Battleship / card games | Act using public history and one's own private information, never the opponent's hidden board/hand. |

Upstream environment implementations and their prompts are the authority for TextArena variants: https://github.com/LeonGuertler/TextArena/tree/main/textarena/envs . Familiar board-game rules are useful priors, not substitutes for those implementations.

Invalid syntax or illegal moves can cause warnings, retries, turn consumption, penalties, or forfeits depending on release and game. A malformed action is not a harmless way to query legality. The adaptor must track the returned active player rather than alternate players blindly; retries, passes, extra turns, and phases can preserve or change initiative.

## Score and progress

Use the evaluator's authoritative score if supplied. Otherwise propose and freeze the following before evaluation:

* Two-player result: win = 1, draw = 0.5, loss/forfeit = 0. If the evaluator means literal win rate, report wins/episodes separately and do not silently count a draw as half a win. Convert native rewards only after verifying their semantics; `(r + 1) / 2` is valid only for documented `-1/0/+1` outcomes.
* Binary single-player puzzles: solved = 1, failed = 0. Valid placements, clues gathered, or moves survived can be progress telemetry without changing this objective.
* Games with genuine numeric scores: normalize using a declared per-game formula and fixed bounds or documented maximum. A generic reward field is not necessarily a numeric game score. Unknown or unbounded score ranges require an explicit evaluator-approved mapping; never normalize by the best result seen in the current run.
* Aggregate using the frozen game weights. Include per-game performance and seat-balanced two-player results so easy puzzles or first-player advantage cannot obscure weaknesses.

Log native outcome/reward, normalized score, terminal reason, move count, invalid actions, seed, game ID, version, and opponent. A wrapper's time/step-budget truncation is separate from an environment's natural terminal loss; freeze its treatment. Infrastructure failures should be identified, not silently replaced with a different game or a fabricated score.

## Pace and tier

**Keep tier 1.** In the intended local Python environment, progression is driven by calls to `step`; no GUI or real-time controller is needed. There is no known common wall-clock turn deadline for local play. Astra can spend seconds to a minute reasoning while the board waits. A game's move/guess budget counts actions, not thinking time. Any exception in the selected release must be caught during the source audit and idle-state check described in `interfaces.md`.

The evaluator may impose a total episode timeout, model-call budget, or action cap; none was supplied. These external limits must be recorded separately. `study.json` uses **`decision_latency_s: -1.0` as an explicit sentinel for no finite in-game deadline**, not a negative duration. JSON cannot represent infinity portably. This does not promise unlimited evaluator runtime. Jev is optional for inexpensive classification; there is no real-time need for it in this suite.

## Source map

* Official README/API examples and license: https://github.com/LeonGuertler/TextArena
* Package metadata, release selection, Python requirements: https://pypi.org/project/textarena/
* Core environment/state implementation: https://github.com/LeonGuertler/TextArena/tree/main/textarena
* Game-specific prompts, transitions, rewards, and assets: https://github.com/LeonGuertler/TextArena/tree/main/textarena/envs
* Observation/action wrappers: https://github.com/LeonGuertler/TextArena/tree/main/textarena/wrappers
* Supplied agent implementations: https://github.com/LeonGuertler/TextArena/tree/main/textarena/agents

All web links in this study are reference destinations to verify when sources are available; no current upstream content or release has been independently fetched in this session.
