# Baba Is AI — study brief

**Recommendation:** use the free Baba Is AI Python environment, with a symbolic state interface and one action per turn. Keep **tier 1**. Score completed puzzles, not movement, time spent thinking, or intermediate rules made.

## Evidence and scope

Study date: 2026-10-30. The user identifies the target as `baba-is-ai`, says seeds select puzzles, and specifies puzzles solved as the score. This is a benchmark/reimplementation inspired by **Baba Is You**, not a request to run the commercial game.

Verified locally: the evaluator `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/baba/bin/python` is **Python 3.12.3**, with only pip installed. `baba-is-ai`, Gymnasium, NumPy, and pygame are absent. No matching source or wheel was found in the searched workspace and local uv cache. HTTPS retrieval of PyPI and GitHub failed with `curl: (6) Could not resolve host`. Consequently **no upstream web page, package metadata, source implementation, or benchmark API was inspected successfully**. All web links below are reference/discovery targets, not claims of live retrieval. General game knowledge is useful planning guidance; exact benchmark mechanics must be checked against its source. `interfaces.md` records the integration blocker rather than inventing environment IDs, action integers, or dependencies.

## Objective and rules

Baba Is You is a grid puzzle in which movable words form the rules of the current board. A sentence such as `BABA IS YOU` gives Baba objects player control; `FLAG IS WIN` makes flags goals; `WALL IS STOP` makes walls block movement. A flag is not intrinsically a goal, a wall is not intrinsically solid, and Baba is not intrinsically the player. Changing the words changes these properties. Official game reference: https://www.hempuli.com/baba/ . Community reference targets: https://babaiswiki.fandom.com/wiki/Rules and https://babaiswiki.fandom.com/wiki/YOU .

Basic sentences read left to right or top to bottom in contiguous aligned cells. Text is normally pushable. Word tiles and the objects they name are distinct: moving the word `ROCK` is not moving an ordinary rock. One word may participate in both a horizontal and a vertical sentence. Text arrangements can therefore modify several rules at once. More elaborate grammar exists in the original game; do not assume the benchmark supports it. Reference: https://babaiswiki.fandom.com/wiki/Rules .

The standard win condition is an object with `YOU` contacting an object with `WIN`; a controlled object can sometimes itself gain `WIN`. Use the benchmark's actual success signal and interaction ordering as authority. Movement commands act on all currently controlled objects, not a selected individual. Push chains, blocking, destructive interactions, and rule changes can make a single command affect much of the board. Reference targets: https://babaiswiki.fandom.com/wiki/WIN and https://babaiswiki.fandom.com/wiki/PUSH .

Typical original-game properties include `STOP` (blocking), `PUSH` (pushable), `DEFEAT` (destroys contacting controlled objects), `SINK` (destructive overlap), `HOT`/`MELT`, `OPEN`/`SHUT`, and `MOVE` (automatic movement on game turns). `X IS Y` may transform objects rather than give a property. These are **a vocabulary to look for, not a verified list of Baba Is AI features**. Exact supported words, transformations, overlap rules, and resolution order belong to package inspection. Do not silently import the full commercial rule system into a small benchmark.

## Controls and pace

| Operation | Intended meaning | Benchmark status |
|---|---|---|
| North / east / south / west | Issue one grid movement command to controlled objects | Map to source-defined enum; integers unverified |
| Wait | Advance a turn without directional movement; useful with moving objects | Expose only if the installed environment supplies it |
| Undo | Restore a previous turn in the commercial game | Not assumed available or legal in this benchmark |
| Restart | Reset a puzzle | Evaluator policy must define whether it is allowed and how actions are counted |
| Reset with scenario seed | Begin an evaluation puzzle | User requirement; native seed-to-level behavior unverified |

The world waits between submitted turns. Automatic objects, if supported, advance on turns rather than while Astra thinks. Seconds or a minute of reasoning do not intrinsically harm the position. A move limit, step-shaped reward, evaluator wall-clock timeout, or model budget can still penalize excessive work; those are separate constraints. Polling or reading a state must never advance it. Failed movement and wait actions may still advance the engine and consume the action budget.

`study.json` uses **`decision_latency_s: -1.0` as an explicit sentinel for no finite in-game thinking deadline**. JSON has no standard infinity literal. It does not mean that the evaluator grants unlimited wall-clock time. The adaptor should translate this sentinel into “no real-time controller needed.”

## Score and progress

For the requested evaluation, a puzzle contributes **1 exactly once if the engine reports success, otherwise 0**. Report total solved and number attempted; also report solved/attempted when comparing runs with different numbers of puzzles. Record seed, selected level, step count, termination reason, and success independently of raw environment reward. A positive shaped reward or `terminated=True` alone is not proof of a win. Truncation is not success unless the same final transition explicitly establishes success.

Intermediate milestones—access to a word, a newly formed useful rule, or reaching a new room—help planning but do not increase the official score. Multiple completions after replaying the same evaluation puzzle must not inflate score. Seeds must select reproducible puzzles from a versioned manifest, not merely reseed a fixed level's random-number generator. Details and leakage controls are in `interfaces.md`.
