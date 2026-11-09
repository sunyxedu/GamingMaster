Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `Wordle` — Wordle normal mode: guess the five-letter hidden word in six guesses. Use 'guess crane', bare 'crane', or {'guess':'crane'}; ASCII case and outer whitespace are normalized. Every answer is legal, as are other words in the pinned guess file. Rules provide absolute paths to both public word files, their version and hashes. Feedback is five green/yellow/gray tiles: greens consume exact letter matches first; remaining copies can yield yellows left to right, gray means no unused copy remains. There is no hard-mode constraint.  The secret is fixed for the round. 'noop' changes no guesses. Invalid actions return an error without using a guess; repeats of legal guesses consume a turn. Terminal actions do nothing. Observations include game/mode, rules, turn (accepted guesses so far), max_turns, history, done, solved, result={solved, guesses_used, max_guesses}, score, and error (null or text). The secret and seed are never in observations. Score is 0 for unfinished/lost games; a solve scores (max_guesses + 1 - guesses_used) / max_guesses. Fewer guesses is better.

Actions:
- guess: Submit a complete guess, e.g. 'guess crane'; see describe() for grammar.
- noop: Leave the board and guess count unchanged; clear any action error.
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Parameterized actions use 'guess <value>'. A bare value or {'guess': value} is also accepted. Invalid actions do not cost guesses.

Model available through `ctx`: `ctx.think(prompt, session_id=...)` — Astra (System-2). Budget: 40 calls
per episode; one call may return a plan of several actions. `ctx.act(action)` applies one action and
returns the next observation. Each episode has a wall-clock deadline of 1800 s, including model
latency (`ctx.budget.remaining()["wall_s"]`); when it passes, or the call budget is used up, the episode
ends with the score reached. Each episode starts cold: fresh `ctx.memory` and fresh module globals.

Evaluation: mean score over the training seeds (the game's own score, see the description), then fewer
System-2 calls, then fewer actions. A candidate can only be selected if none of its training episodes
ends with an error (see the contract: e.g. a TimeoutError from a model call that the harness does not
catch).

What the harness must design: how observations are shown to Astra (full history, a diff, an ASCII map
with a legend, only what changed), what to remember between calls (rules learned, states visited,
plans), how to parse Astra's reply into actions robustly, how many actions to commit per call, when
cheap scripted search or bookkeeping replaces a model call, and how to notice that a plan or belief was
wrong. If the description says that rules or action effects are not given, discovering them by
experiment is part of the task.

A plan must be able to fail and recover. Whatever executes without the model (a compiled build order, a
convention policy, a solver) must check its own effects after each step — `ctx.progress()` counts the
actions since the score and the observation last changed and the actions the game rejected — and, when
the evidence says the plan has stopped working, hand the situation with that evidence to Astra for
re-planning, then verify the new plan the same way. Scenarios the training seeds never show are where a
scripted plan breaks: a candidate with identical scores on every training seed is a scenario-independent
script and will be judged on the unseen validation and held-out seeds.

**Evaluation scope.** The harness is selected on the training seeds of this game and then reported on
unseen seeds of the same game; there is no held-out game. Rules, strategies and search that the game
description or the study establish may be implemented as code once they are verified against
observations — but nothing specific to the training seeds.

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules, bookkeeping and search as code and put planning and adaptation into Astra prompts — but verify
every claim against observations, because sources can be wrong or outdated for this build of the game.

The full documents are readable in this archive under `prior/` (`prior/strategies.md`, `prior/brief.md`, `prior/interfaces.md`, `prior/study.json`, `prior/ADAPTOR.md`). Below: excerpts within a size budget; every cut is marked `…(truncated; full text in prior/<file>)`.

### study/study.json (summary)
- strategy_summary: For Wordle, maintain the exact surviving answer set by simulating duplicate-aware feedback, then choose guesses that split candidates well while respecting remaining turns and hard-mode legality. For classic Mastermind, enumerate codes and use Knuth-style minimax over exact black/white feedback, with a deterministic tie-break. Put arithmetic and state tracking in code; reserve Astra for adapting to rule variations and Jev for optional typed rankings. Verify the active game variant before applying a strategy.
- scoring: Report solved status, guesses_used, and max_guesses as the primary result. Wordle uses the evaluator's six-guess limit; classic Mastermind commonly uses ten guesses. Keep failures distinct from wins and calculate any scalar reward at the evaluator boundary so guesses-used semantics remain explicit.
- interface.observation: JSON with game mode, public rules, turn count, limit, prior guesses and exact feedback, and terminal status. Keep the target word or code hidden until terminal evaluation. Wordle feedback is one green/yellow/gray value per position; Mastermind feedback is exact and misplaced peg counts.
- interface.actions: Wordle submits one valid five-letter guess string. Mastermind submits a four-color sequence for classic rules. Return structured invalid-action errors without consuming a turn; otherwise return updated observation, done status, and score metadata.

### study/strategies.md
# Strong play and implementation priorities

## Wordle

**Track exact constraints and duplicate letters.** After every clue, intersect the candidate answer set with words whose simulated feedback exactly equals the observed feedback. This is the safest core solver and naturally handles repeated letters; informal “green/yellow/gray means simply present/absent” rules often eliminate valid candidates or retain impossible ones. Put feedback simulation, candidate filtering, and bookkeeping in **code**. Sources: https://www.nytimes.com/games/wordle/index.html ; https://en.wikipedia.org/wiki/Wordle .

**Choose informative openers and follow-ups.** A strong first guess covers frequent letters and avoids duplicate letters, balancing vowel and consonant coverage; familiar openers include “SLATE,” “CRANE,” and “TRACE,” but no universal opening is best across answer-list versions and objective functions. Once clues narrow the set, choose a candidate that partitions remaining answers into feedback groups as evenly as possible (expected information gain / entropy); with few turns left, optimize worst-case remaining candidates and prefer a possible answer if it also preserves information. Compute partitions exactly in **code**. Give Astra the objective tradeoff and permission to explain unusual choices, but do not depend on its unaided arithmetic. Jev can rank a supplied shortlist by human plausibility or flag an odd word, but should not replace deterministic scoring. Sources: https://www.nytimes.com/2022/01/03/technology/wordle-word-game.html ; 3Blue1Brown's information-theory explanation and solver: https://www.youtube.com/watch?v=v68zYyaEmEA and https://github.com/3b1b/videos/tree/master/_2022/wordle ; https://www.cut-the-knot.org/ctk/Wordle.shtml .

**Respect the ruleset and remaining turns.** In hard mode, all known greens and yellows must be reused, which can force non-answer probes to be illegal; normal mode permits a “burner” guess chosen to split a large candidate set. Use exact legal-action validation in **code**. Put adaptation to any nonstandard word list, challenge mode, or evaluator-specific scoring in **Astra prompts**. Never assume a daily answer index equals the seeded target.

**Common mistakes.** Treating every gray duplicate as absent despite another copy being yellow/green; guessing a plausible word outside the evaluator's allowed list; overvaluing common vocabulary after candidate constraints are decisive; spending a late turn on information that cannot be converted into a solve; and using online answer scraping that leaks the target or breaks offline reproducibility. Keep all of this in **code** checks and explicit **Astra prompts**. 

## Mastermind

**Use minimax candidate elimination (Knuth).** Enumerate the possible codes, score every legal guess against each surviving code, partition candidates by feedback, and choose the guess minimizing the size of the largest resulting partition. In the classic 4-position, 6-color, repeats-allowed game, Donald Knuth showed a strategy that solves every code in at most five guesses, with the first guess commonly `1122` (two colors, each repeated twice); it is a probe and need not itself be a possible secret. Exact enumeration, feedback calculation, and minimax belong in **code**. Prompt Astra to explain a fallback choice only if variant constraints make the proven strategy inapplicable. Sources: Knuth, “The Computer as Master Mind,” https://www.cs.cmu.edu/~hde/ai-for-games/knuth-mastermind.pdf (copy/scan availability varies); accessible overview https://www.cut-the-knot.org/ctk/Mastermind.shtml ; original game rules https://en.wikipedia.org/wiki/Mastermind_(board_game) .

**Be precise about feedback.** Black/exact pegs are position matches; white/misplaced pegs are matching color multiplicities among unmatched positions. Feedback does not reveal the location of any white peg. Reject candidates inconsistent with the entire history. Put all this in **code**. Jev should not calculate feedback or do candidate arithmetic.

**Use the right tie-break.** A minimax guess can be eliminated from the remaining possible secrets yet still be the best probe. If the largest partitions tie, prefer a guess that is itself a surviving code, then use expected remaining candidates or a deterministic lexical tie-break. Search over all legal guesses when affordable; the classic space is only 6^4=1296 codes. If the number of colors/slots grows, measure search cost and use branch-and-bound or candidate-only search. These are **code** decisions.

**Common mistakes and variants.** Counting matching colors without removing exact matches double-counts pegs; counting unique colors instead of multiplicities breaks repeated-color codes; assuming feedback order conveys positions is wrong; assuming the five-guess guarantee holds for other peg counts, palettes, or no-repeat rules is unsafe. Variant discovery belongs in the adaptor config and **Astra prompts**; exact validation belongs in **code**. Jev has no necessary role here, though it may classify a human-readable rules description when bootstrapping an unknown variant.

## Allocation across the agent

The harness should use **code** for legal action validation, seeded state bookkeeping, feedback, candidate filtering, Wordle partition scoring, and Mastermind minimax. Use **Astra prompts** to select/adapt the objective, diagnose inconsistent observations, and handle evaluator rule descriptions that differ from the standard. Use **Jev questions** only for inexpensive typed preferences over a small supplied set (for example, “which candidate looks like a more ordinary English word?”); do not ask it to infer hidden-state arithmetic or provide the source of truth. For this deterministic puzzle, a correct exact solver should dominate model intuition.

### study/brief.md
# Wordle and Mastermind: game brief

## Scope and evaluator contract

This study covers two related but distinct hidden-answer deduction games. The evaluator description governs the harness contract: tier 1, one target selected from a seed, at most six Wordle guesses, and a score based on solving and guesses used. Classic Mastermind has different rules and scoring, so the evaluator must identify which game is active; this study recommends separate Wordle and Mastermind modes under one study. The local `prompt.md` specifies the harness constraints. The standard rules below are sourced from the linked game references.

## Wordle

Wordle is a single-player daily word puzzle. In the common English version the answer is a five-letter word, and the player submits valid five-letter guesses, with six attempts maximum. After each guess, each letter tile reports green when that letter is in the answer at that position, yellow when it occurs elsewhere (respecting duplicate-letter counts), and gray when no unmatched instance remains. The answer is fixed for the round; feedback is the only information revealed. Win by submitting the answer before attempts run out. A natural evaluator score is `(solved, guesses_used)`, or a scalar that ranks any solve above a failure and rewards fewer guesses; the harness should retain the exact pair to avoid ambiguity.

The official product can be played once per day; the evaluator instead explicitly uses seed-selected targets, so reset(seed) should select a reproducible target without imposing a real-time deadline. The game changes only after a valid guess is submitted. A model taking seconds or a minute does not cost an attempt while the harness waits; decision latency is therefore unbounded for game scoring (record 0 / effectively no deadline, rather than pretending the daily cadence is an action timer). A per-call operational timeout may still be imposed by the runner.

Controls are a five-character guess accepted as a word in the configured allowed-guess list. The solver needs the answer list, guess list, previous guesses, and exact color feedback. Standard hard mode requires each subsequent guess to reuse revealed green positions and include revealed yellow letters; the evaluator description does not specify hard mode, so use normal mode unless the adaptor declares otherwise.

Sources: NYT Wordle rules and play page, https://www.nytimes.com/games/wordle/index.html ; NYT help center, https://help.nytimes.com/hc/en-us/articles/9972553445012-Wordle ; Wordle discussion and strategy overview, https://en.wikipedia.org/wiki/Wordle .

## Mastermind

Classic Mastermind is a two-player code-breaking game. The codemaker chooses a secret sequence of four colored pegs from six colors; in the common classic rules, colors may repeat. The codebreaker submits a four-peg guess each turn. The codemaker returns black/ red markers for correct color in the correct position and white markers for correct color in a different position, without identifying which pegs earned which markers. The codebreaker wins by exactly matching the code, typically within ten guesses; otherwise the codemaker wins. Variants change code length, palette, repetition, feedback, and turn limit, so the evaluator must pin the variant in its state/configuration and report it.

As with Wordle, classic tabletop play is turn-based: the hidden code does not change while the solver thinks. The evaluator's Tier 1 classification is correct. There is no decision deadline in the normal game. The score should record solved status and number of guesses, and preserve failures distinctly. Mastermind's common optimization target is guaranteed solve within five guesses under the classic 4-position/6-color/repeats-allowed setup; don't assume this bound for other variants.

Controls are a sequence of four color identifiers; observations are the exact-count black/white feedback peg counts after each guess. Feedback counts must be computed by matching exact positions first, then counting color overlap among remaining positions (multiset intersection), to avoid double-counting repeated colors.

Sources: Hasbro's Mastermind product/rules, https://instructions.hasbro.com/en-us/instruction/mastermind-game ; rules overview, https://en.wikipedia.org/wiki/Mastermind_(board_game) ; Knuth's classic analysis is indexed as “The Computer as Master Mind,” Journal of Recreational Mathematics 9(1), 1976–77; accessible exposition and algorithm links are collected at https://www.cut-the-knot.org/ctk/Mastermind.shtml .

## Tier and pace

Keep tier 1. Both are finite, turn-based deduction puzzles: the world waits for a guess, and each action is a complete discrete guess. Astra's seconds-to-minute reasoning fits naturally; Jev is optional for typed rankings or recognition but brings no reaction-time advantage. The evaluator's score and episode state, rather than wall-clock speed, should determine progress.

### adaptor/ADAPTOR.md
# Wordle and Mastermind adaptor

This is a local, offline Tier 1 implementation of the selected turn-based games. It
starts no server, process, GUI, network connection, or account. `close()` is
idempotent and has no external resources to release.

## Wordle

`make_game(mode="wordle")` implements normal English Wordle: a five-letter hidden
answer, six guesses, no hard-mode constraint, and duplicate-aware green/yellow/gray
feedback. `reset(seed)` uses Python's seeded PRNG to choose from 2,315 answers. The
bundled historical answer snapshot is from Josh Stephenson's `Wordle-Solver`
repository, commit `48c32e1ac0cb1ec87740cab009214e89b2c636c8` (2025-04-11), which
documents the 2,315 NYT answer words. The 12,972-word allowed guess list comes from
Tab Atkins' `wordle-list` initial snapshot, commit
`ae0403f957ae7312d72929403432becc0b0f5d63` (2022-01-13). Both repositories include
MIT licenses; copies are included in `WORDLIST_LICENSE.txt` and
`GUESSLIST_LICENSE.txt`. These are community-maintained snapshots of Wordle lists,
not a live or NYT-supported feed. The pinned SHA-256 values are:

- `answers.txt`: `f40e47975cf360ff9febe7d49c8e285054d8fa3fb9644cea575922060951d6ee`
- `allowed_guesses.txt`: `feacc549249b5dcb43e2799fbbcda14ebc33d55904fb15d043779104e60c1579`

Both lists are public and their paths, version, counts, and hashes are included in
the observation rules, so a harness can filter candidates without receiving the
secret. Every answer is accepted as a guess; guess-only words are accepted from the
second list. The answer is only included in evaluator-only `info()` after the episode
ends.

## Mastermind

`make_game(mode="mastermind")` defaults to classic Mastermind: four positions, six
color IDs (`0` through `5`), repeats allowed, and ten guesses. A seeded PRNG selects
from the complete configured code space. Feedback first counts exact-position pegs,
then counts matching multiplicities among unmatched positions. `positions`,
`colors`, `repeats`, and `max_guesses` can be passed as explicit JSON kwargs for a
variant; they remain fixed across seeds for that evaluation. The code is only
included in evaluator-only `info()` after completion.

## Interface and scoring

Wordle accepts `guess crane`, the bare string `crane`, or `{"guess":"crane"}`.
Mastermind accepts `guess 0123`, the bare digit string, or `{"guess":[0,1,2,3]}`.
`noop` is the only parameterless action. Invalid actions return an observation with
an error and consume no guess; valid repeated guesses do consume one. Observations
contain public rules, turn count, history and feedback, terminal flags, error, and
the explicit `solved`, `guesses_used`, and `max_guesses` result fields. They never
contain the seed or hidden answer. Scalar score is zero for an unfinished or lost
game and `(max_guesses + 1 - guesses_used) / max_guesses` for a win. `info()` and
`score_breakdown()` retain the primary result fields.

The train, dev, validation, and held-out groups are disjoint. Their seeds were
selected so each group has the same mix of answer words with and without repeated
letters, and Mastermind targets with two, three, or four distinct colors, with and
without repeats. The rules and target-selection process are identical in each group.
There are no opponents in these single-player puzzles. For headroom verification, a
candidate-elimination Wordle solver and Knuth-style Mastermind minimax policy were
run on the 12 train and two dev seeds (14 episodes per game), without consulting
held-out or validation results. Wordle solved 14/14 in a mean 3.64 guesses (mean
score 0.56); Mastermind solved 14/14 in a mean 4.36 guesses (mean score 0.66). A
random legal guess policy solved 0/14 in both games. The detailed episode results
are kept under `runtime/baseline_results.json`.

### study/interfaces.md
# Programmatic interfaces

## Recommendation

Implement a small local, deterministic Python environment for each ruleset and have the generated harness call `reset(seed)` / `step(action)` against it. This matches the supplied evaluator contract, works offline and headless, has no account or service dependency, and exposes exact structured state. It is a harness adapter target, not game-playing code. Keep hidden target out of the observation; expose it only in terminal info for evaluation. Make rules configurable and version the word lists / Mastermind variant so seeded runs are reproducible.

Environment is evaluator Python 3.12.3 at `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/wordle/bin/python`. Required software: that interpreter; standard library is sufficient. No credentials or environment variables are required. The adapter phase should pin a word-list snapshot and checksum, ensure it is bundled locally (no network at episode time), create a seeded PRNG from the provided seed, reset the chosen game mode, and serialize observations/actions as JSON. The word list must distinguish valid guesses from possible answers. A well-known community snapshot is available at https://github.com/tabatkins/wordle-list (large allowed-word list); the evaluator asks for official lists, while NYT answer/guess lists are not published as a supported public API, so validate the chosen snapshot against the evaluator's bundled official data if available. Do not silently substitute an arbitrary system dictionary.

Observation: JSON containing `game`, `mode`, `turn`, `max_turns`, public rule configuration, prior guesses and feedback, and terminal/win flags. Wordle feedback should be five entries from `green/yellow/gray`; Mastermind feedback should be `{exact:n, misplaced:n}`. Hide answer/code until terminal evaluation. Actions: Wordle `{guess: "xxxxx"}`; Mastermind `{guess: [color_id,...]}` or a documented compact color string. Invalid actions should return a structured error without consuming a turn. `step` returns new observation, reward/score metadata, done flag, and terminal info.

## Alternatives

| Interface | Maturity and exposure | Latency / cost | Risk and fit |
|---|---|---|---|
| Local purpose-built environment (chosen) | Simple, mature implementation pattern; exact JSON state and legal actions, deterministic seeded resets | Local calls, negligible latency, free, offline | Lowest operational risk. Must specify evaluator rules and word-list provenance accurately. |
| NYT Wordle website / browser automation | Official game, mature web app; DOM or screenshot can show board and keyboard, but secret answer is hidden | Network/browser launch and UI interaction; free to play, account not generally needed | Online dependency, website changes, possible automation restrictions; not compatible with offline server. NYT terms: https://www.nytimes.com/content/help/rights/terms/terms-of-service.html |
| Browser DOM / reverse-engineered game bundle | Community scripts can inspect game state and embedded word data; e.g. browser userscripts or extracted JavaScript | Low latency after page load; free | Fragile to site updates, online dependency, possible ToS concerns; no supported public game API. Avoid relying on private endpoints or bypassing controls. |
| Community Wordle solvers / datasets | Useful reference implementations, word lists, and analysis; e.g. https://github.com/3b1b/videos/tree/master/_2022/wordle and https://github.com/tabatkins/wordle-list | Local and free once installed | Not a live control API; licenses and list accuracy vary. Check licenses and data provenance before bundling. |
| Mastermind board / mobile game automation | Screen capture plus keyboard/mouse injection can operate a GUI implementation; OCR/vision can read pegs | Variable, slower than local logic; free options exist | Not suitable without GUI, game-specific UI work, and possible app/site ToS. Rules vary by implementation. |
| Emulator hooks / plugins / bot libraries | No universal standard API across commercial Mastermind apps; custom/open-source implementations may expose state | Depends on target; generally unnecessary | Game-specific, often requires installing the target and may expose hidden answer unfairly. No applicable official API identified. |

No server address, account, or API key should be configured for the selected offline environment. If a later phase chooses a specific online implementation, obtain its address and any credentials from the user and document them as environment-variable names only; never commit secret values.

## Adapter contract details

For Wordle, maintain independent `allowed_guesses` and `answers` arrays, normalize to lowercase ASCII, reject words of other lengths, and return deterministic feedback with duplicate-letter accounting. One reliable feedback procedure is: mark greens and remove those answer instances, then scan unmatched guess letters and mark yellow only while a matching answer instance remains. Reset chooses an answer deterministically from the answer set using the supplied seed; avoid direct modulo bias if uniformity matters.

For Mastermind, use an explicit config such as positions=4, colors=6, repeats=true, max_turns=10. Seed the hidden code from that config. Compute exact matches first, then misplaced count as the sum over colors of the minimum remaining multiplicities. Expose only counts. Allow the evaluator to specify alternative config rather than embedding one assumed variant.

Scoring should return at least `{solved: bool, guesses_used: int, max_guesses: int, score: ...}`. Define scalar score in the adapter contract; a clear choice is 0 for failure and `1 - (guesses_used-1)/max_guesses` for success, while retaining the pair as primary reporting. If compatibility expects another convention, map the pair at the outer evaluator boundary.


## Measured model latency (at archive creation, 2026-10-31 05:27)
- System-2 (Astra), one-word answer: 39 s
- System-2, an answer of ~1067 output tokens: 76 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).