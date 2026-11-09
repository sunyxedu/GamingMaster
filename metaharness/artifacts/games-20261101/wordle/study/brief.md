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
