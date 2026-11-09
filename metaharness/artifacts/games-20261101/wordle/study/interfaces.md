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
