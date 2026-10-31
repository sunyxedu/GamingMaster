# Hanabi (card game): study brief

Hanabi is a cooperative, hidden-information, turn-based card game for 2–5 players. All seats work toward five ascending fireworks, one in each color, for a maximum score of 25. The requested evaluation controls every seat, but each decision must still use only that seat's permitted information. It is **tier 1**: an offline simulator changes state only when an action is applied. Seconds or minutes of reasoning consume no game time. Evaluator time/call budgets are separate from the rules. [R1, R2, R3]

## Evidence and verification status

This study was prepared on 2026-10-29. Live requests to GitHub, raw.githubusercontent.com, and PyPI failed with `curl: (6) Could not resolve host`. There is no browser/search tool available in this session. The URLs below are source references for subsequent verification, **not pages successfully retrieved during this study**. Rules, interface descriptions, and community conventions below are based on established prior knowledge; package-version details are explicitly provisional. No game-playing code has been written.

Local checks did verify the evaluator interpreter is Python 3.12.3 on linux-x86_64; pip 26.2.2, Python headers, CMake 3.28.3, and g++ 13.3.0 are present. HLE, OpenSpiel, PettingZoo, NumPy, CFFI, setuptools, and wheel are absent from that interpreter. No relevant cached simulator was found in the project, checked temporary paths, or uv cache. See `interfaces.md` for the dependency-free offline choice.

## Standard game, without rainbow or other variants

* Deck: five colors, ranks 1–5; **3 copies of each 1, 2 copies each of 2/3/4, 1 copy of each 5 per color**. Ten cards per color, fifty total.
* Deal five cards per seat with 2–3 players, four with 4–5. Players see everybody else's hand, never the faces of their own cards. Deal and draws are private to the drawer but visible to the other seats. Begin with eight information/clue tokens, three life/fuse tokens, and five empty stacks.
* On a turn, take exactly one action: play an own-hand card; discard an own-hand card to recover a clue token; or spend a clue token to tell one other seat a color or rank present in its hand. Turns advance clockwise.
* A clue must identify **every** matching card and no nonmatching card. It cannot target yourself, identify both rank and color, or name an absent attribute. Untouched cards receive negative information: a red clue excludes red on every other card in that target's current hand.
* A play succeeds only when its rank is exactly one more than that color's current stack height. A failed play removes the card to discards and loses one life. A successful 5 also recovers one clue token, capped at eight.
* Play and discard remove the selected card and draw one replacement if the deck is nonempty. A clue does not draw. Failed plays also draw if the episode has not already ended.
* The recommended HLE-style profile forbids discarding at eight clues, forbids a clue at zero clues, and allows redundant but truthful clues. Some tabletop interpretations differ on discarding at full clues; freeze this profile for evaluation rather than mixing rules.
* Drawing the final deck card starts the final round: **each seat gets one further turn, including the seat that drew it**. The triggering turn is not one of those extra turns. During that round there are no draws and hands can shrink. End earlier on completion of all five stacks or exhaustion of lives. [R1, R2, R3]

## Score and progress

The visible stack total is the sum of the five stack heights, always 0–25. The recommended benchmark terminal score is **0 on loss of the third life, otherwise the stack total**. This makes a bomb-out a loss and follows the commonly used HLE benchmark convention. The physical rules describe the fuse-out as losing; implementations sometimes retain achieved stack points instead. Publish the scoring profile explicitly and check any imported engine before using it. The adaptor should expose both `fireworks_sum` and `score`; report `score` to the evaluator, without normalization, clue bonuses, or survival bonuses. [R1, R3, R4]

During play, stack total measures immediate progress. Remaining copies and the lowest rank permanently lost in each color determine the achievable ceiling. A score of 20 can be an excellent salvage result after a necessary card is lost, but the evaluation objective remains expected terminal score, not percentage of the remaining ceiling. Use score distribution, bomb-out rate, and frequency of 25 only as supplementary diagnostics.

## Controls and pace

Use structured actions, not keyboard or mouse input: `PLAY` with current hand index, `DISCARD` with current hand index, `REVEAL_COLOR` with another seat and color, or `REVEAL_RANK` with another seat and rank. Select from the observation's legal-action list. A valid but incorrect play is an ordinary move with a life penalty; an out-of-range index or illegal clue is an adaptor validation error and must not silently consume a turn. There is no pass action. `interfaces.md` defines exact JSON fields and terminal behavior.

There is **no finite game-imposed decision deadline**. `study.json` uses **`decision_latency_s: -1.0` as a documented sentinel for unbounded latency**, because standard JSON cannot encode infinity. Do not treat it as a negative timeout. Astra may plan on every turn; Jev is optional and unnecessary for timing. An operational timeout such as 60 seconds per reasoning call is a harness budget, not a game rule.

## Source references

* **R1 — Rules reference:** https://www.ultraboardgames.com/hanabi/game-rules.php — deck, clues, turn actions, ending and scoring. Secondary reproduction; verify against the chosen edition when online.
* **R2 — Spiel des Jahres game listing:** https://www.spiel-des-jahres.de/en/games/hanabi/ — game identity, cooperative premise, player range.
* **R3 — Google DeepMind Hanabi Learning Environment (HLE):** https://github.com/google-deepmind/hanabi-learning-environment — reference simulator and Python API. Legacy organization URL: https://github.com/deepmind/hanabi-learning-environment . Source files of interest: `hanabi_learning_environment/rl_env.py`, `pyhanabi.py`, and `hanabi_lib/hanabi_state.cc`.
* **R4 — Bard et al., The Hanabi Challenge: A New Frontier for AI Research:** https://arxiv.org/abs/1902.00506 — standard AI benchmark, partial observability, self-play versus ad-hoc cooperation, and convention-based reasoning.
