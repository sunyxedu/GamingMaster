# Programmatic interfaces

The reference game is a client-side JavaScript application, not a hosted service with a supported gameplay API. The source is MIT-licensed ([repository](https://github.com/gabrielecirulli/2048)); the practical interfaces below are therefore local and do not require an account.

## Options

| Interface | Maturity / latency | Observation and control | Legal/ToS and cost |
|---|---|---|---|
| **Standalone Python rules engine (chosen)** | Small, deterministic, very mature rules; microseconds to milliseconds per move | Full board as a 4x4 integer matrix, score, move count, terminal flag, and seeded RNG. Call `step("up"|"down"|"left"|"right")`; return a JSON-serializable state and whether a tile spawned. | No website, account, anti-cheat, or ToS issue. Free; Python standard library only. Must faithfully reproduce merge order and 90/10 spawn distribution. |
| Browser DOM + JavaScript source (`gabrielecirulli/2048`) under Playwright/Selenium | Mature browser automation, usually 10–100 ms per key but browser startup is heavier | Read rendered tile elements (`tile-inner` classes encode value and position), score DOM, and game-over state; inject Arrow keys or call the game's JS event path. | Local copy is permitted by its MIT license; automating the public website can violate site policy and is fragile when markup changes. Requires Chromium/driver and extra packages; no GUI is needed with headless mode. |
| Browser screenshot + input (`xdotool`, `pyautogui`, OpenCV/OCR) | Mature desktop tooling but 50–500+ ms and resolution/theme dependent | Screenshot pixels and recognize tile colors/text; inject arrows. | Highest fragility and no need on a server without GUI. X server, fonts, and image/OCR packages add cost; website automation may have ToS concerns. |
| Existing AI/bot implementations (for example [nneonneo/2048-ai](https://github.com/nneonneo/2048-ai), [ov3y/2048-AI](https://github.com/ov3y/2048-AI)) | Community code is useful as a strategy reference; integration quality varies | Usually consume a board array and emit a direction using expectimax/heuristics. Wrap their engine or port the search; inspect licenses and Python/runtime assumptions. | Free/open source when license terms are followed. They are not a stable API, may target the browser DOM, and may not provide seeded reset or exact evaluator telemetry. |
| Reinforcement-learning environment ports (search terms include [Gymnasium](https://gymnasium.farama.org/) and “2048 environment”) | APIs vary and many ports are stale; local step latency is good | Usually `reset(seed)` and `step(action)` with a board observation and reward. Verify dimensions, spawn RNG, merge semantics, and licenses against the reference game. | No network/account after installation, but dependency installation can be difficult offline. Environment mismatch can invalidate scores. |

## Chosen adaptor contract

Use the standalone engine. It is the only option that is fully offline, GUI-free, deterministic under an explicit seed, and independent of browser markup. The adaptor can be implemented with `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/2048/bin/python` and the standard library; no credentials or environment variables are required.

Suggested module contract (the next phase may choose names):

* `reset(seed: int | None) -> dict`: initialize a 4x4 board, score 0, move count 0, and a seeded pseudo-random generator; spawn exactly two starting tiles.
* `observe() -> dict`: return `{"board": [[...], ...], "score": int, "moves": int, "max_tile": int, "game_over": bool, "seed": ...}`. Keep zeros for empty cells. Including the RNG state is useful for exact replay but can be omitted from model prompts.
* `step(action: str) -> dict`: apply one direction. If the direction is a no-op, do not spawn a tile and mark `changed: false`; otherwise merge, add the merge values to score, spawn one tile, increment `moves`, and return the new observation plus `changed` and `spawn` fields.
* `legal_actions() -> list[str]`: directions that change the board. Return an empty list at terminal state.

The engine must preserve the reference line-processing order: compress toward the direction, merge each eligible adjacent pair once, then pad with zeros. A deterministic test suite should compare known boards against the JavaScript implementation's behavior. Do not install packages or fetch assets at runtime; if an adaptor elects to use NumPy, install it into the evaluator venv ahead of execution and record that choice.

## Credentials and setup

No server address, account, API key, or secret is needed for the chosen interface. Setup is: (1) create or vendor the Python engine in the generated harness; (2) run it with the evaluator's Python; (3) expose JSON observations and the four action strings to Astra/Jev; (4) seed reset for each scenario. Browser alternatives would require a local game checkout plus Chromium/Playwright, but they are deliberately not prerequisites.

## Risks and compatibility checks

The main risk is an implementation mismatch: spawning after no-op moves, allowing a tile to merge twice, using the wrong 90/10 distribution, or seeding Python differently from the harness can change scores. Validate merge order with cases such as `[2,2,2,2] -> [4,4]` toward one side and `[2,2,4] -> [8]` after compression. Community bots may carry incompatible licenses or assumptions; browser/screenshot approaches add GUI, latency, and markup risks.
