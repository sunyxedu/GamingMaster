# Programmatic interfaces

The board is a local deterministic puzzle, so a network account is unnecessary. Interfaces below are ordered from most suitable for this Linux, headless, offline benchmark to least suitable.

## 1. Local deterministic JSON API (chosen)

**Maturity/latency.** This is a small adaptor-facing protocol around a pure-Python engine (the benchmark permits an own implementation). It is deterministic, unit-testable, and has sub-millisecond transitions; model latency is the only meaningful delay. No package beyond the evaluator's Python 3.12 standard library is needed. **Exposure.** `reset(seed,width,height,mines,first_click_safe)`, `observe()`, and `act({"type":"reveal|flag|chord","row":r,"col":c})`; observations contain the visible grid, flags, status, remaining mine count, and safe-revealed count. The engine must keep the mine map private until `lost`/`won`. **Risk/cost.** No service, credentials, GUI, license, or ToS risk; the adaptor author must document the exact first-click and chord semantics.

Use this JSON shape so `interfaces.md` is sufficient to build the adaptor:

```json
{"width":9,"height":9,"mines":10,"status":"playing",
 "cells":[["unknown", "revealed:1", "flagged"]],
 "safe_revealed":12,"safe_total":71}
```

Actions are one JSON object per turn. `reveal` on an unknown cell performs flood fill for zeros; `flag` toggles only hidden cells; `chord` is valid on a revealed number and applies the flag-count rule. Return the next observation and a terminal reason. A seed must deterministically select the mine layout; if first-click safety is enabled, derive the layout after the first coordinate while retaining reproducibility.

**Install/setup.** Use `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/minesweeper/bin/python`; no `pip install`, server, account, or environment variable is required. The adaptor phase should (1) implement/choose a documented PRNG such as `random.Random(seed)`, (2) generate exactly `mines` unique coordinates, (3) enforce first-click safety, (4) calculate eight-neighbor counts and flood fill, (5) expose the JSON calls above, and (6) run deterministic reset/action smoke tests. Keep mine coordinates out of model observations.

## 2. Open-source solver/bot libraries

Repositories such as [avineshwar/minesweeper-solver](https://github.com/avineshwar/minesweeper-solver), [AdityaMhatre/Minesweeper](https://github.com/AdityaMhatre/Minesweeper), and [microsoft/minesweeper](https://github.com/microsoft/minesweeper) illustrate board representations and constraint solvers. Maturity varies (many are demos, not maintained libraries); latency is local and usually milliseconds. They expose an in-memory grid or UI-specific callbacks rather than a stable benchmark API. Code licenses must be checked per repository; copying code can create license obligations. They have no service cost, but often assume a browser or desktop UI and therefore are not sufficient alone for this headless run. Use their algorithms as references, not as required dependencies.

## 3. Terminal curses clients

[Simon Tatham's Portable Puzzle Collection](https://www.chiark.greenend.org.uk/~sgtatham/puzzles/) includes a Mines variant with a text UI and source code. A pseudo-terminal can send keystrokes and parse text, and it works without a physical display. This is mature free software, but screen parsing is slower and brittle (terminal dimensions, redraws, and keymaps); the exact random seed and board export controls vary by puzzle. It exposes text/characters rather than structured JSON. License is the collection's free software license; check package metadata before redistribution.

## 4. Browser automation

For a web-hosted clone, [Playwright for Python](https://playwright.dev/python/) or [Selenium](https://www.selenium.dev/documentation/) can click cells and read DOM/accessibility labels. Mature, but needs a browser binary and usually network access; canvas-only games require coordinate clicks and screenshot/OCR. Latency is tens of milliseconds to seconds. Automating a third-party site can violate its Terms of Service, trigger anti-bot systems, or change unexpectedly. There is no reason to use it for this offline benchmark.

## 5. Screenshot plus input injection

[Pillow](https://pillow.readthedocs.io/), [OpenCV](https://docs.opencv.org/), and [PyAutoGUI](https://pyautogui.readthedocs.io/) can capture a clone and synthesize mouse clicks. On a server, [Xvfb](https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml) supplies a virtual display. This is the least reliable option: template/OCR errors, focus and scaling problems, and roughly 50–200 ms or more per action. It exposes pixels only and may violate a site's anti-cheat/ToS. It requires GUI packages and has no advantage here.

## 6. Native Windows game hooks/emulation

Old Windows Minesweeper clones can be run under [Wine](https://www.winehq.org/) and inspected with process/debugger hooks, or driven through Wine's X11 window. This is version-specific, fragile, and potentially disallowed by the game's license or anti-cheat. It adds Wine and a GUI dependency and cannot guarantee offline reproducibility. Do not select it.

### Selection

Select the local deterministic JSON API. It is the only option that is headless, offline, deterministic under `reset(seed)`, and free of credentials and external ToS. The benchmark needs game semantics, not a particular commercial client; an in-process API also gives Astra complete structured observations and makes Jev's typed questions cheap.

### Credentials and environment

No server, account, API key, or secret environment variable is needed. If the harness wants configurable defaults, it may define non-secret `MINESWEEPER_WIDTH`, `MINESWEEPER_HEIGHT`, `MINESWEEPER_MINES`, and `MINESWEEPER_SEED`, but the canonical `reset` arguments must override them. Never put credential values in this study.
