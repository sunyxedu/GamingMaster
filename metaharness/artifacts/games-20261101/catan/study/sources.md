# Source ledger and verification limits

Initial notes: 2026-10-29. Study completed/rechecked: 2026-10-30. This document distinguishes reference URLs from evidence actually retrieved on this machine.

## What was verified on this machine

- Evaluator interpreter: `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/catan/bin/python`, Python **3.12.3**, Linux **x86_64** (`linux-x86_64`). These were read from that interpreter, not the system Python.
- Its package inventory contains pip 26.2.2; import checks found **no** `catanatron`, `catanatron_gym`, `gymnasium`, `gym`, `numpy`, or `networkx`.
- Searches of the workspace, the project's runs, and the user's uv cache found no Catanatron source/package suitable for inspection. This is not a claim about every file on the machine.
- Earlier notes recorded failed HTTP attempts to GitHub raw content and a pip index lookup. On 2026-10-30, fresh requests to `https://github.com/bcollazo/catanatron`, `https://pypi.org/pypi/catanatron/json`, `https://www.catan.com/understand-catan/game-rules`, and `https://catanatron.readthedocs.io/en/latest/` all failed with curl exit 6, `Could not resolve host`. No web page, package metadata, or upstream source was successfully fetched. A failed pip index lookup in this environment **does not establish that the package does not exist**.
- This session can write only in the study workspace and `/tmp`. The evaluator venv lies outside those writable roots. No installation was performed. The adaptor phase needs a context allowed to install into that exact venv, as requested by the user.

Consequently, this is a concrete study and implementation specification based on established base-game rules and prior knowledge, with source URLs for follow-up. It is **not live-verified web research**. No simulator release, dependency set, API signature, source license, bot benchmark, or Python 3.12 compatibility result is certified here. Version-sensitive claims in `interfaces.md` are deliberately marked as candidates or verification gates. Do not treat this report as an already working environment.

No game-playing code was written. The four deliverables are present, and `study.json` is validated with the evaluator's Python. API names in the interface lookup guide are historical lookup hints and may differ from the package ultimately acquired. Quantitative dice examples are derived arithmetic; latency figures for model budgets are recommendations, not simulator benchmarks. Free-software candidates and strategic references need source/license/content verification once artifacts or network access are available.

## Reference URLs

All URLs below are **unretrieved in this session**. Titles describe the intended source, not a claim of having read its current contents. Prefer official rules for rules, pinned upstream source for simulator behavior, and community material only for strategic hypotheses.

| ID | Reference and URL | Intended use |
|---|---|---|
| R1 | Official CATAN rulebook hub — https://www.catan.com/understand-catan/game-rules | Download the base CATAN rulebook and almanac; distinguish editions and expansions. |
| R2 | Official CATAN FAQ — https://www.catan.com/faq | Resolve victory timing, road interruptions, development cards, trade, and robber edge cases. |
| C1 | Catanatron upstream repository — https://github.com/bcollazo/catanatron | README, packaging, license, implementation, tests, built-in players, optional web/Gym interfaces. Pin a commit or release before relying on internals. |
| C2 | Catanatron documentation — https://catanatron.readthedocs.io/ | Candidate documentation entry point; availability and current content could not be checked. Use C1 if absent or stale. |
| C3 | Catanatron distribution metadata — https://pypi.org/project/catanatron/ and https://pypi.org/pypi/catanatron/json | Candidate current package name; verify Requires-Python, dependencies, release files, and repository correspondence. |
| C4 | Historical/alternate core distribution — https://pypi.org/project/catanatron-core/ | Check if the chosen source revision uses a split core distribution. Do not install this and `catanatron` together merely to try both. |
| C5 | Catanatron repository code search — https://github.com/bcollazo/catanatron/search?q=playable_actions&type=code | Locate decision enumeration and API implementation; authentication may be required for online search, but local source search requires none. |
| A1 | JSettlers2 — https://github.com/jdmonin/JSettlers2 | Free Java client/server implementation with robot players and protocol/source for an alternative offline engine. |
| A2 | Pioneers — https://pioneers.sourceforge.net/ and https://sourceforge.net/projects/pioneers/ | Free Catan-style client/server implementation with computer players; investigate headless server/client packages. |
| A3 | OpenSpiel — https://github.com/google-deepmind/open_spiel and https://openspiel.readthedocs.io/en/latest/games.html | General game framework. A standard supported CATAN environment is **not established here**; do not assume its presence. |
| A4 | PettingZoo environment catalog — https://pettingzoo.farama.org/environments/ | General multi-agent framework. A first-party standard CATAN environment is **not established here**. |
| A5 | Colonist — https://colonist.io/ and https://colonist.io/terms-of-service | Online commercial alternative; not suitable for the required offline benchmark. Current bot permission is unverified. |
| A6 | CATAN Universe — https://catanuniverse.com/ | Official digital product, not an established free headless research API. |
| S1 | CATAN community strategy wiki — https://en.wikibooks.org/wiki/Settlers_of_Catan/Strategy | Community discussion of resource production, placement, cities/development cards, ports, and trading. Exact current page contents unverified. |
| S2 | BoardGameGeek CATAN strategy forum — https://boardgamegeek.com/boardgame/13/catan/forums/67 | Community strategic discussion and competing approaches. Forum advice is contextual, not a rules authority. |
| S3 | Colonist strategy blog — https://blog.colonist.io/ | Modern online-player strategy discussions. Timers, negotiation, rankings, and house settings may differ from the simulator. |
| P1 | pip download documentation — https://pip.pypa.io/en/stable/cli/pip_download/ | Preparing Python/platform-specific offline artifacts. |
| P2 | pip install documentation — https://pip.pypa.io/en/stable/cli/pip_install/ | Offline installation with `--no-index`, local wheels, and pinned requirements. |

## Attribution conventions

`[R1]`, `[C1]`, etc. in the deliverables refer to the URLs above; the main files also repeat key URLs so they stand alone. Mathematical production calculations are direct derivations from two fair six-sided dice. Proposed observation formats, evaluation protocols, budgets, and model responsibilities are recommendations by this study, not claims made by an upstream project. Strategic rules of thumb are hypotheses to test against the chosen bot mix, not guaranteed optimal play.
