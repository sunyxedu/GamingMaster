# Source and verification ledger

Date: 2026-10-29. This is a **network-blocked study**, not completed live web research. References supply concrete places to verify the findings. No referenced external page was retrieved successfully; no quotations or current release numbers are asserted.

## Direct observations

- `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/dominion/bin/python -V` returned `Python 3.12.3`.
- Its `python -m pip list` returned only `pip 26.2.2`; `pip show pyminion` reported package not found.
- HTTPS requests to `https://pypi.org/pypi/pyminion/json`, `https://raw.githubusercontent.com/evanofslack/pyminion/master/README.md`, and `https://wiki.dominionstrategy.com/index.php/Dominion` failed with curl exit 6 (`Could not resolve host`). No GitHub branch or file existence was verified by the failed raw-content request.
- Filename searches of the project, `/home/hemlix/.cache`, `/tmp`, and `/opt` found no pyminion or other Dominion simulator source/cache. This is not proof that no copy exists anywhere on the host.
- Java and Node executables are present, but versions/dependencies and suitability for alternate engines were not tested.
- The study sandbox can write only the study directory and `/tmp`, not the evaluator venv. Approval is unavailable. No package installation was attempted after the absent-package/network checks; no files in the venv were changed.
- No game-playing code, bot, or simulator has been written or run. JSON validation of the study artifact is not simulator compatibility validation.

## Source register (all external URLs unretrieved this session)

| Source | URL | Intended use / confidence boundary |
|---|---|---|
| Official publisher rules | https://www.riograndegames.com/wp-content/uploads/2016/09/Dominion-2nd-Edition-Rules.pdf | Authoritative rules; candidate PDF path needs checking |
| Publisher product page | https://www.riograndegames.com/games/dominion-2nd-edition/ | Find current official rules download |
| Community rules and card reference | https://wiki.dominionstrategy.com/index.php/Gameplay | Rules overview |
| Base set / edition list | https://wiki.dominionstrategy.com/index.php/Dominion | Distinguish editions |
| Card list | https://wiki.dominionstrategy.com/index.php/Card_list | Card text and linked FAQs |
| End of game | https://wiki.dominionstrategy.com/index.php/End_of_game | End conditions |
| Strategy overview | https://wiki.dominionstrategy.com/index.php/Strategy | Community strategy starting point |
| Big Money | https://wiki.dominionstrategy.com/index.php/Big_Money | Economy baseline and its limits |
| Engine | https://wiki.dominionstrategy.com/index.php/Engine | Draw/action/payload balance |
| Chapel | https://wiki.dominionstrategy.com/index.php/Chapel | Aggressive trashing |
| Smithy | https://wiki.dominionstrategy.com/index.php/Smithy | Terminal draw and money decks |
| Witch | https://wiki.dominionstrategy.com/index.php/Witch | Curse race |
| Militia | https://wiki.dominionstrategy.com/index.php/Militia | Hand-size attack and timing |
| Sentry | https://wiki.dominionstrategy.com/index.php/Sentry | Deck control and trashing |
| Gardens | https://wiki.dominionstrategy.com/index.php/Gardens | Alternate scoring and pile endings |
| Workshop | https://wiki.dominionstrategy.com/index.php/Workshop | Gaining and Gardens support |
| Throne Room | https://wiki.dominionstrategy.com/index.php/Throne_Room | Repeated action resolution |
| Council Room | https://wiki.dominionstrategy.com/index.php/Council_Room | Draw with opponent benefit |
| Library | https://wiki.dominionstrategy.com/index.php/Library | Draw-to-hand-size sequencing |
| Pyminion repository | https://github.com/evanofslack/pyminion | Candidate Python simulator: README, source, tests, license, packaging metadata |
| Pyminion package index | https://pypi.org/project/pyminion/ | Releases, Requires-Python, dependencies, artifacts |
| Pyminion machine metadata | https://pypi.org/pypi/pyminion/json | Obtain hashes/metadata in a network-enabled provisioning environment |
| DominionSim | https://github.com/Geronimoo/DominionSim | Candidate Java simulation/bot alternative |
| Androminion | https://github.com/mehtank/androminion | Candidate open-source Android/Java VDOM alternative |
| Dominion Online | https://dominion.games/ | Official online service; unsuitable for offline evaluator |
| Temple Gates Games | https://www.templegatesgames.com/dominion/ | Official digital client; unsuitable for free-software/headless requirement |

## Evidence labels used in the deliverables

- Rules: high confidence from established game knowledge; check official rulebook for exact edition wording.
- Strategic advice: community-style synthesis, not measured win-rate claims. Conditional heuristics and baseline policies are explicitly distinguished from optimal play.
- Simulator recommendation: provisional engineering selection. Exact API, dependency set, license, card coverage, reproducibility, tiebreak correctness, and Python 3.12 compatibility are **not verified**.
- Adaptor protocol and test plan: a proposed design owned by this study, not an API claimed to exist upstream.

The next phase needs an approved upstream source archive plus dependency wheelhouse brought onto this machine, or working networking during provisioning. Play itself must remain offline. Re-fetch and archive source/license/metadata in the study directory before claiming implementation readiness.
