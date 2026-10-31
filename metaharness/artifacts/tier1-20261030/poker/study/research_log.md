# Research evidence and limits

Study date: 2026-10-29. This directory contains research and interface specifications, not a game-playing implementation.

## Verified on this machine

- Interpreter `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/poker/bin/python`: CPython **3.12.3**, GCC 13.3.0. Its pip is 26.2.2. `pokerkit`, `rlcard`, `open_spiel`, and `numpy` are not installed in that interpreter.
- No PokerKit source/package or built `gmsim` extension was found in the searched repository, `/tmp`, or the user's uv cache. The repository does contain the `gmsim`/`gmrules` source and an independent PokerKit reference integration.
- Read `scripts/reference/pokerkit_hunl.py`, `tests/test_hunl_reference.py`, `model/games/gmrules/hunl.py`, `data_processing/rs/gmrules/src/games/hunl/{mod,encode}.rs`, `data_processing/rs/gmsim/{README.md,pyproject.toml,python/gmsim/__init__.py}`, and `scripts/slumbot_client.py`, all relative to `/home/hemlix/GamingMaster-mh-tier1`.
- Inspected `tests/fixtures/hunl/pokerkit_reference.json`: recorded source **pokerkit 0.7.6**, 240 completed hands across four configurations and two generation directions, 3,182 recorded events, 158 hands with all five board cards. All recorded payoff pairs sum to zero. This is inspection of previously generated evidence, **not a fresh execution of PokerKit or a newly passed differential test**.
- The reference integration explicitly runs PokerKit 0.7.6 under Python 3.12. This supports that version choice. Its distribution metadata and installation on this evaluator remain to be checked during provisioning.

Evidence hashes (SHA-256):

| Repository file | Hash |
|---|---|
| `scripts/reference/pokerkit_hunl.py` | `0d8079db93d94d0db931e29de9b324180b1936402b8678e77d357f802c5e87d5` |
| `tests/fixtures/hunl/pokerkit_reference.json` | `df7090d14e366dba2adc5aa290e2bd36520360eeae27977da81c69488ce8d54f` |
| `tests/test_hunl_reference.py` | `35cb6b3b28ecf0efc883d00f4e4ab3a16db0ae8ea1c589cee27760ac9f60b42e` |

## Web access limitation

HTTPS requests through the evaluator's Python failed with `URLError: [Errno -3] Temporary failure in name resolution` for:

- <https://pokerkit.readthedocs.io/en/stable/>
- <https://pypi.org/pypi/pokerkit/json>
- <https://raw.githubusercontent.com/uoftcprg/pokerkit/main/README.rst>
- <https://rlcard.org/games.html>
- <https://www.pokertda.com/poker-tda-rules/>
- <https://www.pokernews.com/poker-rules/texas-holdem.htm>
- <https://www.pokernews.com/strategy/heads-up-poker-strategy.htm>

There is no browser/search connector exposed in this session. The web URLs in the deliverables are reference locations based on established documentation and literature; their current contents and availability were **not live-verified**. No quotations or claims of fresh web review are made. Interface facts explicitly tied to local files were checked by reading those files. Strategy numerical recommendations are design priors, not scraped solver outputs. Performance estimates are expectations, not benchmarks.

## Provisioning status

No packages were installed. The study's writable locations exclude the evaluator venv, and network access cannot currently retrieve missing distributions. The adapter phase needs either a staged PokerKit 0.7.6 wheel and its actual dependencies, a provisioned venv, or the documented source-based fallback. Runtime play itself needs no network, GUI, account, or paid service. An engine installation smoke test and live rules checks are explicitly outstanding; see `interfaces.md`.

## Completion audit

Rechecked the existing study notes against this environment while completing `strategies.md` and `study.json`. The evaluator still reports Python 3.12.3 and only pip 26.2.2. Fresh `curl -IL --max-time` requests to PokerKit documentation, the exact PokerKit 0.7.6 PyPI metadata endpoint, and <https://upswingpoker.com/heads-up-poker-strategy/> each failed with curl exit 6, `Could not resolve host`. No web content was retrieved. No browsing/search tool is exposed in this session.

Re-read the local PokerKit reference, its differential-test source, HUNL Rust rules, Python fallback API, build metadata and root license discussion. Independently recounted the committed JSON fixture: 240 trajectories, 3,182 events, 158 complete-board trajectories, all zero-sum; the three hashes above match. Searches of the user's uv cache, `/tmp`, and evaluator venv found reference fixtures but no PokerKit package or built gmsim wheel/extension. These searches establish what was found in those locations, not absence everywhere on the machine.

Completed a concrete opponent specification and corrected the blind-fold acceptance example to `[+50,-50]` in engine seat order (BB, SB). Added explicit duplicate scoreboard isolation and a two-lane alternative for adaptive histories. Strategy citations are marked as unverified reading targets; numerical priors are labeled design choices. No game-playing code, network workaround, package installation, or runtime engine verification was performed.

Final artifact checks ran in the specified evaluator Python: all four required files are nonempty; `study.json` parses as strict JSON, has exactly the requested top-level and interface keys, and uses the required value types; Markdown deliverables contain source URLs. Independently recalculated the strategy guide's five sizing-table rows and nine-out draw example. These checks passed. They validate the study artifacts and arithmetic, not the uninstalled poker engine or the proposed opponent's strength.
