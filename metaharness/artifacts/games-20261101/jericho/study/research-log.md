# Evidence and remaining setup work

Checked 2026-10-30 in `/home/hemlix/GamingMaster-mh-tier1/runs/jericho/study`. Existing `brief.md` and `interfaces.md` were preliminary documents from an earlier attempt; this revision updated them and added `strategies.md` and `study.json`. No game-playing code, adaptor, story binary, or dependency lock was generated.

## Direct local evidence

| Check | Result |
|---|---|
| `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/jericho/bin/python --version` | `Python 3.12.3` |
| Same interpreter, `-m pip show jericho` | Package not found |
| Same interpreter, `-m pip list --disable-pip-version-check` | Only `pip 26.2.2` listed |
| `command -v gcc make` | `/usr/bin/gcc`, `/usr/bin/make` |
| Header existence | `/usr/include/python3.12/Python.h` exists |
| Study `vendor/` file listing | No staged files |
| Applicable parent `AGENTS.md` files | None found at the explicitly checked ancestor paths |

The user identifies the platform as Linux x86_64. A broader file search encountered unrelated permission-denied directories, so it is not evidence that the entire machine has no usable game assets. The study directory contains no provisioned game corpus. Compiler/header presence alone does not verify a Jericho build.

## Web research attempts

`curl -IL --max-time 15` or `20` was attempted for the following URLs. Every request failed with curl exit 6, “Could not resolve host,” before any page content was retrieved:

- https://pypi.org/project/jericho/
- https://raw.githubusercontent.com/microsoft/jericho/master/jericho/jericho.py
- https://jericho-py.readthedocs.io/en/latest/
- https://www.ifarchive.org/indexes/if-archive/games/zcode/
- https://brasslantern.org/beginners/playif.html
- https://www.eristic.net/games/infocom/zork1.html

No browser/search connector is available in this session. These failures do not establish that the URLs are unavailable publicly, that the package is incompatible, or that any game's license is restrictive. They establish that live web verification could not be performed here. All external citations in the deliverables are identified as reference/verification targets, not newly read evidence. No source quotations or purported current release/license findings were invented.

## Installation constraint

The evaluator venv is outside the session's writable roots, which are the study directory and `/tmp`. The tool approval policy is `never`; no escalation or installation into that venv was attempted. Network DNS failure and the empty wheelhouse independently prevent an offline install in this session. `interfaces.md` gives the intended exact-interpreter offline installation command and staging procedure. No credentials are needed.

## Unresolved gates, in order

1. Retrieve upstream Jericho source/metadata and the cited manuals/community material in a network-enabled provisioning environment. Inspect actual release requirements and API signatures; pin a version/commit.
2. Establish an explicitly redistributable, positive-score story corpus matching supported exact builds. Preserve license notices and file hashes. No title is approved yet.
3. Build/resolve an offline wheelhouse for the evaluator's Python 3.12/Linux x86_64, including all required native/NLP assets and build requirements.
4. Install into the specified venv in a phase with write access. Import and exercise real reset/step/score/done behavior; no compatibility success is claimed now.
5. Run the deterministic, scoring, snapshot, input, and offline acceptance checks in `interfaces.md`; freeze the manifest, observation policy, and step budget.

The requested four documents can be used as a design handoff, but cannot honestly be called a fully verified runnable setup until these gates pass.
