# Build the adaptor for **Atari 2600 Frogger**

Working directory: `/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-frogger/adaptor`. Study documents from the previous phase are in `/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-frogger/study`
(read `interfaces.md` first; `study.json` summarises the chosen interface, credentials and scenarios).
User notes (platform, account, constraints — may be empty): Offline headless Linux server (no display, no GPU, no accounts). The evaluator's Python is the venv printed in the prompt; pip works there. Prefer a local emulator interface that exposes symbolic (object-level) game state rather than screenshots.

## Contract
Generated game adaptors — the contract the ``adapt`` phase must satisfy.

An adaptor is a directory (written by Astra, never by hand in this repo):

    <adaptor>/
      game.py            defines make_game(**kwargs) -> TurnGame | RealtimeGame
      ADAPTOR.md         how it connects to the real game, what it observes, how
                         seeds map to scenarios, what score means, its limits
      requirements.txt   (optional) extra pip packages the adaptor needs
      env.example        (optional) environment variables it reads (credentials,
                         server address...). Real values live in the user's .env.
                         An uncommented ``NAME=`` line is a variable the user MUST
                         set (the orchestrator stops and asks for it); keep optional
                         ones commented (``# NAME=default``).
      runtime/           (optional) everything the adaptor writes while it runs: screenshots,
                         caches, logs, server / world directories, generated configs

``game.py`` may import anything installed, ``metaharness.games.base`` and sibling modules of
the adaptor directory. The object returned by ``make_game`` must implement ``TurnGame``
(``step``) or ``RealtimeGame`` (``tick`` + ``tick_s`` + ``ticks``) from ``games/base.py`` and
additionally ``name`` (str). Optional module-level declarations:

    TIER = 1 | 2           must agree with the kind: 2 <=> the game has tick()
    DETERMINISTIC = True   reset(seed) on a fresh instance reproduces the first observation
    SEEDS = {"train": [...], "heldout": [...]}   non-empty, disjoint lists of ints: the
                           scenarios the search trains on and is judged on (declare them when
                           only some seeds are valid, e.g. a fixed set of levels or maps)
    EVAL_KWARGS = {...}    JSON kwargs the evaluator passes to make_game

Optional methods: ``score_breakdown() -> dict[str, float]`` (numeric components of the current
score, e.g. progress vs survival; recorded per episode so the harness designer sees where points
came from) and ``close()``: called on every instance after its episode (and by validation); it
must release every process, server and connection the instance started, and be idempotent.

How the evaluator drives a game: a fresh instance per episode — make_game, reset(seed), play,
read the score, close(). It serialises its own calls into the object (tick / step / observe /
score / done from different threads never overlap); only the adaptor's own background threads
need locking. An exception from tick() / step() ends the episode as a game error: ignore a
malformed or illegal action, or report it in the next observation. ``info()`` is evaluator-only
(never shown to the harness). For an episode spoiled by the infrastructure (server crash, lost
connection — not bad play) report ``info()["valid_trial"] = False`` with
``info()["invalid_reason"]``: the evaluator retries that seed once on a fresh game and otherwise
records an error. Observations must be strict JSON (dicts/lists/str/finite numbers/bools,
< 200 kB); binary perception (screenshots) is exposed as file paths so a harness can hand them
to ``ctx.think(images=[...])``.

Runtime outputs (screenshots, caches, logs, recordings, server / world directories, anything
generated while playing) belong under ``runtime/`` or another directory the source hash skips
(node_modules, .cache, logs, recordings, worlds). The code files elsewhere in the directory
(``*.py *.pyi *.js *.mjs *.cjs *.ts *.sh *.bash``, requirements.txt, package.json) are pinned by
the search manifest (``source_sha256``): one written or changed while the adaptor runs makes a
resumed search refuse to continue.

``validate_adaptor`` is the generic smoke test the orchestrator runs (and shows the model when it
fails): load, construct, the tier the study chose, reset two seeds (the first train and held-out
seed when SEEDS is declared), observe, describe, action spec, apply the noop and every action
once, done/score/observation types after them, tick() time against tick_s, one malformed action,
determinism of reset(seed) on a fresh instance (``DETERMINISTIC``), close(), and that running
changed no code file. Warnings (``[!]``) do not fail it.

`metaharness/games/base.py` (import `ActionSpec` from `metaharness.games.base`):
```python
"""Game interfaces the harness talks to.

The harness never sees engine internals — only what a real game client would
expose: an observation, an action vocabulary, a score, and (tier 2) a clock
that keeps running whether or not the agent has decided.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass
class ActionSpec:
    """What actions exist and what they mean — given to the harness verbatim."""
    names: Sequence[str]
    descriptions: dict[str, str] = field(default_factory=dict)
    noop: str | None = None          # action applied when the agent misses a tick
    notes: str = ""                  # free-text semantics (e.g. "reversing = death")

    def describe(self) -> str:
        lines = [f"- {n}: {self.descriptions.get(n, '')}".rstrip() for n in self.names]
        if self.noop:
            lines.append(f"(real-time games: ticks apply '{self.noop}' until the first action is posted; "
                         "afterwards a tick without a fresh action repeats the last posted action)")
        if self.notes:
            lines.append(self.notes)
        return "\n".join(lines)


def close_game(game: Any, timeout_s: float = 120.0) -> str | None:
    """Release a game instance the evaluator no longer needs (its optional ``close()``).

    Generated adaptors may own servers, emulators or connections; the evaluator calls this once
    per instance after reading the final score. Runs in a helper thread so a hanging ``close()``
    cannot stall the run beyond ``timeout_s``. Returns an error description, or None."""
    fn = getattr(game, "close", None)
    if not callable(fn):
        return None
    err: list[str] = []

    def _close() -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — reported, never fatal
            err.append(f"{type(e).__name__}: {e}"[:500])

    th = threading.Thread(target=_close, name="game-close", daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        return f"close() did not return within {timeout_s:.0f}s"
    return err[0] if err else None


class TurnGame(Protocol):
    """Tier 1: the world waits for the agent. Unknown rules are part of the task."""
    name: str
    action_spec: ActionSpec

    def reset(self, seed: int) -> Any: ...
    def observe(self) -> Any: ...
    def step(self, action: Any) -> Any: ...
    @property
    def done(self) -> bool: ...
    @property
    def score(self) -> float: ...
    def info(self) -> dict[str, Any]: ...
    def describe(self) -> str:
        """What the harness is told about the game *a priori* (may hide rules)."""
        ...


class RealtimeGame(Protocol):
    """Tier 2: the world ticks every ``tick_s`` seconds. The controller sets
    the *current* action through a mailbox; each tick applies the latest one
    (or ``action_spec.noop`` if none). Late decisions act on a later state."""
    name: str
    action_spec: ActionSpec
    tick_s: float

    def reset(self, seed: int) -> Any: ...
    def observe(self) -> Any: ...
    def tick(self, action: Any) -> None: ...
    @property
    def done(self) -> bool: ...
    @property
    def score(self) -> float: ...
    @property
    def ticks(self) -> int: ...
    def info(self) -> dict[str, Any]: ...
    def describe(self) -> str: ...

```

## Requirements
* Tier 2: real-time — implement RealtimeGame: tick(action) applies the *current* action, tick_s = the real cadence, ticks counts applied ticks; the runner calls tick on its own clock.
* `reset(seed)` must start a fresh episode and vary it by seed as described in study.json "scenarios":
  Call reset(seed) with held-out integer seeds and keep mode=0, difficulty=0 for the primary benchmark; vary seeds so traffic/platform phases and initial object arrangements are unseen at evaluation. Add secondary scenario sets for ALE mode=1 (harder single-player variation), mode=2/Speedy Frogger, and difficulty=1 (floating objects wrap rather than killing the frog) without mixing their scores. Run enough frames to include multiple frogs and the five-home level transition, and record seeds, mode, difficulty, ROM hash, frameskip, and sticky-action probability with every episode.
  If only some seeds are valid (a fixed set of levels / maps), declare them in `SEEDS`. Every seed must be a
  distinct episode (use the game's own seeded randomness), and train and held-out seeds must come from the same
  mix of scenarios: held-out seeds measure new instances of the game the search trained on, so a scenario
  family that appears only among held-out seeds measures transfer instead. Declare 8+ seeds per group when
  episodes are short.
* `score` must reflect playing well as described in study.json "scoring": Report cumulative ALE reward and the decoded on-screen score as the primary score because ALE computes Frogger reward as the change in the BCD/decimal score at RAM 0xCC/0xCE. Also report score_delta, frogs_home, deaths, reserve_frogs, time_bonus, level, and terminal_reason: the manual awards 1 for a forward jump, 5 for a home, 100 for five homes, 20 for a lady frog, 20 for a fly, and 2 per second remaining. Sources: https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/Frogger.cpp and https://atariage.com/manual_html_page.php?SoftwareLabelID=194
* `describe()` tells a model everything it may know a priori: rules, objective, action semantics,
  observation fields. `action_spec` lists every legal action name with a one-line meaning and a `noop`.
  Actions with parameters use a compact string grammar documented in `describe()` (e.g. `"move 3 4"`).
* Observations are strict JSON (no NaN / inf, sets or objects) and compact (< 200 kB): structured state
  where the interface gives it; otherwise text; screenshots as file paths under `runtime/` in this directory.
* Everything the adaptor needs lives in this directory (downloads, builds, caches under `runtime/`), never
  in /tmp or elsewhere: the evaluator loads only this directory. Prefer the interface the study chose and
  its maintained packages over building your own bridge to the game.
* If the game needs a running client/server/account, script the setup (`setup.sh` or Python) and run it.
  Every process, server or connection an instance starts is released by its `close()`.
  If something needs the user (account login, purchase, a GUI step), write `NEEDS.md` listing exactly
  what and stop (the orchestrator removes it once the adaptor validates).
* Write `ADAPTOR.md`, `requirements.txt` (if any), `env.example`. The evaluator's interpreter is
  `/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-frogger/.venv/bin/python` (Python 3.12.3; `metaharness` is importable there). Install
  requirements into *that* environment: `uv pip install --python /home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-frogger/.venv/bin/python ...` (a uv-made
  environment has no pip), or `/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-frogger/.venv/bin/python -m pip install ...` where it has one.
* Test it yourself: `/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-frogger/.venv/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-frogger/adaptor', expected_tier=2).to_markdown())"`
  must print PASS (read its [!] warnings too). Then play a few random actions and check the observation and
  score make sense.
* Time: about 90 minutes. Get a minimal adaptor that validates first, then improve its fidelity: what is on disk is validated when the time is up.
## Validation feedback from the orchestrator
# Adaptor validation: FAIL

- [x] load game.py and make_game() — name=Atari 2600 Frogger tier=tier2 realtime=True
- [x] TIER agrees with the game's kind (2 <=> tick()) — TIER=2; the game is real-time (has tick)
- [x] tier 2, as the study decided — the game is real-time (has tick), tier2
- [x] SEEDS: non-empty, disjoint train / held-out lists of ints — 8 train (101, 103, 107, 109, 113, …), 8 held-out (139, 149, 151, 157, 163, …)
- [x] action_spec is an ActionSpec with >=1 action
- [x] realtime: tick_s > 0 and noop action set — tick_s=0.016666666666666666 noop=NOOP
- [x] describe() returns a non-trivial string — 1319 chars
- [x] seed 101: observe() JSON-serialisable — 1495 bytes
- [x] seed 101: done is bool and score is a number after reset — done=False score=0.0
- [x] seed 101: applied 6 actions without exception
- [x] seed 101: done is bool and score is a finite number after the actions — done=False score=0.0
- [x] seed 101: observations after the actions are strict JSON
- [x] seed 101: ticks counter advanced — ticks=6 applied=6
- [x] seed 101: tick() keeps the declared pace — median tick() 0.2 ms, tick_s 17 ms
- [x] seed 101: info() is a dict
- [x] seed 139: observe() JSON-serialisable — 1496 bytes
- [x] seed 139: done is bool and score is a number after reset — done=False score=0.0
- [x] seed 139: applied 6 actions without exception
- [x] seed 139: done is bool and score is a finite number after the actions — done=False score=0.0
- [x] seed 139: observations after the actions are strict JSON
- [x] seed 139: ticks counter advanced — ticks=6 applied=6
- [x] seed 139: tick() keeps the declared pace — median tick() 0.2 ms, tick_s 17 ms
- [x] seed 139: info() is a dict
- [x] malformed action '__not_an_action__' does not raise
- [x] close() releases the instance (called after every episode)
- [x] DETERMINISTIC: reset(seed) on a fresh instance reproduces the first observation
- [ ] different seeds give different episodes (the same actions on two seeds diverge) — seeds 101 and 139: identical observations through 300 identical actions — held-out seeds would replay the training seeds and a fixed action sequence could be memorised; let the seed drive the game's own randomness (the platform's standard stochasticity, e.g. randomised starts or action noise, or a seeded scenario)
- [!] seeds 101 and 139 give identical first observations — fine for a fixed opening position if the seed still changes the episode (checked next)
