# Build the adaptor for **NetHack (NetHack Learning Environment)**

Working directory: `runs/nethack/adaptor`. Study documents from the previous phase are in `runs/nethack/study`
(read `interfaces.md` first; `study.json` summarises the chosen interface, credentials and scenarios).
User notes (platform, account, constraints — may be empty): Linux x86_64 server without GUI; everything must run offline on this machine with free software. The evaluator's Python is the venv at /home/hemlix/GamingMaster-mh-tier1/runs/venvs/nethack (install there). Tier 1: the agent plays turn by turn. pip nle (wheels first; a source build may need cmake/bison/flex); textual observation; score = in-game score or depth reached.

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
                           only some seeds are valid, e.g. a fixed set of levels or maps); optional
                           "dev" (the proposer's own trials) and "validation" (top-k re-evaluation)
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
* Tier 1: turn-based — implement TurnGame: step(action) blocks until the game has applied the action and returns the next observation.
* `reset(seed)` must start a fresh episode and vary it by seed as described in study.json "scenarios":
  For reset(seed), use held-out integer seeds and vary NetHack's randomized role/race/alignment, starting inventory, dungeon layouts, monster/item placements and branch choices when the pinned NLE build permits it. Include ordinary score episodes and staircase/challenge curriculum episodes, while never training on a fixed map or seed.
  Declare `SEEDS = {"train": [...], "heldout": [...], "dev": [...], "validation": [...]}` (disjoint; 1-2 dev
  seeds; 4-6 train / held-out / validation seeds when an episode is expensive, 8-12 each when an episode is
  cheap — a mean over few seeds is noisy and lets a harness overfit the training scenarios) whenever seeds
  select scenarios, and always when only some seeds are valid (a fixed set of levels / maps): the search
  trains on `train`, the proposer may try drafts on `dev`, finalists are re-evaluated on `validation`, the
  result is reported on `heldout`. Scenario groups must be drawn alike (same difficulty mix in every group).
* `score` must reflect playing well as described in study.json "scoring": Report the terminal NetHack score from the environment/info as the primary score because it combines gold, experience, dungeon progress and achievements; also report maximum dungeon depth and an ascension/success flag so survival and true completion are visible.
  It must keep headroom: pick opponents, difficulty and scenario length so that neither a trivial policy nor a
  competent scripted one reaches the maximum (a search cannot tell candidates apart on a saturated score).
* `describe()` tells a model everything it may know a priori: rules, objective, action semantics,
  observation fields. `action_spec` lists every legal action name with a one-line meaning and a `noop`.
  Actions with parameters use a compact string grammar documented in `describe()` (e.g. `"move 3 4"`).
* Observations are strict JSON (no NaN / inf, sets or objects) and compact (< 200 kB): structured state
  where the interface gives it; otherwise text; screenshots as file paths under `runtime/` in this directory.
* If the game needs a running client/server/account, script the setup (`setup.sh` or Python) and run it.
  Every process, server or connection an instance starts is released by its `close()`.
  If something needs the user (account login, purchase, a GUI step), write `NEEDS.md` listing exactly
  what and stop (the orchestrator removes it once the adaptor validates).
* Write `ADAPTOR.md`, `requirements.txt` (if any), `env.example`. The evaluator's interpreter is
  `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/nethack/bin/python` (Python 3.12.3; `metaharness` is importable there). Install
  requirements into *that* environment: `uv pip install --python /home/hemlix/GamingMaster-mh-tier1/runs/venvs/nethack/bin/python ...` (a uv-made
  environment has no pip), or `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/nethack/bin/python -m pip install ...` where it has one.
* Test it yourself: `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/nethack/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('runs/nethack/adaptor', expected_tier=1).to_markdown())"`
  must print PASS (read its [!] warnings too). Then play a few random actions and check the observation and
  score make sense.
