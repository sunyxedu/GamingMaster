# Build the adaptor for **Atari 2600 Asteroids**

Working directory: `/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-asteroids/adaptor`. Study documents from the previous phase are in `/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-asteroids/study`
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
  Keep the ROM, one-player mode 0, difficulty 0, frame budget and sticky-action policy fixed for the main score suite. Local fixed-input checks found identical RAM trajectories for reset seeds 0, 1, 2, 3, 42 and 123, so seed alone is not meaningful randomization here. For held-out robustness, define a versioned scenario catalogue before evaluation and map seed to explicit mode/difficulty/speed/defense variants, or to a measured bounded control prelude whose frames, rewards and life losses remain in the reported budget; report variants separately because they change the task. Do not randomize RAM, silently discard prelude outcomes, or memorize seed schedules. Add fast modes, defense groups and difficulty 3 only after mode mapping and enemy-craft observation coverage are validated.
  If only some seeds are valid (a fixed set of levels / maps), declare them in `SEEDS`. Every seed must be a
  distinct episode (use the game's own seeded randomness), and train and held-out seeds must come from the same
  mix of scenarios: held-out seeds measure new instances of the game the search trained on, so a scenario
  family that appears only among held-out seeds measures transfer instead. Declare 8+ seeds per group when
  episodes are short.
* `score` must reflect playing well as described in study.json "scoring": Report sum of original unclipped ALE rewards within a fixed 18000-emulated-frame budget or true game over. Asteroids ALE corrects 100000-point display rollover. Report surviving frames/seconds, lives lost, displayed score, settings and termination/truncation reason separately; do not replace score with survival or reward shaping.
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
  `/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-asteroids/.venv/bin/python` (Python 3.12.3; `metaharness` is importable there). Install
  requirements into *that* environment: `uv pip install --python /home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-asteroids/.venv/bin/python ...` (a uv-made
  environment has no pip), or `/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-asteroids/.venv/bin/python -m pip install ...` where it has one.
* Test it yourself: `/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-asteroids/.venv/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-asteroids/adaptor', expected_tier=2).to_markdown())"`
  must print PASS (read its [!] warnings too). Then play a few random actions and check the observation and
  score make sense.
* Time: about 90 minutes. Get a minimal adaptor that validates first, then improve its fidelity: what is on disk is validated when the time is up.
