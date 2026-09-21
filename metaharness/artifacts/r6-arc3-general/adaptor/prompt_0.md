# Build the adaptor for **ARC-AGI-3**

Working directory: `/Users/hemlix/Documents/GamingMaster/runs/r6-arc3/adaptor`. Study documents from the previous phase are in `/Users/hemlix/Documents/GamingMaster/runs/r6-arc3/study`
(read `interfaces.md` first; `study.json` summarises the chosen interface, credentials and scenarios).

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

``game.py`` may import anything installed plus ``metaharness.games.base``. The
object returned by ``make_game`` must implement ``TurnGame`` (``step``) or
``RealtimeGame`` (``tick`` + ``tick_s`` + ``ticks``) from ``games/base.py`` and
additionally ``name`` (str) and ``TIER`` may be declared at module level as
``TIER = 1 | 2``. Observations must be JSON-serialisable (dicts/lists/str/
numbers/bools); binary perception (screenshots) is exposed as file paths so a
harness can hand them to ``ctx.think(images=[...])``.

``validate_adaptor`` is the generic smoke test the orchestrator runs (and shows
the model when it fails): load, construct, reset two seeds, observe, describe,
action spec, apply the noop and every action once, done/score types, and
determinism of ``reset(seed)`` when the adaptor claims it (``DETERMINISTIC``).

`metaharness/games/base.py` (import `ActionSpec` from `metaharness.games.base`):
```python
"""Game interfaces the harness talks to.

The harness never sees engine internals — only what a real game client would
expose: an observation, an action vocabulary, a score, and (tier 2) a clock
that keeps running whether or not the agent has decided.
"""
from __future__ import annotations

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
            lines.append(f"(no decision in time => '{self.noop}')")
        if self.notes:
            lines.append(self.notes)
        return "\n".join(lines)


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
  Use the immutable versioned eight-game manifest in interfaces.md; nonnegative seed modulo 8 selects the game, with engine seed fixed at 0. Suggested development residues 0-3 and held-out residues 4-7. Every seed sharing a game prefix belongs to the same split; seed 8 is another CD82 episode, not an unseen scenario. Two seed-0 initializations and one seed-1 initialization yielded identical initial frames for every cached game, so RNG seed changes alone are not demonstrated novelty. Hold out entire game families and clear game-specific memory between evaluator episodes while transferring learned mechanics between levels within an episode.
* `score` must reflect playing well as described in study.json "scoring": Primary episode score is max observed levels_completed since evaluator reset(seed), bounded by win_levels, so retries cannot farm duplicate completions. Incremental reward is the increase of this high-water mark. Track current progress, numbered actions, explicit retries/undo, elapsed time, WIN, completion fraction, and truncation separately. The cached games have 6,10,7,8,6,7,9,7 levels (60 total). Optional SDK RHAE is distinct from this user-requested level score; see https://docs.arcprize.org/methodology.
* `describe()` tells a model everything it may know a priori: rules, objective, action semantics,
  observation fields. `action_spec` lists every legal action name with a one-line meaning and a `noop`.
  Actions with parameters use a compact string grammar documented in `describe()` (e.g. `"move 3 4"`).
* Observations are JSON-serialisable and compact (< 200 kB): structured state where the interface
  gives it; otherwise text; screenshots as file paths under this directory.
* If the game needs a running client/server/account, script the setup (`setup.sh` or Python) and run it.
  If something needs the user (account login, purchase, a GUI step), write `NEEDS.md` listing exactly
  what and stop.
* Write `ADAPTOR.md`, `requirements.txt` (if any), `env.example`. The evaluator's interpreter is
  `/Users/hemlix/Documents/GamingMaster/.venv/bin/python` (Python 3.14.5; `metaharness` is importable there). Install
  requirements into *that* environment: `/Users/hemlix/Documents/GamingMaster/.venv/bin/python -m pip install ...`.
* Test it yourself: `/Users/hemlix/Documents/GamingMaster/.venv/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('/Users/hemlix/Documents/GamingMaster/runs/r6-arc3/adaptor').to_markdown())"`
  must print PASS. Then play a few random actions and check the observation and score make sense.
