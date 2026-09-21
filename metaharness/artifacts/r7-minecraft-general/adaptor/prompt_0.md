# Build the adaptor for **Minecraft**

Working directory: `/Users/hemlix/Documents/GamingMaster/runs/r7-minecraft/adaptor`. Study documents from the previous phase are in `/Users/hemlix/Documents/GamingMaster/runs/r7-minecraft/study`
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
* Tier 2: real-time — implement RealtimeGame: tick(action) applies the *current* action, tick_s = the real cadence, ticks counts applied ticks; the runner calls tick on its own clock.
* `reset(seed)` must start a fresh episode and vary it by seed as described in study.json "scenarios":
  Use unseen natural vanilla 1.21.1 world seeds with stable digest-derived world seed, safe-spawn selection and equal time-of-day strata 1000/9000. Prequalify starts by a fixed terrain/resource predicate independent of policy performance: dry safe surface, at least 12 reachable log blocks and 24 shallow ordinary stone blocks nearby within 48 blocks, without lava crossings or drops above two blocks; use a bounded deterministic candidate search near world spawn and record inadmissibility. Coal, iron, animals, wood species, hills and water vary naturally. Restore a pristine stopped per-seed baseline including a fresh player, not a played world or teleport-only reset. Hold out entire world seeds and repeat trials because terrain seeds do not make mob AI, drops and network timing bitwise deterministic. Keep Normal difficulty, initial clear weather, empty inventory, score and 300-second duration fixed for compared agents.
* `score` must reflect playing well as described in study.json "scoring": Custom absolute score, clamped to [0,100]. Resource points use per-episode maxima of simultaneously held server-confirmed personal inventory counts: min(unstripped Overworld logs,10) + 0.5*min(cobblestone,20) + min(combined coal+charcoal,5) + min(raw_iron,5), maximum 30. Latch first verified possession of crafting_table=5, wooden_pickaxe=5, stone_pickaxe=10, stone_axe=5, furnace=5, torch=5, one cooked-food item=5, iron_ingot=5 and shield=5, maximum 50. Cooked-food whitelist: cooked_beef, cooked_porkchop, cooked_mutton, cooked_chicken, cooked_rabbit, cooked_cod, cooked_salmon, baked_potato. Legitimate looting counts as equipment progress. Add 10*seconds_survived/300 and, only at an alive time-limit finish, 10*final_health/20; subtract 20 on terminal death. Exclude setup time, crafting previews/cursor/container slots and duplicate transfers; retain earned credit after consumption. Report components and milestone times as well as score; optional reward is score delta. Infrastructure failures are invalid trials. Full healthy idling earns 20, while resources and equipment provide the remaining 80.
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
* Test it yourself: `/Users/hemlix/Documents/GamingMaster/.venv/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('/Users/hemlix/Documents/GamingMaster/runs/r7-minecraft/adaptor').to_markdown())"`
  must print PASS. Then play a few random actions and check the observation and score make sense.
