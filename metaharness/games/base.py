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
