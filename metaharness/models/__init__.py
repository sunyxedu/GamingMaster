from .base import (
    ModelCall,
    ModelUnavailable,
    System1Answer,
    System1Model,
    System2Model,
    System2Result,
    TraceSink,
)
from .jev import Choice, JevClient, Noul, Score
from .mock import HeuristicSystem1, ScriptedSystem2

__all__ = [
    "ModelCall", "ModelUnavailable", "System1Answer", "System1Model", "System2Model",
    "System2Result", "TraceSink", "Choice", "JevClient", "Noul", "Score",
    "HeuristicSystem1", "ScriptedSystem2", "make_system1", "make_system2",
]


def make_system2(name: str, **kw) -> System2Model:
    """``astra`` (codex CLI, gpt-6-astra) | ``claude`` (claude CLI) | ``scripted``."""
    if name == "astra":
        from .astra import AstraCodex
        return AstraCodex(**kw)
    if name == "claude":
        from .claude_cli import ClaudeCLI
        return ClaudeCLI(**kw)
    if name == "scripted":
        return ScriptedSystem2(**kw)
    raise ValueError(f"unknown system2 backend {name!r}")


def make_system1(name: str, **kw) -> System1Model:
    """``jev`` (TypeSafe System-One API) | ``heuristic`` (offline mock)."""
    if name == "jev":
        return JevClient(**kw)
    if name == "heuristic":
        return HeuristicSystem1(**kw)
    raise ValueError(f"unknown system1 backend {name!r}")
