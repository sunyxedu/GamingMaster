"""Model backends + profile resolution.

``make_system2("astra")`` looks up ``[system2.astra]`` in ``models.toml`` and
builds the backend it names (codex / claude / scripted). The backend classes
know nothing about specific models; the profile does.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .base import (
    ModelCall,
    ModelTimeout,
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
    "ModelCall", "ModelTimeout", "ModelUnavailable", "System1Answer", "System1Model", "System2Model",
    "System2Result", "TraceSink", "Choice", "JevClient", "Noul", "Score",
    "HeuristicSystem1", "ScriptedSystem2", "make_system1", "make_system2",
    "load_profiles", "profile_names",
]

DEFAULT_PROFILES = Path(__file__).resolve().parent.parent / "models.toml"


def load_profiles(path: Optional[Path | str] = None) -> dict[str, dict[str, dict[str, Any]]]:
    p = Path(path or os.environ.get("METAHARNESS_MODELS") or DEFAULT_PROFILES)
    try:
        import tomllib
    except ImportError:  # python < 3.11
        import tomli as tomllib  # type: ignore[no-redef]
    with open(p, "rb") as fh:
        data = tomllib.load(fh)
    data.setdefault("system2", {})
    data.setdefault("system1", {})
    return data


def profile_names(kind: str) -> list[str]:
    return sorted(load_profiles()[kind])


def _profile(kind: str, name: str) -> dict[str, Any]:
    profiles = load_profiles()[kind]
    if name not in profiles:
        raise ValueError(f"unknown {kind} profile {name!r}; known: {sorted(profiles)} (see {DEFAULT_PROFILES})")
    return dict(profiles[name])


def make_system2(name: str, **overrides: Any) -> System2Model:
    """Build the System-2 model named by a ``[system2.<name>]`` profile.
    ``overrides`` (e.g. ``reasoning_effort=``) win over the profile."""
    prof = _profile("system2", name)
    backend = prof.pop("backend")
    prof.update({k: v for k, v in overrides.items() if v is not None})
    if backend == "codex":
        from .codex_cli import CodexCLI
        if not prof.get("model"):
            raise ValueError(f"system2 profile {name!r} (codex) needs a model id in {DEFAULT_PROFILES}")
        return CodexCLI(model=prof["model"], reasoning_effort=prof.get("reasoning_effort"))
    if backend == "claude":
        from .claude_cli import ClaudeCLI
        return ClaudeCLI(model=prof.get("model") or None, reasoning_effort=prof.get("reasoning_effort"))
    if backend == "scripted":
        return ScriptedSystem2(**{k: v for k, v in prof.items() if k in ("replies", "latency_s")})
    raise ValueError(f"unknown system2 backend {backend!r} in profile {name!r}")


def make_system1(name: str, **overrides: Any) -> System1Model:
    prof = _profile("system1", name)
    backend = prof.pop("backend")
    prof.update({k: v for k, v in overrides.items() if v is not None})
    if backend == "typesafe":
        if not prof.get("model"):
            raise ValueError(f"system1 profile {name!r} (typesafe) needs a model id in {DEFAULT_PROFILES}")
        return JevClient(model=prof["model"])
    if backend == "heuristic":
        return HeuristicSystem1(fn=prof.get("fn"), latency_s=prof.get("latency_s", 0.3))
    raise ValueError(f"unknown system1 backend {backend!r} in profile {name!r}")
