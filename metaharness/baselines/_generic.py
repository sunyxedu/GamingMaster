"""Game-agnostic helpers shared by the seed baselines. Baselines must not
reference any game's observation keys — the proposer learns from them."""
from __future__ import annotations

import json
import re
from typing import Any, Sequence


def obs_text(obs: Any, limit: int = 6000) -> str:
    """Compact, generic rendering of an observation for a text model."""
    if isinstance(obs, dict) and isinstance(obs.get("grid"), str):
        head = {k: v for k, v in obs.items() if k != "grid"}
        s = json.dumps(head, default=str) + "\n" + obs["grid"]
    elif isinstance(obs, dict) and isinstance(obs.get("view"), str):
        head = {k: v for k, v in obs.items() if k != "view"}
        s = json.dumps(head, default=str) + "\n" + obs["view"]
    else:
        s = json.dumps(obs, default=str, separators=(",", ":"))
    return s if len(s) <= limit else s[:limit] + " …(truncated)"


def parse_action(text: str, names: Sequence[str], default: str | None = None) -> str | None:
    """Last action name mentioned in ``text`` (longest names first); keeps
    numeric arguments that follow it (e.g. 'ACTION6 10 12')."""
    if not text:
        return default
    best = None
    for n in sorted(names, key=len, reverse=True):
        for m in re.finditer(re.escape(n) + r"(?:\s+-?\d+){0,2}", text, flags=re.I):
            if best is None or m.end() > best.end():
                best = m
    if best is None:
        return default
    tok = best.group(0).split()
    canon = next(n for n in names if n.lower() == tok[0].lower())
    return " ".join([canon] + tok[1:])
