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
    """Last action name mentioned in ``text`` (longest names first; names may contain spaces); keeps
    numeric arguments that follow it (e.g. 'ACTION6 10 12')."""
    if not text:
        return default
    best: tuple[int, int, str, str] | None = None        # (end, start, canonical name, args)
    for n in sorted(names, key=len, reverse=True):
        for m in re.finditer(r"(?<![A-Za-z0-9_])" + re.escape(n) + r"((?:\s+-?\d+){0,2})", text, flags=re.I):
            key = (m.end(), -m.start())
            if best is None or key > (best[0], -best[1]):
                best = (m.end(), m.start(), n, m.group(1))
    if best is None:
        return default
    return " ".join([best[2]] + best[3].split())
