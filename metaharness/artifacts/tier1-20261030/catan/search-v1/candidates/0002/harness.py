"""Code-first Catan harness.

The engine already provides the authoritative legal action list.  This harness
uses that list for routine moves (especially setup and affordable builds) and
asks Astra only about choices whose arguments or race context matter.
"""
from __future__ import annotations

import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


PIP = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
TERRAIN = {"WOOD": 0, "BRICK": 1, "SHEEP": 2, "WHEAT": 3, "ORE": 4}


def _aid(a: Any) -> str:
    return str(a.get("id", "")) if isinstance(a, dict) else ""


def _typ(a: Any) -> str:
    return str(a.get("type", "")).lower() if isinstance(a, dict) else ""


def _param(a: Any) -> Any:
    return a.get("parameters") if isinstance(a, dict) else None


def _legal(obs: Any) -> list[dict[str, Any]]:
    xs = obs.get("legal_actions", []) if isinstance(obs, dict) else []
    return [x for x in xs if isinstance(x, dict) and _aid(x)]


def _vertex_score(obs: dict[str, Any], vertex: int) -> float:
    board = obs.get("board", {})
    hs = board.get("hexes", [])
    vs = board.get("vertices", [])
    try:
        v = next(x for x in vs if int(x.get("id")) == int(vertex))
    except Exception:
        return 0.0
    terrains = []
    score = 0.0
    for hi in v.get("adjacent_hexes", []):
        try:
            h = hs[int(hi)]
        except Exception:
            continue
        terrain = h.get("terrain")
        if terrain in TERRAIN and not h.get("robber", False):
            score += 1.8 * PIP.get(int(h.get("number") or 0), 0)
            terrains.append(terrain)
    score += 1.5 * len(set(terrains))
    if "WHEAT" in terrains:
        score += 1.8
    if "ORE" in terrains:
        score += 1.2
    if "WOOD" in terrains and "BRICK" in terrains:
        score += 1.0
    # A port is useful, but production dominates a speculative port.
    try:
        vport = v.get("port")
        if vport:
            score += 1.0
    except Exception:
        pass
    return score


def _vertex_from(a: dict[str, Any]) -> int | None:
    p = _param(a)
    if isinstance(p, int):
        return p
    if isinstance(p, str) and p.lstrip("-").isdigit():
        return int(p)
    if isinstance(p, (list, tuple)) and p and isinstance(p[0], int):
        return int(p[0])
    return None


def _vnum(a: dict[str, Any]) -> int:
    v = _vertex_from(a)
    return -1 if v is None else v


def _edge_endpoints(a: dict[str, Any]) -> tuple[int, int] | None:
    p = _param(a)
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        try:
            return int(p[0]), int(p[1])
        except Exception:
            return None
    if isinstance(p, str):
        q = re.findall(r"-?\d+", p)
        if len(q) >= 2:
            return int(q[0]), int(q[1])
    return None


def _compact(obs: Any) -> str:
    """Keep Astra's context below the latency cliff while retaining race data."""
    if not isinstance(obs, dict):
        return repr(obs)
    keep = {k: obs.get(k) for k in (
        "phase", "pending_decision", "turn_number", "current_player", "last_event",
        "self", "opponents", "awards", "legal_actions", "terminal", "termination",
    )}
    b = obs.get("board", {})
    if isinstance(b, dict):
        keep["board"] = {
            "hexes": b.get("hexes", []), "vertices": b.get("vertices", []),
            "edges": b.get("edges", []),
        }
    return json.dumps(keep, separators=(",", ":"), default=str)[:14000]


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.session = None
        self.plan = "Build productive settlements and cities; protect an immediate win."
        self.last_obs = None

    def _model_action(self, game, obs, legal: list[dict[str, Any]]) -> str | None:
        names = ", ".join(f"{_aid(a)}={_typ(a)}({_param(a)})" for a in legal)
        prompt = f"""You are the tactical planner for a Catanatron game. The legal IDs below are authoritative and expire after this step.
Choose exactly one ID from the list. Do not invent coordinates or actions. Prefer an immediate legal win; otherwise advance the shortest resource/build path, deny an opponent who can win next, and avoid ending a turn if an affordable useful action exists. For robber choose the best block of the strongest visible opponent. For development cards/trades, select the option that completes a build or award race.
Persistent plan: {self.plan}
Legal IDs: {names}
State: {_compact(obs)}
Reply with just the ID (for example a7) and optionally one short sentence."""
        try:
            r = self.ctx.think(prompt, session_id=self.session, timeout_s=120.0)
            self.session = getattr(r, "session_id", self.session)
            text = str(getattr(r, "text", ""))
            ids = {_aid(a) for a in legal}
            found = re.findall(r"\ba\d+\b", text, flags=re.I)
            for x in reversed(found):
                if x.lower() in ids:
                    tail = re.sub(r".*?\b" + re.escape(x) + r"\b", "", text, count=1, flags=re.I | re.S).strip()
                    if tail:
                        self.plan = tail[:360]
                    return x.lower()
            # A malformed answer is harmless: fall back to the first legal ID.
        except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError):
            return None
        except Exception:
            return None
        return None

    def _routine(self, obs: dict[str, Any], legal: list[dict[str, Any]]) -> str | None:
        types = {_typ(a) for a in legal}
        # Singleton/forced decisions must never consume a model call.
        if len(legal) == 1:
            return _aid(legal[0])
        # Snake setup: maximize pip-weighted production and resource diversity.
        initial = "initial" in str(obs.get("phase", "")).lower() or "initial" in str(obs.get("pending_decision", "")).lower()
        if initial and "build_settlement" in types:
            xs = [(_vertex_score(obs, _vnum(a)), _aid(a))
                  for a in legal if _typ(a) == "build_settlement"]
            if xs:
                return max(xs)[1]
        if initial and "build_road" in types:
            choices = []
            for a in legal:
                if _typ(a) != "build_road":
                    continue
                ep = _edge_endpoints(a)
                val = max((_vertex_score(obs, x) for x in (ep or ())), default=0.0)
                choices.append((val, _aid(a)))
            if choices:
                return max(choices)[1]
        # Exact, affordable actions are already filtered by the adaptor. Build VP
        # first, then a settlement, then a road; this avoids Astra wasting calls.
        for t in ("build_city", "build_settlement", "build_road"):
            xs = [a for a in legal if _typ(a) == t]
            if xs:
                if t == "build_settlement":
                    return max(xs, key=lambda a: _vertex_score(obs, _vnum(a)))["id"]
                return _aid(xs[0])
        # Rolling and ending are deterministic; development/trade/robber need context.
        if "roll" in types:
            return next(_aid(a) for a in legal if _typ(a) == "roll")
        if "discard" in types:
            return _aid(legal[0])
        return None

    def run_episode(self, game) -> None:
        if getattr(self.ctx, "system2", None) is None:
            return
        obs = game.observe()
        while not self.ctx.done and not (isinstance(obs, dict) and obs.get("terminal")):
            legal = _legal(obs)
            if not legal:
                return
            action = self._routine(obs, legal)
            if action is None:
                action = self._model_action(game, obs, legal)
            if action is None or action not in {_aid(a) for a in legal}:
                action = _aid(legal[0])
            try:
                obs = self.ctx.act(action)
            except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError):
                return
            except Exception:
                # A stale/invalid action should not kill an episode; the adaptor
                # reports it in the next observation and supplies fresh IDs.
                try:
                    obs = game.observe()
                except Exception:
                    return


def build(ctx):
    return Harness(ctx)
