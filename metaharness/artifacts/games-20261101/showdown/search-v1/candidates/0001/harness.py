"""Robust Astra harness for Gen 9 Random Battle.

The model is used for the strategic choice, while a small deterministic policy
keeps the protocol moving when a response is malformed, unavailable, or too
slow.  All actions are selected from the current request rather than from a
stale action list.
"""
from __future__ import annotations

import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


# A compact type chart is enough to rank candidate attacks.  The simulator is
# still authoritative; this is only a fallback when Astra gives no usable line.
_WEAK = {
    "Normal": {"Fighting"}, "Fire": {"Water", "Ground", "Rock"},
    "Water": {"Electric", "Grass"}, "Electric": {"Ground"},
    "Grass": {"Fire", "Ice", "Poison", "Flying", "Bug"},
    "Ice": {"Fire", "Fighting", "Rock", "Steel"},
    "Fighting": {"Flying", "Psychic", "Fairy"},
    "Poison": {"Ground", "Psychic"}, "Ground": {"Water", "Grass", "Ice"},
    "Flying": {"Electric", "Ice", "Rock"}, "Psychic": {"Bug", "Ghost", "Dark"},
    "Bug": {"Fire", "Flying", "Rock"}, "Rock": {"Water", "Grass", "Fighting", "Ground", "Steel"},
    "Ghost": {"Ghost", "Dark"}, "Dragon": {"Ice", "Dragon", "Fairy"},
    "Dark": {"Fighting", "Bug", "Fairy"}, "Steel": {"Fire", "Fighting", "Ground"},
    "Fairy": {"Poison", "Steel"},
}
_IMMUNE = {("Normal", "Ghost"), ("Fighting", "Ghost"), ("Electric", "Ground"),
           ("Poison", "Steel"), ("Ground", "Flying"), ("Psychic", "Dark"),
           ("Ghost", "Normal"), ("Dragon", "Fairy")}


def _species(details: Any) -> str:
    if not isinstance(details, str):
        return ""
    return details.split(",", 1)[0].strip()


def _types_for(name: str, dex: dict[str, Any]) -> list[str]:
    x = dex.get(name, {}) if isinstance(dex, dict) else {}
    ts = x.get("types", []) if isinstance(x, dict) else []
    return [str(t) for t in ts] if isinstance(ts, list) else []


def _effect(move_type: str, target_types: list[str]) -> float:
    if not move_type or not target_types:
        return 1.0
    value = 1.0
    for target in target_types:
        if (move_type, target) in _IMMUNE:
            return 0.0
        if target in _WEAK and move_type in _WEAK[target]:
            value *= 2.0
        if move_type in _WEAK and target in _WEAK[move_type]:
            value *= 0.5
    return value


def _canonical_action(text: Any, legal: list[str]) -> str | None:
    """Find an action token anywhere in arbitrary model output.

    Matching against *legal* prevents stale Tera/switch choices.  Long forms
    are checked first, then a permissive regex handles punctuation and markdown.
    """
    if not isinstance(text, str):
        return None
    low = text.lower()
    found: list[tuple[int, int, str]] = []
    for action in legal:
        m = re.search(r"(?<![a-z0-9])" + re.escape(action.lower()) + r"(?![a-z0-9])", low)
        if m:
            # Earliest mention wins; at one position prefer the more specific
            # Tera form over its bare move prefix.
            found.append((m.start(), -len(action), action))
    if found:
        found.sort()
        return found[0][2]
    patterns = [
        r"\b(move\s+[1-4](?:\s+terastallize)?)\b",
        r"\b(switch\s+[1-6])\b",
    ]
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            token = re.sub(r"\s+", " ", m.group(1).strip())
            for action in legal:
                if action.lower() == token:
                    return action
    return None


def _fallback(obs: dict[str, Any]) -> str | None:
    legal = [str(x) for x in (obs.get("legal_actions") or []) if isinstance(x, str)]
    if not legal:
        return None
    req = obs.get("request") or {}
    side = ((req.get("side") or {}).get("pokemon") or []) if isinstance(req, dict) else []
    active_req = (req.get("active") or []) if isinstance(req, dict) else []
    # Forced replacement/revival: select the healthiest legal teammate.
    if not any(a.startswith("move ") for a in legal):
        best = None
        best_ratio = -1.0
        for a in legal:
            m = re.fullmatch(r"switch\s+([1-6])", a)
            if not m:
                continue
            i = int(m.group(1)) - 1
            p = side[i] if i < len(side) and isinstance(side[i], dict) else {}
            cond = str(p.get("condition", "0/1"))
            try:
                hp, mx = cond.split("/", 1)
                ratio = float(hp) / max(1.0, float(mx))
            except Exception:
                ratio = 0.0
            if ratio > best_ratio:
                best_ratio, best = ratio, a
        return best or legal[0]

    moves = (active_req[0].get("moves") if active_req and isinstance(active_req[0], dict) else []) or []
    dex = obs.get("move_dex") or {}
    own = next((p for p in side if isinstance(p, dict) and p.get("active")), {})
    own_types = _types_for(_species(own.get("details")), obs.get("species_dex") or {})
    foes = obs.get("opponent_revealed") or []
    foe = next((p for p in foes if isinstance(p, dict) and p.get("ident") == (obs.get("active") or {}).get("p2")), None)
    foe_types = _types_for(_species((foe or {}).get("details")), obs.get("species_dex") or {})
    scored: list[tuple[float, str]] = []
    for action in legal:
        m = re.fullmatch(r"move\s+([1-4])(\s+terastallize)?", action)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        mv = moves[idx] if idx < len(moves) and isinstance(moves[idx], dict) else {}
        ident = str(mv.get("id", "")).lower()
        info = dex.get(ident, {}) if isinstance(dex, dict) else {}
        bp = float(info.get("base_power") or 0)
        category = str(info.get("category", ""))
        typ = str(info.get("type", ""))
        acc = info.get("accuracy", 100)
        try:
            accv = 1.0 if acc is True else float(acc) / 100.0
        except Exception:
            accv = 1.0
        if category.lower() == "status" or bp <= 0:
            # Recovery is valuable at low HP; hazards/setup are modestly useful.
            desc = str(info.get("description", "")).lower()
            score = 24.0 if any(k in desc for k in ("recover", "heals", "restores")) else 8.0
            if "hazard" in desc or "raises" in desc:
                score += 4.0
        else:
            stab = 1.5 if typ in own_types else 1.0
            score = bp * accv * _effect(typ, foe_types) * stab
            if "terastallize" in action:
                # Preserve Tera unless it creates an obvious high-power line.
                score *= 0.86
        scored.append((score, action))
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
    return legal[0]


def _render(obs: dict[str, Any]) -> str:
    """Expose the useful private/public ledger while keeping prompts bounded."""
    req = obs.get("request") or {}
    side = ((req.get("side") or {}).get("pokemon") or []) if isinstance(req, dict) else []
    # The complete own roster and active move list are strategically important.
    compact = {
        "turn": obs.get("turn"),
        "legal_actions": obs.get("legal_actions", []),
        "request_active": req.get("active", []) if isinstance(req, dict) else [],
        "own_team": side,
        "opponent_revealed": obs.get("opponent_revealed", []),
        "active": obs.get("active", {}),
        "public_summary": obs.get("public_summary", {}),
        "recent_events": (obs.get("recent_events") or [])[-45:],
        "last_error": obs.get("last_error"),
        "move_dex": obs.get("move_dex", {}),
        "species_dex": obs.get("species_dex", {}),
    }
    raw = json.dumps(compact, ensure_ascii=False, separators=(",", ":"), default=str)
    return raw if len(raw) <= 24000 else raw[:24000] + "…"


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        ctx = self.ctx
        # A missing System-2 backend should degrade to the scripted policy,
        # rather than abandoning a valid turn-game episode.
        model_ok = getattr(ctx, "system2", None) is not None
        try:
            obs = game.observe()
        except Exception:
            obs = None
        session = None
        calls = 0
        while not getattr(ctx, "done", False) and not getattr(game, "done", False):
            if not isinstance(obs, dict):
                try:
                    obs = game.observe()
                except Exception:
                    break
            legal = [str(x) for x in (obs.get("legal_actions") or []) if isinstance(x, str)]
            if not legal:
                break
            action = None
            # A fresh strategic call is made while budget and deadline permit.
            if model_ok and calls < 36:
                try:
                    remaining = ctx.budget.remaining()
                    if int(remaining.get("system2_calls", 0)) > 0 and float(remaining.get("wall_s", 0)) > 100:
                        intro = ("You play Pokémon Showdown Gen 9 Random Battle as p1. "
                                 "Choose exactly one legal action from the supplied list. "
                                 "Use the protocol ledger, respect forced switches, speed, typing, "
                                 "revealed sets and the one-use Tera. Prefer a guaranteed KO or safe "
                                 "progress; preserve a needed counter. Reply with an action token; "
                                 "a brief reason may follow.\n\n")
                        prompt = intro + "Current state (JSON):\n" + _render(obs)
                        res = ctx.think(prompt, session_id=session, timeout_s=100.0)
                        calls += 1
                        session = getattr(res, "session_id", None) or session
                        action = _canonical_action(getattr(res, "text", ""), legal)
                except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError):
                    model_ok = False
                except Exception:
                    # Garbage stand-in objects and backend quirks must not abort an episode.
                    model_ok = False
            if action is None:
                action = _fallback(obs)
            if action is None:
                break
            try:
                nxt = ctx.act(action)
                obs = nxt if isinstance(nxt, dict) else game.observe()
            except Exception:
                # Hidden trapping can invalidate an apparently legal switch.  Refresh the
                # request and continue with a locally selected legal command.
                try:
                    obs = game.observe()
                except Exception:
                    break


def build(ctx):
    return Harness(ctx)
