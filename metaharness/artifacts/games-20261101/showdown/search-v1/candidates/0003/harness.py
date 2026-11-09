"""Checkpointed Astra policy for the Showdown turn game.

The model proposes a short line, but code owns legality and invalidates that line
whenever the protocol shows a meaningful event.  A small typed policy keeps the
battle moving through malformed replies and model failures.
"""
from __future__ import annotations

import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


_WEAK = {
    "Normal": {"Fighting"}, "Fire": {"Water", "Ground", "Rock"},
    "Water": {"Electric", "Grass"}, "Electric": {"Ground"},
    "Grass": {"Fire", "Ice", "Poison", "Flying", "Bug"},
    "Ice": {"Fire", "Fighting", "Rock", "Steel"},
    "Fighting": {"Flying", "Psychic", "Fairy"},
    "Poison": {"Ground", "Psychic"}, "Ground": {"Water", "Grass", "Ice"},
    "Flying": {"Electric", "Ice", "Rock"}, "Psychic": {"Bug", "Ghost", "Dark"},
    "Bug": {"Fire", "Flying", "Rock"},
    "Rock": {"Water", "Grass", "Fighting", "Ground", "Steel"},
    "Ghost": {"Ghost", "Dark"}, "Dragon": {"Ice", "Dragon", "Fairy"},
    "Dark": {"Fighting", "Bug", "Fairy"},
    "Steel": {"Fire", "Fighting", "Ground"},
    "Fairy": {"Poison", "Steel"},
}
_IMMUNE = {("Normal", "Ghost"), ("Fighting", "Ghost"),
           ("Electric", "Ground"), ("Poison", "Steel"),
           ("Ground", "Flying"), ("Psychic", "Dark"),
           ("Ghost", "Normal"), ("Dragon", "Fairy")}
_EVENT_WORDS = ("|faint|", "|switch|", "|damage|", "|heal|", "|status|",
                "|boost|", "|unboost|", "|weather|", "|fieldstart|",
                "|fieldend|", "|terastallize|", "|cant|")


def _species(details: Any) -> str:
    return str(details).split(",", 1)[0].strip() if details else ""


def _types(name: str, dex: Any) -> list[str]:
    item = dex.get(name, {}) if isinstance(dex, dict) else {}
    vals = item.get("types", []) if isinstance(item, dict) else []
    return [str(x) for x in vals] if isinstance(vals, list) else []


def _effect(attacking: str, targets: list[str]) -> float:
    value = 1.0
    for target in targets:
        if (attacking, target) in _IMMUNE:
            return 0.0
        if target in _WEAK and attacking in _WEAK[target]:
            value *= 0.5
        if attacking in _WEAK and target in _WEAK[attacking]:
            value *= 2.0
    return value


def _hp_ratio(p: Any) -> float:
    if not isinstance(p, dict):
        return 0.0
    raw = str(p.get("condition", "0/1"))
    try:
        a, b = raw.split("/", 1)
        return float(a) / max(1.0, float(re.match(r"[0-9.]+", b).group(0)))
    except Exception:
        return 0.0


def _fallback(obs: dict[str, Any]) -> str | None:
    """Choose a legal, conservative action without pretending hidden sets are known."""
    legal = [x for x in obs.get("legal_actions", []) if isinstance(x, str)]
    if not legal:
        return None
    req = obs.get("request") if isinstance(obs.get("request"), dict) else {}
    side = ((req.get("side") or {}).get("pokemon") or [])
    # Forced replacements and Revival Blessing: preserve the healthiest available slot.
    if not any(x.startswith("move ") for x in legal):
        choices = []
        for action in legal:
            m = re.fullmatch(r"switch ([1-6])", action)
            if m:
                i = int(m.group(1)) - 1
                choices.append((_hp_ratio(side[i]) if i < len(side) else 0.0, action))
        return max(choices, default=(0.0, legal[0]))[1]

    active = (req.get("active") or [])
    moves = active[0].get("moves", []) if active and isinstance(active[0], dict) else []
    own = next((p for p in side if isinstance(p, dict) and p.get("active")), {})
    own_types = _types(_species(own.get("details")), obs.get("species_dex", {}))
    foe_id = (obs.get("active") or {}).get("p2")
    foe = next((p for p in (obs.get("opponent_revealed") or [])
                if isinstance(p, dict) and p.get("ident") == foe_id), {})
    foe_types = _types(_species(foe.get("details")), obs.get("species_dex", {}))
    dex = obs.get("move_dex") if isinstance(obs.get("move_dex"), dict) else {}
    scored: list[tuple[float, str]] = []
    for action in legal:
        m = re.fullmatch(r"move ([1-4])( terastallize)?", action)
        if not m:
            continue
        index = int(m.group(1)) - 1
        move = moves[index] if index < len(moves) and isinstance(moves[index], dict) else {}
        info = dex.get(str(move.get("id", "")).lower(), {})
        category = str(info.get("category", ""))
        power = float(info.get("base_power") or 0)
        typ = str(info.get("type", ""))
        try:
            accuracy = 1.0 if info.get("accuracy") is True else float(info.get("accuracy", 100)) / 100.0
        except Exception:
            accuracy = 1.0
        if category.lower() == "status" or power <= 0:
            desc = str(info.get("description", "")).lower()
            score = 22.0 if any(k in desc for k in ("recover", "heals", "restores")) else 7.0
            if any(k in desc for k in ("hazard", "raises", "status", "paraly")):
                score += 4.0
        else:
            score = power * accuracy * _effect(typ, foe_types)
            if typ in own_types:
                score *= 1.5
            # A Tera action is only preferred when it materially improves a line.
            if m.group(2):
                score *= 0.88
        scored.append((score, action))
    if scored:
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[0][1]
    return legal[0]


def _compact(obs: dict[str, Any]) -> str:
    req = obs.get("request") if isinstance(obs.get("request"), dict) else {}
    side = ((req.get("side") or {}).get("pokemon") or [])
    state = {
        "turn": obs.get("turn"), "legal": obs.get("legal_actions", []),
        "active_request": req.get("active", []), "own_team": side,
        "revealed": obs.get("opponent_revealed", []), "active": obs.get("active", {}),
        "summary": obs.get("public_summary", {}),
        "events": (obs.get("recent_events") or [])[-35:],
        "last_error": obs.get("last_error"), "move_dex": obs.get("move_dex", {}),
        "species_dex": obs.get("species_dex", {}),
    }
    raw = json.dumps(state, ensure_ascii=False, separators=(",", ":"), default=str)
    return raw[:26000] + ("…" if len(raw) > 26000 else "")


def _extract(text: Any, legal: list[str]) -> list[str]:
    """Extract only currently legal tokens; arbitrary model prose is harmless."""
    if not isinstance(text, str):
        return []
    low = text.lower()
    found: list[str] = []
    # Longest first prevents `move 1` winning inside `move 1 terastallize`.
    for action in sorted(legal, key=len, reverse=True):
        pat = r"(?<![a-z0-9])" + re.escape(action.lower()) + r"(?![a-z0-9])"
        if re.search(pat, low):
            found.append(action)
    # Preserve the model's textual order, and cap a speculative line.
    if not found:
        return []
    positions = []
    for action in found:
        m = re.search(r"(?<![a-z0-9])" + re.escape(action.lower()) + r"(?![a-z0-9])", low)
        positions.append((m.start() if m else 10**9, action))
    positions.sort()
    out = []
    for _, action in positions:
        if action not in out:
            out.append(action)
    return out[:3]


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(ctx, "system2", None) is None:
            return
        try:
            obs = game.observe()
        except Exception:
            return
        session = None
        plan: list[str] = []
        calls = 0
        previous_events: list[str] = []
        while not getattr(ctx, "done", False) and not getattr(game, "done", False):
            if not isinstance(obs, dict):
                try:
                    obs = game.observe()
                except Exception:
                    break
            legal = [x for x in obs.get("legal_actions", []) if isinstance(x, str)]
            if not legal:
                break
            events = [str(x) for x in (obs.get("recent_events") or [])]
            if any(any(word in event for word in _EVENT_WORDS) for event in events):
                plan = []
            action = plan.pop(0) if plan and plan[0] in legal else None
            if action is None:
                plan = []
                # Reserve calls for actual decisions and leave a margin for the deadline.
                do_model = calls < 30
                try:
                    rem = ctx.budget.remaining()
                    do_model = do_model and int(rem.get("system2_calls", 0)) > 0 and float(rem.get("wall_s", 0)) > 120
                except Exception:
                    do_model = False
                if do_model:
                    prompt = ("You are the strategic controller for a Gen 9 Random Battle. "
                              "Use only the current private request and public protocol. "
                              "Choose a robust line: immediate KO/safe damage first, then a "
                              "switch, recovery, hazard, setup or Tera only when justified. "
                              "Respect forced replacement and current slot numbering. "
                              "Return one line beginning ACTION: with an exact legal token. "
                              "Optionally put up to two conditional follow-ups after PLAN:. "
                              "Do not invent actions; if uncertain choose the safest legal attack.\n\n"
                              "STATE JSON:\n" + _compact(obs))
                    try:
                        res = ctx.think(prompt, session_id=session, timeout_s=105.0)
                        calls += 1
                        session = getattr(res, "session_id", None) or session
                        choices = _extract(getattr(res, "text", ""), legal)
                        if choices:
                            action = choices[0]
                            plan = choices[1:]
                    except (BudgetExceeded, ModelUnavailable, TimeoutError, RuntimeError):
                        calls = 30
                    except Exception:
                        calls = 30
                if action is None:
                    action = _fallback(obs)
            if action is None:
                break
            try:
                nxt = ctx.act(action)
                obs = nxt if isinstance(nxt, dict) else game.observe()
            except Exception:
                # A hidden trap or stale choice is recoverable: refresh and use a legal fallback.
                plan = []
                try:
                    obs = game.observe()
                except Exception:
                    break


def build(ctx):
    return Harness(ctx)
