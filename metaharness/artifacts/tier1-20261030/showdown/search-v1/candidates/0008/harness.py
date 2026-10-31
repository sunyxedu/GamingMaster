"""Line-planning and commitment harness for Gen 9 Random Battle.

Astra proposes a short legal line; the controller validates each command against
that request and keeps a small deterministic tactical policy moving when the
line becomes stale or inference is unavailable.  Voluntary switches are given
a brief commitment window to avoid oscillation between otherwise healthy team
members.
"""
from __future__ import annotations

import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable

_CHART = {
    "Normal": {"Rock": .5, "Steel": .5, "Ghost": 0},
    "Fire": {"Fire": .5, "Water": .5, "Rock": .5, "Dragon": .5, "Grass": 2, "Ice": 2, "Bug": 2, "Steel": 2},
    "Water": {"Fire": 2, "Water": .5, "Grass": .5, "Dragon": .5, "Ground": 2, "Rock": 2},
    "Electric": {"Water": 2, "Electric": .5, "Grass": .5, "Dragon": .5, "Ground": 0, "Flying": 2},
    "Grass": {"Fire": .5, "Water": 2, "Grass": .5, "Poison": .5, "Flying": .5, "Bug": .5, "Dragon": .5, "Steel": .5, "Ground": 2, "Rock": 2},
    "Ice": {"Fire": .5, "Water": .5, "Grass": 2, "Ice": .5, "Ground": 2, "Flying": 2, "Dragon": 2, "Steel": .5},
    "Fighting": {"Normal": 2, "Ice": 2, "Rock": 2, "Dark": 2, "Steel": 2, "Poison": .5, "Flying": .5, "Psychic": .5, "Bug": .5, "Fairy": .5, "Ghost": 0},
    "Poison": {"Grass": 2, "Poison": .5, "Ground": .5, "Rock": .5, "Ghost": .5, "Steel": 0, "Fairy": 2},
    "Ground": {"Fire": 2, "Electric": 2, "Poison": 2, "Rock": 2, "Steel": 2, "Grass": .5, "Bug": .5, "Flying": 0},
    "Flying": {"Grass": 2, "Fighting": 2, "Bug": 2, "Electric": .5, "Rock": .5, "Steel": .5},
    "Psychic": {"Fighting": 2, "Poison": 2, "Psychic": .5, "Steel": .5, "Dark": 0},
    "Bug": {"Grass": 2, "Psychic": 2, "Dark": 2, "Fire": .5, "Fighting": .5, "Poison": .5, "Flying": .5, "Ghost": .5, "Steel": .5, "Fairy": .5},
    "Rock": {"Fire": 2, "Ice": 2, "Flying": 2, "Bug": 2, "Fighting": .5, "Ground": .5, "Steel": .5},
    "Ghost": {"Psychic": 2, "Ghost": 2, "Dark": .5, "Normal": 0},
    "Dragon": {"Dragon": 2, "Steel": .5, "Fairy": 0},
    "Dark": {"Psychic": 2, "Ghost": 2, "Fighting": .5, "Dark": .5, "Fairy": .5},
    "Steel": {"Ice": 2, "Rock": 2, "Fairy": 2, "Fire": .5, "Water": .5, "Electric": .5, "Steel": .5},
    "Fairy": {"Fighting": 2, "Dragon": 2, "Dark": 2, "Fire": .5, "Poison": .5, "Steel": .5},
}


def _name(value: Any) -> str:
    return str(value or "").split(":", 1)[-1].split(",", 1)[0].strip()


def _types(obs: dict, value: Any) -> list[str]:
    ent = (obs.get("species_dex") or {}).get(_name(value), {})
    return [str(x) for x in ent.get("types", [])] if isinstance(ent, dict) else []


def _effect(atk: str, defs: list[str]) -> float:
    out = 1.0
    row = _CHART.get(atk, {})
    for d in defs:
        out *= float(row.get(d, 1.0))
    return out


def _hp(p: Any) -> float:
    if not isinstance(p, dict):
        return 0.0
    c = str(p.get("condition", "0/1"))
    if c.startswith("fnt") or c.startswith("0 fnt"):
        return 0.0
    m = re.match(r"\s*([0-9.]+)\s*/\s*([0-9.]+)", c)
    if m:
        return max(0.0, min(1.0, float(m.group(1)) / max(1.0, float(m.group(2)))))
    try:
        return max(0.0, min(1.0, float(c.rstrip("%")) / 100.0))
    except Exception:
        return 0.0


def _foe(obs: dict) -> dict:
    ident = (obs.get("active") or {}).get("p2")
    for p in obs.get("opponent_revealed") or []:
        if isinstance(p, dict) and p.get("ident") == ident:
            return p
    return {}


def _info(obs: dict, move: dict) -> dict:
    dex = obs.get("move_dex") or {}
    return dex.get(str(move.get("id", "")).lower(), {}) if isinstance(dex, dict) else {}


def _move_score(obs: dict, i: int, tera: bool = False) -> float:
    req = obs.get("request") or {}
    active = (req.get("active") or [{}])[0]
    moves = active.get("moves") or []
    if i < 0 or i >= len(moves) or not isinstance(moves[i], dict):
        return -1e9
    mv = moves[i]
    if mv.get("disabled") or not mv.get("pp", 1):
        return -1e9
    inf = _info(obs, mv)
    typ = str(inf.get("type", "Normal"))
    try:
        power = float(inf.get("base_power") or 0)
    except Exception:
        power = 0.0
    try:
        acc = 1.0 if inf.get("accuracy") in (True, None) else float(inf.get("accuracy", 100)) / 100.0
    except Exception:
        acc = 1.0
    side = ((req.get("side") or {}).get("pokemon") or [])
    own = next((p for p in side if isinstance(p, dict) and p.get("active")), {})
    foe = _foe(obs)
    own_t, foe_t = _types(obs, own.get("details")), _types(obs, foe.get("details"))
    eff = _effect(typ, foe_t)
    if tera:
        tera_t = active.get("canTerastallize")
        if not tera_t:
            return -1e9
        if str(mv.get("id", "")).lower() == "terablast":
            typ, eff = str(tera_t), _effect(str(tera_t), foe_t)
        stab = 1.5 if typ == tera_t or tera_t in own_t else 1.0
    else:
        stab = 1.5 if typ in own_t else 1.0
    if power:
        score = power * acc * stab * eff
        if eff == 0:
            return -100.0
        if eff >= 2:
            score += 12
        if int(inf.get("priority", 0) or 0) > 0 and _hp(own) < .55:
            score += 18
        if tera:
            score -= 14
            if eff >= 2:
                score += 18
        return score
    text = (str(inf.get("name", "")) + " " + str(inf.get("description", ""))).lower()
    hp, foehp = _hp(own), _hp(foe)
    score = 7.0
    if any(x in text for x in ("recover", "roost", "heals", "restores", "strength sap")):
        score += 56 if hp < .55 else -10
    if any(x in text for x in ("stealth rock", "spikes", "sticky web")):
        cond = (obs.get("public_summary") or {}).get("side_conditions", {}).get("p1")
        score += 20 if not cond else -5
    if any(x in text for x in ("toxic", "will-o-wisp", "thunder wave", "nuzzle", "spore", "sleep powder")):
        score += 20 if foehp > .35 else 2
    if any(x in text for x in ("calm mind", "nasty plot", "swords dance", "dragon dance", "bulk up")):
        score += 14 if hp > .75 else -12
    return score - (15 if tera else 0)


def _rank(obs: dict) -> tuple[str | None, list[dict[str, Any]]]:
    legal = [x for x in obs.get("legal_actions") or [] if isinstance(x, str)]
    if not legal:
        return None, []
    if not any(x.startswith("move ") for x in legal):
        side = ((obs.get("request") or {}).get("side") or {}).get("pokemon", [])
        vals = []
        for a in legal:
            m = re.fullmatch(r"switch ([1-6])", a)
            if m:
                j = int(m.group(1)) - 1
                vals.append((_hp(side[j]) if 0 <= j < len(side) else 0.0, a))
        return (max(vals)[1] if vals else legal[0]), [{"action": a, "score": round(s, 2), "kind": "forced"} for s, a in vals]
    req = obs.get("request") or {}
    side = ((req.get("side") or {}).get("pokemon") or [])
    foe_t = _types(obs, _foe(obs).get("details"))
    ranked: list[tuple[float, str, str]] = []
    for a in legal:
        m = re.fullmatch(r"move ([1-4])( terastallize)?", a)
        if m:
            ranked.append((_move_score(obs, int(m.group(1)) - 1, bool(m.group(2))), a, "move"))
            continue
        m = re.fullmatch(r"switch ([1-6])", a)
        if m:
            j = int(m.group(1)) - 1
            p = side[j] if 0 <= j < len(side) and isinstance(side[j], dict) else {}
            if p.get("active"):
                continue
            pt = _types(obs, p.get("details"))
            incoming = max((_effect(t, pt) for t in foe_t), default=1.0)
            score = (2.0 - min(2.0, incoming)) * 40.0 + _hp(p) * 34.0 + (18 if incoming == 0 else 0)
            ranked.append((score, a, "switch"))
    ranked.sort(key=lambda x: x[0], reverse=True)
    if not ranked:
        return legal[0], []
    moves = [x for x in ranked if x[2] == "move"][:6]
    switches = [x for x in ranked if x[2] == "switch"][:3]
    shown = sorted(moves + switches, key=lambda x: x[0], reverse=True)[:6]
    fallback = max(moves, key=lambda x: x[0])[1] if moves else ranked[0][1]
    # Code takes an obvious defensive exit itself.  Astra is reserved for
    # close switches, where role and hidden coverage matter.
    if moves and switches:
        bm = max(moves, key=lambda x: x[0])
        bs = max(switches, key=lambda x: x[0])
        if bm[0] < 45 and bs[0] > bm[0] + 25:
            fallback = bs[1]
    return fallback, [{"action": a, "score": round(s, 1), "kind": k} for s, a, k in shown]


def _fork_worthy(options: list[dict[str, Any]], fallback: str, legal: list[str]) -> bool:
    """Ask the model only at a real tactical fork, rather than every reveal."""
    if not options or not any(a.startswith("move ") for a in legal):
        return False
    vals = [(float(x.get("score", -1e9)), str(x.get("action", "")), str(x.get("kind", ""))) for x in options]
    moves = sorted((v for v in vals if v[2] == "move"), reverse=True)
    switches = sorted((v for v in vals if v[2] == "switch"), reverse=True)
    if not moves:
        return True
    best = moves[0][0]
    # A close move choice benefits from set/role reasoning; a clear high-power
    # move should be executed without spending 50-100 seconds on Astra.
    close_moves = len(moves) > 1 and moves[1][0] >= best * 0.84
    weak = best < 72
    switch_fork = bool(switches and switches[0][0] >= best - 8)
    tera_fork = bool("terastallize" in fallback and best < 150)
    return close_moves or weak or switch_fork or tera_fork


def _prompt(obs: dict, options: list[dict[str, Any]]) -> str:
    req = obs.get("request") or {}
    side = ((req.get("side") or {}).get("pokemon") or [])
    own = [{"details": p.get("details"), "condition": p.get("condition"), "active": p.get("active"),
            "status": p.get("status"), "ability": p.get("ability"), "item": p.get("item"),
            "moves": p.get("moves", [])} for p in side if isinstance(p, dict)]
    active = (req.get("active") or [{}])[0]
    state = {"turn": obs.get("turn"), "legal": obs.get("legal_actions", []),
             "active_moves": active.get("moves", []), "can_tera": active.get("canTerastallize"),
             "own_team": own, "foe": _foe(obs), "active": obs.get("active", {}),
             "revealed": obs.get("opponent_revealed", []), "options": options,
             "events": (obs.get("recent_events") or [])[-14:],
             "field": (obs.get("public_summary") or {}).get("side_conditions", {})}
    raw = json.dumps(state, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(raw) > 22000:
        raw = raw[:22000] + "…"
    return ("Play Gen 9 Random Battle as p1. Pick a SHORT LINE of up to three actions. "
            "Each action must be one currently legal token; later tokens are tentative and will "
            "be checked again. Prefer a KO or safe progress, preserve a needed counter, and use "
            "Tera only for a decisive exchange. Forced replacement means switch. Output exactly "
            "LINE: token ; token ; token, with no alternative commands in the reason.\nSTATE:\n" + raw)


def _line(text: Any) -> list[str]:
    if not isinstance(text, str):
        return []
    # Restrict parsing to the LINE field to avoid extracting alternatives from prose.
    s = text.lower().split("line:", 1)[-1] if "line:" in text.lower() else text.lower()
    out = []
    for m in re.finditer(r"\b(move\s+[1-4](?:\s+terastallize)?|switch\s+[1-6])\b", s):
        tok = re.sub(r"\s+", " ", m.group(1)).strip()
        if tok not in out:
            out.append(tok)
    return out[:3]


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        ctx = self.ctx
        try:
            obs = game.observe()
        except Exception:
            return
        plan: list[str] = []
        session = None
        calls = 0
        steps = 0
        last_sig = None
        switch_step = -99
        while not getattr(ctx, "done", False) and not getattr(game, "done", False):
            if not isinstance(obs, dict):
                try:
                    obs = game.observe()
                except Exception:
                    break
            legal = [x for x in obs.get("legal_actions") or [] if isinstance(x, str)]
            if not legal:
                break
            fallback, options = _rank(obs)
            if fallback is None:
                break
            # A line is invalidated by a new active identity or forced request.
            sig = ((obs.get("active") or {}).get("p1"), (obs.get("active") or {}).get("p2"),
                   tuple(legal))
            if sig != last_sig and any(a.startswith("switch ") for a in legal) and not any(a.startswith("move ") for a in legal):
                plan = []
            last_sig = sig
            # A proposed line is deliberately one action long.  A switch
            # reorders request slots, so retaining later tokens is unsafe.
            plan = [plan[0]] if plan and plan[0] in legal else []
            foe_id = (obs.get("active") or {}).get("p2")
            new_foe = foe_id != getattr(self, "last_foe", None)
            checkpoint = (foe_id, steps // 3)
            # First action gets one strategic look; later calls require a
            # meaningful fork and a changed checkpoint.  This is the key
            # code-first difference from 0006.
            ask = (not plan and calls < 10 and getattr(ctx, "system2", None) is not None
                   and (steps == 0 or (new_foe and _fork_worthy(options, fallback, legal))))
            if ask:
                try:
                    rem = ctx.budget.remaining()
                    if int(rem.get("system2_calls", 0)) > 0 and float(rem.get("wall_s", 0)) > 170:
                        res = ctx.think(_prompt(obs, options), session_id=session, timeout_s=82.0)
                        calls += 1
                        session = getattr(res, "session_id", None) or session
                        proposed = _line(getattr(res, "text", ""))
                        # Accept only the current first token, and only if it
                        # was in the small candidate set shown to Astra.
                        allowed = {str(x.get("action")) for x in options}
                        if proposed and proposed[0] in legal and proposed[0] in allowed:
                            plan = [proposed[0]]
                except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
                    calls += 1
                    plan = []
                except Exception:
                    calls += 1
                    plan = []
                self.checkpoint = checkpoint
            self.last_foe = foe_id
            while plan and plan[0] not in legal:
                plan.pop(0)
            action = plan.pop(0) if plan else fallback
            # Commitment window: after a voluntary switch, do not immediately
            # reverse it while attacks remain available unless the active is critical.
            if action.startswith("switch ") and any(a.startswith("move ") for a in legal) and steps - switch_step < 2:
                own = next((p for p in ((obs.get("request") or {}).get("side") or {}).get("pokemon", [])
                            if isinstance(p, dict) and p.get("active")), {})
                if _hp(own) > .18:
                    action = fallback
            if action.startswith("switch ") and any(a.startswith("move ") for a in legal):
                switch_step = steps
            try:
                nxt = ctx.act(action)
                obs = nxt if isinstance(nxt, dict) else game.observe()
            except Exception:
                try:
                    obs = game.observe()
                    fresh, _ = _rank(obs if isinstance(obs, dict) else {})
                    if fresh is None:
                        break
                    nxt = ctx.act(fresh)
                    obs = nxt if isinstance(nxt, dict) else game.observe()
                    plan = []
                except Exception:
                    break
            steps += 1


def build(ctx):
    return Harness(ctx)
