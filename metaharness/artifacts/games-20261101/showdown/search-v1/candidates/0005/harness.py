"""Candidate-ranking controller for Gen 9 Random Battle.

The engine-facing loop is deliberately boring: every submitted command is
validated against the current request, and a local policy remains available
when Astra is slow or returns arbitrary prose.  Astra is shown a short list of
locally generated attacks and switches, so its reasoning is spent on the
uncertain matchup rather than reconstructing the protocol.
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


def _species(value: Any) -> str:
    return str(value or "").split(":", 1)[-1].split(",", 1)[0].strip()


def _types(obs: dict, value: Any) -> list[str]:
    name = _species(value)
    ent = (obs.get("species_dex") or {}).get(name, {})
    return [str(x) for x in ent.get("types", [])] if isinstance(ent, dict) else []


def _effect(attacking: str, defending: list[str]) -> float:
    out = 1.0
    row = _CHART.get(attacking, {})
    for typ in defending:
        out *= float(row.get(typ, 1.0))
    return out


def _hp(p: Any) -> float:
    if not isinstance(p, dict):
        return 0.0
    raw = str(p.get("condition", "0/1"))
    if raw.startswith("fnt") or raw.startswith("0 fnt"):
        return 0.0
    try:
        a, b = raw.split("/", 1)
        return max(0.0, min(1.0, float(a) / max(1.0, float(re.match(r"[0-9.]+", b).group(0)))))
    except Exception:
        try:
            return max(0.0, min(1.0, float(raw.rstrip("%")) / 100.0))
        except Exception:
            return 0.0


def _active_foe(obs: dict) -> dict:
    ident = (obs.get("active") or {}).get("p2")
    for p in obs.get("opponent_revealed") or []:
        if isinstance(p, dict) and p.get("ident") == ident:
            return p
    return {}


def _move_info(obs: dict, move: dict) -> dict:
    dex = obs.get("move_dex") or {}
    return dex.get(str(move.get("id", "")).lower(), {}) if isinstance(dex, dict) else {}


def _attack_score(obs: dict, idx: int, tera: bool = False) -> float:
    req = obs.get("request") or {}
    active = (req.get("active") or [{}])[0]
    moves = active.get("moves") or []
    if idx < 0 or idx >= len(moves) or not isinstance(moves[idx], dict):
        return -1e9
    move = moves[idx]
    if move.get("disabled") or not move.get("pp", 1):
        return -1e9
    info = _move_info(obs, move)
    typ = str(info.get("type", ""))
    try:
        power = float(info.get("base_power") or 0)
    except Exception:
        power = 0.0
    try:
        accuracy = 1.0 if info.get("accuracy") in (True, None) else float(info.get("accuracy", 100)) / 100.0
    except Exception:
        accuracy = 1.0
    side = ((req.get("side") or {}).get("pokemon") or [])
    own = next((p for p in side if isinstance(p, dict) and p.get("active")), {})
    foe = _active_foe(obs)
    own_types = _types(obs, own.get("details"))
    foe_types = _types(obs, foe.get("details"))
    eff = _effect(typ, foe_types)
    if tera:
        tera_type = active.get("canTerastallize")
        if not tera_type:
            return -1e9
        if str(move.get("id", "")).lower() == "terablast":
            typ, eff = str(tera_type), _effect(str(tera_type), foe_types)
        stab = 1.5 if typ == tera_type or tera_type in own_types else 1.0
    else:
        stab = 1.5 if typ in own_types else 1.0
    if power > 0:
        score = power * accuracy * stab * eff
        if eff == 0:
            return -100.0
        # Priority is a small but meaningful tie-break when the active is frail.
        if int(info.get("priority", 0) or 0) > 0 and _hp(own) < .5:
            score += 18
        if eff >= 2:
            score += 12
        if tera:
            score -= 12
            if eff >= 2 or (str(move.get("id", "")).lower() == "terablast" and eff > 1):
                score += 18
        return score
    text = (str(info.get("name", "")) + " " + str(info.get("description", ""))).lower()
    hp = _hp(own)
    foehp = _hp(foe)
    score = 7.0
    if any(x in text for x in ("recover", "roost", "heals", "restores", "strength sap")):
        score += 58 if hp < .55 else -10
    if any(x in text for x in ("stealth rock", "spikes", "sticky web")):
        conditions = (obs.get("public_summary") or {}).get("side_conditions", {}).get("p1", {})
        score += 20 if not conditions else -5
    if any(x in text for x in ("toxic", "will-o-wisp", "thunder wave", "nuzzle", "spore", "sleep powder")):
        score += 20 if foehp > .35 else 2
    if any(x in text for x in ("calm mind", "nasty plot", "swords dance", "dragon dance", "bulk up")):
        score += 13 if hp > .75 else -12
    if tera:
        score -= 15
    return score


def _candidates(obs: dict) -> tuple[str | None, list[dict[str, Any]]]:
    legal = [x for x in obs.get("legal_actions") or [] if isinstance(x, str)]
    if not legal:
        return None, []
    if not any(x.startswith("move ") for x in legal):
        side = ((obs.get("request") or {}).get("side") or {}).get("pokemon", [])
        vals = []
        for a in legal:
            m = re.fullmatch(r"switch ([1-6])", a)
            if m:
                i = int(m.group(1)) - 1
                vals.append((_hp(side[i]) if 0 <= i < len(side) else 0.0, a))
        return (max(vals)[1] if vals else legal[0]), [{"action": a, "score": round(s, 2), "kind": "forced switch"} for s, a in vals]

    req = obs.get("request") or {}
    side = ((req.get("side") or {}).get("pokemon") or [])
    foe = _active_foe(obs)
    foe_types = _types(obs, foe.get("details"))
    scored: list[tuple[float, str, str]] = []
    for a in legal:
        m = re.fullmatch(r"move ([1-4])( terastallize)?", a)
        if m:
            score = _attack_score(obs, int(m.group(1)) - 1, bool(m.group(2)))
            scored.append((score, a, "attack"))
            continue
        m = re.fullmatch(r"switch ([1-6])", a)
        if m:
            i = int(m.group(1)) - 1
            p = side[i] if 0 <= i < len(side) and isinstance(side[i], dict) else {}
            if p.get("active"):
                continue
            pt = _types(obs, p.get("details"))
            # Defensive score rewards a useful resistance and healthy entry.
            incoming = max((_effect(t, pt) for t in foe_types), default=1.0)
            defense = (2.0 - min(2.0, incoming)) * 40.0 + _hp(p) * 32.0
            if incoming == 0:
                defense += 18
            scored.append((defense, a, "switch"))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return legal[0], []
    # Keep the best defensive switch visible even when several attacks score
    # above it; otherwise Astra cannot make the central attack-versus-switch
    # decision this shortlist is designed to expose.
    attacks = [x for x in scored if x[2] == "attack"][:4]
    switches = [x for x in scored if x[2] == "switch"][:1]
    chosen = sorted(attacks + switches, key=lambda x: x[0], reverse=True)[:5]
    if not chosen:
        chosen = scored[:5]
    fallback = max((x for x in scored if x[2] == "attack"), default=scored[0], key=lambda x: x[0])[1]
    return fallback, [{"action": a, "score": round(s, 1), "kind": k} for s, a, k in chosen]


def _extract(text: Any, legal: list[str]) -> str | None:
    if not isinstance(text, str):
        return None
    low = text.lower()
    hits = []
    for action in legal:
        m = re.search(r"(?<![a-z0-9])" + re.escape(action.lower()) + r"(?![a-z0-9])", low)
        if m:
            hits.append((m.start(), -len(action), action))
    return sorted(hits)[0][2] if hits else None


def _prompt(obs: dict, options: list[dict[str, Any]]) -> str:
    req = obs.get("request") or {}
    side = ((req.get("side") or {}).get("pokemon") or [])
    own = [{"details": p.get("details"), "condition": p.get("condition"), "active": p.get("active"),
            "status": p.get("status"), "ability": p.get("ability"), "item": p.get("item"),
            "moves": p.get("moves", [])} for p in side if isinstance(p, dict)]
    active = (req.get("active") or [{}])[0]
    state = {
        "turn": obs.get("turn"), "legal": obs.get("legal_actions", []),
        "active_moves": active.get("moves", []), "can_tera": active.get("canTerastallize"),
        "own_team": own, "foe": _active_foe(obs), "active": obs.get("active", {}),
        "revealed": obs.get("opponent_revealed", []), "options": options,
        "events": (obs.get("recent_events") or [])[-18:],
        "field": (obs.get("public_summary") or {}).get("side_conditions", {}),
    }
    raw = json.dumps(state, ensure_ascii=False, separators=(",", ":"), default=str)
    return ("You are the tactical controller for Gen 9 Random Battle. Pick exactly one command "
            "from CURRENT LEGAL. The OPTIONS were generated from the private request: use them "
            "as a safety filter, but correct their ranking when the revealed foe, speed/priority, "
            "status, hazards, or a concrete KO changes the answer. Preserve Tera unless it wins a "
            "decisive exchange. Forced replacement must be a switch. Reply with `CHOICE: <token>` "
            "and at most one short reason. Never mention a token that is not legal.\nSTATE:\n" + raw)[:30000]


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        ctx = self.ctx
        try:
            obs = game.observe()
        except Exception:
            return
        session = None
        calls = 0
        steps = 0
        last_checkpoint = None
        failures = 0
        while not getattr(ctx, "done", False) and not getattr(game, "done", False):
            if not isinstance(obs, dict):
                try:
                    obs = game.observe()
                except Exception:
                    break
            legal = [x for x in obs.get("legal_actions") or [] if isinstance(x, str)]
            if not legal:
                break
            fallback, options = _candidates(obs)
            if fallback is None:
                break
            active = obs.get("active") or {}
            revealed = obs.get("opponent_revealed") or []
            # A checkpoint is a new foe/own active, new revealed move/species,
            # or a periodic six-action review. Damage alone need not pay for a
            # 60-second call; it is still present in the next compact ledger.
            reveal_sig = tuple((p.get("ident"), tuple(p.get("moves") or []), p.get("ability"))
                               for p in revealed if isinstance(p, dict))
            checkpoint = (active.get("p1"), active.get("p2"), reveal_sig,
                          steps // 6)
            ask = checkpoint != last_checkpoint and calls < 18
            if ask and getattr(ctx, "system2", None) is not None:
                try:
                    rem = ctx.budget.remaining()
                    ask = int(rem.get("system2_calls", 0)) > 0 and float(rem.get("wall_s", 0)) > 125
                except Exception:
                    ask = False
            action = fallback
            if ask:
                try:
                    result = ctx.think(_prompt(obs, options), session_id=session, timeout_s=88.0)
                    calls += 1
                    failures = 0
                    session = getattr(result, "session_id", None) or session
                    selected = _extract(getattr(result, "text", ""), legal)
                    if selected is not None:
                        action = selected
                except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
                    calls += 1
                    failures += 1
                except Exception:
                    calls += 1
                    failures += 1
                last_checkpoint = checkpoint
            try:
                nxt = ctx.act(action)
                obs = nxt if isinstance(nxt, dict) else game.observe()
            except Exception:
                # Hidden trapping or a stale request is recoverable. Refresh and
                # post a freshly ranked legal action once; never propagate it.
                try:
                    obs = game.observe()
                    fresh, _ = _candidates(obs if isinstance(obs, dict) else {})
                    if fresh is None:
                        break
                    nxt = ctx.act(fresh)
                    obs = nxt if isinstance(nxt, dict) else game.observe()
                except Exception:
                    break
            steps += 1


def build(ctx):
    return Harness(ctx)
