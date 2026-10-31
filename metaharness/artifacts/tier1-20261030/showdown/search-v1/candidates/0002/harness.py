"""Robust, code-first Random Battle harness.

The model is useful for the genuinely uncertain choice (usually a first-turn
switch or a Tera decision), but the simulator must keep moving when a model
answer is malformed or unavailable.  A small type/power scorer therefore
always supplies a legal fallback and is also used to validate model choices.
"""
from __future__ import annotations

import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


# The common Gen 9 chart.  Unknown types deliberately have neutral effect.
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


def _types(dex: dict, ident: str | None) -> list[str]:
    if not ident:
        return []
    # Ident is generally "p2: Species"; details may include level/gender.
    name = ident.split(":", 1)[-1].strip()
    name = name.split(",", 1)[0].strip()
    ent = dex.get(name) or dex.get(name.replace("-", " "))
    return list(ent.get("types", [])) if isinstance(ent, dict) else []


def _effect(move_type: str, target_types: list[str]) -> float:
    value = 1.0
    row = _CHART.get(move_type, {})
    for t in target_types:
        value *= float(row.get(t, 1.0))
    return value


def _hp_fraction(p: dict) -> float:
    c = str(p.get("condition", ""))
    if not c or c.startswith("fnt"):
        return 0.0
    try:
        a, b = c.split("/", 1)
        return max(0.0, min(1.0, float(a) / float(b.split()[0])))
    except Exception:
        try:
            return float(c.rstrip("%")) / 100.0
        except Exception:
            return 1.0


def _opponent(obs: dict) -> dict:
    ident = (obs.get("active") or {}).get("p2")
    for p in obs.get("opponent_revealed", []) or []:
        if p.get("ident") == ident:
            return p
    for p in reversed(obs.get("opponent_revealed", []) or []):
        if p.get("ident") == ident or ident is None:
            return p
    return {}


def _move_score(obs: dict, idx: int, tera: bool = False) -> float:
    req = obs.get("request") or {}
    active = ((req.get("active") or [{}])[0])
    moves = active.get("moves") or []
    if idx >= len(moves):
        return -1e9
    m = moves[idx]
    if m.get("disabled") or not m.get("pp", 1):
        return -1e9
    dex = obs.get("move_dex") or {}
    md = dex.get(m.get("id"), {})
    typ = md.get("type", "Normal")
    power = float(md.get("base_power") or 0)
    acc = md.get("accuracy", 100)
    if acc is True or acc is None:
        acc = 100
    try:
        acc = float(acc)
    except Exception:
        acc = 100
    own = next((p for p in (req.get("side") or {}).get("pokemon", []) if p.get("active")), {})
    opp = _opponent(obs)
    target_types = _types(obs.get("species_dex") or {}, opp.get("ident") or opp.get("details"))
    own_types = _types(obs.get("species_dex") or {}, own.get("ident") or own.get("details"))
    eff = _effect(typ, target_types)
    if tera:
        tera_type = active.get("canTerastallize")
        if not tera_type:
            return -1e9
        # Tera Blast changes type; ordinary moves gain the assigned STAB.
        if m.get("id") == "terablast":
            typ = tera_type
            eff = _effect(typ, target_types)
        stab = 1.5 if tera_type in own_types or typ == tera_type else 1.0
    else:
        stab = 1.5 if typ in own_types else 1.0
    hp = _hp_fraction(own)
    opphp = _hp_fraction(opp)
    if power:
        # This is intentionally a ranking, not a claimed exact damage calc.
        score = power * (acc / 100.0) * stab * eff
        if eff == 0:
            score = -40
        if eff >= 2 and opphp < .55:
            score += 35
        if power >= 100 and eff >= 1.5:
            score += 8
    else:
        desc = str(md.get("description", "")).lower()
        n = str(md.get("name", m.get("move", ""))).lower()
        score = 8.0
        if any(x in n + " " + desc for x in ("recover", "roost", "slack off", "rest", "moonlight", "strength sap")):
            score += 55 if hp < .45 else -12
        if any(x in n + " " + desc for x in ("stealth rock", "spikes", "sticky web")):
            score += 18 if not (obs.get("public_summary") or {}).get("side_conditions", {}).get("p1") else -5
        if any(x in n + " " + desc for x in ("will-o-wisp", "toxic", "thunder wave", "nuzzle", "spore", "sleep powder")):
            score += 22 if opphp > .35 else 4
        if any(x in n + " " + desc for x in ("calm mind", "nasty plot", "swords dance", "dragon dance", "bulk up")):
            score += 15 if hp > .75 and opphp > .45 else -8
        if any(x in n + " " + desc for x in ("protect", "substitute")):
            score += 5
    # Spending Tera is reserved for a clear matchup improvement.
    if tera:
        score -= 7
        if eff >= 2 or (m.get("id") == "terablast" and eff > 1):
            score += 14
        if hp < .3 and eff >= 1.5:
            score += 10
    return score


def _scripted(obs: dict) -> str:
    legal = list(obs.get("legal_actions") or [])
    if not legal:
        return "noop"
    moves = [a for a in legal if a.lower().startswith("move ")]
    move_choice = None
    move_score = -1e18
    if moves:
        # Evaluate each slot once; a Tera variant only wins when clearly better.
        for a in moves:
            mt = re.match(r"move\s+(\d+)(?:\s+terastallize)?$", a, re.I)
            if not mt:
                continue
            i = int(mt.group(1)) - 1
            s = _move_score(obs, i, a.lower().endswith("terastallize"))
            if s > move_score:
                move_score, move_choice = s, a
    # Forced replacement or a deliberately defensive switch: choose the
    # healthiest legal teammate, with a small type-coverage bonus.
    req = obs.get("request") or {}
    opp = _opponent(obs)
    ot = _types(obs.get("species_dex") or {}, opp.get("ident") or opp.get("details"))
    pokes = (req.get("side") or {}).get("pokemon", [])
    candidates = []
    for a in legal:
        m = re.match(r"switch\s+(\d+)$", a, re.I)
        if not m:
            continue
        j = int(m.group(1)) - 1
        p = pokes[j] if 0 <= j < len(pokes) else {}
        if p.get("active"):
            continue
        pt = _types(obs.get("species_dex") or {}, p.get("ident") or p.get("details"))
        hp = _hp_fraction(p)
        # Prefer healthy pieces and those with useful revealed type coverage;
        # incoming hidden moves remain uncertain.
        defense = max((_effect(t, ot) for t in pt), default=1.0)
        candidates.append((hp + (1.0 if defense < 1 else 0.0) + len(pt) * .01, a))
    if candidates:
        # Voluntary switching is worthwhile when the active piece is nearly
        # gone or every attack is resisted/utility-only.  Otherwise retain
        # momentum and take the best ranked move.
        active_self = next((p for p in pokes if p.get("active")), {})
        hp = _hp_fraction(active_self)
        if move_choice is None or hp < .22 or move_score < 24:
            return max(candidates)[1]
    if move_choice is not None:
        return move_choice
    return legal[0]


def _compact(obs: dict) -> dict:
    """Keep Astra's prompt small enough that slow output does not dominate."""
    req = obs.get("request") or {}
    side = (req.get("side") or {}).get("pokemon", [])
    active = (req.get("active") or [{}])[0]
    return {
        "turn": obs.get("turn"),
        "legal": obs.get("legal_actions", []),
        "active_moves": active.get("moves", []),
        "can_tera": active.get("canTerastallize"),
        "team": [{"details": p.get("details"), "condition": p.get("condition"), "active": p.get("active"), "moves": p.get("moves", []), "ability": p.get("ability"), "item": p.get("item")} for p in side],
        "opponent": obs.get("opponent_revealed", []),
        "active_ids": obs.get("active"),
        "events": (obs.get("recent_events") or [])[-20:],
    }


def _model_action(text: Any, legal: list[str]) -> str | None:
    if not isinstance(text, str):
        return None
    by_key = {re.sub(r"\s+", " ", a.strip()).casefold(): a for a in legal}
    lines = [re.sub(r"[`*]", "", x).strip() for x in text.splitlines()]
    for line in reversed(lines):
        key = re.sub(r"\s+", " ", line).casefold()
        if key in by_key:
            return by_key[key]
    # Accept a command embedded in a short explanation, while preferring the
    # longest form so "move 1" does not steal "move 1 terastallize".
    for m in re.finditer(r"\b(?:move\s+[1-4](?:\s+terastallize)?|switch\s+[1-6])\b", text, re.I):
        key = re.sub(r"\s+", " ", m.group(0)).casefold()
        if key in by_key:
            return by_key[key]
    return None


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.session = None
        self.steps = 0
        self.model_calls = 0
        self.last_signature = None

    def run_episode(self, game) -> None:
        ctx = self.ctx
        obs = game.observe()
        # Model calls are deliberately sparse: the local scorer handles every
        # request, while Astra reconsiders after new information or every six
        # accepted actions.  This also leaves ample time for long battles.
        while not ctx.done and not getattr(game, "done", False):
            legal = list((obs or {}).get("legal_actions") or [])
            if not legal:
                break
            action = _scripted(obs)
            current_enemy = (obs or {}).get("active", {}).get("p2")
            should_ask = (self.steps == 0 or self.steps % 12 == 0 or
                          (current_enemy is not None and current_enemy != self.last_signature))
            self.last_signature = current_enemy
            rem = {}
            try:
                rem = ctx.budget.remaining()
            except Exception:
                pass
            if (should_ask and self.model_calls < 8 and getattr(ctx, "system2", None) is not None and
                    rem.get("system2_calls", 1) > 0 and rem.get("wall_s", 999) > 90):
                prompt = ("You play Gen 9 Random Battle. Choose one command from the legal list. "
                          "Use the private team data and revealed opponent; preserve Tera unless it "
                          "secures a KO or survival. Reply with exactly one legal command.\n" +
                          json.dumps(_compact(obs), ensure_ascii=False, separators=(",", ":")))
                try:
                    self.model_calls += 1
                    res = ctx.think(prompt, session_id=self.session, timeout_s=90.0)
                    self.session = getattr(res, "session_id", self.session)
                    picked = _model_action(getattr(res, "text", ""), legal)
                    if picked is not None:
                        action = picked
                except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
                    pass
                except Exception:
                    # Stand-in models and transient adapters are allowed to
                    # return arbitrary objects; the scripted action is enough.
                    pass
            try:
                obs = ctx.act(action)
            except Exception:
                # Hidden trapping can invalidate a visible switch. Re-read the
                # request and submit a fresh legal fallback instead of failing
                # the whole episode (the old candidate crashed here).
                try:
                    obs = ctx.observe()
                except Exception:
                    break
                legal2 = list((obs or {}).get("legal_actions") or [])
                if not legal2:
                    break
                fallback = _scripted(obs)
                if fallback == action and len(legal2) > 1:
                    fallback = legal2[0]
                try:
                    obs = ctx.act(fallback)
                except Exception:
                    break
            self.steps += 1


def build(ctx):
    return Harness(ctx)
