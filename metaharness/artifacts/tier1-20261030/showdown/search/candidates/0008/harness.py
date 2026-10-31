"""Threat-barrier controller for Gen 9 Random Battle.

The model supplies a small ranked action set.  Code owns legality, an opening
information barrier, and a very conservative one-step repeat learned from the
last protocol result.
"""
from __future__ import annotations
import json
import re
from metaharness.core.contract import BudgetExceeded, ModelUnavailable


def ident(x):
    return re.sub(r"[^a-z0-9]", "", str(x or "").lower())


def hpfrac(p):
    c = str((p or {}).get("condition", "0"))
    if "fnt" in c:
        return 0.0
    m = re.match(r"(\d+)\s*/\s*(\d+)", c)
    try:
        return float(m.group(1)) / max(1, float(m.group(2))) if m else 0.0
    except (ValueError, AttributeError):
        return 0.0


def team(o):
    return (o.get("request") or {}).get("side", {}).get("pokemon", [])


def own_active(o):
    return next((p for p in team(o) if p.get("active")), {})


def foe_active(o):
    key = (o.get("active") or {}).get("p2", "")
    ps = (o.get("public_summary") or {}).get("pokemon", {})
    if key in ps:
        return ps[key]
    return next((p for p in o.get("opponent_revealed", []) if p.get("ident") == key), {"ident": key})


def own_moves(o):
    a = ((o.get("request") or {}).get("active") or [{}])
    return a[0].get("moves", []) if a else []


def species(p):
    return str(p.get("details", "")).split(",")[0].strip()


CHART = {
    "Normal": ("", "Rock Steel", "Ghost"), "Fire": ("Grass Ice Bug Steel", "Fire Water Rock Dragon", ""),
    "Water": ("Fire Ground Rock", "Water Grass Dragon", ""), "Electric": ("Water Flying", "Electric Grass Dragon", "Ground"),
    "Grass": ("Water Ground Rock", "Fire Grass Poison Flying Bug Dragon Steel", ""), "Ice": ("Grass Ground Flying Dragon", "Fire Water Ice Steel", ""),
    "Fighting": ("Normal Ice Rock Dark Steel", "Poison Flying Psychic Bug Fairy", "Ghost"), "Poison": ("Grass Fairy", "Poison Ground Rock Ghost", "Steel"),
    "Ground": ("Fire Electric Poison Rock Steel", "Grass Bug", "Flying"), "Flying": ("Grass Fighting Bug", "Electric Rock Steel", ""),
    "Psychic": ("Fighting Poison", "Psychic Steel", "Dark"), "Bug": ("Grass Psychic Dark", "Fire Fighting Poison Flying Ghost Steel Fairy", ""),
    "Rock": ("Fire Ice Flying Bug", "Fighting Ground Steel", ""), "Ghost": ("Psychic Ghost", "Dark", "Normal"),
    "Dragon": ("Dragon", "Steel", "Fairy"), "Dark": ("Psychic Ghost", "Fighting Dark Fairy", ""),
    "Steel": ("Ice Rock Fairy", "Fire Water Electric Steel", ""), "Fairy": ("Fighting Dragon Dark", "Fire Poison Steel", ""),
}


def effectiveness(atk, defs):
    a, r, im = [set(x.split()) for x in CHART.get(atk, ("", "", ""))]
    v = 1.0
    for t in defs or []:
        v *= 0 if t in im else 2 if t in a else .5 if t in r else 1
    return v


def ptypes(o, p):
    d = (o.get("species_dex") or {}).get(species(p), {})
    return d.get("types", [])


RECOVERY = {"recover", "roost", "slackoff", "softboiled", "milkdrink", "shoreup", "synthesis", "moonlight", "morningsun", "rest"}
SETUP = {"calmmind", "swordsdance", "dragondance", "nastyplot", "shellsmash", "bellydrum", "quiverdance", "agility", "cottonguard", "irondefense", "amnesia", "curse", "coil", "growth", "bulkup", "honeclaws", "workup"}


def move_data(o, mid):
    dex = o.get("move_dex") or {}
    return dex.get(mid) or dex.get(ident(mid)) or {}


def move_score(o, action):
    """A deliberately rough ordering used only when Astra is unavailable."""
    p, q = own_active(o), foe_active(o)
    if not action.startswith("move "):
        return -1000
    try:
        i = int(action.split()[1]) - 1
    except (ValueError, IndexError):
        return -1000
    mm = own_moves(o)
    if not 0 <= i < len(mm):
        return -1000
    m = mm[i]; mid = m.get("id", m.get("move", "")); d = move_data(o, mid)
    if d.get("category") == "Status":
        if ident(mid) in RECOVERY and hpfrac(p) < .42: return 45
        if ident(mid) in SETUP: return 12
        return 8
    typ = d.get("type", "Normal")
    eff = effectiveness(typ, ptypes(o, q))
    if eff == 0: return -250
    bp = float(d.get("base_power") or 55)
    stab = 1.5 if typ in ptypes(o, p) else 1.0
    acc = d.get("accuracy", 100)
    acc = 1 if acc is True else float(acc or 100) / 100
    score = bp * eff * stab * acc
    if eff >= 2: score += 25
    if hpfrac(q) > 0 and score > hpfrac(q) * 100: score += 45
    if "terastallize" in action: score -= 10
    return score


def fallback(o, banned=()):
    legal = [a for a in o.get("legal_actions", []) if a != "noop" and a not in banned]
    if not legal:
        legal = [a for a in o.get("legal_actions", []) if a != "noop"]
    if not legal:
        return None
    rows = []
    for a in legal:
        if a.startswith("move "):
            s = move_score(o, a)
        else:
            try: p = team(o)[int(a.split()[1]) - 1]
            except (ValueError, IndexError): p = {}
            # Healthy, non-fainted teammates are useful; avoid gratuitous switches.
            s = hpfrac(p) * 35 - (8 if p.get("active") else 0)
            if p.get("reviving"): s += 50
        rows.append((s, a))
    return max(rows, key=lambda x: x[0])[1]


def is_opening_barrier(o, action):
    """Unknown foe + no damaging reveal: do not spend a turn on setup/status."""
    if (o.get("turn") or 0) > 2:
        return False
    revealed = foe_active(o).get("moves", [])
    # move_dex is intentionally limited to our moves, so protocol damage is
    # the reliable evidence that a newly revealed opposing move was damaging.
    damaging = any((move_data(o, m).get("category") or "") != "Status" for m in revealed)
    if any("|-damage|p1" in str(line) or "|-damage|p1a:" in str(line) for line in o.get("recent_events", [])):
        damaging = True
    if damaging or not action.startswith("move "):
        return False
    try: i = int(action.split()[1]) - 1
    except (ValueError, IndexError): return True
    mm = own_moves(o)
    return 0 <= i < len(mm) and (move_data(o, mm[i].get("id", "")).get("category") == "Status")


def action_ok(o, action, banned=()):
    legal = o.get("legal_actions", [])
    if action not in legal or action in banned or action == "noop":
        return False
    if is_opening_barrier(o, action):
        return False
    if action.startswith("move "):
        try: i = int(action.split()[1]) - 1
        except (ValueError, IndexError): return False
        mm = own_moves(o)
        if i < 0 or i >= len(mm): return False
        mid = mm[i].get("id", mm[i].get("move", "")); d = move_data(o, mid)
        if hpfrac(own_active(o)) < .20 and d.get("category") == "Status" and ident(mid) not in RECOVERY:
            return False
        if d.get("category") != "Status" and effectiveness(d.get("type", "Normal"), ptypes(o, foe_active(o))) == 0:
            return False
    return True


SYSTEM = """You are playing Gen 9 Random Battle singles against a strong MCTS opponent. Return one compact JSON object only:
{\"actions\":[{\"action\":\"exact legal action\",\"reason\":\"short\"},{\"action\":\"backup legal action\",\"reason\":\"short\"}],\"memory\":\"<=100 words\"}
Rank at most three legal actions. First perform a threat audit: revealed attacks, priority/speed clues, boosts,
hazards, status, and the strongest plausible unrevealed coverage. Preserve a unique check. A setup move is good
only when the active foe cannot punish it and the resulting payoff is concrete. Use Tera only for a decisive KO
or survival. No team preview exists; hidden moves are unknown. Request slots reorder after switches. Protocol and
legal_actions are authoritative. Choose a robust action that still works if the opponent switches. The harness
will reject illegal actions, known immunities, and opening setup before any damaging foe move is revealed.
"""


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.memo = ""
        self.history = []
        self.banned = set()
        self.repeat = None
        self.last_illegal = 0
        self.calls = 0

    def compact(self, o):
        req = o.get("request") or {}
        rows = []
        for i, p in enumerate(team(o), 1):
            rows.append({k: p.get(k) for k in ("ident", "details", "condition", "active", "moves", "ability", "item", "teraType", "terastallized", "reviving")}|{"slot": i, "types": ptypes(o, p)})
        return {"turn": o.get("turn"), "legal_actions": o.get("legal_actions", []), "active": o.get("active"),
                "request": {"forceSwitch": req.get("forceSwitch"), "active": req.get("active"), "side": {"pokemon": rows}},
                "foe": foe_active(o), "opponent_revealed": o.get("opponent_revealed", []),
                "field": (o.get("public_summary") or {}), "recent_events": o.get("recent_events", [])[-35:],
                "move_dex": o.get("move_dex", {}), "species_dex": o.get("species_dex", {}),
                "last_error": o.get("last_error"), "memory": self.memo, "history": self.history[-5:],
                "progress": self.ctx.progress()}

    def parse(self, text, legal):
        objs = []
        dec = json.JSONDecoder()
        s = str(text or "")
        for m in re.finditer(r"\{", s):
            try:
                x, _ = dec.raw_decode(s[m.start():])
                if isinstance(x, dict): objs.append(x)
            except (ValueError, TypeError, RecursionError):
                continue
        for obj in objs:
            arr = obj.get("actions") if isinstance(obj.get("actions"), list) else [obj]
            for row in arr:
                if isinstance(row, dict) and row.get("action") in legal:
                    return row.get("action"), obj
        low = s.lower()
        for a in legal:
            if a.lower() in low:
                return a, {}
        return None, {}

    def update(self, before, action, after):
        pr = self.ctx.progress()
        if after.get("last_error") or pr.get("illegal_actions", 0) > self.last_illegal:
            self.banned.add(action); self.repeat = None
        else:
            self.banned.clear()
        self.last_illegal = pr.get("illegal_actions", 0)
        bp, ap = own_active(before), own_active(after)
        bq, aq = foe_active(before), foe_active(after)
        loss = hpfrac(bp) - hpfrac(ap)
        gain = hpfrac(bq) - hpfrac(aq)
        self.history.append({"turn": before.get("turn"), "action": action, "own": species(bp), "foe": species(bq),
                             "hp_before": [round(hpfrac(bp), 3), round(hpfrac(bq), 3)],
                             "hp_after": [round(hpfrac(ap), 3), round(hpfrac(aq), 3)], "progress": pr})
        self.history = self.history[-10:]
        # One repeat is earned only by an observed, non-lethal, same-matchup hit.
        if action.startswith("move ") and "terastallize" not in action and gain >= .15 and loss < .20 and species(bp) == species(ap) and species(bq) == species(aq):
            self.repeat = {"action": action, "own": bp.get("ident"), "foe": bq.get("ident"), "foe_hp": hpfrac(aq)}
        else:
            self.repeat = None
        if pr.get("since_obs_change", 0) >= 2 or pr.get("since_score_change", 0) >= 4:
            self.repeat = None
        self.ctx.log("verified_action", action=action, source="harness", hp_loss=loss, foe_damage=gain, progress=pr)

    def run_episode(self, game):
        ctx = self.ctx
        try: o = ctx.observe()
        except (BudgetExceeded, ModelUnavailable): return
        for _ in range(220):
            if ctx.done or not isinstance(o, dict) or o.get("done"): return
            legal = [a for a in o.get("legal_actions", []) if a != "noop" and a not in self.banned]
            if not legal:
                self.banned.clear(); legal = [a for a in o.get("legal_actions", []) if a != "noop"]
            if not legal: return
            action = None; source = "fallback"
            if len(legal) == 1:
                action, source = legal[0], "only_legal"
            if action is None and self.repeat:
                r = self.repeat
                if r["action"] in legal and own_active(o).get("ident") == r["own"] and foe_active(o).get("ident") == r["foe"] and hpfrac(own_active(o)) >= .35:
                    action, source = r["action"], "verified_repeat"
                self.repeat = None
            budget = ctx.budget.remaining()
            if action is None and budget.get("system2_calls", 0) > 1 and budget.get("wall_s", 0) > 285:
                try:
                    res = ctx.think(SYSTEM + "\nSTATE=" + json.dumps(self.compact(o), ensure_ascii=False, separators=(",", ":"), default=str), timeout_s=min(260.0, max(20.0, budget.get("wall_s", 0) - 25.0)), session_id=None)
                    self.calls += 1
                    action, data = self.parse(getattr(res, "text", ""), legal)
                    if action and not action_ok(o, action, self.banned):
                        action = None
                    if action:
                        source = "astra"; self.memo = str(data.get("memory", self.memo))[:1200]
                    else:
                        self.ctx.log("model_rejected", reason="malformed or safety barrier")
                except (BudgetExceeded, ModelUnavailable): return
                except (TimeoutError, RuntimeError) as exc:
                    self.ctx.log("model_failure", error=type(exc).__name__)
            if action is None:
                action = fallback(o, self.banned)
            if not action or ctx.done: return
            before = o
            try:
                out = ctx.act(action); o = out if isinstance(out, dict) else ctx.observe()
            except (BudgetExceeded, ModelUnavailable): return
            except Exception as exc:
                self.banned.add(action); self.repeat = None
                self.ctx.log("action_error_recovery", action=action, error=str(exc)[:200])
                if ctx.done: return
                try: o = ctx.observe()
                except Exception: return
                continue
            self.update(before, action, o)


def build(ctx):
    return Harness(ctx)
