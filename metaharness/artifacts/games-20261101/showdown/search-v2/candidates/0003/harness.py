"""Empirical tactical replay: observe a favorable exchange before reusing it.

Only player-visible observations are consumed. Damage estimates are explicitly
rough emergency fallbacks, never claimed to be an engine or a hidden-set oracle.
"""
from __future__ import annotations

import json
import math
import re
from collections import deque

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


def ident(x):
    return re.sub(r"^(p[12])[a-z]:", r"\1:", str(x or ""))


def norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x or "").lower())


def species(p):
    return p.get("details", "").split(",")[0]


def hp(p):
    s = str(p.get("condition", "0"))
    if "fnt" in s:
        return 0.0
    try:
        a = s.split()[0].split("/")
        return float(a[0]) / float(a[1]) if len(a) > 1 else float(a[0]) / 100
    except (ValueError, ZeroDivisionError, IndexError):
        return 0.0


def status(p):
    a = str(p.get("condition", "")).split()
    return a[1] if len(a) > 1 else p.get("status", "")


def team(o):
    return (o.get("request") or {}).get("side", {}).get("pokemon", [])


def active(o):
    return next((p for p in team(o) if p.get("active")), {})


def enemy(o):
    who = o.get("active", {}).get("p2")
    return next((p for p in o.get("opponent_revealed", []) if p.get("ident") == who),
                o.get("public_summary", {}).get("pokemon", {}).get(who, {}))


def legal(o):
    return [a for a in o.get("legal_actions", []) if isinstance(a, str) and a != "noop"]


def events(o, initial=False):
    x = o.get("battle_log" if initial else "recent_events", [])
    return x.splitlines() if isinstance(x, str) else x if isinstance(x, list) else []


def parse_json(text):
    if not isinstance(text, str):
        return {}
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            value, _ = dec.raw_decode(text[m.start():])
            if isinstance(value, dict) and "action" in value:
                return value
        except (ValueError, TypeError):
            pass
    # Garbage / plain exact action is deliberately a ONE-step plan.
    choices = re.findall(r"\b(?:move [1-4](?: terastallize)?|switch [1-6])\b", text.lower())
    return {"action": choices[-1]} if choices else {}


# Standard type table; all rows are attack types. Ability immunities below are
# applied only when actually revealed, except our own exact private ability.
TYPE = {
    "Normal": ("", "Rock Steel", "Ghost"),
    "Fire": ("Grass Ice Bug Steel", "Fire Water Rock Dragon", ""),
    "Water": ("Fire Ground Rock", "Water Grass Dragon", ""),
    "Electric": ("Water Flying", "Electric Grass Dragon", "Ground"),
    "Grass": ("Water Ground Rock", "Fire Grass Poison Flying Bug Dragon Steel", ""),
    "Ice": ("Grass Ground Flying Dragon", "Fire Water Ice Steel", ""),
    "Fighting": ("Normal Ice Rock Dark Steel", "Poison Flying Psychic Bug Fairy", "Ghost"),
    "Poison": ("Grass Fairy", "Poison Ground Rock Ghost", "Steel"),
    "Ground": ("Fire Electric Poison Rock Steel", "Grass Bug", "Flying"),
    "Flying": ("Grass Fighting Bug", "Electric Rock Steel", ""),
    "Psychic": ("Fighting Poison", "Psychic Steel", "Dark"),
    "Bug": ("Grass Psychic Dark", "Fire Fighting Poison Flying Ghost Steel Fairy", ""),
    "Rock": ("Fire Ice Flying Bug", "Fighting Ground Steel", ""),
    "Ghost": ("Psychic Ghost", "Dark", "Normal"),
    "Dragon": ("Dragon", "Steel", "Fairy"),
    "Dark": ("Psychic Ghost", "Fighting Dark Fairy", ""),
    "Steel": ("Ice Rock Fairy", "Fire Water Electric Steel", ""),
    "Fairy": ("Fighting Dragon Dark", "Fire Poison Steel", ""),
}
RECOVER = set("recover roost slackoff softboiled milkdrink moonlight morningsun synthesis shoreup healorder strengthsap".split())
PIVOT = set("uturn voltswitch flipturn partingshot chillyreception teleport shedtail batonpass".split())
CHARGE = set("meteorbeam electroshot solarbeam solarblade skullbash skyattack fly bounce dig dive phantomforce shadowforce".split())


def types(o, p, offensive=False):
    ts = o.get("species_dex", {}).get(species(p), {}).get("types", [])
    tera = p.get("terastallized") or p.get("teraTypeRevealed")
    return ts if offensive or not tera or tera == "Stellar" else [tera]


def effectiveness(o, move_type, p, attacker=None):
    ability = norm(p.get("ability", p.get("baseAbility", "")))
    bypass = norm((attacker or {}).get("ability")) in ("moldbreaker", "teravolt", "turboblaze")
    if not bypass:
        immune = {"Ground": ("levitate", "eartheater"),
                  "Water": ("waterabsorb", "stormdrain", "dryskin"),
                  "Electric": ("voltabsorb", "lightningrod", "motordrive"),
                  "Fire": ("flashfire", "wellbakedbody"), "Grass": ("sapsipper",)}
        if ability in immune.get(move_type, ()):
            return 0.0
    sup, resist, immune = [s.split() for s in TYPE.get(move_type, ("", "", ""))]
    mult = 1.0
    for t in types(o, p):
        mult *= 0 if t in immune else 2 if t in sup else .5 if t in resist else 1
    if ability == "wonderguard" and mult <= 1 and not bypass:
        return 0.0
    return mult


def boost(n):
    n = max(-6, min(6, int(n or 0)))
    return (2 + n) / 2 if n >= 0 else 2 / (2 - n)


def level(p):
    m = re.search(r"(?:^|, )L(\d+)", p.get("details", ""))
    return int(m.group(1)) if m else 80


def est_stat(o, p, name):
    if name in p.get("stats", {}):
        return max(1, p["stats"][name])
    b = o.get("species_dex", {}).get(species(p), {}).get("base_stats", {}).get(name, 85)
    # Random Battle often uses 85 EVs; hidden sets may differ. This is only a
    # ranking estimate, not an exact KO certificate.
    return ((2 * b + 31 + 21) * level(p) // 100) + (level(p) + 10 if name == "hp" else 5)


def maxhp(o, p):
    try:
        return int(str(p.get("condition", "")).split()[0].split("/")[1]) if str(p.get("ident", "")).startswith("p1") else est_stat(o, p, "hp")
    except (IndexError, ValueError):
        return est_stat(o, p, "hp")


def damage(o, user, foe, mid, ledger):
    """Approximate mean damage fraction, with known type/item/boost effects."""
    m = o.get("move_dex", {}).get(norm(mid), {})
    if not m or m.get("category") == "Status":
        return 0.0
    category = m.get("category", "Physical")
    mt = m.get("type", "Normal")
    tera = user.get("terastallized") or user.get("teraTypeRevealed")
    ub = ledger.get(user.get("ident"), {}).get("boosts", user.get("boosts", {}))
    fb = ledger.get(foe.get("ident"), {}).get("boosts", foe.get("boosts", {}))
    if norm(mid) == "terablast" and tera:
        mt = tera
        if est_stat(o, user, "atk") * boost(ub.get("atk", 0)) > est_stat(o, user, "spa") * boost(ub.get("spa", 0)):
            category = "Physical"
    eff = effectiveness(o, mt, foe, user)
    if not eff:
        return 0.0
    bp = float(m.get("base_power", 0) or 0)
    if norm(mid) in ("seismictoss", "nightshade"):
        return level(user) / maxhp(o, foe)
    if norm(mid) in ("superfang", "naturesmadness", "ruination"):
        return .5 * hp(foe)
    if norm(mid) in ("counter", "mirrorcoat", "metalburst", "comeuppance"):
        return 0.02  # cannot presume the needed incoming hit
    if norm(mid) in ("grassknot", "lowkick", "heavyslam", "heatcrash"):
        bp = 80  # weight is not supplied
    if norm(mid) in ("storedpower", "powertrip"):
        bp = 20 + 20 * sum(max(0, n) for n in ub.values())
    if norm(mid) in ("eruption", "waterspout", "dragonenergy"):
        bp = 150 * hp(user)
    if norm(mid) == "knockoff" and foe.get("item", "unknown"):
        bp *= 1.5
    if norm(mid) in ("bulletseed", "rockblast", "iciclespear", "tailslap", "pinmissile", "scaleshot"):
        bp *= 4.5 if norm(user.get("ability")) == "skilllink" else 4 if norm(user.get("item")) == "loadeddice" else 3
    attack = "atk" if category == "Physical" else "spa"
    defense = "def" if category == "Physical" or norm(mid) in ("psyshock", "psystrike", "secretsword") else "spd"
    if norm(mid) == "bodypress":
        attack = "def"
    a = est_stat(o, user, attack) * boost(ub.get(attack, 0))
    d = est_stat(o, foe, defense) * boost(fb.get(defense, 0))
    if norm(foe.get("ability")) == "unaware":
        a = est_stat(o, user, attack)
    if norm(user.get("ability")) == "unaware":
        d = est_stat(o, foe, defense)
    if norm(mid) == "foulplay":
        a = est_stat(o, foe, "atk") * boost(fb.get("atk", 0))
    item, ability = norm(user.get("item")), norm(user.get("ability"))
    if (attack == "atk" and item == "choiceband") or (attack == "spa" and item == "choicespecs"):
        a *= 1.5
    if ability in ("hugepower", "purepower") and attack == "atk":
        a *= 2
    if status(user) == "brn" and category == "Physical" and ability != "guts" and norm(mid) != "facade":
        a *= .5
    if status(user) and ability == "guts" and attack == "atk":
        a *= 1.5
    if norm(foe.get("item")) == "assaultvest" and defense == "spd":
        d *= 1.5
    original = types(o, user, offensive=True)
    stab = (2 if mt in original else 1.5) if tera == mt else 1.5 if mt in original else 1
    if ability == "adaptability" and stab > 1:
        stab += .5
    modifier = 1.3 if item == "lifeorb" else 1.0
    if ability == "waterbubble" and mt == "Water":
        modifier *= 2
    if ability == "technician" and bp <= 60:
        modifier *= 1.5
    weather = norm(o.get("public_summary", {}).get("weather"))
    if weather in ("raindance", "primordialsea"):
        modifier *= 1.5 if mt == "Water" else .5 if mt == "Fire" else 1
    if weather in ("sunnyday", "desolateland"):
        modifier *= 1.5 if mt == "Fire" else .5 if mt == "Water" else 1
    if norm(mid) in CHARGE and item != "powerherb":
        modifier *= .50
    accuracy = m.get("accuracy", 100)
    accuracy = 1.0 if accuracy is True else float(accuracy or 100) / 100
    if ability == "noguard" or norm(foe.get("ability")) == "noguard":
        accuracy = 1.0
    return max(0, (((2 * level(user) / 5 + 2) * bp * a / max(d, 1) / 50 + 2)
                   * .925 * stab * eff * modifier * accuracy / maxhp(o, foe)))


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ledger = {}
        self.recent = deque(maxlen=90)
        self.outcomes = deque(maxlen=10)
        self.certificate = None
        self.memo = ""
        self.reason = "Opening: identify win condition and safe progress."
        self.age = 0
        self.calls = 0
        self.steps = 0
        self.failure_streak = 0
        self.blocked = set()
        self.last_exchange = None
        self.no_gain = 0

    def ingest(self, o, initial=False):
        for line in events(o, initial):
            if not isinstance(line, str):
                continue
            fields = line.split("|")
            if len(fields) < 2:
                continue
            tag = fields[1]
            if tag not in ("", "upkeep", "t:", "request"):
                self.recent.append(line)
            if tag == "-clearallboost":
                for value in self.ledger.values():
                    value["boosts"] = {}
                continue
            if len(fields) < 3:
                continue
            who = ident(fields[2])
            p = self.ledger.setdefault(who, {"boosts": {}}) if who.startswith(("p1:", "p2:")) else {}
            if tag in ("switch", "drag"):
                p["boosts"] = {}
            elif tag in ("-boost", "-unboost", "-setboost") and len(fields) >= 5:
                try:
                    n = int(fields[4]); stat = fields[3]
                    old = p.setdefault("boosts", {}).get(stat, 0)
                    p["boosts"][stat] = max(-6, min(6, n if tag == "-setboost" else old + n * (-1 if tag == "-unboost" else 1)))
                except (ValueError, TypeError):
                    pass
            elif tag == "-clearboost":
                p["boosts"] = {}
            elif tag == "-clearnegativeboost":
                p["boosts"] = {k: max(0, v) for k, v in p.get("boosts", {}).items()}
            elif tag in ("-ability", "-item", "-enditem") and len(fields) >= 4:
                p["ability" if tag == "-ability" else "item"] = "" if tag == "-enditem" else fields[3]
            elif tag == "-terastallize" and len(fields) >= 4:
                p["teraTypeRevealed"] = fields[3]
            if tag in ("-damage", "-heal"):
                for extra in fields[4:]:
                    if extra.startswith("[from] item:"):
                        p["item"] = extra.split(":", 1)[1].strip()
                    elif extra.startswith("[from] ability:"):
                        p["ability"] = extra.split(":", 1)[1].strip()
    def enrich(self, o, p):
        result = dict(p)
        result.update(self.ledger.get(p.get("ident"), {}))
        # Exact private request overrides historical item/ability/tera data.
        if str(p.get("ident", "")).startswith("p1:"):
            for k in ("item", "ability", "terastallized"):
                if k in p:
                    result[k] = p[k]
        return result

    def replay_action(self, o):
        c = self.certificate
        if not c or self.age >= 5 or o.get("last_error"):
            return None
        me, foe = active(o), enemy(o)
        if c["self"] != me.get("ident") or c["foe"] != foe.get("ident"):
            return None
        action = c["action"]
        if action not in legal(o) or action in self.blocked:
            return None
        moves = (o.get("request") or {}).get("active", [])
        moves = moves[0].get("moves", []) if moves else []
        i = int(action.split()[1]) - 1
        if i >= len(moves) or norm(moves[i].get("id")) != c["move"]:
            return None
        # Samples include rounded enemy HP and stochastic damage. Widen both
        # sides before estimating how many more exchanges are survivable.
        outgoing = max(0, c["direct_out"] * .75 - .02)
        progress = max(0, c["net_out"] * .75 - .02)
        incoming = c["incoming"] * 1.30 + .035
        first = c["first"] == "p1"
        if outgoing < .035 or progress < .025:
            return None
        if first and outgoing >= hp(foe) + .01:
            return action
        if hp(me) <= incoming + .025:
            return None
        # Require a favorable race, with one spare exchange when slower.
        turns_to_ko = math.ceil(hp(foe) / progress)
        turns_alive = math.ceil(max(0, hp(me) - .025) / incoming)
        if turns_to_ko < turns_alive or (first and turns_to_ko <= turns_alive):
            return action
        return None

    def exchange(self, before, after, action):
        if not action.startswith("move ") or "terastallize" in action:
            return None
        me, foe = active(before), enemy(before)
        if me.get("ident") != active(after).get("ident") or foe.get("ident") != enemy(after).get("ident"):
            return None
        moves = (before.get("request") or {}).get("active", [])
        moves = moves[0].get("moves", []) if moves else []
        i = int(action.split()[1]) - 1
        if i >= len(moves):
            return None
        mid = norm(moves[i].get("id", moves[i].get("move")))
        md = before.get("move_dex", {}).get(mid, {})
        if md.get("category") not in ("Physical", "Special") or mid in CHARGE or mid in PIVOT:
            return None
        if mid in ("counter", "mirrorcoat", "metalburst", "comeuppance", "suckerpunch", "upperhand", "fakeout", "firstimpression", "futuresight", "doomdesire", "endeavor"):
            return None
        identities = {me.get("ident"): hp(me), foe.get("ident"): hp(foe)}
        direct_out = direct_in = 0.0
        order = []
        last_attacker = ""
        own_move_seen = False
        enemy_attack_immune = False
        for line in events(after):
            f = line.split("|")
            if len(f) < 3:
                continue
            tag, target = f[1], ident(f[2])
            if tag == "move":
                last_attacker = target
                order.append(target[:2])
                if target == me.get("ident") and len(f) > 3 and norm(f[3]) == mid:
                    own_move_seen = True
            if tag == "-immune" and target == me.get("ident") and last_attacker == foe.get("ident"):
                enemy_attack_immune = True
            if tag in ("-crit", "-miss", "-activate", "-enditem", "-prepare"):
                # Critical rolls, one-use damage modifiers, substitutes,
                # protection and ability triggers are not stable calibration.
                return None
            if tag in ("-damage", "-heal") and len(f) > 3 and target in identities:
                value = hp({"condition": f[3]})
                removed = max(0, identities[target] - value)
                if tag == "-damage" and not any("[from]" in x for x in f[4:]):
                    if target == foe.get("ident") and last_attacker == me.get("ident"):
                        direct_out += removed
                    if target == me.get("ident") and last_attacker == foe.get("ident"):
                        direct_in += removed
                identities[target] = value
        if not own_move_seen or "p1" not in order or "p2" not in order or direct_out <= .03:
            return None
        if direct_in <= 0 and not enemy_attack_immune:
            return None  # a recovery/setup turn does not measure attack safety
        old = self.certificate or {}
        incoming = max(direct_in, max(0, hp(me) - hp(active(after))))
        if old.get("self") == me.get("ident") and old.get("foe") == foe.get("ident"):
            incoming = max(incoming, old.get("incoming", 0))
        return {"self": me.get("ident"), "foe": foe.get("ident"), "action": action,
                "move": mid, "first": order[0], "incoming": incoming,
                "direct_out": direct_out, "net_out": hp(foe) - hp(enemy(after))}

    def fallback(self, o):
        """Emergency ranking, used only when Astra cannot supply a fresh choice."""
        actions = [x for x in legal(o) if x not in self.blocked] or legal(o)
        if not actions:
            return None
        me, foe = self.enrich(o, active(o)), self.enrich(o, enemy(o))
        moves = (o.get("request") or {}).get("active", [])
        moves = moves[0].get("moves", []) if moves else []
        ranked = []
        for a in actions:
            if not a.startswith("move ") or "terastallize" in a:
                continue
            idx = int(a.split()[1]) - 1
            if idx >= len(moves):
                continue
            mid = norm(moves[idx].get("id", moves[idx].get("move")))
            val = damage(o, me, foe, mid, self.ledger)
            if mid in RECOVER and hp(me) < .55:
                val = .24 + (.55 - hp(me)) * .3
            if mid == "rest" and hp(me) < .3 and status(me) != "slp":
                val = .16
            if mid == "sleeptalk" and status(me) == "slp":
                val = .35
            if mid in ("haze", "clearsmog") and sum(max(0, x) for x in foe.get("boosts", {}).values()) >= 3:
                val += .35
            if mid in ("toxic", "willowisp", "thunderwave", "spore", "sleeppowder") and not status(foe):
                val += .12
            if mid in ("stealthrock", "spikes"):
                cond = o.get("public_summary", {}).get("side_conditions", {}).get("p2", {})
                if not any(norm(k) == mid for k in cond):
                    val += .13
            # Known priority can convert a low-HP opponent into a safe finish.
            md = o.get("move_dex", {}).get(mid, {})
            if val >= hp(foe) + .03 and md.get("priority", 0) > 0:
                val += .25
            if self.outcomes and self.outcomes[-1].get("action") == a and self.no_gain >= 2:
                val -= .3
            ranked.append((val, a))
        if ranked:
            return max(ranked)[1]
        # Forced replacement: rank surviving entrants by best attack and resist
        # profile, using only currently visible enemy information.
        for a in actions:
            if not a.startswith("switch "):
                continue
            i = int(a.split()[1]) - 1
            if i >= len(team(o)):
                continue
            p = self.enrich(o, team(o)[i])
            if hp(p) == 0:  # Revival Blessing explicitly exposes these choices.
                val = sum(p.get("stats", {}).values()) / 1000
            else:
                best = max([damage(o, p, foe, m, self.ledger) for m in p.get("moves", [])] or [0])
                weaknesses = max([effectiveness(o, t, p, foe) for t in types(o, foe, True)] or [1])
                val = best + .35 * hp(p) - .12 * weaknesses
                hazards = o.get("public_summary", {}).get("side_conditions", {}).get("p1", {})
                if norm(p.get("item")) != "heavydutyboots" and any(norm(k) == "stealthrock" for k in hazards):
                    val -= .125 * effectiveness(o, "Rock", p)
            ranked.append((val, a))
        return max(ranked)[1] if ranked else actions[0]

    def prompt(self, o, progress):
        own = []
        for i, p in enumerate(team(o), 1):
            p = self.enrich(o, p)
            own.append({"slot": i, **{k: p[k] for k in ("ident", "details", "condition", "active", "stats", "moves", "item", "ability", "teraType", "terastallized", "boosts") if k in p}})
        request = (o.get("request") or {})
        summary = o.get("public_summary", {})
        snap = {
            "turn": o.get("turn"), "active": o.get("active"), "legal_actions": legal(o),
            "request_flags": {k: v for k, v in request.items() if k != "side"},
            "team": own,
            "opponents": [self.enrich(o, p) for p in o.get("opponent_revealed", [])],
            "public_effects": {k: v for k, v in summary.items() if k != "pokemon"},
            "active_public": {k: v for k, v in summary.get("pokemon", {}).items() if k in o.get("active", {}).values()},
            "species_dex": o.get("species_dex", {}), "move_dex": o.get("move_dex", {}),
            "last_error": o.get("last_error"), "blocked_actions": sorted(self.blocked),
            "verified_outcomes": list(self.outcomes)[-5:], "empirical_exchange": self.last_exchange,
        }
        # Never slice an observation JSON: all own sets, revealed opponents,
        # legal commands, mechanics references and latest events remain present.
        return (
            "Play Pokemon Showdown Gen 9 Random Battle singles as p1 against a strong search bot. "
            "Win by fainting all six. Only the supplied observations are available; no preview or hidden sets. "
            "Exact own HP/stats and rounded opponent HP are given. Protocol overrides summaries. "
            "Consider strongest opponent reply (attack/switch/setup), speed/priority, ability immunities, hazards, "
            "status, healing and Tera. Preserve the unique counter to a revealed sweeper; stop passive losing trades. "
            "Tera should change a decisive survival/KO or enable a sweep; don't save it forever.\n"
            "Return compact JSON only: {\"action\":\"move 1\", \"memo\":\"short strategic memory\"}. "
            "action must be exactly ONE current legal command, including tera if desired. "
            "memo <=40 words records win condition, preserved counter, important hidden-set inference and uncertainty. "
            "Choose the strongest current tactical action. Do not output an action sequence or reusable rules. "
            "Check observed HP exchanges: if the faster foe's last hit exceeds our current HP, a repeat attack "
            "needs a concrete reason (priority, Tera, recovery, changed state); otherwise switch or sacrifice deliberately. "
            "Code may repeat a damaging attack after observing a favorable exchange with the same opponent: "
            "it checks conservative damage margins, observed move order, current HP/PP, and changed mechanics after EVERY turn. "
            "Any unsafe or novel situation returns here. Your previous memo is advice, not a commitment.\n"
            f"Reason for consultation: {self.reason}\nMemory: {self.memo}\n"
            f"Progress (score deliberately omitted): {json.dumps({k:v for k,v in progress.items() if k != 'score'})}\n"
            f"Budget: {json.dumps(self.ctx.budget.remaining())}\n"
            "CURRENT STATE:\n" + json.dumps(snap, separators=(",", ":"), ensure_ascii=False) +
            "\nRECENT AUTHORITATIVE PROTOCOL:\n" + "\n".join(list(self.recent)[-65:])
        )

    def assess(self, before, after, action, bp, ap):
        """Verify each action and calibrate a possible repeat from real damage."""
        me0, me1, e0, e1 = active(before), active(after), enemy(before), enemy(after)
        same_us = me0.get("ident") == me1.get("ident")
        same_foe = e0.get("ident") == e1.get("ident")
        entrant_before = next((p for p in team(before) if p.get("ident") == me1.get("ident")), me0)
        loss = max(0, hp(entrant_before) - hp(me1))
        dealt = hp(e0) - hp(e1) if same_foe else None
        self.outcomes.append({"action": action, "self": species(me0), "foe": species(e0),
                              "own_hp_loss": round(loss, 3), "enemy_net_hp_loss": None if dealt is None else round(dealt, 3),
                              "turn": before.get("turn"), "error": after.get("last_error")})
        self.age += 1
        self.last_exchange = self.exchange(before, after, action)
        cause = []
        rejected = bool(after.get("last_error")) or ap.get("illegal_actions", 0) > bp.get("illegal_actions", 0)
        if rejected:
            self.blocked.add(action)
            cause.append("action rejected: " + str(after.get("last_error")))
        else:
            self.blocked.clear()
        if ap.get("since_obs_change", 0) > 0:
            cause.append("observation did not change")
        if not same_foe or not same_us:
            cause.append("active matchup changed")
        if status(me0) != status(me1) or status(e0) != status(e1):
            cause.append("status changed")
        if self.certificate and loss > self.certificate["incoming"] * 1.30 + .05:
            cause.append("damage exceeded the empirical envelope")
        for line in events(after):
            f = line.split("|")
            if len(f) < 2:
                continue
            tag = f[1]
            target = f[2] if len(f) > 2 else ""
            if tag in ("-terastallize", "replace", "detailschange", "-formechange", "-transform", "-swapboost", "-copyboost", "-invertboost", "-clearallboost"):
                cause.append("type/form/boost mechanics changed")
            if tag in ("-boost", "-unboost", "-setboost", "-clearboost"):
                cause.append("stat stages changed")
            if tag in ("-start", "-end", "-fieldstart", "-fieldend", "-sidestart", "-sideend"):
                cause.append("field/volatile condition changed")
            if tag == "-weather" and "[upkeep]" not in line:
                cause.append("weather changed")
            if (tag == "-immune" and target.startswith("p2")) or tag == "-fail":
                cause.append("our action failed or hit immunity")
            if tag in ("-ability", "-item", "-enditem"):
                cause.append("ability/item information changed")
        if same_foe and action.startswith("move") and dealt is not None and dealt <= .01:
            self.no_gain += 1
        else:
            self.no_gain = 0
        if self.no_gain >= 2 or ap.get("since_score_change", 0) >= 4:
            cause.append("no enemy HP progress / progress counter stalled")
        if cause:
            self.certificate = None
            self.reason = "; ".join(dict.fromkeys(cause))
        else:
            self.certificate = self.last_exchange
            self.reason = "Observed exchange does not justify another safe attack, or replay limit reached."

    def run_episode(self, game):
        ctx = self.ctx
        o = ctx.observe()
        if not isinstance(o, dict):
            return
        self.ingest(o, initial=True)
        while not ctx.done and not o.get("done"):
            choices = legal(o)
            if not choices:
                return
            # Check counters before every step, including fallback operation.
            progress = ctx.progress()
            action = self.replay_action(o)
            source = "empirical_replay"
            if len(choices) == 1 and not o.get("last_error"):
                action, source = choices[0], "only_legal"
            if action is None:
                rem = ctx.budget.remaining()
                # Keep one call and ~2 minutes unspent so reaching a call limit
                # cannot prematurely close a battle that code can finish.
                can_think = rem.get("system2_calls", 0) > 1 and (rem.get("wall_s", 0) > 240 or self.calls == 0) and self.calls < 28
                if can_think:
                    self.calls += 1
                    self.age = 0
                    try:
                        result = ctx.think(self.prompt(o, progress), session_id=None, timeout_s=180.0)
                        self.certificate = None
                        data = parse_json(result.text)
                        proposed = data.get("action")
                        if isinstance(proposed, str) and proposed.strip().lower() in choices and proposed.strip().lower() not in self.blocked:
                            action = proposed.strip().lower()
                            self.memo = str(data.get("memo", self.memo))[:900]
                            self.failure_streak = 0
                            source = "astra"
                        else:
                            self.reason = "Malformed or illegal model answer; review next observed outcome."
                            self.failure_streak += 1
                    except (BudgetExceeded, ModelUnavailable):
                        return
                    except (TimeoutError, RuntimeError):
                        self.certificate = None
                        self.reason = "Astra timed out or failed; emergency legal action was required."
                        self.failure_streak += 1
                if action is None:
                    action, source = self.fallback(o), "emergency"
            if action not in legal(o):
                self.certificate = None
                self.reason = "Proposed action is no longer legal."
                action = self.fallback(o)
                if action is None:
                    return
            before = o
            try:
                ctx.log("policy_action", source=source, action=action, certificate=self.certificate, reason=self.reason)
                after = ctx.act(action)
            except (BudgetExceeded, ModelUnavailable):
                return
            except Exception as exc:
                self.certificate = None
                self.blocked.add(action)
                self.reason = "Action exception: " + str(exc)[:250]
                self.failure_streak += 1
                if ctx.done or self.failure_streak >= 4:
                    return
                o = ctx.observe()
                if not isinstance(o, dict):
                    return
                continue
            o = after if isinstance(after, dict) else ctx.observe()
            if not isinstance(o, dict):
                return
            self.steps += 1
            if o.get("done") or ctx.done:
                return
            self.ingest(o)
            self.assess(before, o, action, progress, ctx.progress())
            ctx.memory["battle_memo"] = self.memo
            ctx.memory["recent_outcomes"] = list(self.outcomes)
            if o.get("last_error"):
                self.failure_streak += 1
                if self.failure_streak >= 4:
                    return
            else:
                self.failure_streak = 0


def build(ctx):
    return Harness(ctx)
