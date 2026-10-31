"""Resource-plan Catan controller.

This controller is deliberately deterministic.  It treats the adaptor's legal
list as authoritative, but keeps a small resource and route model so that
maritime trades are made for a build that is actually reachable.  The model is
kept out of the low-level loop: a stale or malformed model answer cannot block
a turn-game episode.
"""
from __future__ import annotations

import json

PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
RES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
COSTS = {
    "build_road": {"WOOD": 1, "BRICK": 1},
    "build_settlement": {"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1},
    "build_city": {"ORE": 3, "WHEAT": 2},
    "buy_development_card": {"ORE": 1, "SHEEP": 1, "WHEAT": 1},
}


def _obj(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            y = json.loads(x)
            return y if isinstance(y, dict) else {}
        except Exception:
            return {}
    return {}


def _acts(o):
    xs = o.get("legal_actions", []) if isinstance(o, dict) else []
    return [a for a in xs if isinstance(a, dict) and a.get("id") is not None]


def _vertex_id(a):
    p = a.get("parameters")
    if isinstance(p, int):
        return p
    if isinstance(p, (list, tuple)):
        for x in p:
            if isinstance(x, int):
                return x
    return None


def _endpoints(a):
    p = a.get("parameters")
    if isinstance(p, (list, tuple)):
        return [x for x in p if isinstance(x, int)]
    return []


def _board(o):
    b = o.get("board", {}) if isinstance(o, dict) else {}
    hs = b.get("hexes", []) if isinstance(b.get("hexes", []), list) else []
    vs = b.get("vertices", []) if isinstance(b.get("vertices", []), list) else []
    return b, hs, {v.get("id"): v for v in vs if isinstance(v, dict)}


def _vertex_score(v, hs, missing=()):
    if not isinstance(v, dict):
        return -100.0
    total = 0.0
    kinds = set()
    for i in v.get("adjacent_hexes", []):
        if not isinstance(i, int) or i < 0 or i >= len(hs):
            continue
        h = hs[i] if isinstance(hs[i], dict) else {}
        terrain, number = h.get("terrain"), h.get("number")
        if not terrain or not isinstance(number, (int, float)) or h.get("robber"):
            continue
        kinds.add(terrain)
        total += PIPS.get(int(number), 0)
        if terrain in ("WHEAT", "ORE"):
            total += 0.55
        if terrain in missing:
            total += 1.4
    total += 0.75 * len(kinds)
    if v.get("port"):
        total += 0.6
    return total


def _my_terrains(o, hs, vertices):
    out = set()
    for v in vertices.values():
        if v.get("owner") == "RED" and v.get("building") in ("SETTLEMENT", "CITY"):
            for i in v.get("adjacent_hexes", []):
                if isinstance(i, int) and 0 <= i < len(hs):
                    t = hs[i].get("terrain") if isinstance(hs[i], dict) else None
                    if t:
                        out.add(t)
    return out


def _initial_settlement(o, acts, memory):
    b, hs, vertices = _board(o)
    # During the snake placement, the second site should complement the first
    # rather than merely duplicate its highest pip intersection.
    old = _my_terrains(o, hs, vertices)
    missing = tuple(t for t in RES if t not in old)
    scored = []
    for a in acts:
        if "settlement" not in str(a.get("type", "")):
            continue
        v = vertices.get(_vertex_id(a), {})
        s = _vertex_score(v, hs, missing)
        if old:
            kinds = set()
            for i in v.get("adjacent_hexes", []):
                if isinstance(i, int) and 0 <= i < len(hs):
                    t = hs[i].get("terrain") if isinstance(hs[i], dict) else None
                    if t:
                        kinds.add(t)
            # Brick is especially valuable for the first expansion and ore
            # prevents an all-wood/sheep opening from becoming trade-locked.
            if "BRICK" in kinds and "BRICK" not in old:
                s += 2.1
            if "ORE" in kinds and "ORE" not in old:
                s += 1.7
            s += 1.1 * len(kinds - old)
        scored.append((s, str(a.get("id")), a))
    return max(scored, key=lambda z: (z[0], z[1]))[2] if scored else (acts[0] if acts else None)


def _initial_road(o, acts):
    _, hs, vertices = _board(o)
    scored = []
    for a in acts:
        if "road" not in str(a.get("type", "")):
            continue
        ends = _endpoints(a)
        s = 0.0
        for e in ends:
            s += _vertex_score(vertices.get(e, {}), hs)
            v = vertices.get(e, {})
            if v.get("building") is None:
                s += 0.5
        scored.append((s, str(a.get("id")), a))
    return max(scored, key=lambda z: (z[0], z[1]))[2] if scored else (acts[0] if acts else None)


def _trade_parts(a):
    p = a.get("parameters")
    if not isinstance(p, (list, tuple)) or len(p) < 2:
        return None, None
    # Catanatron represents 4:1 as four inputs plus output, and a 3:1/2:1
    # trade with None placeholders.  The output is always the final value.
    out = p[-1] if isinstance(p[-1], str) else None
    ins = [x for x in p[:-1] if isinstance(x, str) and x in RES]
    return (ins[0] if ins else None), out


def _trade_score(o, a, acts):
    """Value a trade by the best build it enables after the trade."""
    selfv = o.get("self", {}) if isinstance(o, dict) else {}
    have = selfv.get("resources", {}) if isinstance(selfv.get("resources", {}), dict) else {}
    ins, out = _trade_parts(a)
    if not ins or out not in RES or have.get(ins, 0) < len([x for x in (a.get("parameters") or [])[:-1] if x == ins]):
        return -1000.0
    nxt = {r: int(have.get(r, 0) or 0) for r in RES}
    for _ in range(len([x for x in (a.get("parameters") or [])[:-1] if x == ins])):
        nxt[ins] -= 1
    nxt[out] += 1
    best = -1.0
    for typ, cost in COSTS.items():
        deficit = sum(max(0, n - nxt.get(r, 0)) for r, n in cost.items())
        # A trade completing a VP action is strong; a road is useful only when
        # it opens a productive settlement site.
        weight = {"build_city": 8.0, "build_settlement": 7.0, "build_road": 2.5, "buy_development_card": 1.2}[typ]
        best = max(best, weight - 1.8 * deficit)
    # Avoid cycling a surplus into a resource that does not reduce any deficit.
    return best


def _robber(o, acts):
    b, hs, vertices = _board(o)
    opp = {q.get("id") for q in o.get("opponents", []) if isinstance(q, dict)}
    scored = []
    for a in acts:
        if a.get("type") != "move_robber":
            continue
        p = a.get("parameters")
        coord = p[0] if isinstance(p, list) and p else None
        victim = p[1] if isinstance(p, list) and len(p) > 1 else None
        h = next((x for x in hs if isinstance(x, dict) and x.get("coordinate") == coord), {})
        pressure = PIPS.get(h.get("number"), 0) * (2.0 if victim in opp else 0.15)
        for vid in h.get("vertices", []):
            v = vertices.get(vid, {})
            if v.get("owner") in opp:
                pressure += 1.5 if v.get("building") == "CITY" else 0.75
        for q in o.get("opponents", []):
            if isinstance(q, dict) and q.get("id") == victim:
                pressure += min(6, float(q.get("resource_count", 0) or 0)) * 0.25
                pressure += float(q.get("public_vp", 0) or 0) * 0.15
        scored.append((pressure, str(a.get("id")), a))
    return max(scored, key=lambda z: (z[0], z[1]))[2] if scored else (acts[0] if acts else None)


def _normal(o, acts):
    selfv = o.get("self", {}) if isinstance(o, dict) else {}
    true_vp = float(selfv.get("true_vp", selfv.get("public_vp", 0)) or 0)
    b, hs, vertices = _board(o)
    rank = {
        "build_city": 110, "build_settlement": 106, "build_road": 72,
        "play_year_of_plenty": 68, "play_road_building": 67,
        "play_monopoly": 63, "play_knight_card": 60,
        "buy_development_card": 45, "maritime_trade": 20,
        "roll": 10, "end_turn": 0, "discard": 130,
    }
    # If at nine points, every legal VP-producing build beats development
    # cards and speculative trades.
    if true_vp >= 9:
        rank["build_city"] += 30
        rank["build_settlement"] += 30
    candidates = []
    for a in acts:
        typ = str(a.get("type", ""))
        val = rank.get(typ, 5)
        if typ == "build_city":
            val += _vertex_score(vertices.get(_vertex_id(a), {}), hs) * 1.5
        elif typ == "build_settlement":
            val += _vertex_score(vertices.get(_vertex_id(a), {}), hs) * 1.3
        elif typ == "build_road":
            ends = _endpoints(a)
            val += max((_vertex_score(vertices.get(e, {}), hs) for e in ends), default=0) * 1.6
            # Roads ending at an empty productive vertex are a route investment.
            if any(vertices.get(e, {}).get("building") is None for e in ends):
                val += 2.0
        elif typ == "maritime_trade":
            val += _trade_score(o, a, acts) * 8.0
            # Trades with no build payoff are intentionally below end_turn.
            if _trade_score(o, a, acts) < 0:
                val = -10
        elif typ == "buy_development_card":
            res = selfv.get("resources", {})
            # A card is a fallback when no affordable board build is listed.
            if any(x.get("type") in ("build_city", "build_settlement") for x in acts):
                val -= 18
            if int(res.get("ORE", 0) or 0) >= 3 and int(res.get("WHEAT", 0) or 0) >= 2:
                val -= 8
        candidates.append((val, str(a.get("id")), a))
    return max(candidates, key=lambda z: (z[0], z[1]))[2] if candidates else (acts[0] if acts else None)


def _choose(o, memory):
    acts = _acts(o)
    if not acts:
        return None
    # Detect forced decisions from the actual action types, since phase names
    # vary slightly between adaptor revisions.
    types = {str(a.get("type", "")) for a in acts}
    if "discard" in types and len(types) == 1:
        return acts[0]
    if "move_robber" in types:
        return _robber(o, acts)
    phase = str(o.get("phase", o.get("pending_decision", ""))).lower()
    if "initial_settlement" in phase or any("initial_settlement" in t for t in types):
        return _initial_settlement(o, acts, memory)
    if "initial_road" in phase or any("initial_road" in t for t in types):
        return _initial_road(o, acts)
    return _normal(o, acts)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(ctx, "done", False):
            return
        memory = getattr(ctx, "memory", {})
        try:
            obs = _obj(game.observe())
        except Exception:
            return
        while not getattr(ctx, "done", False) and not bool(obs.get("terminal", False)):
            action = _choose(obs, memory)
            if not action:
                return
            try:
                obs = _obj(ctx.act(action.get("id")))
            except Exception:
                # The adaptor reports stale/illegal IDs in the next view.  A
                # fresh observation lets the loop recover without an error.
                try:
                    obs = _obj(game.observe())
                except Exception:
                    return


def build(ctx):
    return Harness(ctx)
