"""A legality-first Catan controller.

The previous controller asked Astra to rediscover the current legal action on
every step.  This controller treats the observation as an executable state:
all actions come from the current legal list, and inexpensive board/resource
features choose among them.  This is deliberately model-free; a model call
cannot be useful if its answer is stale by the time it is posted.
"""
from __future__ import annotations

import json
from collections import Counter


PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
RES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")


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


def _actions(o):
    a = o.get("legal_actions", []) if isinstance(o, dict) else []
    return [x for x in a if isinstance(x, dict) and x.get("id") is not None]


def _param_vertex(a):
    p = a.get("parameters")
    if isinstance(p, int):
        return p
    if isinstance(p, (list, tuple)):
        for x in p:
            if isinstance(x, int):
                return x
    return None


def _resource_score(v, board):
    """Production plus diversity, with wheat/ore slightly favored."""
    hs = board.get("hexes", [])
    total, kinds = 0.0, set()
    for hid in (v.get("adjacent_hexes", []) if isinstance(v, dict) else []):
        if not isinstance(hid, int) or hid >= len(hs):
            continue
        h = hs[hid]
        if h.get("robber"):
            continue
        n, t = h.get("number"), h.get("terrain")
        if isinstance(n, (int, float)) and t:
            total += PIPS.get(int(n), 0)
            kinds.add(t)
            if t in ("WHEAT", "ORE"):
                total += 0.55
    return total + 0.8 * len(kinds)


def _initial_settlement(obs, acts):
    board = obs.get("board", {})
    verts = board.get("vertices", [])
    byid = {v.get("id"): v for v in verts if isinstance(v, dict)}
    scored = []
    for a in acts:
        if a.get("type") not in ("build_settlement", "build_initial_settlement"):
            continue
        i = _param_vertex(a)
        v = byid.get(i, {})
        scored.append((_resource_score(v, board), str(a.get("id")), a))
    if scored:
        return max(scored, key=lambda z: (z[0], z[1]))[2]
    return acts[0] if acts else None


def _initial_road(obs, acts):
    """Prefer a road whose endpoint opens the best nearby empty vertex."""
    board = obs.get("board", {})
    verts = {v.get("id"): v for v in board.get("vertices", []) if isinstance(v, dict)}
    mine = {v.get("id") for v in board.get("vertices", [])
            if isinstance(v, dict) and v.get("owner") == "RED"}
    best = []
    for a in acts:
        if "road" not in str(a.get("type", "")):
            continue
        p = a.get("parameters")
        ends = []
        def collect(x):
            if isinstance(x, int): ends.append(x)
            elif isinstance(x, (list, tuple)):
                for y in x: collect(y)
        collect(p)
        score = 0.0
        for e in ends:
            if e in mine:
                score += 1.0
            score += _resource_score(verts.get(e, {}), board)
        best.append((score, str(a.get("id")), a))
    return max(best, key=lambda z: (z[0], z[1]))[2] if best else (acts[0] if acts else None)


def _robber(obs, acts):
    board = obs.get("board", {})
    hs = board.get("hexes", [])
    verts = {v.get("id"): v for v in board.get("vertices", []) if isinstance(v, dict)}
    opp = {x.get("id") for x in obs.get("opponents", []) if isinstance(x, dict)}
    scored = []
    for a in acts:
        if a.get("type") != "move_robber":
            continue
        p = a.get("parameters")
        coord = p[0] if isinstance(p, list) and p else None
        victim = p[1] if isinstance(p, list) and len(p) > 1 else None
        hid = None
        for h in hs:
            if h.get("coordinate") == coord:
                hid = h.get("id")
                break
        pressure = 0.0
        if isinstance(hid, int):
            h = hs[hid] if hid < len(hs) else {}
            pressure += PIPS.get(h.get("number"), 0) * (2 if victim in opp else 0.25)
            for v in h.get("vertices", []):
                vv = verts.get(v, {})
                if vv.get("owner") in opp:
                    pressure += 1.5 if vv.get("building") == "CITY" else 0.8
        # A victim with more cards is a better steal target; this is public.
        for q in obs.get("opponents", []):
            if isinstance(q, dict) and q.get("id") == victim:
                pressure += min(4, float(q.get("resource_count", 0))) * 0.2
        scored.append((pressure, str(a.get("id")), a))
    return max(scored, key=lambda z: (z[0], z[1]))[2] if scored else (acts[0] if acts else None)


def _choose(obs, previous=None):
    acts = _actions(obs)
    if not acts:
        return None
    phase = str(obs.get("phase", obs.get("pending_decision", ""))).lower()
    if "initial_settlement" in phase:
        a = _initial_settlement(obs, acts)
    elif "initial_road" in phase:
        a = _initial_road(obs, acts)
    elif "robber" in phase:
        a = _robber(obs, acts)
    else:
        # Exact tactical priorities.  The engine has already filtered costs,
        # distance rules, bank shortages, and card timing for us.
        rank = {
            "play_year_of_plenty": 100, "play_monopoly": 99,
            "play_road_building": 98, "play_knight_card": 97,
            "build_city": 90, "build_settlement": 88, "build_road": 70,
            "maritime_trade": 60, "buy_development_card": 50,
            "roll": 20, "end_turn": 0, "discard": 110,
        }
        # If multiple actions have the same type, cities with more productive
        # vertices and settlements with better production win the tie.
        board = obs.get("board", {})
        verts = {v.get("id"): v for v in board.get("vertices", []) if isinstance(v, dict)}
        def key(a):
            typ = str(a.get("type", ""))
            val = rank.get(typ, 10)
            if typ in ("build_city", "build_settlement"):
                val += _resource_score(verts.get(_param_vertex(a), {}), board)
            return (val, str(a.get("id")))
        a = max(acts, key=key)
    # Never repeat an action if the observation still exposes alternatives:
    # repeated stale outputs were the dominant baseline failure.  Keep the
    # same type as a fallback, but choose another ID when possible.
    if previous and a and a.get("id") == previous.get("id") and len(acts) > 1:
        same = [x for x in acts if x.get("type") == a.get("type") and x.get("id") != previous.get("id")]
        if same:
            a = same[0]
    return a


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        if getattr(ctx, "done", False):
            return
        obs = _obj(game.observe())
        previous = None
        while not ctx.done and not bool(obs.get("terminal", False)):
            a = _choose(obs, previous)
            if not a:
                return
            try:
                obs = _obj(ctx.act(a["id"]))
            except Exception:
                # A transient adaptor rejection supplies a fresh legal list;
                # retrying never lets a parse/model error escape the harness.
                try:
                    obs = _obj(game.observe())
                except Exception:
                    return
            previous = a


def build(ctx):
    return Harness(ctx)
