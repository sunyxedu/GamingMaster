"""Joint opening portfolio search with the archived legality-first executor.

Only public observations are used. No simulator imports, random draws, or model
calls. Production forecasts rank opening pairs; engine actions remain authoritative.
"""
from __future__ import annotations

import json


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


def _opening_board(obs):
    board = obs.get("board", {})
    verts = {v["id"]: v for v in board.get("vertices", []) if isinstance(v, dict) and "id" in v}
    hs = {h["id"]: h for h in board.get("hexes", []) if isinstance(h, dict) and "id" in h}
    adj = {n: set() for n in verts}
    for e in board.get("edges", []):
        p = e.get("id")
        if isinstance(p, list) and len(p) == 2:
            u, v = p
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
    prod = {}
    for n, v in verts.items():
        p = [0.0] * 5
        for h in v.get("adjacent_hexes", []):
            h = hs.get(h, {})
            r = h.get("terrain")
            if r in RES:
                # Setup values long-run production, not temporary robber state.
                p[RES.index(r)] += PIPS.get(h.get("number"), 0)
        prod[n] = tuple(p)
    return verts, adj, prod


def _portfolio(nodes, verts, prod):
    """Expected build delay from a public production portfolio.

    Production is in expected cards per four-roll round. Bank conversion is a
    continuous planning approximation, never a simulated legal action. Actual
    actions always come from the observation.
    """
    p = [sum(prod[n][i] for n in nodes) / 9.0 for i in range(5)]
    ports = {verts[n].get("port") for n in nodes}
    generic = any(x in ports for x in ("ANY", "3:1", "GENERIC"))
    rates = [2 if r in ports else (3 if generic else 4) for r in RES]

    def delay(cost):
        # Cards expected from a round can cover deficits via surplus exchange.
        # This ignores stochastic waiting and must not be mistaken for a rule.
        lo, hi = 0.0, 40.0
        for _ in range(16):
            t = (lo + hi) / 2
            deficit = sum(max(0.0, cost[i] - p[i] * t) for i in range(5))
            trade = sum(max(0.0, p[i] * t - cost[i]) / rates[i] for i in range(5))
            if trade >= deficit:
                hi = t
            else:
                lo = t
        return hi

    # Different productive continuations must be possible. Road + settlement
    # requires two wood and brick; cities/development need ore and wheat.
    expand = delay((2, 2, 1, 1, 0))
    city = delay((0, 0, 0, 2, 3))
    dev = delay((0, 0, 1, 1, 1))
    # A mix of speed and robustness avoids selecting mere resource coverage
    # on 2/12 tokens, while still pricing absent resources through actual ports.
    score = (13.0 / (1.0 + expand) + 9.0 / (1.0 + city)
             + 5.0 / (1.0 + dev) + 0.75 * sum(p))
    coverage = sum(min(1.0, x / 0.38) for x in p)
    score += 0.45 * coverage
    return score, (expand, city, dev)


def _available_sites(verts, adj):
    return {n for n, v in verts.items()
            if not v.get("owner") and not any(verts.get(j, {}).get("owner") for j in adj.get(n, ())) }


def _expansion_option(n, chosen, available, verts, adj, prod):
    """Best nearby distance-valid site from a prospective setup settlement."""
    frontier = {n}
    seen = {n}
    best = 0.0
    for distance in range(1, 4):
        frontier = {j for u in frontier for j in adj.get(u, ())
                    if j not in seen and not verts.get(j, {}).get("owner")}
        seen.update(frontier)
        if distance < 2:
            continue
        for j in frontier & available:
            if j in chosen or any(j in adj.get(k, ()) for k in chosen):
                continue
            gain = sum(prod[j]) + 1.6 * sum(1 for i in range(5) if prod[j][i] and not any(prod[k][i] for k in chosen))
            best = max(best, gain / (distance + 1.0))
    return best


def _opening_choices(obs, acts):
    verts, adj, prod = _opening_board(obs)
    mine = obs.get("self", {}).get("id", "RED")
    own = [n for n, v in verts.items() if v.get("owner") == mine]
    candidates = [(a, _param_vertex(a)) for a in acts
                  if a.get("type") in ("build_settlement", "build_initial_settlement")
                  and _param_vertex(a) in verts]
    available = _available_sites(verts, adj)
    rows = []
    for a, n in candidates:
        if own:
            nodes = own + [n]
            value, waits = _portfolio(nodes, verts, prod)
            route = max((_expansion_option(k, nodes, available, verts, adj, prod) for k in nodes), default=0)
            # Second settlement grants these cards once, improving the first
            # settlement bundle especially when wood/brick are present.
            start = sum(bool(prod[n][i]) for i in range(4))
            rows.append((value + 0.16 * route + 0.08 * start, a, nodes, waits))
        else:
            pairs = []
            for _, m in candidates:
                if n == m or m in adj.get(n, ()):
                    continue
                value, waits = _portfolio([n, m], verts, prod)
                pairs.append((value, m, waits))
            pairs.sort(reverse=True)
            if pairs:
                # Several complementary fallbacks matter when opponents place
                # between the two picks. Equal snake turns still use this
                # conservative policy; no opponent hidden state is consulted.
                top = pairs[:3]
                value = sum(q[0] * w for q, w in zip(top, (0.55, 0.30, 0.15))) / sum((0.55, 0.30, 0.15)[:len(top)])
                m = pairs[0][1]
                route = max(_expansion_option(k, [n, m], available, verts, adj, prod) for k in (n, m))
                # First pick secures a sound economic anchor in case every
                # preferred partner is taken. Full pair production dominates.
                anchor = _portfolio([n], verts, prod)[0]
                rows.append((value + 0.22 * anchor + 0.16 * route, a, [n, m], pairs[0][2]))
            else:
                value, waits = _portfolio([n], verts, prod)
                rows.append((value, a, [n], waits))
    return sorted(rows, key=lambda x: (x[0], str(x[1].get("id"))), reverse=True)


def _initial_settlement(obs, acts):
    rows = _opening_choices(obs, acts)
    return rows[0][1] if rows else (acts[0] if acts else None)


def _initial_road(obs, acts):
    """Look beyond the unbuildable adjacent node to a reachable third site."""
    verts, adj, prod = _opening_board(obs)
    mine = obs.get("self", {}).get("id", "RED")
    own = [n for n, v in verts.items() if v.get("owner") == mine]
    available = _available_sites(verts, adj)
    base = _portfolio(own, verts, prod)[0] if own else 0.0
    occupied_edges = {tuple(sorted(e.get("id", []))): e.get("owner")
                      for e in obs.get("board", {}).get("edges", []) if e.get("owner")}
    scored = []
    for a in acts:
        p = a.get("parameters")
        if a.get("type") != "build_road" or not isinstance(p, list) or len(p) != 2:
            continue
        roots = [n for n in p if n not in own]
        seen = set(p)
        frontier = set(roots)
        best = 0.0
        for d in range(1, 5):
            for n in frontier & available:
                gain = _portfolio(own + [n], verts, prod)[0] - base
                best = max(best, gain / (1.0 + 0.65 * d))
            nxt = set()
            for u in frontier:
                for v in adj.get(u, ()):
                    edge_owner = occupied_edges.get(tuple(sorted((u, v))))
                    if v not in seen and not verts.get(v, {}).get("owner") and edge_owner in (None, mine):
                        nxt.add(v)
            seen.update(nxt)
            frontier = nxt
        scored.append((best, str(a.get("id")), a))
    return max(scored, key=lambda q: (q[0], q[1]))[2] if scored else (acts[0] if acts else None)


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
    if "initial" not in phase and previous and a and a.get("id") == previous.get("id") and len(acts) > 1:
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
            if "initial_settlement" in str(obs.get("phase", "")):
                rows = _opening_choices(obs, _actions(obs))
                try:
                    ctx.log("opening_portfolios", top=[{"id": r[1]["id"], "nodes": r[2],
                            "value": round(r[0], 3), "projected_rounds": [round(t, 2) for t in r[3]]}
                            for r in rows[:5]])
                except Exception:
                    pass
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
