"""Observation-only economic planning on top of candidate 0003.

No game internals, simulations of the engine, random draws, or model calls.
A plan is a resource cost and board objective, never a cached action ID.
"""
from __future__ import annotations

import json
from collections import defaultdict
import heapq


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
                cards = min(6, float(q.get("resource_count", 0) or 0))
                pressure += cards * 0.2
                # Production alone can select a convenient victim while a
                # public leader is one build away.  Add a bounded threat term:
                # public VP is observable, and the extra bonus only matters in
                # the endgame so ordinary robber choices remain production-led.
                vp = float(q.get("public_vp", 0) or 0)
                pressure += 0.35 * vp
                if vp >= 8:
                    pressure += 2.5
                elif vp >= 7:
                    pressure += 1.0
        scored.append((pressure, str(a.get("id")), a))
    return max(scored, key=lambda z: (z[0], z[1]))[2] if scored else (acts[0] if acts else None)


COST = {
    'build_city': (0, 0, 0, 2, 3),
    'build_settlement': (1, 1, 1, 1, 0),
    'buy_development_card': (0, 0, 1, 1, 1),
    'build_road': (1, 1, 0, 0, 0),
}


def _vec(d):
    return tuple(int(d.get(r, 0) or 0) for r in RES)


def _edge(p):
    if isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, int) for v in p):
        return tuple(sorted(p))
    return None


def _afford(hand, cost):
    return all(h >= c for h, c in zip(hand, cost))


def _trade(hand, a):
    """The adaptor pads 2:1 / 3:1 trades with null, not resource cards."""
    p = a.get('parameters')
    if not isinstance(p, (list, tuple)) or len(p) < 2 or p[-1] not in RES:
        return None
    h = list(hand)
    for r in p[:-1]:
        if r in RES:
            h[RES.index(r)] -= 1
        elif r is not None:
            return None
    if min(h) < 0:
        return None
    h[RES.index(p[-1])] += 1
    return tuple(h)


class Position:
    def __init__(self, obs):
        self.obs = obs
        self.me = obs.get('self', {})
        self.color = self.me.get('id', 'RED')
        self.vp = float(self.me.get('true_vp', self.me.get('public_vp', 0)))
        self.board = obs.get('board', {})
        self.vertices = {v['id']: v for v in self.board.get('vertices', [])}
        self.hexes = {h['id']: h for h in self.board.get('hexes', [])}
        self.edges = {_edge(e['id']): e.get('owner') for e in self.board.get('edges', []) if _edge(e.get('id'))}
        self.adj = defaultdict(list)
        for u, v in self.edges:
            self.adj[u].append(v)
            self.adj[v].append(u)
        self.mine = {v for v, data in self.vertices.items() if data.get('owner') == self.color}
        self.blocked = {v for v, data in self.vertices.items() if data.get('owner') not in (None, self.color)}
        self.roads = {e for e, owner in self.edges.items() if owner == self.color}
        self.hand = _vec(self.me.get('resources', {}))
        self.bank = _vec(obs.get('bank_counts', {r: 19 for r in RES}))
        ports = self.me.get('ports', [])
        self.rates = [2 if r in ports else 3 if None in ports else 4 for r in RES]
        # Current legal trades can verify or improve inferred port rates.
        for a in _actions(obs):
            if a['type'] == 'maritime_trade':
                p = a.get('parameters') or []
                given = [r for r in p[:-1] if r in RES]
                if given and len(set(given)) == 1:
                    i = RES.index(given[0])
                    self.rates[i] = min(self.rates[i], len(given))
        self.production = [0.0] * 5
        for v in self.mine:
            mult = 2 if self.vertices[v].get('building') == 'CITY' else 1
            for i, p in enumerate(self.vertex_production(v)):
                self.production[i] += mult * p * 4 / 36
        self.pieces = self.me.get('pieces', {})

    def vertex_production(self, v):
        p = [0.0] * 5
        for hid in self.vertices.get(v, {}).get('adjacent_hexes', []):
            h = self.hexes.get(hid, {})
            r = h.get('terrain')
            if r in RES:
                # A currently blocked tile still has long-run value.
                p[RES.index(r)] += PIPS.get(h.get('number'), 0) * (0.4 if h.get('robber') else 1)
        return p

    def value(self, v):
        p = self.vertex_production(v)
        # Value of an extra building falls toward the endgame; VP does not.
        horizon = max(0.5, 10 - self.vp)
        production = sum(p) + 0.22 * (p[3] + p[4])
        scarcity = sum(min(1.2, x * 0.25) for i, x in enumerate(p) if self.production[i] < 0.25)
        return 1.0 + 0.014 * horizon * (production + scarcity)

    def settlement_sites(self):
        return {v for v in self.vertices if not self.vertices[v].get('owner')
                and all(not self.vertices[n].get('owner') for n in self.adj[v])}

    def routes(self, extra=()):
        """Shortest new-road path from our public network, with blocked vertices."""
        owned = self.roads | set(extra)
        starts = self.mine | {v for e in owned for v in e if v not in self.blocked}
        best = {v: (0, ()) for v in starts}
        heap = [(0, v, ()) for v in starts]
        heapq.heapify(heap)
        while heap:
            n, v, path = heapq.heappop(heap)
            if best.get(v) != (n, path) or v in self.blocked:
                continue
            for w in self.adj[v]:
                e = tuple(sorted((v, w)))
                if self.edges[e] not in (None, self.color) or w in self.blocked:
                    continue
                new = path if e in owned else path + (e,)
                dist = len(new)
                if dist > 3:
                    continue
                if w not in best or (dist, new) < best[w]:
                    best[w] = (dist, new)
                    heapq.heappush(heap, (dist, w, new))
        return best

    def longest(self, roads, color=None):
        """Conservative trail estimate matching this adaptor's observed awards.

        In the archive, an edge incident to an opponent building is excluded
        from the reported length, even as an endpoint. Removing these edges
        matches all post-initialization public maxima in the parent traces.
        Actual award ownership is always taken from the next observation.
        """
        color = color or self.color
        blocked = {v for v, d in self.vertices.items() if d.get('owner') not in (None, color)}
        adj = defaultdict(list)
        eligible = [e for e in sorted(roads) if not any(v in blocked for v in e)]
        for i, (u, v) in enumerate(eligible):
            adj[u].append((v, 1 << i))
            adj[v].append((u, 1 << i))
        def dfs(v, used):
            if used and v in blocked:
                return 0
            best = 0
            for w, bit in adj[v]:
                if not used & bit:
                    best = max(best, 1 + dfs(w, used | bit))
            return best
        return max((dfs(v, 0) for v in list(adj)), default=0)

    def extensions(self, roads):
        starts = self.mine | {v for e in roads for v in e if v not in self.blocked}
        return [e for e, owner in self.edges.items() if owner is None and e not in roads
                and any(v in starts and v not in self.blocked for v in e)]

    def road_award(self):
        """Only pay for the award when no more than two new roads can take it."""
        awards = self.obs.get('awards', {})
        if awards.get('longest_road_owner') == self.color or self.pieces.get('roads', 15) <= 0:
            return None
        rivals = set(self.edges.values()) - {None, self.color}
        rival = max((self.longest({e for e, c in self.edges.items() if c == color}, color) for color in rivals), default=0)
        threshold = max(5, rival + 1)
        own = self.longest(self.roads)
        if own + 2 < threshold:
            return None
        choices = []
        for e in self.extensions(self.roads):
            r1 = self.roads | {e}
            n = self.longest(r1)
            if n >= threshold:
                choices.append((e,))
            elif self.pieces.get('roads', 15) >= 2:
                for f in self.extensions(r1):
                    if self.longest(r1 | {f}) >= threshold:
                        choices.append((e, f))
        if not choices:
            return None
        return min(choices, key=lambda path: (len(path), -sum(sum(self.vertex_production(v)) for e in path for v in e), path))

    def goals(self):
        # A goal: (kind, target vertex, complete resource cost, ordered roads, value).
        out = []
        if self.pieces.get('cities', 4):
            for v in sorted(self.mine):
                if self.vertices[v].get('building') == 'SETTLEMENT':
                    out.append(('build_city', v, COST['build_city'], (), self.value(v)))
        if self.pieces.get('settlements', 5):
            routes = self.routes()
            for v in self.settlement_sites():
                if v not in routes:
                    continue
                n, path = routes[v]
                if n > self.pieces.get('roads', 15):
                    continue
                cost = (n + 1, n + 1, 1, 1, 0)
                value = self.value(v)
                port = self.vertices[v].get('port')
                if port in RES and self.rates[RES.index(port)] > 2:
                    value += min(0.5, self.production[RES.index(port)] * 0.2)
                out.append(('build_settlement', v, cost, path, value))
        acts = _actions(self.obs)
        # When the hand can pay but no purchase is legal, the deck may be empty.
        if not _afford(self.hand, COST['buy_development_card']) or any(a['type'] == 'buy_development_card' for a in acts):
            awards = self.obs.get('awards', {})
            needed = max(3, int(awards.get('largest_army_size', 0)) + 1) - int(self.me.get('played_knights', 0))
            knights = int(self.me.get('development_cards', {}).get('KNIGHT', 0))
            army_value = 0 if awards.get('largest_army_owner') == self.color else 1.12 / max(2, needed + knights)
            value = 0.26 + army_value
            if self.vp >= 9:
                value = 0.30
            out.append(('buy_development_card', None, COST['buy_development_card'], (), value))
        award = self.road_award()
        if award:
            n = len(award)
            out.append(('road_award', None, (n, n, 0, 0, 0), award, 2.05))
        return out

    def eta(self, hand, cost):
        """Expected rounds to resource feasibility, including exact exchange ratios.

        This is an economic forecast, not a hidden-state game simulation.
        Integer cards prevent unrealistically pooling fractional bank trades.
        """
        def feasible(t):
            stock = [int(h + p * t + 1e-7) for h, p in zip(hand, self.production)]
            deficits = sum(max(0, c - h) for h, c in zip(stock, cost))
            surplus = sum(max(0, h - c) // rate for h, c, rate in zip(stock, cost, self.rates))
            return deficits <= surplus
        if feasible(0):
            return 0.0
        lo, hi = 0.0, 32.0
        if not feasible(hi):
            return 64.0
        for _ in range(13):
            mid = (lo + hi) / 2
            if feasible(mid):
                hi = mid
            else:
                lo = mid
        return hi

    def utility(self, goal, hand):
        kind, v, cost, path, value = goal
        eta = self.eta(hand, cost)
        # Guarantee a known winning build/award takes precedence over production.
        points = 2 if kind == 'road_award' else 0 if kind == 'buy_development_card' else 1
        if self.vp + points >= 10:
            value += 8.0
        return value / (1.0 + eta + 0.12 * len(path))


def _target_action(goal, acts):
    kind, vertex, cost, path, value = goal
    if path:
        return next((a for a in acts if a['type'] == 'build_road' and _edge(a.get('parameters')) == path[0]), None)
    return next((a for a in acts if a['type'] == kind and (vertex is None or _param_vertex(a) == vertex)), None)


def _free_road(pos, acts):
    roads = [a for a in acts if a['type'] == 'build_road']
    award = pos.road_award()
    if award:
        chosen = next((a for a in roads if _edge(a.get('parameters')) == award[0]), None)
        if chosen:
            return chosen
    sites = pos.settlement_sites()
    def value(a):
        e = _edge(a.get('parameters'))
        if not e:
            return -1e6
        routes = pos.routes((e,))
        potential = max((pos.value(v) / (1 + n) for v, (n, path) in routes.items() if v in sites), default=0)
        return potential + 0.08 * pos.longest(pos.roads | {e})
    return max(roads, key=value) if roads else None


def _economic_choice(obs):
    acts = _actions(obs)
    if not acts:
        return None, None
    if len(acts) == 1:
        return acts[0], None
    phase = str(obs.get('phase', obs.get('pending_decision', ''))).lower()
    # Preserve the leader's opening and robber policy for this experiment.
    if 'initial_settlement' in phase:
        return _initial_settlement(obs, acts), None
    if 'initial_road' in phase:
        return _initial_road(obs, acts), None
    if 'robber' in phase:
        return _robber(obs, acts), None
    pos = Position(obs)
    types = {a['type'] for a in acts}
    if types == {'build_road'}:
        return _free_road(pos, acts), 'free road'
    # Prefer a known immediate win before optional card play.
    if pos.vp >= 9:
        builds = [a for a in acts if a['type'] in ('build_city', 'build_settlement')]
        if builds:
            return max(builds, key=lambda a: pos.value(_param_vertex(a))), 'winning VP build'
    knight = next((a for a in acts if a['type'] == 'play_knight_card'), None)
    if knight:
        return knight, 'knight / army'
    roll = next((a for a in acts if a['type'] == 'roll'), None)
    if roll:
        return roll, None
    goals = pos.goals()
    if not goals:
        return next((a for a in acts if a['type'] == 'end_turn'), acts[0]), None
    ranked = sorted(goals, key=lambda g: pos.utility(g, pos.hand), reverse=True)
    goal = ranked[0]
    best_now = pos.utility(goal, pos.hand)

    plenty = [a for a in acts if a['type'] == 'play_year_of_plenty']
    if plenty:
        def plenty_value(a):
            hand = list(pos.hand)
            for r in a.get('parameters') or []:
                if r in RES:
                    hand[RES.index(r)] += 1
            return max(pos.utility(g, hand) for g in goals)
        a = max(plenty, key=plenty_value)
        if plenty_value(a) > best_now + 0.06:
            return a, 'plenty completes economic goal'
    monopoly = [a for a in acts if a['type'] == 'play_monopoly']
    if monopoly:
        estimates = [0.0] * 5
        for opp in obs.get('opponents', []):
            prod = [0.6] * 5
            for v, data in pos.vertices.items():
                if data.get('owner') == opp.get('id'):
                    mult = 2 if data.get('building') == 'CITY' else 1
                    for i, p in enumerate(pos.vertex_production(v)):
                        prod[i] += mult * p
            for i, p in enumerate(prod):
                estimates[i] += opp.get('resource_count', 0) * p / sum(prod)
        def mono_value(a):
            r = a.get('parameters')
            if r not in RES:
                return -1e6
            i = RES.index(r)
            h = list(pos.hand)
            h[i] += int(estimates[i])
            return max(pos.utility(g, h) for g in goals) + 0.04 * estimates[i]
        a = max(monopoly, key=mono_value)
        if estimates[RES.index(a['parameters'])] >= 2.0 and mono_value(a) > best_now + 0.08:
            return a, 'monopoly estimated from public production and hand counts'
    road_card = next((a for a in acts if a['type'] == 'play_road_building'), None)
    if road_card and pos.pieces.get('roads', 15):
        return road_card, 'free road investment'

    target = _target_action(goal, acts)
    if target:
        return target, (goal[0], goal[1], goal[3])
    trades = [a for a in acts if a['type'] == 'maritime_trade']
    # Keep a chosen goal across its trades. The forecast is recomputed each step;
    # spending into the goal makes its ranking better, not a stale cached plan.
    current_eta = pos.eta(pos.hand, goal[2])
    candidates = []
    for a in trades:
        h = _trade(pos.hand, a)
        if h is None:
            continue
        p = a['parameters']
        ri = RES.index(p[-1])
        # Never buy a resource the complete goal already has enough of.
        if pos.hand[ri] >= goal[2][ri]:
            continue
        new_eta = pos.eta(h, goal[2])
        deficit_before = sum(max(0, c - x) for c, x in zip(goal[2], pos.hand))
        deficit_after = sum(max(0, c - x) for c, x in zip(goal[2], h))
        # A partial conversion can protect a large hand from seven-discards
        # without changing its forecast completion time. It must reduce the
        # goal's actual deficit, so this does not authorize resource churn.
        protects_hand = (sum(pos.hand) > 7 and deficit_after < deficit_before
                         and new_eta <= current_eta + 0.004)
        if ((current_eta == 0 and new_eta == 0 and deficit_after < deficit_before)
                or new_eta + 0.04 < current_eta or protects_hand):
            # Prefer fewer spent cards and retaining scarce resources.
            surplus = sum(min(x, 4) / (0.4 + prod) for x, prod in zip(h, pos.production))
            candidates.append((-new_eta, -deficit_after, surplus, a))
    if candidates:
        return max(candidates, key=lambda z: z[:3])[3], ('trade for', goal[0], goal[1])
    # An unavailable build is never guessed: the legal list is authoritative.
    # If the top forecast is affordable but not actionable (e.g. bank shortage),
    # choose a different immediately executable objective instead of stalling.
    if current_eta == 0:
        for other in ranked[1:]:
            target = _target_action(other, acts)
            if target:
                return target, ('alternate', other[0], other[1])
    end = next((a for a in acts if a['type'] == 'end_turn'), None)
    return end or acts[0], ('save for', goal[0], goal[1])


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.done:
            return
        obs = _obj(game.observe())
        failures = 0
        while not ctx.done and not obs.get('terminal', False):
            acts = _actions(obs)
            if not acts:
                return
            try:
                chosen, plan = _economic_choice(obs)
            except Exception as exc:
                # Defensive fallback covers unexpected observation schemas.
                ctx.log('economic_fallback', error=type(exc).__name__)
                chosen = next((a for a in acts if a['type'] in ('roll', 'end_turn', 'discard')), acts[0])
                plan = None
            if not chosen or chosen['id'] not in {a['id'] for a in acts}:
                chosen = acts[0]
            if plan:
                ctx.memory['economic_goal'] = plan
            old_count = obs.get('action_count')
            try:
                nxt = _obj(ctx.act(chosen['id']))
                if not nxt and not ctx.done:
                    nxt = _obj(ctx.observe())
            except Exception:
                if ctx.done:
                    return
                failures += 1
                if failures >= 3:
                    return
                try:
                    obs = _obj(ctx.observe())
                except Exception:
                    return
                continue
            if old_count is not None and nxt.get('action_count') == old_count:
                failures += 1
                if failures >= 3:
                    return
            else:
                failures = 0
            obs = nxt


def build(ctx):
    return Harness(ctx)
