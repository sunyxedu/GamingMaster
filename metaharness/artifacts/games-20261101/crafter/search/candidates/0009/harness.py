"""Observation-driven achievement planner with sparse strategic consultation.
Only ctx supplies observations/actions/inference; no simulator access or seed logic.
"""
from __future__ import annotations
import collections
import heapq
import json
import re

DIRS = ((1, 0, 'move_right'), (0, 1, 'move_down'), (-1, 0, 'move_left'), (0, -1, 'move_up'))
FLOOR = {'grass', 'path', 'sand'}
HOSTILE = {'zombie', 'skeleton'}
# Recipe hypotheses; successes replace costs with observed inventory consumption.
RECIPES = {'place_table': {'wood': 2}, 'place_furnace': {'stone': 4},
           'place_stone': {'stone': 1}, 'place_plant': {'sapling': 1}}
for tier in ('wood', 'stone', 'iron'):
    for tool in ('pickaxe', 'sword'):
        RECIPES['make_' + tier + '_' + tool] = ({'wood': 1} if tier == 'wood' else
            {'wood': 1, 'stone': 1} if tier == 'stone' else {'wood': 1, 'coal': 1, 'iron': 1})


def plus(p, d):
    return p[0] + d[0], p[1] + d[1]


def dist(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.world = {}
        self.visible = {}
        self.visits = collections.Counter()
        self.recipes = {k: dict(v) for k, v in RECIPES.items()}
        self.failures = collections.Counter()
        self.blocked = {}
        self.interact_fail = collections.Counter()
        self.history = collections.deque(maxlen=16)
        self.step = 0
        self.last_progress = 0
        self.calls = 0
        self.last_call = -1000
        self.preference = 'progress'
        self.prefer_until = 0
        self.explore_target = None
        self.placing = None
        self.hunger_target = 0
        self.thirst_target = 0
        self.last_sleep = -1000
        self.no_effect = 0
        self.stocked = False
        self.advice_start = {}
        self.plant_probes = 0
        self.future = None
        self.verified_recipes = set()
        self.bad_placements = collections.Counter()
        self.previous_daylight = None
        self.daylight_trend = 0.0
        # A late expedition can become a two-cell walk while a hostile follows.
        # Keep a short positional trace and a persistent safe anchor so low-health
        # recovery has a deterministic escape route independent of Astra advice.
        self.position_trace = collections.deque(maxlen=10)
        self.anchor = None
        self.panic_until = -1

    def ingest(self, obs):
        if isinstance(obs, str):
            try:
                obs = json.loads(obs)
            except Exception:
                obs = {}
        if not isinstance(obs, dict):
            obs = {}
        self.obs = obs
        self.step = int(obs.get('step', self.step))
        self.inv = dict(obs.get('inventory', {}))
        self.player = obs.get('player', {})
        light = self.player.get('daylight', 1.0)
        if self.previous_daylight is not None:
            self.daylight_trend = light - self.previous_daylight
        self.previous_daylight = light
        self.pos = tuple(self.player.get('position', [32, 32]))
        self.face = tuple(self.player.get('facing', [0, 1]))
        self.position_trace.append(self.pos)
        self.ach = {k for k, v in obs.get('achievements', {}).items() if v}
        self.ach.update(obs.get('unlocked', []))
        self.visible = {}
        rows = obs.get('local_map', {}).get('rows', [])
        for row in rows:
            for cell in row:
                if not isinstance(cell, dict) or 'x' not in cell or 'y' not in cell:
                    continue
                p = (cell['x'], cell['y'])
                c = dict(cell)
                c['material'] = str(c.get('material', 'unknown')).lower()
                c['object'] = str(c.get('object', '')).lower()
                if c['object'].startswith('plant_'):
                    c['ripe'] = any(word in c['object'] for word in ('ripe', 'mature', 'grown')) and 'growing' not in c['object']
                    c['object'] = 'plant'
                c['seen_step'] = self.step
                self.world[p] = c
                self.visible[p] = c
                if c.get('material') in ('table', 'furnace') and self.anchor is None:
                    self.anchor = p
        self.visits[self.pos] += 1
        self.ctx.memory['achievements'] = sorted(self.ach)

    def panic_escape(self):
        """Retreat from a nearby hostile when health/night makes exploration unsafe."""
        enemies = [p for p, c in self.visible.items() if c.get('object') in HOSTILE]
        if not enemies:
            return None
        nearest = min(enemies, key=lambda p: dist(self.pos, p))
        # A known station is a useful destination, but never walk through a mob.
        if self.anchor is not None and self.anchor != self.pos:
            costs, first = self.path_cache.get(False, ({}, {}))
            if self.anchor in first:
                q = first[self.anchor]
                if q is not None and all(dist(q, e) >= 2 for e in enemies):
                    self.intent_target = ('retreat', self.anchor)
                    return self.movement(q)
        options = []
        for dx, dy, action in DIRS:
            q = (self.pos[0] + dx, self.pos[1] + dy)
            if not self.walkable(q):
                continue
            separation = min(dist(q, e) for e in enemies)
            exits = sum(self.walkable((q[0] + ex, q[1] + ey)) for ex, ey, _ in DIRS)
            # Penalize immediately reversing a two-cycle; this lets the player
            # turn around an obstacle instead of paying repeated contact damage.
            reverse = len(self.position_trace) >= 2 and q == self.position_trace[-2]
            options.append((-separation, self.danger(q), reverse, -exits, action))
        if options:
            self.intent_target = ('retreat', nearest)
            return min(options)[-1]
        return self.face_do(nearest)

    def amount(self, key):
        return self.inv.get(key, 0)

    def material(self, p):
        return self.world.get(p, {}).get('material', 'unknown')

    def obj(self, p):
        # Do not freeze mobile creatures outside the current field of view.
        return self.visible.get(p, {}).get('object', '')

    def mineable(self, mat):
        if mat == 'tree':
            return True
        if mat in ('stone', 'coal'):
            return any(self.amount(t + '_pickaxe') for t in ('wood', 'stone', 'iron'))
        if mat == 'iron':
            return self.amount('stone_pickaxe') or self.amount('iron_pickaxe')
        if mat == 'diamond':
            return self.amount('iron_pickaxe')
        return False

    def walkable(self, p):
        return (self.material(p) in FLOOR and self.obj(p) in ('', 'player') and
                self.blocked.get(p, -1) < self.step)

    def danger(self, p):
        score = 0
        for q, c in self.visible.items():
            if c.get('object') in HOSTILE:
                d = dist(p, q)
                score += max(0, 5 - d) * (4 if self.amount('health') < 5 else 1)
            elif c.get('object') == 'arrow' and dist(p, q) <= 1:
                score += 8
        return score

    def paths(self, mining=False):
        costs = {self.pos: 0}
        first = {self.pos: None}
        queue = [(0, self.pos)]
        while queue:
            cost, p = heapq.heappop(queue)
            if cost != costs[p]:
                continue
            for dx, dy, action in DIRS:
                q = (p[0] + dx, p[1] + dy)
                if q not in self.world:
                    continue
                floor = self.walkable(q)
                mine = mining and self.mineable(self.material(q)) and not self.obj(q)
                if not floor and not mine:
                    continue
                nc = cost + (1 if floor else 4) + self.danger(q) + min(self.visits[q], 20) * .015
                if nc < costs.get(q, 1e9):
                    costs[q] = nc
                    first[q] = q if p == self.pos else first[p]
                    heapq.heappush(queue, (nc, q))
        return costs, first

    def movement(self, q):
        delta = (q[0] - self.pos[0], q[1] - self.pos[1])
        action = next((a for dx, dy, a in DIRS if (dx, dy) == delta), None)
        if action and not self.walkable(q) and self.mineable(self.material(q)):
            return 'do' if self.face == delta else action
        return action

    def face_do(self, q):
        delta = (q[0] - self.pos[0], q[1] - self.pos[1])
        if self.face == delta:
            return 'do'
        return next((a for dx, dy, a in DIRS if (dx, dy) == delta), None)

    def target(self, kind, max_cost=1e6, mining=True):
        candidates = []
        source = self.visible if kind in ('cow', 'zombie', 'skeleton') else self.world
        for q, c in source.items():
            matches = c.get('object') == kind if kind in ('cow', 'zombie', 'skeleton', 'plant') else c.get('material') == kind
            if kind == 'plant' and matches:
                matches = bool(c.get('ripe', False) or c.get('grown', False) or
                               (q not in self.visible and self.step - c.get('seen_step', self.step) > 220))
            if matches and self.interact_fail[(kind, q)] < 8:
                candidates.append(q)
        if not candidates:
            return None
        costs, first = self.path_cache[mining]
        options = []
        for q in candidates:
            for dx, dy, _ in DIRS:
                p = (q[0] + dx, q[1] + dy)
                if p in costs and (p == self.pos or self.walkable(p)):
                    penalty = self.danger(p) * (0 if kind in HOSTILE else 1)
                    options.append((costs[p] + penalty, q, p))
        if not options:
            return None
        cost, q, p = min(options)
        if cost > max_cost:
            return None
        self.intent_target = (kind, q)
        return self.face_do(q) if p == self.pos else self.movement(first[p])

    def explore(self):
        costs, first = self.path_cache[True]
        if self.explore_target in costs and self.pos != self.explore_target and self.walkable(self.explore_target):
            return self.movement(first[self.explore_target])
        candidates = []
        for p, cost in costs.items():
            if p == self.pos or not self.walkable(p):
                continue
            unseen = sum((p[0] + dx, p[1] + dy) not in self.world for dx in range(-4, 5) for dy in range(-4, 5))
            if unseen:
                value = cost - unseen * .27 + self.visits[p] * .7
                candidates.append((value, p))
        if candidates:
            self.explore_target = min(candidates)[1]
            return self.movement(first[self.explore_target])
        # If enclosed by minable rock/trees, expose one tile at a time.
        for mat in ('stone', 'tree', 'coal', 'iron'):
            if self.mineable(mat):
                action = self.target(mat, mining=True)
                if action:
                    return action
        options = [q for q in first if q != self.pos and first[q] is not None]
        if options:
            q = min(options, key=lambda q: self.visits[q] * 4 + costs[q])
            return self.movement(first[q])
        return 'noop'

    def near(self, mat, p=None):
        p = self.pos if p is None else p
        return any(self.material((p[0] + dx, p[1] + dy)) == mat for dx, dy, _ in DIRS)

    def station(self, furnace=False):
        costs, first = self.path_cache[True]
        options = [(c, p) for p, c in costs.items() if self.walkable(p) and self.near('table', p) and
                   (not furnace or self.near('furnace', p))]
        if not options:
            return None
        _, p = min(options)
        return ('ready', p) if p == self.pos else (self.movement(first[p]), p)

    def place(self, action, table_near=False):
        if self.placing and self.placing[0] != action:
            self.placing = None
        if self.placing:
            _, stand, target, back = self.placing
            delta = (target[0] - stand[0], target[1] - stand[1])
            legal = self.walkable(target) and self.bad_placements[(action, target)] < 2 and (action != 'place_plant' or self.material(target) == 'grass')
            if not legal:
                self.placing = None
            elif self.pos == stand and self.face == delta:
                self.placing = None
                return action
            elif self.pos == back:
                return self.movement(stand)
            elif back in self.path_cache[False][0] and self.pos != back:
                return self.movement(self.path_cache[False][1][back])
            else:
                self.placing = None
        options = []
        costs, first = self.path_cache[False]
        for stand in costs:
            if table_near and not self.near('table', stand):
                continue
            if dist(stand, self.pos) > 8:
                continue
            for dx, dy, _ in DIRS:
                target = (stand[0] + dx, stand[1] + dy)
                back = (stand[0] - dx, stand[1] - dy)
                if not self.walkable(target) or self.bad_placements[(action, target)] >= 2:
                    continue
                if action == 'place_plant' and self.material(target) != 'grass':
                    continue
                if stand == self.pos and self.face == (dx, dy):
                    return action
                if back in costs and self.walkable(back):
                    options.append((costs[back] + 2 + self.danger(stand), stand, target, back))
        if options:
            _, stand, target, back = min(options)
            self.placing = (action, stand, target, back)
            return self.movement(stand) if self.pos == back else self.movement(first[back])
        return None

    def signature(self, action):
        return (action, tuple(self.amount(k) for k in ('wood', 'stone', 'coal', 'iron', 'sapling')),
                self.near('table'), self.near('furnace'))

    def craft_goal(self, action, depth=0):
        if depth > 8:
            return None
        for resource, needed in self.recipes.get(action, {}).items():
            if self.amount(resource) < needed:
                return self.acquire(resource, depth + 1)
        if action.startswith('make_'):
            furnace = 'iron' in action
            station = self.station(furnace)
            if station:
                if station[0] != 'ready':
                    travel = self.path_cache[True][0].get(station[1], 1000)
                    # Portable workbenches cost less than repeated cross-map journeys.
                    if not furnace and travel > 14:
                        if self.amount('wood') >= self.recipes[action].get('wood', 0) + 2:
                            local_table = self.place('place_table')
                            if local_table:
                                return local_table
                        else:
                            nearby_wood = self.target('tree', max_cost=8)
                            if nearby_wood:
                                return nearby_wood
                    return station[0]
                if self.failures[self.signature(action)] < 2:
                    return action
                return None
            if furnace and self.station(False):
                return self.craft_goal('place_furnace', depth + 1)
            return self.craft_goal('place_table', depth + 1)
        if action == 'place_furnace' and not self.near('table'):
            station = self.station(False)
            if station and station[0] != 'ready':
                return station[0]
            if not station:
                return self.craft_goal('place_table', depth + 1)
        return self.place(action, table_near=action == 'place_furnace')

    def acquire(self, resource, depth=0):
        if depth > 8:
            return None
        if resource in ('stone', 'coal') and not self.mineable(resource):
            return self.craft_goal('make_wood_pickaxe', depth + 1)
        if resource == 'iron' and not self.mineable(resource):
            return self.craft_goal('make_stone_pickaxe', depth + 1)
        if resource == 'diamond' and not self.mineable(resource):
            return self.craft_goal('make_iron_pickaxe', depth + 1)
        mat = {'wood': 'tree', 'sapling': 'grass'}.get(resource, resource)
        return self.target(mat) or self.explore()

    def combat(self):
        enemies = [(dist(self.pos, p), p, c) for p, c in self.visible.items() if c.get('object') in HOSTILE]
        if not enemies:
            return None
        # Prefer a target already in our facing tile, then a wounded opponent.
        enemies.sort(key=lambda item: (item[0], item[1] != plus(self.pos, self.face), item[2].get('health', 9)))
        d, q, c = enemies[0]
        crowd = sum(e[0] <= 3 for e in enemies)
        strength = 5 if self.amount('iron_sword') else 3 if self.amount('stone_sword') else 2 if self.amount('wood_sword') else 1
        if d <= 1:
            can_finish = c.get('health', 9) <= strength and plus(self.pos, self.face) == q
            if can_finish or (self.amount('health') >= 5 and crowd <= 2):
                self.intent_target = (c['object'], q)
                return self.face_do(q)
            escapes = [(self.danger(plus(self.pos, direction)) - .3 * sum(self.walkable(plus(plus(self.pos, direction), d2)) for d2 in DIRS), -dist(plus(self.pos, direction), q), direction[2])
                       for direction in DIRS if self.walkable(plus(self.pos, direction))]
            if escapes:
                return min(escapes)[2]
            return self.face_do(q)
        if d <= 3 and (self.amount('health') < 5 or crowd >= 3):
            escapes = [(self.danger(plus(self.pos, direction)), direction[2]) for direction in DIRS
                       if self.walkable(plus(self.pos, direction))]
            if escapes and min(escapes)[0] < self.danger(self.pos):
                return min(escapes)[1]
        # Hold a safe facing stance for an approaching lone zombie instead of
        # stepping into its attack range with our back turned.
        if d == 2 and crowd == 1 and c.get('object') == 'zombie':
            if q[0] == self.pos[0] or q[1] == self.pos[1]:
                delta = ((q[0] > self.pos[0]) - (q[0] < self.pos[0]),
                         (q[1] > self.pos[1]) - (q[1] < self.pos[1]))
                if self.face == delta:
                    return 'do'
        return None

    def recovery_action(self):
        """Choose survival actions before a strategic objective is allowed.

        Hunger and thirst are slow drains, so waiting until zero is already a
        death sentence during a long route.  Keep a reserve and search for a
        known recovery tile with a much larger radius than ordinary objectives.
        If no tile is known, exploration is still preferable to crafting or
        mining: the next observation can reveal water, a cow, or a crop.
        """
        drink = self.amount('drink')
        food = self.amount('food')
        health = self.amount('health')
        # At these levels every non-recovery action is an emergency.  The
        # reserve is deliberately hysteretic (we refill to nine) so a route
        # cannot immediately spend the newly found food or water.
        urgent = drink <= 3 or food <= 3 or health <= 3
        if drink <= 7 or 'collect_drink' not in self.ach:
            action = self.target('water', max_cost=180 if urgent else 75)
            if action:
                return action, 'recovery_water' if urgent else 'reserve_water'
            if urgent:
                # Keep revealing the frontier until water is found; a model
                # objective must not steal these ticks.
                action = self.explore()
                if action:
                    return action, 'search_water'
        if food <= 7 or 'eat_cow' not in self.ach:
            action = self.target('plant', max_cost=120 if urgent else 65)
            if action is None:
                action = self.target('cow', max_cost=150 if urgent else 70)
            if action:
                return action, 'recovery_food' if urgent else 'reserve_food'
            if urgent:
                action = self.explore()
                if action:
                    return action, 'search_food'
        return None

    def choose(self):
        self.intent_target = None
        self.path_cache = {False: self.paths(False), True: self.paths(True)}
        emergency = self.combat()
        # Combat's normal kiting is useful at full health, but at low health or
        # deep night a marginal attack is worse than preserving all achievements
        # already earned.  Enter a bounded retreat mode and resume only after a
        # few safe ticks; this is deliberately independent of model advice.
        hostile = [p for p, c in self.visible.items() if c.get('object') in HOSTILE]
        hp = self.amount('health')
        daylight = float(self.player.get('daylight', 1.0) or 0.0)
        nearest = min((dist(self.pos, p) for p in hostile), default=99)
        if hostile and ((hp <= 4 and nearest <= 6) or (daylight < 0.24 and hp <= 6 and nearest <= 8)):
            self.panic_until = max(self.panic_until, self.step + 14)
        if hostile and self.step < self.panic_until:
            escaped = self.panic_escape()
            if escaped:
                return escaped, 'panic_retreat'
        if emergency:
            return emergency, 'combat'
        if self.player.get('sleeping'):
            return 'sleep', 'sleeping'
        drink = self.amount('drink')
        food = self.amount('food')
        # Start refilling well before the critical point.  A resource can be
        # ten or more actions away, and each action consumes a tick.
        if drink <= 7:
            self.thirst_target = 9
        if drink >= 9:
            self.thirst_target = 0
        if food <= 7:
            self.hunger_target = 9
        if food >= 9:
            self.hunger_target = 0
        # This guard runs before Astra preferences and all crafting.  It is
        # intentionally deterministic so an old/stale adviser cannot spend a
        # whole night walking while the vitals silently reach zero.
        recovery = self.recovery_action()
        if recovery:
            return recovery
        daylight = self.player.get('daylight', 1.0)
        # Dev observations showed roughly 11 sleep ticks per restored energy.
        # Do not start a long sleep shortly before dusk: it cannot be cancelled
        # reliably by movement, and a sleeping hit removed seven health.
        estimated_sleep = 12 * max(0, 9 - self.amount('energy')) + 5
        safe_day = daylight >= .75 and daylight + min(0, self.daylight_trend) * estimated_sleep >= .75
        if safe_day and (self.amount('energy') <= 3 or ('wake_up' not in self.ach and self.amount('energy') <= 7)):
            if not any(c.get('object') in HOSTILE for c in self.visible.values()):
                if food >= 3 and drink >= 3:
                    self.last_sleep = self.step
                    return 'sleep', 'rest'
        if safe_day and self.amount('health') <= 4 and food >= 5 and drink >= 5 and self.amount('energy') < 8 and not any(c.get('object') in HOSTILE for c in self.visible.values()):
            return 'sleep', 'heal'
        # Sparse model preference is advisory; safety and prerequisites stay in code.
        if self.step < self.prefer_until:
            action = None
            if self.preference == 'explore':
                needed = 'stone' if not self.amount('stone_pickaxe') else 'iron' if not self.amount('iron_pickaxe') else 'diamond'
                # Discovery completes exploration; never walk past the new resource
                # merely because a model's tick allowance has not expired.
                if self.target(needed) is not None:
                    self.prefer_until = 0
                else:
                    action = self.explore()
            elif self.preference in ('wood', 'stone', 'coal', 'iron', 'diamond', 'sapling'):
                if self.amount(self.preference) > self.advice_start.get(self.preference, 0):
                    self.prefer_until = 0
                else:
                    action = self.acquire(self.preference)
            elif self.preference in RECIPES:
                if self.preference in self.ach:
                    self.prefer_until = 0
                else:
                    action = self.craft_goal(self.preference)
            elif self.preference in ('cow', 'plant', 'zombie', 'skeleton', 'water'):
                action = self.target(self.preference, max_cost=80)
            if action:
                return action, 'advice:' + self.preference
        # Carry a small expedition reserve so upgrading does not require one
        # return trip per ingredient. Gathering is bounded to nearby resources.
        if self.amount('wood_sword') and not self.stocked:
            if self.amount('wood') >= 6:
                self.stocked = True
            else:
                action = self.target('tree', max_cost=12)
                if action:
                    return action, 'wood_reserve'
                self.stocked = True
        if self.amount('wood_sword') and not self.amount('stone_sword') and self.amount('stone') < 3:
            action = self.target('stone', max_cost=10)
            if action:
                return action, 'stone_reserve'
        # Plant early enough for a crop to mature during crafting/exploration.
        if self.amount('wood_sword') and 'place_plant' not in self.ach and self.plant_probes < 25:
            if self.amount('sapling'):
                action = self.place('place_plant')
            else:
                action = self.target('grass', max_cost=2, mining=False)
                if action == 'do':
                    self.plant_probes += 1
            if action:
                return action, 'early_garden'
        # A stone sword is a larger survival upgrade than the second pickaxe.
        for tool in ('wood_pickaxe', 'wood_sword', 'stone_sword', 'stone_pickaxe'):
            if not self.amount(tool):
                action = self.craft_goal('make_' + tool)
                if action:
                    return action, tool
        if 'place_stone' not in self.ach and self.amount('stone') >= 1:
            action = self.craft_goal('place_stone')
            if action:
                return action, 'place_stone'
        if 'place_plant' not in self.ach:
            if self.amount('sapling'):
                action = self.craft_goal('place_plant')
            else:
                action = self.target('grass', max_cost=4, mining=False)
            if action:
                return action, 'planting'
        if 'place_furnace' not in self.ach:
            action = self.craft_goal('place_furnace')
            if action:
                return action, 'furnace'
        for tool in ('iron_pickaxe', 'iron_sword'):
            if not self.amount(tool):
                action = self.craft_goal('make_' + tool)
                if action:
                    return action, tool
        if 'collect_diamond' not in self.ach:
            return self.acquire('diamond'), 'diamond'
        if 'eat_plant' not in self.ach:
            action = self.target('plant')
            if action:
                return action, 'eat_plant'
            if self.amount('sapling') and sum(c.get('object') == 'plant' for c in self.visible.values()) < 4:
                action = self.place('place_plant')
                if action:
                    return action, 'garden'
        for monster in ('zombie', 'skeleton'):
            if 'defeat_' + monster not in self.ach and self.amount('health') >= 6:
                action = self.target(monster, max_cost=30)
                if action:
                    return action, 'hunt_' + monster
        return self.explore(), 'explore'

    def consult(self, reason):
        remaining = self.ctx.budget.remaining()
        if self.future is not None or self.calls >= 8 or remaining.get('system2_calls', 0) <= 1 or remaining.get('wall_s', 0) < 180:
            return
        self.calls += 1
        self.last_call = self.step
        resources = {}
        for kind in ('tree', 'stone', 'coal', 'iron', 'diamond', 'water', 'table', 'furnace'):
            pts = [p for p, c in self.world.items() if c.get('material') == kind]
            resources[kind] = sorted(pts, key=lambda p: dist(p, self.pos))[:8]
        state = {'step': self.step, 'position': self.pos, 'inventory': self.inv,
                 'achievements': sorted(self.ach), 'resources': resources,
                 'visible_objects': [c for c in self.visible.values() if c.get('object') and c.get('object') != 'player'],
                 'recent_transitions': list(self.history), 'recipe_hypotheses': self.recipes,
                 'failed_recipes': [{'action': k[0], 'inventory_wood_stone_coal_iron_sapling': k[1], 'table': k[2], 'furnace': k[3], 'count': v} for k, v in list(self.failures.items())[-8:]],
                 'reason': reason}
        prompt = ('You are the strategic supervisor of a Crafter controller. Maximize unique achievements out of 22. '
                  'Code already handles pathfinding, facing, survival, combat, recipe dependencies and one-tick observation feedback. '
                  'Choose a useful short objective, especially to repair stalled progress. Do not give movement lists. '
                  'A recipe hypothesis may be incorrect; failures and observed deltas are evidence. '
                  'Return a JSON object with objective (progress, explore, wood, stone, coal, iron, diamond, sapling, '
                  'cow, plant, zombie, skeleton, water, or a make_/place_ action), ticks (8..80), '
                  'and optionally recipes mapping action to resource costs if a correction is needed. '
                  'Use progress to resume the standard dependency plan. Keep response under 150 words.\n' + json.dumps(state, separators=(',', ':')))
        request_step = self.step
        timeout = min(150.0, remaining['wall_s'] - 20)

        def request():
            # Only the model request and pure parsing run on the background thread.
            # No shared controller state is mutated here; the main loop applies advice.
            try:
                result = self.ctx.think(prompt, session_id=None, timeout_s=timeout)
                match = re.search(r'\{[\s\S]*\}', result.text)
                data = json.loads(match.group(0)) if match else {}
                return request_step, data if isinstance(data, dict) else {}, None
            except Exception as exc:
                return request_step, {}, type(exc).__name__
        try:
            self.future = self.ctx.spawn(request)
        except Exception as exc:
            self.ctx.log('strategy_failed', error=type(exc).__name__)

    def receive_advice(self):
        if self.future is None or not self.future.done():
            return
        future = self.future
        self.future = None
        try:
            request_step, data, error = future.result()
            if error:
                self.ctx.log('strategy_failed', error=error)
                return
            objective = data.get('objective', 'progress')
            valid = {'progress', 'explore', 'wood', 'stone', 'coal', 'iron', 'diamond', 'sapling', 'cow', 'plant', 'zombie', 'skeleton', 'water'} | set(RECIPES)
            if isinstance(objective, str) and objective in valid and self.step - request_step <= 160:
                ticks = max(8, min(80, int(data.get('ticks', 35))))
                self.preference = objective
                self.advice_start = dict(self.inv)
                self.prefer_until = self.step + ticks
            changes = data.get('recipes', {})
            if isinstance(changes, dict):
                for key, costs in changes.items():
                    if key not in self.recipes or key in self.verified_recipes or not isinstance(costs, dict) or not costs:
                        continue
                    if all(k in ('wood', 'stone', 'coal', 'iron', 'sapling') and type(v) is int and 0 < v <= 9 for k, v in costs.items()):
                        self.recipes[key] = dict(costs)
            self.ctx.log('strategy', objective=self.preference, until=self.prefer_until, age_ticks=self.step-request_step)
        except Exception as exc:
            self.ctx.log('strategy_failed', error=type(exc).__name__)

    def run_episode(self, game):
        self.names = set(game.action_spec.names)
        self.ingest(self.ctx.observe())
        while not self.ctx.done and not self.obs.get('done', False):
            if len(self.ach) >= 22:
                break
            self.receive_advice()
            # Advice is nonblocking, including backend admission/queue latency.
            if self.step >= 100 and self.step - self.last_call >= 180 and (
                self.calls == 0 or self.step - self.last_progress > 160 or self.no_effect >= 8):
                self.consult('stalled' if self.step - self.last_progress > 160 else 'strategic review')
                if self.ctx.done:
                    break
            action, reason = self.choose()
            if action not in self.names:
                action = 'noop'
            before_inv = dict(self.inv)
            before_ach = set(self.ach)
            before_pos = self.pos
            front = plus(self.pos, self.face)
            before_cell = dict(self.visible.get(front, {}))
            signature = self.signature(action)
            target = self.intent_target
            try:
                obs = self.ctx.act(action)
            except Exception as exc:
                if self.ctx.done:
                    break
                self.ctx.log('action_failed', action=action, error=type(exc).__name__)
                self.no_effect += 1
                if self.no_effect >= 12:
                    return
                self.ingest(self.ctx.observe())
                continue
            self.ingest(obs if obs is not None else self.ctx.observe())
            delta = {k: v - before_inv.get(k, 0) for k, v in self.inv.items() if v != before_inv.get(k, 0)}
            gained = sorted(self.ach - before_ach)
            if gained:
                self.last_progress = self.step
                self.ctx.log('achievement', names=gained, step=self.step)
            meaningful = bool(gained or any(k not in ('food', 'drink', 'energy', 'health') for k in delta))
            self.no_effect = 0 if meaningful or self.pos != before_pos else self.no_effect + 1
            if action in self.recipes:
                success = action in gained or (action.startswith('make_') and delta.get(action[5:], 0) > 0) or any(v < 0 and k in self.recipes[action] for k, v in delta.items())
                if success:
                    costs = {k: -v for k, v in delta.items() if v < 0 and k in ('wood', 'stone', 'coal', 'iron', 'sapling')}
                    if costs:
                        self.recipes[action] = costs
                        self.verified_recipes.add(action)
                else:
                    self.failures[signature] += 1
                    if action.startswith('place_'):
                        self.bad_placements[(action, front)] += 1
            if action.startswith('move_') and self.pos == before_pos:
                d = next(((dx, dy) for dx, dy, a in DIRS if a == action), None)
                q = plus(before_pos, d) if d else before_pos
                if self.material(q) in FLOOR and not self.obj(q):
                    self.blocked[q] = self.step + 5
            if action == 'do' and target:
                after_cell = self.visible.get(front, {})
                unchanged = all(after_cell.get(k) == v for k, v in before_cell.items() if k != 'seen_step') and not meaningful and not delta.get('drink', 0) and not delta.get('food', 0)
                if unchanged:
                    self.interact_fail[target] += 1
                else:
                    self.interact_fail[target] = 0
            if meaningful or gained or action in self.recipes or self.step % 20 == 0:
                self.history.append({'step': self.step, 'action': action, 'goal': reason, 'delta': delta, 'gained': gained, 'position': self.pos})
            # A hard cap prevents accidental infinite loops even with imperfect fake observations.
            if self.step >= 9800:
                break


def build(ctx):
    return Harness(ctx)
