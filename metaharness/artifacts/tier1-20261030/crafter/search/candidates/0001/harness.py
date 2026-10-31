"""Observation-driven options: cheap closed-loop skills, sparse strategic reviews."""
from __future__ import annotations
import json
import heapq
import re
from collections import Counter, deque

DIRS = ((0, -1, 'move_up'), (1, 0, 'move_right'), (0, 1, 'move_down'), (-1, 0, 'move_left'))
WALK = {'grass', 'path', 'sand'}
RESOURCE = {'wood': 'tree', 'stone': 'stone', 'coal': 'coal', 'iron': 'iron', 'diamond': 'diamond', 'drink': 'water'}
# These are explicit hypotheses. Inventory deltas verify them on every success.
RECIPES = {
    'place_table': {'wood': 2}, 'place_furnace': {'stone': 4},
    'place_stone': {'stone': 1}, 'place_plant': {'sapling': 1},
    'make_wood_pickaxe': {'wood': 1}, 'make_wood_sword': {'wood': 1},
    'make_stone_pickaxe': {'wood': 1, 'stone': 1}, 'make_stone_sword': {'wood': 1, 'stone': 1},
    'make_iron_pickaxe': {'wood': 1, 'coal': 1, 'iron': 1},
    'make_iron_sword': {'wood': 1, 'coal': 1, 'iron': 1},
}

def add(p, d):
    return p[0] + d[0], p[1] + d[1]

def dist(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])

class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.tiles = {}
        self.entities = {}
        self.plants = {}
        self.plant_since = {}
        self.cow_memory = {}
        self.seen = {}
        self.visits = Counter()
        self.failures = Counter()
        self.blocked = {}
        self.events = deque(maxlen=24)
        self.recipes = {k: dict(v) for k, v in RECIPES.items()}
        self.verified = {}
        self.target = None
        self.target_kind = None
        self.frontier = None
        self.frontier_kind = None
        self.goal = None
        self.goal_until = 0
        self.heading = 'north'
        self.last_review = 0
        self.reviews = 0
        self.last_progress = 0
        self.recover = None
        self.last_action = None
        self.last_reason = ''
        self.reason = ''
        self.routes = None
        self.raw_plan = deque()
        self.recipe_failed = False

    def update(self, obs):
        self.obs = obs
        self.p = tuple(obs.get('player', {}).get('position', (0, 0)))
        self.f = tuple(obs.get('player', {}).get('facing', (0, 1)))
        self.inv = obs.get('inventory', {})
        self.ach = obs.get('achievements', {})
        self.tick = int(obs.get('step', 0))
        self.health = self.inv.get('health', obs.get('player', {}).get('health', 9))
        self.entities = {}
        for row in obs.get('local_map', {}).get('rows', []):
            for cell in row:
                if not isinstance(cell, dict) or 'x' not in cell:
                    continue
                p = (cell['x'], cell['y'])
                self.tiles[p] = cell.get('material', 'unknown')
                self.seen[p] = self.tick
                obj = str(cell.get('object', '')).lower()
                if obj and obj != 'player':
                    self.entities[p] = dict(cell, object=obj)
                if obj.startswith('plant'):
                    self.plants[p] = dict(cell)
                    self.plant_since.setdefault(p, self.tick)
                elif p in self.plants:
                    del self.plants[p]
                    self.plant_since.pop(p, None)
                if obj == 'cow':
                    self.cow_memory[p] = self.tick
                elif p in self.cow_memory:
                    del self.cow_memory[p]
        self.hostiles = {p: c for p, c in self.entities.items() if c['object'] in ('zombie', 'skeleton')}
        self.visits[self.p] += 1
        self.routes = None

    def have(self, key):
        return self.inv.get(key, 0)

    def unlocked(self, key):
        return self.ach.get(key, 0) > 0

    def pick(self):
        return 3 if self.have('iron_pickaxe') else 2 if self.have('stone_pickaxe') else 1 if self.have('wood_pickaxe') else 0

    def sword(self):
        return 3 if self.have('iron_sword') else 2 if self.have('stone_sword') else 1 if self.have('wood_sword') else 0

    def mineable(self, mat):
        return mat == 'tree' or (mat in ('stone', 'coal') and self.pick() >= 1) or (mat == 'iron' and self.pick() >= 2) or (mat == 'diamond' and self.pick() >= 3)

    def walkable(self, p):
        return self.tiles.get(p) in WALK and p not in self.entities and p not in self.plants and self.blocked.get(p, -1) < self.tick

    def danger(self, p):
        return sum(max(0, 3 - dist(p, q)) * (1 if self.sword() >= 1 and self.health >= 6 else 4) for q in self.hostiles)

    def paths(self):
        if self.routes is not None:
            return self.routes
        costs = {self.p: 0}
        prev = {}
        heap = [(0, self.p)]
        while heap:
            c, p = heapq.heappop(heap)
            if c != costs[p]:
                continue
            for dx, dy, _ in DIRS:
                q = (p[0] + dx, p[1] + dy)
                if q not in self.tiles or self.blocked.get(q, -1) >= self.tick:
                    continue
                if q in self.entities or q in self.plants:
                    continue
                mat = self.tiles[q]
                if mat in WALK:
                    step = 1
                elif self.mineable(mat):
                    step = 4
                else:
                    continue
                nc = c + step + self.danger(q)
                if nc < costs.get(q, 1e20):
                    costs[q] = nc
                    prev[q] = p
                    heapq.heappush(heap, (nc, q))
        self.routes = costs, prev
        return self.routes

    def navigate(self, target):
        if target == self.p:
            return None
        costs, prev = self.paths()
        if target not in costs:
            return None
        q = target
        while prev.get(q) != self.p:
            q = prev.get(q)
            if q is None:
                return None
        d = (q[0] - self.p[0], q[1] - self.p[1])
        if self.mineable(self.tiles.get(q)):
            return 'do' if d == self.f else self.move(d)
        return self.move(d)

    def move(self, d):
        return next((name for dx, dy, name in DIRS if (dx, dy) == tuple(d)), 'noop')

    def interact(self, target):
        if dist(self.p, target) == 1:
            d = (target[0] - self.p[0], target[1] - self.p[1])
            if self.f == d:
                return 'do'
            # Nonwalkable targets permit a safe facing-only collision.
            if not self.walkable(target):
                return self.move(d)
        costs, _ = self.paths()
        stands = [add(target, d) for d in DIRS if add(target, d) in costs and self.tiles.get(add(target, d)) in WALK]
        if not stands:
            return None
        stand = min(stands, key=lambda q: costs[q] + (0 if q != self.p else 2))
        if stand != self.p:
            return self.navigate(stand)
        # For grass, orientation is obtained by approaching a standing tile from behind.
        for dx, dy, _ in DIRS:
            q = (self.p[0] + dx, self.p[1] + dy)
            if self.walkable(q):
                return self.move((dx, dy))
        return None

    def nearest(self, targets, limit=1e9):
        costs, _ = self.paths()
        choices = []
        for q in targets:
            if self.blocked.get(q, -1) >= self.tick:
                continue
            options = [costs.get(add(q, d), 1e9) for d in DIRS]
            c = min(options) + self.danger(q)
            if c < limit:
                choices.append((c + (0 if q == self.target else .2), q))
        return min(choices)[1] if choices else None

    def gather(self, resource, limit=1e9):
        self.reason = 'gather_' + resource
        if resource in ('cow', 'zombie', 'skeleton'):
            targets = [p for p, c in self.entities.items() if c['object'] == resource]
        elif resource == 'plant':
            targets = [p for p, c in self.plants.items() if c.get('ripe') or c.get('grown') or c.get('object') in ('plant_ripe', 'plant_mature')]
        else:
            mat = RESOURCE.get(resource, resource)
            if mat not in ('water', 'grass') and not self.mineable(mat):
                return None
            targets = [p for p, m in self.tiles.items() if m == mat and p not in self.entities and p != self.p]
        q = self.nearest(targets, limit)
        if q is None:
            return None
        self.target = q
        self.target_kind = resource
        return self.interact(q)

    def near(self, mat, p=None):
        p = self.p if p is None else p
        return any(self.tiles.get(add(p, d)) == mat for d in DIRS)

    def bench(self, iron=False):
        costs, _ = self.paths()
        stands = [p for p in costs if self.tiles.get(p) in WALK and self.near('table', p) and (not iron or self.near('furnace', p))]
        if not stands:
            return None
        q = min(stands, key=lambda p: costs[p])
        return self.navigate(q) if q != self.p else ''

    def place(self, action, beside_table=False):
        self.reason = action
        valid = {'grass'} if action == 'place_plant' else {'grass', 'sand', 'path'}
        # Search (position, facing), not just position. Moving toward an empty
        # placement target changes position; moving toward a table only turns.
        # Keeping facing in the search eliminates two-tile orientation loops.
        start = (self.p, self.f)
        heap = [(0, start, '')]
        costs = {start: 0}
        while heap:
            cost, (p, facing), first = heapq.heappop(heap)
            if cost != costs[(p, facing)]:
                continue
            front = add(p, facing)
            legal = (self.tiles.get(front) in valid and front not in self.entities
                     and front not in self.plants
                     and self.failures[(action, p, front)] < 2
                     and (not beside_table or self.near('table', p)))
            if legal:
                return first or action
            if cost > 60:
                continue
            for dx, dy, name in DIRS:
                q = (p[0]+dx, p[1]+dy)
                d = (dx,dy)
                if q in self.entities or q in self.plants:
                    continue
                if self.walkable(q):
                    dest = q
                elif self.tiles.get(q) in ('tree','stone','coal','iron','diamond','table','furnace'):
                    dest = p
                else:
                    continue
                state = (dest, d)
                newcost = cost + 1 + self.danger(dest)
                if newcost < costs.get(state, 1e20):
                    costs[state] = newcost
                    heapq.heappush(heap, (newcost, state, first or name))
        return None

    def craft(self, item, depth=0):
        if depth > 5:
            return None
        action = 'make_' + item
        self.reason = action
        req = self.recipes.get(action, {})
        for res, count in req.items():
            if self.have(res) < count:
                if res == 'iron' and self.pick() < 2:
                    return self.craft('stone_pickaxe', depth + 1)
                if res in ('stone', 'coal') and self.pick() < 1:
                    return self.craft('wood_pickaxe', depth + 1)
                return self.gather(res) or self.explore(res)
        iron = item.startswith('iron_')
        b = self.bench(iron)
        if b is not None:
            if b:
                return b
            key = (action, tuple(sorted(self.inv.items())), self.p)
            if self.failures[key] < 2:
                self.reason = action
                return action
            self.recipe_failed = True
            return None
        if not any(m == 'table' for m in self.tiles.values()):
            if self.have('wood') < self.recipes['place_table']['wood']:
                return self.gather('wood') or self.explore('wood')
            return self.place('place_table')
        if iron:
            if self.have('stone') < self.recipes['place_furnace']['stone']:
                return self.gather('stone') or self.explore('stone')
            return self.place('place_furnace', beside_table=True)
        return None

    def explore(self, resource=None):
        self.reason = 'explore_' + str(resource)
        costs, _ = self.paths()
        if self.frontier_kind != resource:
            self.frontier = None
            self.frontier_kind = resource
        if resource == 'cow':
            remembered = [p for p, t in self.cow_memory.items() if self.tick-t < 150 and p in costs and dist(self.p,p)>2]
            if remembered:
                return self.navigate(min(remembered, key=lambda p: costs[p]))
        if self.frontier in costs and self.p != self.frontier and dist(self.p, self.frontier) > 1:
            return self.navigate(self.frontier)
        options = []
        desired = {'iron', 'diamond', 'coal', 'stone'}
        heading = {'north': (0, -1), 'south': (0, 1), 'east': (1, 0), 'west': (-1, 0)}.get(self.heading, (0, -1))
        for p, c in costs.items():
            if p == self.p or c > 90:
                continue
            unseen = sum((p[0] + dx, p[1] + dy) not in self.tiles for dx in (-4, 0, 4) for dy in (-4, 0, 4))
            if not unseen:
                continue
            mineral = sum(self.tiles.get(add(p, d)) in ('stone', 'coal', 'iron', 'lava', 'diamond') for d in DIRS)
            biome = mineral * (2 if resource in desired else -.5)
            directional = ((p[0]-self.p[0])*heading[0] + (p[1]-self.p[1])*heading[1]) * .05
            score = c + self.visits[p] * 3 - unseen * 2 - biome - directional
            options.append((score, p))
        if options:
            self.frontier = min(options)[1]
            return self.navigate(self.frontier)
        # If surrounded, mine an adjacent legal block, or walk to the least visited reachable tile.
        if costs:
            options = [(self.visits[p] * 5 + c, p) for p, c in costs.items() if p != self.p]
            if options:
                return self.navigate(min(options)[1])
        return 'noop'

    def safety(self):
        # Fight adjacent threats before any crafting/navigation action.
        adjacent = [p for p in self.hostiles if dist(self.p, p) == 1]
        if adjacent:
            self.reason = 'defend'
            q = min(adjacent, key=lambda p: (p != add(self.p, self.f), self.hostiles[p].get('health', 5)))
            if self.health >= 4 or self.sword() >= 2 or len(adjacent) == 1:
                return self.interact(q)
            moves = [(self.danger(add(self.p, d)), d) for d in DIRS if self.walkable(add(self.p, d))]
            if moves:
                return min(moves)[1][2]
            return self.interact(q)
        if self.obs.get('player', {}).get('sleeping'):
            self.reason = 'sleeping'
            return 'noop'
        if self.have('drink') <= 3:
            self.recover = 'drink'
        if self.recover == 'drink':
            if self.have('drink') >= 8:
                self.recover = None
            else:
                a = self.gather('drink')
                if a:
                    return a
        if self.have('food') <= 6:
            a = self.gather('plant', 35) or self.gather('cow', 45)
            if a:
                return a
        if (self.have('energy') <= 3 or (not self.unlocked('wake_up') and self.have('energy') <= 8 and self.sword() >= 2)) and not self.hostiles and self.have('food') >= 3 and self.have('drink') >= 3:
            self.reason = 'sleep'
            return 'sleep'
        if self.have('drink') <= 1:
            return self.explore('water')
        if self.have('food') <= 3:
            return self.explore('cow')
        return None

    def farming(self):
        if not self.have('sapling') and not self.unlocked('collect_sapling'):
            front = add(self.p, self.f)
            if self.tiles.get(front) == 'grass' and front not in self.entities:
                self.reason = 'collect_sapling'
                return 'do'
            return self.gather('grass', 12)
        if self.have('sapling') and (not self.unlocked('place_plant') or len(self.plants) < 3):
            return self.place('place_plant')
        if not self.unlocked('eat_plant'):
            return self.gather('plant', 50)
        return None

    def goal_action(self, goal):
        if goal.startswith('make_'):
            return self.craft(goal[5:])
        if goal.startswith('collect_'):
            return self.gather(goal[8:])
        if goal in ('eat_cow', 'eat_plant', 'defeat_zombie', 'defeat_skeleton'):
            return self.gather(goal.split('_', 1)[1], 50)
        if goal in ('place_stone', 'place_table', 'place_furnace', 'place_plant'):
            for res, count in self.recipes[goal].items():
                if self.have(res) < count:
                    return self.gather(res)
            return self.place(goal, goal == 'place_furnace')
        if goal == 'explore':
            return self.explore('diamond' if self.pick() >= 3 else 'iron')
        return None

    def choose(self):
        a = self.safety()
        if a:
            return a
        # Cheap opportunistic achievements cannot distract distant resource progression.
        if not self.unlocked('collect_drink') or self.have('drink') < 7:
            a = self.gather('drink', 3)
            if a:
                return a
        if self.sword() and (not self.unlocked('eat_cow') or self.have('food') <= 6):
            a = self.gather('cow', 4)
            if a:
                return a
        if self.raw_plan:
            self.reason = 'astra_probe'
            return self.raw_plan.popleft()
        if self.goal and self.tick < self.goal_until and not self.unlocked(self.goal):
            a = self.goal_action(self.goal)
            if a:
                return a
        # Provision several dependent recipes in one trip. The first dev run showed
        # single-item procurement repeatedly traversing the same 15-tile commute.
        if not self.unlocked('place_table') and self.have('wood') < 8:
            return self.gather('wood') or self.explore('wood')
        if self.have('wood_pickaxe') and self.have('wood_sword') and not self.unlocked('make_stone_pickaxe') and self.have('stone') < 7:
            return self.gather('stone') or self.explore('stone')
        if self.pick() >= 2 and self.unlocked('place_furnace') and not self.unlocked('make_iron_pickaxe'):
            if self.have('wood') < 2:
                return self.gather('wood') or self.explore('wood')
            if self.have('coal') < 2:
                return self.gather('coal') or self.explore('coal')
            if self.have('iron') < 2:
                return self.gather('iron') or self.explore('iron')
        # Dependency compiler, re-evaluated from actual inventory every single step.
        for item in ('wood_pickaxe', 'wood_sword', 'stone_pickaxe', 'stone_sword'):
            if not self.unlocked('make_' + item):
                a = self.craft(item)
                if a:
                    return a
        if not self.unlocked('collect_sapling') or not self.unlocked('place_plant'):
            a = self.farming()
            if a:
                return a
        if not self.unlocked('place_stone'):
            if self.have('stone') < 1:
                a = self.gather('stone')
            else:
                a = self.place('place_stone')
            if a:
                return a
        if not self.unlocked('place_furnace'):
            if self.have('stone') < self.recipes['place_furnace']['stone']:
                a = self.gather('stone') or self.explore('stone')
            else:
                a = self.place('place_furnace', beside_table=True)
            if a:
                return a
        for item in ('iron_pickaxe', 'iron_sword'):
            if not self.unlocked('make_' + item):
                a = self.craft(item)
                if a:
                    return a
        if not self.unlocked('collect_diamond'):
            a = self.gather('diamond') or self.explore('diamond')
            if a:
                return a
        a = self.farming()
        if a:
            return a
        for mob in ('cow', 'zombie', 'skeleton'):
            ach = ('eat_' if mob == 'cow' else 'defeat_') + mob
            if not self.unlocked(ach) and self.health >= 5:
                a = self.gather(mob, 40)
                if a:
                    return a
        return self.explore('plant')

    def review(self):
        budget = self.ctx.budget.remaining()
        if self.reviews >= 6 or budget.get('system2_calls', 0) <= 1 or budget.get('wall_s', 0) < 180:
            return
        if self.tick < 100 or self.tick - self.last_review < 220:
            return
        self.last_review = self.tick
        self.reviews += 1
        interests = {}
        for mat in ('tree', 'water', 'table', 'furnace', 'stone', 'coal', 'iron', 'diamond'):
            points = sorted((p for p, m in self.tiles.items() if m == mat), key=lambda p: dist(self.p, p))[:8]
            interests[mat] = points
        rows = []
        legend = {'grass': '.', 'path': '_', 'sand': ':', 'tree': 'T', 'stone': 'S', 'coal': 'C', 'iron': 'I', 'diamond': 'D', 'water': '~', 'lava': 'L', 'table': 'B', 'furnace': 'F'}
        for y in range(self.p[1]-4, self.p[1]+5):
            row = ''
            for x in range(self.p[0]-4, self.p[0]+5):
                p = (x,y)
                row += '@' if p == self.p else self.entities[p]['object'][0].lower() if p in self.entities else legend.get(self.tiles.get(p), '?')
            rows.append(row)
        data = {'step': self.tick, 'player': self.obs.get('player'), 'inventory': self.inv,
                'unlocked': [k for k,v in self.ach.items() if v], 'missing': [k for k,v in self.ach.items() if not v],
                'local_map': rows, 'nearby_known_resources': interests, 'plants': list(self.plants.values())[:8],
                'recent_events': list(self.events), 'recipe_hypotheses': self.recipes, 'verified_recipes': self.verified,
                'last_option': self.last_reason, 'ticks_since_unlock': self.tick-self.last_progress}
        prompt = ('You are strategic supervisor of a Crafter controller. Maximize the 22 unique achievements. '
                  'Code executes navigation, mining, crafting prerequisites, combat and survival with fresh observations every tick. '
                  'Do not micromanage movement. Identify a neglected achievable objective or correct a failed recipe. '
                  'Default progression is wood tools -> stone tools -> sapling/plant -> furnace -> iron tools -> diamond. '
                  'Map rows north to south, columns west to east, centered on player; legend '+json.dumps(legend)+'. '
                  'Objects c=cow,z=zombie,s=skeleton,p=plant. Table and furnace must be cardinally adjacent for iron tools. '
                  'Recipe numbers are hypotheses and successful consumption overrides them. '
                  'Return ONLY JSON: {"goal":"achievement name or auto or explore", "horizon":120, '
                  '"heading":"north/east/south/west", "recipes":{}, "actions":[], "reason":"one short sentence"}. '
                  'Recipes may override make_* or place_* costs with resource:integer; only if evidence warrants. '
                  'Optional actions is at most 6 bare legal actions for a specific probe, normally empty. '
                  'Prefer auto if default progression is sound. No prose. State:\n'+json.dumps(data,separators=(',',':')))
        try:
            res = self.ctx.think(prompt, session_id=None, timeout_s=min(150, budget['wall_s']-20))
            text = res.text
            start = text.find('{')
            reply, _ = json.JSONDecoder().raw_decode(text[start:]) if start >= 0 else ({}, 0)
            if not isinstance(reply, dict):
                return
            goal = reply.get('goal')
            if goal in self.ach or goal in ('auto', 'explore'):
                self.goal = None if goal == 'auto' else goal
                self.goal_until = self.tick + max(20, min(180, int(reply.get('horizon',120))))
            if reply.get('heading') in ('north','east','south','west'):
                self.heading = reply['heading']
                self.frontier = None
            recipes = reply.get('recipes', {})
            if isinstance(recipes, dict):
                for action, req in recipes.items():
                    if action in self.recipes and action not in self.verified and isinstance(req, dict) and req and all(k in ('wood','stone','coal','iron','sapling') and isinstance(v,int) and 0 < v <= 9 for k,v in req.items()):
                        self.recipes[action] = req
            actions = reply.get('actions', [])
            if isinstance(actions, list):
                self.raw_plan = deque(a for a in actions[:6] if isinstance(a,str) and a in self.names)
            self.ctx.log('strategic_review', step=self.tick, response=reply)
        except Exception as exc:
            self.ctx.log('review_fallback', error=type(exc).__name__, step=self.tick)

    def record(self, old, action, new):
        inv0, inv1 = old.get('inventory', {}), new.get('inventory', {})
        delta = {k:v-inv0.get(k,0) for k,v in inv1.items() if v != inv0.get(k,0)}
        p = tuple(old.get('player',{}).get('position',(0,0)))
        f = tuple(old.get('player',{}).get('facing',(0,1)))
        front = add(p, f)
        gained = [k for k,v in new.get('achievements',{}).items() if v > old.get('achievements',{}).get(k,0)]
        unlocked = [k for k in gained if not old.get('achievements',{}).get(k,0)]
        if unlocked:
            self.last_progress = self.tick
            self.ctx.log('option_unlock', step=self.tick, achievements=unlocked, option=self.reason)
        if gained or delta or action.startswith(('make_', 'place_')):
            self.events.append({'t':self.tick,'action':action,'option':self.reason,'delta':delta,'achievements':gained})
        if action in self.recipes:
            succeeded = action in gained or (action.startswith('make_') and delta.get(action[5:],0)>0)
            if succeeded:
                cost = {k:-v for k,v in delta.items() if v<0 and k in ('wood','stone','coal','iron','sapling')}
                if cost:
                    self.recipes[action] = cost
                    self.verified[action] = cost
            else:
                self.failures[(action,p,front)] += 1
                self.failures[(action,tuple(sorted(inv0.items())),p)] += 1
                self.recipe_failed = True
        if action.startswith('move_') and tuple(new.get('player',{}).get('position',p)) == p:
            d = next(((x,y) for x,y,n in DIRS if n==action),(0,0))
            q = add(p,d)
            if self.tiles.get(q) in WALK and q not in self.entities:
                self.blocked[q] = self.tick + 8
        if action == 'do' and not gained and not any(k not in ('food','drink','energy','health') for k in delta):
            key = ('do',front,self.tiles.get(front))
            self.failures[key] += 1
            if self.failures[key] >= 5 and self.tiles.get(front) != 'grass' and front not in self.entities and self.tiles.get(front) != 'water':
                self.blocked[front] = self.tick + 100
        self.last_action = action
        self.last_reason = self.reason

    def run_episode(self, game):
        self.names = set(game.action_spec.names)
        obs = self.ctx.observe()
        if not isinstance(obs,dict):
            return
        self.update(obs)
        while not self.ctx.done and not obs.get('done',False):
            if len([k for k,v in self.ach.items() if v]) >= 22:
                break
            if self.ctx.budget.remaining().get('wall_s',0) < 2:
                break
            self.review()
            if self.ctx.done:
                break
            action = self.choose()
            if action not in self.names:
                action = 'noop'
            try:
                new = self.ctx.act(action)
            except Exception as exc:
                self.ctx.log('action_failure',action=action,error=type(exc).__name__)
                if self.ctx.done:
                    break
                # No repeated illegal action loops: leave with earned score.
                return
            if not isinstance(new,dict):
                new = self.ctx.observe()
            if not isinstance(new,dict):
                break
            self.record(obs,action,new)
            obs = new
            self.update(obs)


def build(ctx):
    return Harness(ctx)
