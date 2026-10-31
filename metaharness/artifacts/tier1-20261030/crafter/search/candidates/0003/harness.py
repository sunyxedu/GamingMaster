"""Camp-and-expedition controller. All world knowledge comes from observations."""
from __future__ import annotations

import heapq
import json
from collections import Counter, deque

DIRS = ((1, 0, 'move_right'), (0, 1, 'move_down'), (-1, 0, 'move_left'), (0, -1, 'move_up'))
WALK = {'grass', 'path', 'sand'}
RECIPES = {
    'wood_pickaxe': {'wood': 1}, 'wood_sword': {'wood': 1},
    'stone_pickaxe': {'wood': 1, 'stone': 1}, 'stone_sword': {'wood': 1, 'stone': 1},
    'iron_pickaxe': {'wood': 1, 'coal': 1, 'iron': 1},
    'iron_sword': {'wood': 1, 'coal': 1, 'iron': 1},
}


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.tiles = {}
        self.entities = {}
        self.threat_memory = {}
        self.plants = {}
        self.visits = Counter()
        self.failures = Counter()
        self.blocked = {}
        self.events = deque(maxlen=14)
        self.recipes = {k: dict(v) for k, v in RECIPES.items()}
        self.camp = None
        self.table = None
        self.furnace = None
        self.calls = 0
        self.last_review = 0
        self.last_unlock = 0
        self.focus = 'balanced'
        self.heading = (1, 0)
        self.drinking = False
        self.feeding = False
        self.distance = {}
        self.first = {}
        self.last_intent = ''
        self.step = 0
        self.plant_cooldown = {}
        self.placement_targets = {}
        self.verified_recipes = set()
        self.farm_effort = 0
        self.last_farm = 0

    def update(self, obs):
        if isinstance(obs, str):
            obs = json.loads(obs)
        self.obs = obs
        self.inv = obs.get('inventory', {})
        self.ach = obs.get('achievements', {})
        self.pos = tuple(obs['player']['position'])
        self.face = tuple(obs['player'].get('facing', (0, 1)))
        self.step = int(obs.get('step', self.step + 1))
        self.entities = {}
        visible = set()
        for row in obs.get('local_map', {}).get('rows', []):
            for cell in row:
                p = (cell['x'], cell['y'])
                visible.add(p)
                self.tiles[p] = cell.get('material', 'unknown')
                obj = cell.get('object', '')
                if obj and obj != 'player':
                    self.entities[p] = dict(cell)
                if obj.startswith('plant'):
                    self.plants[p] = obj
                else:
                    self.plants.pop(p, None)
                if cell.get('material') == 'table':
                    if self.table is None:
                        self.table = p
                if cell.get('material') == 'furnace':
                    if self.furnace is None:
                        self.furnace = p
        self.visits[self.pos] += 1
        self.enemies = {p: c for p, c in self.entities.items() if c.get('object') in ('zombie', 'skeleton')}
        # Threats just outside the moving view must not vanish from route costs.
        # Visible empty tiles invalidate memories immediately; old unseen threats expire.
        self.threat_memory = {p: t for p, t in self.threat_memory.items()
                              if p not in visible and self.step - t <= 12}
        self.threat_memory.update({p: self.step for p in self.enemies})
        self.hp = self.inv.get('health', obs['player'].get('health', 9))
        self.pick = 3 if self.inv.get('iron_pickaxe') else 2 if self.inv.get('stone_pickaxe') else 1 if self.inv.get('wood_pickaxe') else 0
        self.sword = 3 if self.inv.get('iron_sword') else 2 if self.inv.get('stone_sword') else 1 if self.inv.get('wood_sword') else 0

    def unlocked(self, name):
        return self.ach.get(name, 0) > 0

    def mineable(self, material):
        return material == 'tree' or (material in ('stone', 'coal') and self.pick >= 1) or (material == 'iron' and self.pick >= 2) or (material == 'diamond' and self.pick >= 3)

    def walkable(self, p):
        return self.tiles.get(p) in WALK and p not in self.entities and p not in self.plants and self.blocked.get(p, 0) <= self.step

    def paths(self):
        """One observation-conditioned Dijkstra tree; mining is an explicit edge cost."""
        self.distance = {self.pos: 0.0}
        self.first = {self.pos: None}
        queue = [(0.0, self.pos)]
        count = 0
        while queue and count < 2200:
            cost, p = heapq.heappop(queue)
            if cost != self.distance.get(p):
                continue
            count += 1
            for dx, dy, action in DIRS:
                q = (p[0] + dx, p[1] + dy)
                mat = self.tiles.get(q)
                if mat not in WALK and not self.mineable(mat):
                    continue
                if self.blocked.get(q, 0) > self.step or q in self.plants:
                    continue
                entity = self.entities.get(q, {}).get('object')
                if entity and q != self.pos:
                    continue
                price = 1.0 if mat in WALK else 4.0
                price += min(0.3, self.visits[q] * .01)
                for e in self.threat_memory:
                    d = dist(q, e)
                    if d <= 3:
                        price += (4 - d) * (3 if self.hp >= 6 and self.sword else 9)
                new = cost + price
                if new < self.distance.get(q, float('inf')):
                    self.distance[q] = new
                    self.first[q] = q if p == self.pos else self.first[p]
                    heapq.heappush(queue, (new, q))

    def face_action(self, target):
        d = (target[0] - self.pos[0], target[1] - self.pos[1])
        if d == self.face:
            return 'do'
        return next((a for dx, dy, a in DIRS if (dx, dy) == d), None)

    def go(self, target):
        if target == self.pos:
            return None
        q = self.first.get(target)
        if q is None:
            return None
        d = (q[0] - self.pos[0], q[1] - self.pos[1])
        if self.tiles.get(q) not in WALK:
            return self.face_action(q)
        return next((a for dx, dy, a in DIRS if (dx, dy) == d), None)

    def approach(self, target):
        if dist(self.pos, target) == 1:
            return self.face_action(target)
        choices = [(self.distance[q], q) for dx, dy, _ in DIRS
                   for q in [(target[0] + dx, target[1] + dy)]
                   if q in self.distance]
        if not choices:
            return None
        return self.go(min(choices)[1])

    def target_cost(self, target):
        return min((self.distance.get((target[0] + dx, target[1] + dy), 1e6) for dx, dy, _ in DIRS), default=1e6)

    def interact(self, targets, intent, maxcost=150):
        candidates = [(self.target_cost(p), p) for p in targets
                      if self.failures[(intent, p)] < 8 and self.blocked.get(p, 0) <= self.step]
        if not candidates:
            return None
        cost, target = min(candidates)
        if cost > maxcost:
            return None
        self.last_intent = intent
        self.target = target
        return self.approach(target)

    def gather(self, material, maxcost=150):
        return self.interact([p for p, m in self.tiles.items() if m == material], 'gather_' + material, maxcost)

    def explore(self, purpose='resources'):
        """Move to a reachable visibility frontier, preferring short safe expeditions."""
        choices = []
        for p, cost in self.distance.items():
            if self.tiles.get(p) not in WALK or p == self.pos or cost > 75:
                continue
            unseen = sum((p[0] + dx, p[1] + dy) not in self.tiles
                         for dx, dy in ((-4, -4), (-4, 0), (-4, 4), (0, -4), (0, 4), (4, -4), (4, 0), (4, 4)))
            if not unseen:
                continue
            displacement = ((p[0] - self.pos[0]) * self.heading[0] + (p[1] - self.pos[1]) * self.heading[1])
            tether = 0 if self.camp is None else max(0, dist(p, self.camp) - 24) * .15
            value = cost + tether - unseen * 2 - displacement * .1 + self.visits[p] * .2
            choices.append((value, p))
        if choices:
            target = min(choices)[1]
            self.last_intent = 'explore_' + purpose
            self.target = target
            return self.go(target)
        # No frontier: remove a reachable mineable boundary, never wander randomly.
        return self.interact([p for p, m in self.tiles.items() if self.mineable(m)], 'open_boundary', 80)

    def choose_camp(self):
        water = [p for p, m in self.tiles.items() if m == 'water']
        stone = [p for p, m in self.tiles.items() if m == 'stone']
        trees = [p for p, m in self.tiles.items() if m == 'tree']
        choices = []
        for p, cost in self.distance.items():
            if cost > 25 or self.tiles.get(p) != 'grass':
                continue
            # Keep all cardinal station/turning tiles open, plus surrounding farm space.
            if any(self.tiles.get((p[0] + dx, p[1] + dy)) not in WALK for dx, dy, _ in DIRS):
                continue
            wd = min((dist(p, q) for q in water), default=30)
            if wd > 12 and self.step < 100:
                continue
            sd = min((dist(p, q) for q in stone), default=20)
            td = min((dist(p, q) for q in trees), default=15)
            value = wd * 2 + sd * .5 + td * .3 + cost * .3
            value += sum(max(0, 6 - dist(p, q)) * 5 for q in self.enemies)
            choices.append((value, p))
        if choices:
            self.camp = min(choices)[1]
            self.table = add(self.camp, (1, 0))
            self.furnace = add(self.camp, (0, 1))
            self.ctx.log('camp_selected', position=list(self.camp), step=self.step)
            return True
        return False

    def place(self, kind, target):
        self.last_intent = 'place_' + kind
        self.target = target
        if target in self.entities or self.tiles.get(target) not in WALK:
            return None
        if kind == 'plant' and self.tiles.get(target) != 'grass':
            return None
        if self.failures[(self.last_intent, target)] >= 4:
            return None
        if add(self.pos, self.face) == target:
            return 'place_' + kind
        # Walk into the adjacent stand tile facing the placement tile. Facing is not free.
        choices = []
        for dx, dy, action in DIRS:
            stand = (target[0] - dx, target[1] - dy)
            behind = (target[0] - 2 * dx, target[1] - 2 * dy)
            if not self.walkable(stand) and stand != self.pos:
                continue
            if self.pos == behind:
                choices.append((0, action))
            elif behind in self.distance and self.walkable(behind):
                act = self.go(behind)
                if act:
                    choices.append((self.distance[behind] + 1, act))
        return min(choices)[1] if choices else None

    def place_any(self, kind, candidates):
        # Keep a placement target while aligning facing. Re-selecting a tile relative
        # to the moving player can move the goal every tick and never place anything.
        current = self.placement_targets.get(kind)
        candidates = list(candidates)
        if current in candidates:
            a = self.place(kind, current)
            if a:
                return a
        self.placement_targets.pop(kind, None)
        front = add(self.pos, self.face)
        candidates.sort(key=lambda p: (p != front, self.target_cost(p)))
        for p in candidates:
            a = self.place(kind, p)
            if a:
                self.placement_targets[kind] = p
                return a
        return None

    def near_station(self, iron=False):
        targets = []
        for p, cost in self.distance.items():
            if self.tiles.get(p) not in WALK:
                continue
            if self.table is None or max(abs(p[0] - self.table[0]), abs(p[1] - self.table[1])) > 1:
                continue
            if iron and (self.furnace is None or max(abs(p[0] - self.furnace[0]), abs(p[1] - self.furnace[1])) > 1):
                continue
            targets.append((cost, p))
        return min(targets)[1] if targets else None

    def craft(self, item):
        if self.inv.get(item, 0):
            return None
        costs = self.recipes[item]
        for material, amount in costs.items():
            if self.inv.get(material, 0) < amount:
                action = self.gather('tree' if material == 'wood' else material)
                return action or self.explore(material)
        if self.table is None or self.tiles.get(self.table) != 'table':
            return self.establish()
        if item.startswith('iron') and (self.furnace is None or self.tiles.get(self.furnace) != 'furnace'):
            return self.build_furnace()
        p = self.near_station(item.startswith('iron'))
        if p is None:
            return None
        self.last_intent = 'craft_' + item
        self.target = p
        if p != self.pos:
            return self.go(p)
        signature = ('make_' + item, tuple(sorted(self.inv.items())), self.pos)
        if self.failures[signature] >= 2:
            return None
        return 'make_' + item

    def establish(self):
        if self.inv.get('wood', 0) < 4:
            return self.gather('tree') or self.explore('wood')
        if self.camp is None and not self.choose_camp():
            return self.explore('camp_with_water')
        if self.table and self.tiles.get(self.table) != 'table':
            return self.place('table', self.table)
        return None

    def build_furnace(self):
        if self.inv.get('stone', 0) < 4:
            return self.gather('stone') or self.explore('stone')
        if self.camp is None:
            return self.establish()
        return self.place('furnace', self.furnace)

    def plant(self):
        if not self.camp:
            return None
        self.farm_effort += 1
        self.last_farm = self.step
        if self.inv.get('sapling', 0) == 0:
            return self.interact([p for p, m in self.tiles.items() if m == 'grass' and p not in self.entities and p not in self.plants and dist(p, self.camp) < 7], 'gather_sapling', 30)
        slots = [p for p, m in self.tiles.items() if m == 'grass' and 2 <= dist(p, self.camp) <= 4 and p not in self.entities and p not in self.plants and p not in (self.table, self.furnace) and self.failures[('place_plant', p)] < 4]
        return self.place_any('plant', slots)

    def food(self, maxcost=100):
        ripe = [p for p, obj in self.plants.items() if 'ripe' in obj and self.plant_cooldown.get(p, 0) < self.step]
        a = self.interact(ripe, 'eat_plant', maxcost)
        if a:
            return a
        return self.interact([p for p, c in self.entities.items() if c.get('object') == 'cow'], 'eat_cow', maxcost)

    def survival(self):
        close = sorted(self.enemies, key=lambda p: dist(self.pos, p))
        adjacent = [p for p in close if dist(self.pos, p) == 1]
        if adjacent:
            # Finish a wounded enemy; fleeing from a one-hit enemy caused avoidable deaths.
            target = min(adjacent, key=lambda p: (self.enemies[p].get('health', 5), add(self.pos, self.face) != p))
            damage = (1, 2, 3, 5)[self.sword]
            lethal = self.enemies[target].get('health', 5) <= damage
            if lethal or (self.hp >= 5 and len(adjacent) == 1) or (self.sword >= 2 and self.hp >= 3):
                self.last_intent = 'combat'
                self.target = target
                return self.face_action(target)
            choices = []
            for dx, dy, a in DIRS:
                p = add(self.pos, (dx, dy))
                if self.walkable(p):
                    choices.append((min(dist(p, q) for q in close), -self.visits[p], a))
            if choices:
                return max(choices)[2]
            self.last_intent = 'combat'
            self.target = target
            return self.face_action(target)
        food, drink, energy = (self.inv.get(k, 9) for k in ('food', 'drink', 'energy'))
        self.drinking = drink <= 4 or (self.drinking and drink < 9)
        if self.drinking:
            a = self.gather('water', 160)
            if a:
                return a
        self.feeding = food <= 4 or (self.feeding and food < 7)
        if self.feeding:
            a = self.food()
            if a:
                return a
        safe = not close or dist(self.pos, close[0]) > 5
        if self.obs['player'].get('sleeping') and safe:
            return 'sleep'
        if safe and drink >= 3 and food >= 3 and (energy <= 3 or (not self.unlocked('wake_up') and energy <= 7)):
            self.last_intent = 'rest'
            return 'sleep'
        if drink <= 1 and not self.gather('water', 160):
            return self.explore('water')
        if food <= 1:
            return self.food(200) or self.explore('food')
        return None

    def policy(self):
        self.last_intent = 'idle'
        self.target = self.pos
        a = self.survival()
        if a:
            return a
        # Deterministic opening; no latency barrier before securing useful achievements.
        if not self.unlocked('place_table'):
            return self.establish() or self.explore('camp')
        for item in ('wood_sword', 'wood_pickaxe'):
            if not self.inv.get(item):
                a = self.craft(item)
                if a:
                    return a
        if not self.unlocked('collect_drink'):
            a = self.gather('water', 60)
            if a:
                return a
        if not self.unlocked('eat_cow'):
            a = self.interact([p for p, c in self.entities.items() if c.get('object') == 'cow'], 'eat_cow', 10)
            if a:
                return a
        # Provision one round trip for both stone tools, furnace and a placement.
        stone_tools_missing = [x for x in ('stone_sword', 'stone_pickaxe') if not self.inv.get(x)]
        if stone_tools_missing:
            need_wood = sum(self.recipes[x].get('wood', 0) for x in stone_tools_missing)
            if self.inv.get('wood', 0) < need_wood:
                return self.gather('tree') or self.explore('wood')
            need_stone = sum(self.recipes[x].get('stone', 0) for x in stone_tools_missing)
            # Stop gathering early if health is low; a sword upgrade takes priority.
            if self.hp >= 6:
                need_stone += (0 if self.unlocked('place_furnace') else 4)
                need_stone += (0 if self.unlocked('place_stone') else 1)
            if self.inv.get('stone', 0) < need_stone:
                return self.gather('stone') or self.explore('stone')
            for item in stone_tools_missing:
                a = self.craft(item)
                if a:
                    return a
        # Farming never preempts the stone sword and has a bounded work allowance.
        plant_goal = 3 if self.focus == 'farm' else 1 if self.focus == 'mine' else 2
        farm_due = (not self.unlocked('place_plant') or len(self.plants) < plant_goal)
        if farm_due and self.farm_effort < 65:
            a = self.plant()
            if a:
                return a
        if not self.unlocked('place_stone'):
            if self.inv.get('stone', 0) == 0:
                return self.gather('stone') or self.explore('stone')
            candidates = [p for p, mat in self.tiles.items() if mat in WALK
                          and p not in self.entities and p not in self.plants
                          and p not in (self.table, self.furnace, self.camp)
                          and (not self.camp or dist(p, self.camp) >= 3)
                          and self.target_cost(p) < 15]
            a = self.place_any('stone', candidates)
            if a:
                return a
        if not self.unlocked('place_furnace'):
            a = self.build_furnace()
            if a:
                return a
        if not self.unlocked('eat_plant'):
            a = self.interact([p for p, obj in self.plants.items() if 'ripe' in obj], 'eat_plant', 100)
            if a:
                return a
        if not self.unlocked('collect_coal'):
            a = self.gather('coal')
            if a:
                return a
        if self.focus == 'hunt' and self.hp >= 6 and self.sword >= 2:
            a = self.interact([p for p, c in self.enemies.items() if not self.unlocked('defeat_' + c['object'])], 'hunt', 25)
            if a:
                return a
        # Return-to-camp is explicit: a gathered load is converted into tools promptly.
        iron_missing = [x for x in ('iron_pickaxe', 'iron_sword') if not self.inv.get(x)]
        if iron_missing:
            for material in ('wood', 'coal', 'iron'):
                amount = self.recipes[iron_missing[0]].get(material, 0)
                if self.inv.get(material, 0) < amount:
                    return self.gather('tree' if material == 'wood' else material) or self.explore(material)
        for item in ('iron_pickaxe', 'iron_sword'):
            if not self.inv.get(item):
                a = self.craft(item)
                if a:
                    return a
        if not self.unlocked('collect_diamond'):
            a = self.gather('diamond')
            if a:
                return a
        if self.hp >= 6 and self.sword >= 2:
            targets = [p for p, c in self.enemies.items() if not self.unlocked('defeat_' + c['object'])]
            a = self.interact(targets, 'hunt', 35)
            if a:
                return a
        if not self.unlocked('eat_cow'):
            a = self.interact([p for p, c in self.entities.items() if c.get('object') == 'cow'], 'eat_cow', 80)
            if a:
                return a
        if not self.unlocked('eat_plant') and self.camp:
            if not self.plants:
                a = self.plant()
                if a:
                    return a
            if dist(self.pos, self.camp) > 4:
                a = self.go(self.camp)
                if a:
                    return a
        return self.explore('diamond' if self.pick == 3 else 'iron') or 'noop'

    def review(self):
        remaining = self.ctx.budget.remaining()
        if self.calls >= 5 or remaining.get('system2_calls', 0) <= 1 or remaining.get('wall_s', 0) < 200:
            return
        self.calls += 1
        self.last_review = self.step
        resources = {}
        for mat in ('water', 'tree', 'stone', 'coal', 'iron', 'diamond'):
            ps = [p for p, m in self.tiles.items() if m == mat]
            resources[mat] = [list(p) for p in sorted(ps, key=lambda p: self.target_cost(p))[:6]]
        state = {'step': self.step, 'position': self.pos, 'camp': self.camp,
                 'inventory': self.inv, 'missing': [k for k, v in self.ach.items() if not v],
                 'resources': resources, 'plants': [[*p, obj] for p, obj in self.plants.items()],
                 'recent': list(self.events), 'current_intent': self.last_intent,
                 'recipes': self.recipes, 'focus': self.focus}
        prompt = (
            'You advise a Crafter camp-and-expedition controller. Score = unique achievements /22. '
            'Code manages every movement, facing, path, survival interrupt, crafting and resource acquisition. '
            'It establishes a camp with table/furnace near water and plants crops, then upgrades tools and mines. '
            'Recommend exploration heading and one priority. Recipes shown are hypotheses; correct only if evidence warrants. '
            'Output ONLY compact JSON: {"heading":"north|east|south|west", '
            '"focus":"balanced|farm|mine|hunt", "reason":"brief", '
            '"recipe_updates":{"iron_pickaxe":{"wood":1,"coal":1,"iron":1}}}. '
            'Omit recipe_updates unless needed. Do not output primitive actions.\n' + json.dumps(state, separators=(',', ':')))
        try:
            reply = self.ctx.think(prompt, timeout_s=min(150.0, remaining['wall_s'] - 20))
            txt = reply.text
            start, end = txt.find('{'), txt.rfind('}')
            data = json.loads(txt[start:end + 1]) if start >= 0 else {}
            if not isinstance(data, dict):
                return
            h = data.get('heading')
            self.heading = {'north': (0, -1), 'south': (0, 1), 'east': (1, 0), 'west': (-1, 0)}.get(h, self.heading)
            if data.get('focus') in ('balanced', 'farm', 'mine', 'hunt'):
                self.focus = data['focus']
            updates = data.get('recipe_updates', {})
            if isinstance(updates, dict):
                for item, costs in updates.items():
                    if item in self.recipes and item not in self.verified_recipes and isinstance(costs, dict) and costs and all(k in ('wood', 'stone', 'coal', 'iron') and type(v) is int and 1 <= v <= 5 for k, v in costs.items()):
                        self.recipes[item] = costs
            self.ctx.log('camp_review', decision=data, step=self.step)
        except Exception as exc:
            self.ctx.log('review_failed', error=type(exc).__name__)

    def run_episode(self, game):
        obs = self.ctx.observe()
        if obs is None:
            obs = game.observe()
        self.update(obs)
        names = set(game.action_spec.names)
        while not self.ctx.done and not self.obs.get('done', False) and self.step < 9500:
            if self.ctx.budget.remaining().get('wall_s', 0) <= 2:
                break
            self.paths()
            if (self.calls == 0 and self.step >= 120) or (self.step - self.last_review >= 400 and self.step - self.last_unlock >= 160):
                self.review()
                if self.ctx.done:
                    break
            action = self.policy()
            if self.step % 25 == 0:
                self.ctx.log('camp_status', step=self.step, intent=self.last_intent, camp=self.camp, position=self.pos, inventory=self.inv, farm_effort=self.farm_effort)
            if action not in names:
                action = 'noop'
            old_inv = dict(self.inv)
            old_ach = dict(self.ach)
            old_pos = self.pos
            old_tiles = self.tiles.get(self.target)
            intent, target = self.last_intent, self.target
            signature = (action, tuple(sorted(self.inv.items())), self.pos)
            try:
                obs = self.ctx.act(action)
            except Exception as exc:
                if self.ctx.done:
                    break
                self.events.append({'step': self.step, 'action_error': type(exc).__name__, 'action': action})
                self.failures[signature] += 1
                if self.failures[signature] >= 3:
                    break
                continue
            if obs is None:
                obs = self.ctx.observe()
            if obs is None:
                break
            self.update(obs)
            delta = {k: v - old_inv.get(k, 0) for k, v in self.inv.items() if v != old_inv.get(k, 0)}
            new = [k for k, v in self.ach.items() if v and not old_ach.get(k)]
            if new:
                self.last_unlock = self.step
                self.ctx.log('camp_unlock', step=self.step, achievements=new, intent=intent)
            if new or action.startswith(('make_', 'place_')) or any(k not in ('health', 'energy', 'food', 'drink') for k in delta):
                self.events.append({'step': self.step, 'action': action, 'delta': delta, 'new': new})
            useful = bool(new) or any(k not in ('health', 'energy') and v > 0 for k, v in delta.items()) or self.tiles.get(target) != old_tiles
            if action == 'do' and not useful and intent not in ('combat', 'hunt', 'eat_cow', 'gather_sapling'):
                self.failures[(intent, target)] += 1
                if intent == 'eat_plant':
                    self.plant_cooldown[target] = self.step + 80
            elif useful:
                self.failures[(intent, target)] = 0
            if action.startswith(('make_', 'place_')) and not new and not any(v < 0 for k, v in delta.items() if k in ('wood', 'stone', 'coal', 'iron', 'sapling')):
                self.failures[signature] += 1
                self.failures[(intent, target)] += 1
            if action.startswith('make_') and useful:
                costs = {k: -v for k, v in delta.items() if k in ('wood', 'stone', 'coal', 'iron') and v < 0}
                if costs:
                    self.recipes[action[5:]] = costs
                    self.verified_recipes.add(action[5:])
            if action.startswith('move_') and self.pos == old_pos:
                d = next(((dx, dy) for dx, dy, a in DIRS if a == action), (0, 0))
                q = add(old_pos, d)
                if self.tiles.get(q) in WALK and q not in self.entities and q not in self.plants:
                    self.blocked[q] = self.step + 5
            if len([k for k, v in self.ach.items() if v]) >= 22:
                break


def build(ctx):
    return Harness(ctx)
