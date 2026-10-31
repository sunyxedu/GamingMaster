"""Model-led, transactional macro plans over a verified movement graph.

All state comes from observations; there is no NLE dependency or seed policy.
"""
from collections import Counter, deque
import json
import re

DIRS = {'n': (0, -1), 'e': (1, 0), 's': (0, 1), 'w': (-1, 0),
        'ne': (1, -1), 'se': (1, 1), 'sw': (-1, 1), 'nw': (-1, -1)}
KEYS = dict(zip('kljhunby', ['move '+d for d in DIRS]))
KEYS.update({k.upper(): a.replace('move ', 'run ') for k, a in list(KEYS.items())})
KEYS.update({'e': 'eat', 's': 'search', '.': 'wait', '>': 'down', '<': 'up'})
FOOD = ('food ration', 'lembas', 'cram ration', 'pancake', 'apple', 'orange',
        'pear', 'melon', 'banana', 'carrot', 'slime mold', 'fortune cookie',
        'cream pie', 'candy bar', 'clove of garlic', 'sprig of wolfsbane')
OPS = {'go', 'door', 'search', 'fight', 'rest', 'eat', 'explore', 'act', 'descend'}


def rows(v):
    if isinstance(v, str):
        return v.splitlines()
    if not isinstance(v, (list, tuple)):
        return []
    out = []
    for r in v:
        if isinstance(r, str):
            out.append(r)
        elif isinstance(r, (list, tuple)):
            out.append(''.join(chr(c) if isinstance(c, int) and 0 < c < 256 else ' ' for c in r))
    return out


def number(v, default=0):
    try:
        return int(v)
    except (ValueError, TypeError, OverflowError):
        return default


def xy(v):
    if isinstance(v, (list, tuple)) and len(v) == 2:
        p = (number(v[0], -1), number(v[1], -1))
        if 0 <= p[0] < 79 and 0 <= p[1] < 21:
            return p
    return None


def dist(p, q):
    return max(abs(p[0]-q[0]), abs(p[1]-q[1]))


def parse(text, names):
    """Strict data-only parser: bad fields are ignored, never evaluated."""
    obj = {}
    s = str(text).strip()
    try:
        obj = json.JSONDecoder().raw_decode(s[s.index('{'):])[0]
    except (ValueError, TypeError):
        pass
    if not isinstance(obj, dict):
        obj = {}
    raw = obj.get('plan', [])
    if not isinstance(raw, list):
        raw = []
    plan = []
    for item in raw[:10]:
        if not isinstance(item, dict) or not isinstance(item.get('op'), str) or item['op'] not in OPS:
            continue
        op = item['op']
        q = {'op': op, 'n': max(1, min(120, number(item.get('n', 40), 40))), 'used': 0}
        if op in ('go', 'door', 'fight') and xy(item.get('xy')) is None:
            continue
        if xy(item.get('xy')) is not None:
            q['xy'] = xy(item['xy'])
        if op == 'act':
            action = item.get('action')
            if not isinstance(action, str) or action not in names:
                continue
            q['action'] = action
            q['n'] = min(12, q['n'])
        if op == 'eat':
            letter = item.get('letter', '')
            if isinstance(letter, str) and letter in KEYS:
                q['letter'] = letter
        plan.append(q)
    if not plan and s in names:
        plan = [{'op': 'act', 'action': s, 'n': 1, 'used': 0}]
    combat = obj.get('fight', [])
    if not isinstance(combat, list):
        combat = []
    return plan, {g for g in combat if isinstance(g, int) and 0 <= g < 6000}, str(obj.get('note', ''))[:400]


class Level:
    def __init__(self):
        self.terrain = {}
        self.visits = Counter()
        self.searched = Counter()
        self.blocked = Counter()
        self.doors = Counter()
        self.stairs = {}
        self.probed = Counter()


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.levels = {}
        self.plan = deque()
        self.history = deque(maxlen=18)
        self.outcomes = deque(maxlen=12)
        self.calls = 0
        self.steps = 0
        self.last = None
        self.fight = set()
        self.note = ''
        self.reason = 'opening: select a useful multi-goal plan'
        self.pending_dir = None
        self.pending_food = None
        self.last_eat = -1000
        self.stale = 0
        self.prompt_count = 0
        self.last_prompt = None
        self.batch_steps = 0
        self.batch_hp = 0
        self.last_call_step = -100
        self.failures = 0
        self.last_damage = -1000
        self.no_model = False
        self.identity = ''

    def at(self, key, p, default=0):
        try:
            return number(self.obs[key][p[1]][p[0]], default)
        except (KeyError, IndexError, TypeError):
            return default

    def update(self, obs):
        self.obs = obs if isinstance(obs, dict) else {}
        self.stats = self.obs.get('stats', {})
        if not isinstance(self.stats, dict):
            self.stats = {}
        self.pos = (number(self.stats.get('x')), number(self.stats.get('y')))
        self.key = (number(self.stats.get('dungeon_number')), number(self.stats.get('dungeon_level', self.stats.get('depth', 1))))
        self.level = self.levels.setdefault(self.key, Level())
        self.map = [r[:79].ljust(79) for r in rows(self.obs.get('map', self.obs.get('chars', [])))[:21]]
        self.map += [' '*79] * (21-len(self.map))
        self.msg = str(self.obs.get('message', ''))
        if not self.identity and 'welcome to NetHack' in self.msg:
            self.identity = self.msg
        self.tty = '\n'.join(rows(self.obs.get('tty', self.obs.get('tty_chars', []))))
        self.hp = number(self.stats.get('hp'), 1)
        self.maxhp = max(1, number(self.stats.get('hp_max'), 1))
        self.turn = number(self.stats.get('time'))
        self.hunger = number(self.stats.get('hunger'), 1)
        self.monsters = {}
        for y, row in enumerate(self.map):
            for x, c in enumerate(row):
                p = (x, y)
                if c == ' ':
                    continue
                special = self.at('specials', p)
                glyph = self.at('glyphs', p, -1)
                monster = (c.isalpha() or c in ':;&@') and p != self.pos and not special & 32
                if monster:
                    self.monsters[p] = {'xy': p, 'char': c, 'glyph': glyph,
                                        'color': self.at('colors', p), 'pet': bool(special & 8)}
                    self.level.terrain.setdefault(p, '.')
                elif p != self.pos:
                    # Brown -/| and these archived glyphs are open doorways.
                    terrain = '/' if (c in '|-' and self.at('colors', p) == 3) or glyph in (2371, 2372) else c
                    old = self.level.terrain.get(p)
                    self.level.terrain[p] = terrain
                    if old != terrain and terrain in '.#/<>':
                        # New geometry overrides an earlier failed probe into stone.
                        for edge in list(self.level.blocked):
                            if edge[1] == p:
                                del self.level.blocked[edge]
                    if c in '<>':
                        self.level.stairs[p] = c
        self.level.terrain.setdefault(self.pos, '.')
        self.level.visits[self.pos] += 1
        if self.msg and (not self.history or self.history[-1] != self.msg):
            self.history.append(self.msg)
        low = self.msg.lower()
        if 'stairs leading down' in low or 'staircase down' in low:
            self.level.stairs[self.pos] = '>'
        self.hostiles = {p: m for p, m in self.monsters.items() if not m['pet']}
        if self.last is not None:
            key, pos, hp, turn, action, prompt = self.last
            changed = pos != self.pos or key != self.key or turn != self.turn
            self.stale = 0 if changed else self.stale+1
            if key != self.key:
                self.invalidate('level changed: plan coordinates are obsolete')
            if self.hp < hp:
                self.last_damage = self.steps
            if key == self.key and not prompt and action.startswith('move '):
                dx, dy = DIRS[action[5:]]
                q = (pos[0]+dx, pos[1]+dy)
                dynamic_block = 'in the way' in low or 'swap places' in low
                if dynamic_block:
                    self.level.blocked.pop((pos, q), None)
                if self.pos == pos and self.turn == turn and not self.prompt_kind() and q not in self.monsters and not dynamic_block:
                    self.level.blocked[(pos, q)] += 1
                if self.pos == pos and any(s in low for s in ('door is closed', 'door is locked', 'door resists')):
                    self.level.terrain[q] = '+'
                if self.pos == q:
                    self.level.blocked.pop((pos, q), None)
            if action == 'search' and not prompt:
                for dx, dy in DIRS.values():
                    self.level.searched[(pos[0]+dx, pos[1]+dy)] += 1
            if changed or self.msg:
                self.outcomes.append({'a': action, 'xy': self.pos, 'hp': self.hp, 'turn': self.turn, 'msg': self.msg[:150]})

    def invalidate(self, reason):
        self.plan.clear()
        self.reason = reason

    def prompt_kind(self):
        s = (self.msg+'\n'+self.tty[:240]).lower()
        if '--more--' in self.tty.lower() or '--more--' in self.msg.lower():
            return 'more'
        if 'what direction' in s:
            return 'direction'
        if 'what do you want to eat' in s:
            return 'eat'
        if re.search(r'\[[yn][ynq ]*\]', s) or '[yn' in s or '(y/n)' in s:
            return 'yesno'
        if '(end)' in self.tty.lower() or re.search(r'\(\d+ of \d+\)', self.tty):
            return 'more'
        return None

    def food(self):
        letters = self.obs.get('inventory_letters', [])
        choices = []
        for i, item in enumerate(self.obs.get('inventory', [])):
            if not isinstance(item, str) or not any(f in item.lower() for f in FOOD):
                continue
            try:
                k = chr(letters[i]) if isinstance(letters[i], int) else str(letters[i])
            except (ValueError, IndexError, TypeError):
                continue
            if k in KEYS:
                choices.append((0 if 'ration' in item or 'lembas' in item else 1, k))
        return min(choices)[1] if choices else None

    def prompt_action(self):
        kind = self.prompt_kind()
        if not kind:
            self.prompt_count = 0
            return None
        sig = (kind, self.msg)
        self.prompt_count = self.prompt_count+1 if sig == self.last_prompt else 1
        self.last_prompt = sig
        if kind == 'more':
            return 'more'
        if kind == 'direction':
            return 'move '+self.pending_dir if self.pending_dir in DIRS else 'wait'
        if kind == 'eat':
            k = self.pending_food or self.food()
            self.pending_food = None
            return KEYS.get(k, 'more')
        s = (self.msg+' '+self.tty[:180]).lower()
        yes = 'continue eating' in s or 'finish eating' in s or ('eat' in s and any(f in s for f in FOOD))
        # Never approve attacking a peaceful monster or an unknown corpse automatically.
        return 'move nw' if yes else 'move se'

    def passable(self, p):
        return p in self.level.terrain and self.level.terrain[p] not in ' |-=+^{}\\0`'

    def neighbors(self, p):
        for d, (dx, dy) in DIRS.items():
            q = (p[0]+dx, p[1]+dy)
            if not self.passable(q) or q in self.hostiles or self.level.blocked[(p, q)] > 0:
                continue
            if dx and dy:
                if self.level.terrain.get(p) == '/' or self.level.terrain.get(q) == '/':
                    continue
                if not self.passable((p[0]+dx, p[1])) and not self.passable((p[0], p[1]+dy)):
                    continue
            yield q, 'move '+d

    def paths(self):
        paths = {self.pos: []}
        queue = deque([self.pos])
        while queue:
            p = queue.popleft()
            for q, a in self.neighbors(p):
                if q not in paths:
                    paths[q] = paths[p]+[a]
                    queue.append(q)
        return paths

    def candidates(self, paths):
        choices = []
        for p, path in paths.items():
            c = self.level.terrain.get(p, ' ')
            if self.level.stairs.get(p) == '>':
                choices.append({'kind': 'downstairs', 'xy': p, 'distance': len(path)})
            if self.map[p[1]][p[0]] in '$%':
                choices.append({'kind': 'gold' if self.map[p[1]][p[0]] == '$' else 'food/corpse', 'xy': p, 'distance': len(path)})
            blanks = []
            for d in ('n', 'e', 's', 'w'):
                dx, dy = DIRS[d]
                q = (p[0]+dx, p[1]+dy)
                if 1 <= q[0] < 79 and 0 <= q[1] < 21 and q not in self.level.terrain and self.level.probed[(p, q)] < 1:
                    blanks.append(d)
            if blanks:
                choices.append({'kind': 'frontier', 'xy': p, 'probe': blanks, 'distance': len(path)})
        for p, c in self.level.terrain.items():
            if c != '+' or self.level.doors[p] >= 12:
                continue
            approach = [q for q in paths if abs(q[0]-p[0])+abs(q[1]-p[1]) == 1]
            if approach:
                choices.append({'kind': 'closed door', 'xy': p, 'distance': min(len(paths[q]) for q in approach)+1})
        priority = {'downstairs': 0, 'gold': 1, 'closed door': 2, 'frontier': 3, 'food/corpse': 4}
        return sorted(choices, key=lambda a: (priority[a['kind']], a['distance']))[:32]

    def go(self, p, paths):
        if p == self.pos:
            return None
        if p in paths and paths[p]:
            return paths[p][0]
        return None

    def door(self, p, paths):
        if self.level.terrain.get(p) != '+':
            return None
        if self.level.doors[p] >= 12:
            return None
        if abs(self.pos[0]-p[0])+abs(self.pos[1]-p[1]) == 1:
            d = next(d for d, delta in DIRS.items() if delta == (p[0]-self.pos[0], p[1]-self.pos[1]))
            self.pending_dir = d
            self.level.doors[p] += 1
            return 'kick'
        approach = [q for q in paths if abs(q[0]-p[0])+abs(q[1]-p[1]) == 1]
        if approach:
            return self.go(min(approach, key=lambda q: len(paths[q])), paths)
        return None

    def explore(self, paths):
        choices = self.candidates(paths)
        # A bounded macro authorized by Astra; only geometric goal ranking lives here.
        goals = []
        for q in choices:
            p = q['xy']; kind = q['kind']
            if kind == 'gold' and p != self.pos:
                goals.append((q['distance']-8, kind, p, q))
            elif kind == 'frontier':
                goals.append((q['distance']+self.level.visits[p]*0.15, kind, p, q))
            elif kind == 'closed door':
                goals.append((q['distance']+4+self.level.doors[p], kind, p, q))
        if goals:
            _, kind, p, q = min(goals)
            if kind == 'closed door':
                return self.door(p, paths)
            if self.pos != p:
                return self.go(p, paths)
            d = q['probe'][0]
            dx, dy = DIRS[d]
            target = (p[0]+dx, p[1]+dy)
            self.level.probed[(p, target)] += 1
            return 'move '+d
        # Search promising unexplored walls, distributing attempts over coverage.
        search = []
        for p, path in paths.items():
            adj = [(p[0]+dx, p[1]+dy) for dx, dy in DIRS.values()]
            walls = [q for q in adj if self.level.terrain.get(q) in ('|', '-')]
            if not walls:
                continue
            coverage = min(self.level.searched[q] for q in walls)
            if coverage < 24:
                corridor = self.level.terrain.get(p) in ('#', '/')
                search.append((coverage*2+len(path)+(0 if corridor else 5), p))
        if search:
            p = min(search)[1]
            return self.go(p, paths) if p != self.pos else 'search'
        return None

    def safety(self):
        # Interrupt the transaction on evidence that its risk assumptions changed.
        if not self.plan:
            return
        self.batch_hp = max(self.batch_hp, self.hp)
        op = self.plan[0]
        adjacent = {p: m for p, m in self.hostiles.items() if dist(p, self.pos) <= 1}
        approved = self.fight.copy()
        if op['op'] == 'fight':
            target = self.hostiles.get(op.get('xy'))
            if target:
                approved.add(target['glyph'])
            if 'glyph' in op:
                approved.add(op['glyph'])
        unknown = [m for m in adjacent.values() if m['glyph'] not in approved]
        if unknown and self.steps-self.last_call_step >= 3:
            self.invalidate('unapproved adjacent creature; choose fight, retreat, or wait for pet')
        elif self.hp < max(4, self.batch_hp*0.60) and self.steps-self.last_call_step >= 3 and op['op'] != 'rest':
            self.invalidate('HP loss exceeds transaction guard')
        elif self.hunger >= 3 and self.steps-self.last_call_step >= 8 and self.steps-self.last_eat > 30:
            self.invalidate('weak/faint hunger; replan for food')
        elif self.stale >= 8:
            self.invalidate('eight commands without movement or game-time progress')

    def execute(self):
        paths = self.paths()
        while self.plan:
            task = self.plan[0]
            if task['used'] >= task['n']:
                self.plan.popleft()
                continue
            op = task['op']; p = task.get('xy', self.pos)
            a = None
            if op == 'go':
                a = self.go(p, paths)
            elif op == 'door':
                a = self.door(p, paths)
            elif op == 'search':
                a = self.go(p, paths) if p != self.pos else 'search'
            elif op == 'rest':
                if self.hp < self.maxhp and not any(dist(p, self.pos) <= 2 for p in self.hostiles) and self.hunger < 3:
                    a = 'wait'
            elif op == 'eat':
                if task['used'] == 0:
                    self.pending_food = task.get('letter') or self.food()
                    if self.pending_food:
                        self.last_eat = self.steps
                        a = 'eat'
            elif op == 'fight':
                if 'glyph' not in task and p in self.hostiles:
                    task['glyph'] = self.hostiles[p]['glyph']
                opponents = [q for q, m in self.hostiles.items() if m['glyph'] == task.get('glyph') and dist(q, self.pos) <= 2]
                if opponents:
                    q = min(opponents, key=lambda q: dist(q, self.pos))
                    if dist(q, self.pos) == 1:
                        delta = (q[0]-self.pos[0], q[1]-self.pos[1])
                        a = 'move '+next(d for d in DIRS if DIRS[d] == delta)
                    else:
                        a = 'wait'
            elif op == 'descend':
                stairs = [q for q in paths if self.level.stairs.get(q) == '>']
                if stairs:
                    p = min(stairs, key=lambda q: len(paths[q]))
                    a = 'down' if p == self.pos else self.go(p, paths)
                else:
                    a = self.explore(paths)
            elif op == 'explore':
                a = self.explore(paths)
            elif op == 'act':
                a = task['action']
            if a:
                task['used'] += 1
                # During navigation, approved adjacent foes are handled one blow at a time.
                if op in ('explore', 'descend', 'search', 'door'):
                    foes = [(q, m) for q, m in self.hostiles.items() if dist(q, self.pos) == 1 and m['glyph'] in self.fight]
                    if foes and self.hp >= max(4, self.maxhp*0.35):
                        q, _ = foes[0]
                        delta = (q[0]-self.pos[0], q[1]-self.pos[1])
                        return 'move '+next(d for d in DIRS if DIRS[d] == delta)
                return a
            if op == 'go' and p != self.pos:
                self.history.append('Executor: go to %s is unreachable in the learned graph from %s.' % (p, self.pos))
            self.plan.popleft()
        return None

    def prompt(self, game):
        inv = []
        letters = self.obs.get('inventory_letters', [])
        for i, item in enumerate(self.obs.get('inventory', [])):
            if not item:
                continue
            k = chr(letters[i]) if i < len(letters) and isinstance(letters[i], int) and 0 < letters[i] < 256 else '?'
            inv.append({'letter': k, 'item': item, 'input_action': KEYS.get(k)})
        paths = self.paths()
        maptext = '\n'.join('%02d %s' % (i, r.rstrip()) for i, r in enumerate(self.map) if r.strip())
        state = {'stats': self.stats, 'message': self.msg, 'identity': self.identity, 'inventory': inv,
                 'creatures': list(self.monsters.values()), 'goals': self.candidates(paths),
                 'reason_for_call': self.reason, 'last_plan_note': self.note,
                 'outcomes': list(self.outcomes), 'recent_messages': list(self.history)[-8:],
                 'wall_s': round(self.ctx.budget.remaining().get('wall_s', 0)),
                 'blocked_edges': [list(edge) for edge, count in sorted(self.level.blocked.items(), key=lambda item: dist(item[0][0], self.pos)) if count > 0][:16],
                 'search_coverage': [{'xy': p, 'searched': n} for p, n in self.level.searched.most_common(12)]}
        return (
            'You control NetHack through bounded, verified macro plans. Maximize score by gaining experience, gold, '
            'exploring and descending while surviving. You own strategy; code only executes your plan. '
            'Return ONLY compact JSON, no analysis, under 400 tokens. Each call costs roughly a minute: '
            'plan 40-120 useful turns when safe. Coordinates are [x,y], zero based, y increases south.\n'
            'Available JSON: {"plan":[{"op":"explore","n":80}],"fight":[glyph_ids],"note":"brief intent"}. '
            'Plan may contain up to 10 ordered tasks:\n'
            'go xy: shortest safe known path; door xy: approach cardinally and kick a closed door until open; '
            'search xy n: navigate then search repeatedly; fight xy n: melee that creature, tracking nearby motion, '
            'stop when gone; rest n: wait until healed, interrupted by nearby creatures; eat letter: eat supported '
            'inventory item; explore n: geometry-only frontier/door/gold exploration and hidden-door search; '
            'descend n: find downstairs and descend, exploring until found; act action n: exact exposed command. '
            'All n bound command count (default 40), max120; go/door/fight stop early on completion. '
            'fight list authorizes melee with these observed glyph species during exploration; otherwise new '
            'adjacent creatures interrupt the plan. Large HP loss also interrupts. Exploration does NOT descend '
            'automatically. Rest heals without food if not weak but costs hunger/time. Choose realistic goals.\n'
            'Pets have pet:true; never attack pets or peaceful shopkeepers. Tourist may lack a wielded weapon. '
            'Only these commands exist: '+', '.join(self.names)+'. No pickup/wield/quaff/zap/read/escape key. '
            'The inventory input_action maps item letters to exposed commands, usable only in prompts. '
            'Code handles More, directions and safe eating confirmations. Do not issue more outside prompts. '
            'Do not eat unknown corpses. Gold is normally picked up by walking over it. Avoid traps/water/boulders. '
            'Known doors restrict diagonal movement; paths already handle this. Unknown creatures may be peaceful; '
            'Blocked_edges records failed movement probes; if a known-clear route contradicts them, an act move '
            'can retry that edge and successful movement repairs the graph. '
            'use message, glyph, color and map to classify. When no gold/stairs visible, explore rather than '
            'repeatedly waiting. At low HP use nearby pet, safe retreat and rest.\n'
            'STATE\n'+json.dumps(state, separators=(',', ':'))+'\nMAP (row labels; columns below)\n'
            '   '+''.join(str(i//10 or ' ') for i in range(79))+'\n'
            '   '+''.join(str(i%10) for i in range(79))+'\n'+maptext+
            '\nTTY status/prompts\n'+ '\n'.join(rows(self.obs.get('tty', []))[:2]+rows(self.obs.get('tty', []))[-2:]))

    def replan(self, game):
        remaining = self.ctx.budget.remaining()
        if self.no_model or remaining.get('system2_calls', 0) <= 1 or remaining.get('wall_s', 0) < 85:
            return False
        self.last_call_step = self.steps
        self.calls += 1
        self.ctx.log('macro_request', call=self.calls, stats=self.stats, reason=self.reason)
        try:
            result = self.ctx.think(self.prompt(game), session_id=None,
                                    timeout_s=min(210.0, max(1.0, remaining['wall_s']-8)))
            plan, fight, note = parse(result.text, self.names)
        except Exception as e:
            self.failures += 1
            self.ctx.log('macro_call_failed', error=type(e).__name__)
            if self.failures >= 3:
                self.no_model = True
            return False
        self.fight = fight
        self.note = note
        self.plan = deque(plan)
        self.batch_steps = 0
        self.batch_hp = self.hp
        self.ctx.log('macro_plan', plan=plan, fight=sorted(fight), note=note)
        return bool(plan)

    def fallback(self):
        # A degraded-mode policy, used only for invalid output, backend failure or exhausted budget.
        self.batch_hp = self.hp
        self.batch_steps = 0
        self.fight = {m['glyph'] for p, m in self.hostiles.items() if m['char'] in 'xr:o' and dist(p, self.pos) <= 2}
        if self.hunger >= 2 and self.food() and self.steps-self.last_eat > 30:
            task = {'op': 'eat', 'n': 1}
        elif self.hp < self.maxhp*0.7 and not any(dist(p, self.pos) <= 3 for p in self.hostiles) and self.hunger < 3:
            task = {'op': 'rest', 'n': 80}
        else:
            task = {'op': 'descend', 'n': 80}
        task['used'] = 0
        self.plan = deque([task])

    def run_episode(self, game):
        self.names = list(game.action_spec.names)
        try:
            obs = self.ctx.observe()
        except Exception:
            return
        # Bank a little verified map/gold progress before a potentially queued first call.
        # The ordinary contact/HP guards still stop this setup transaction immediately.
        self.plan = deque([{'op': 'explore', 'n': 24, 'used': 0}])
        updated_steps = -1
        while not self.ctx.done and self.steps < 12000:
            if updated_steps != self.steps:
                self.update(obs)
                updated_steps = self.steps
            if self.obs.get('done') or self.obs.get('truncated'):
                break
            if self.ctx.budget.remaining().get('wall_s', 0) <= 1:
                break
            action = self.prompt_action()
            prompt = action is not None
            if self.prompt_count >= 12:
                self.ctx.log('unresolvable_prompt', message=self.msg, stats=self.stats)
                break
            if not prompt:
                self.safety()
                if self.batch_steps >= 160:
                    self.invalidate('160-command transaction complete')
                if not self.plan:
                    if not self.replan(game):
                        self.fallback()
                    if self.ctx.done:
                        break
                action = self.execute()
                if action is None:
                    self.reason = 'plan completed or goal unreachable; choose next goals'
                    remaining = self.ctx.budget.remaining()
                    if self.no_model or remaining.get('system2_calls', 0) <= 1 or remaining.get('wall_s', 0) < 85:
                        self.ctx.log('no_reachable_goal', stats=self.stats)
                        break
                    # One retry next iteration; record why rather than silently wandering.
                    if self.steps == self.last_call_step:
                        self.fallback()
                        action = self.execute() or 'search'
                    else:
                        continue
            if action not in self.names:
                action = 'wait' if 'wait' in self.names else self.names[0]
            self.last = (self.key, self.pos, self.hp, self.turn, action, prompt)
            self.steps += 1
            self.batch_steps += 1
            if self.steps % 25 == 0:
                self.ctx.log('macro_checkpoint', steps=self.steps, stats=self.stats, action=action, tasks=len(self.plan))
            try:
                obs = self.ctx.act(action)
                if obs is None and not self.ctx.done:
                    obs = self.ctx.observe()
            except Exception as e:
                self.ctx.log('macro_action_failed', action=action, error=type(e).__name__)
                if self.ctx.done:
                    break
                self.invalidate('action error')
                try:
                    obs = self.ctx.observe()
                except Exception:
                    break
                self.failures += 1
                if self.failures >= 6:
                    break
            if self.stale >= 40:
                self.ctx.log('no_progress_stop', stats=self.stats, message=self.msg)
                break
        self.ctx.memory['macro_summary'] = {'calls': self.calls, 'steps': self.steps, 'note': self.note}


def build(ctx):
    return Harness(ctx)
