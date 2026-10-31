"""An observation-driven explorer with a sparse Astra tactical adviser.

No NLE imports or engine access: all beliefs come from ctx observations.
"""
from collections import Counter, deque
import json
import re

DIRS = {'n': (0, -1), 'e': (1, 0), 's': (0, 1), 'w': (-1, 0),
        'ne': (1, -1), 'se': (1, 1), 'sw': (-1, 1), 'nw': (-1, -1)}
KEYS = dict(zip('kljh unby'.replace(' ', ''),
                ['move n', 'move e', 'move s', 'move w', 'move ne', 'move se', 'move sw', 'move nw']))
KEYS.update({k.upper(): 'run ' + v.split()[1] for k, v in list(KEYS.items())})
KEYS.update({'e': 'eat', 's': 'search', '.': 'wait', '>': 'down', '<': 'up', '\r': 'more'})
SAFE_FOOD = ('food ration', 'lembas wafer', 'cram ration', 'pancake', 'fortune cookie',
             'apple', 'orange', 'pear', 'melon', 'banana', 'carrot', 'slime mold',
             'clove of garlic', 'sprig of wolfsbane', 'candy bar', 'cream pie')
SAFE_CORPSES = ('newt', 'lichen', 'lizard', 'jackal', 'fox', 'goblin', 'hobgoblin',
                'grid bug', 'giant rat', 'sewer rat')
WEAK = set('x:ro')


def lines(value):
    if isinstance(value, str):
        return value.splitlines()
    if isinstance(value, (list, tuple)):
        return [v if isinstance(v, str) else ''.join(chr(int(c)) for c in v) for v in value]
    return []


def dist(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def parse_reply(text, names):
    """Accept only JSON data or an exact legal action; never execute model code."""
    text = str(text).strip()
    obj = None
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        start, end = text.find('{'), text.rfind('}')
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
            except ValueError:
                pass
    if not isinstance(obj, dict):
        obj = {'actions': [text]} if text in names else {}
    actions = obj.get('actions', [])
    if not isinstance(actions, list):
        actions = []
    obj['actions'] = [a for a in actions[:12] if isinstance(a, str) and a in names]
    return obj


class Level:
    def __init__(self):
        self.terrain = {}
        self.visits = Counter()
        self.searched = Counter()
        self.blocked = Counter()
        self.doors = Counter()
        self.stairs = {}
        self.corpse_age = {}
        self.steps = 0


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.levels = {}
        self.recent = deque(maxlen=16)
        self.queue = deque()
        self.calls = 0
        self.last_call = -1000
        self.last_call_level = None
        self.last_danger = None
        self.mode = 'explore'
        self.target = None
        self.target_expiry = 0
        self.avoid = set()
        self.fight = set()
        self.note = ''
        self.pending_dir = None
        self.food_letter = None
        self.eat_retry = -100
        self.steps = 0
        self.last = None
        self.still = 0
        self.same_prompt = 0
        self.last_prompt = ''
        self.no_progress = 0
        self.last_progress = None
        self.recovering = False

    def update(self, obs):
        self.obs = obs
        self.s = obs.get('stats', {})
        self.pos = (int(self.s.get('x', 0)), int(self.s.get('y', 0)))
        self.levelkey = (self.s.get('dungeon_number', 0), self.s.get('dungeon_level', self.s.get('depth', 1)))
        self.level = self.levels.setdefault(self.levelkey, Level())
        self.rows = lines(obs.get('map', obs.get('chars', [])))
        self.rows = [r.ljust(79)[:79] for r in self.rows[:21]]
        self.rows += [' ' * 79] * (21 - len(self.rows))
        self.msg = str(obs.get('message', ''))
        self.tty = '\n'.join(lines(obs.get('tty', [])))
        self.monsters = []
        self.pets = []
        self.objects = []
        for y, row in enumerate(self.rows):
            for x, ch in enumerate(row):
                p = (x, y)
                if ch == ' ':
                    continue
                special = self.array('specials', p)
                glyph = self.array('glyphs', p)
                color = self.array('colors', p)
                monster = (ch.isalpha() or ch in ':;&@') and p != self.pos and not special & 32
                if monster:
                    m = {'xy': p, 'char': ch, 'glyph': glyph, 'color': color}
                    (self.pets if special & 8 else self.monsters).append(m)
                elif p != self.pos:
                    self.level.terrain[p] = ch
                    if ch in '<>':
                        self.level.stairs[p] = ch
                    if ch in '$%':
                        self.objects.append((p, ch))
                    if ch == '%':
                        self.level.corpse_age.setdefault(p, self.s.get('time', 0))
                if ch in '-|' and color == 3:
                    self.level.terrain[p] = '/'  # observed brown doorway, not wall
        self.level.terrain.setdefault(self.pos, '.')
        self.level.visits[self.pos] += 1
        self.level.steps += 1
        if 'staircase down' in self.msg.lower() or 'stairs leading down' in self.msg.lower():
            self.level.stairs[self.pos] = '>'
        if self.last:
            oldkey, oldpos, oldhp, oldturn, action, oldmsg = self.last
            moved = oldkey != self.levelkey or oldpos != self.pos
            self.still = 0 if moved else self.still + 1
            if oldkey != self.levelkey:
                self.queue.clear()
                self.target = None
            if self.s.get('hp', oldhp) < oldhp:
                self.queue.clear()
            if action.startswith('move ') and not moved and oldkey == self.levelkey:
                dx, dy = DIRS[action.split()[1]]
                dest = (oldpos[0] + dx, oldpos[1] + dy)
                # Only geometry evidence bans an edge; attacking/pet swaps must remain possible.
                if self.s.get('time', 0) == oldturn and not self.is_prompt():
                    self.level.blocked[(oldpos, dest)] += 1
                    self.queue.clear()
                if any(w in self.msg.lower() for w in ('door is closed', 'door is locked', 'door resists')):
                    self.level.terrain[dest] = '+'
                    self.level.doors[dest] += 1
            self.recent.append({'action': action, 'xy': self.pos, 'hp': self.s.get('hp'),
                                'turn': self.s.get('time'), 'message': self.msg[:220]})
        progress = (self.levelkey, len(self.level.terrain), self.s.get('score', obs.get('score', 0)),
                    len(self.level.visits))
        self.no_progress = self.no_progress + 1 if progress == self.last_progress else 0
        self.last_progress = progress

    def array(self, name, p):
        try:
            return int(self.obs.get(name, [])[p[1]][p[0]])
        except (IndexError, TypeError, ValueError):
            return 0

    def char(self, p):
        x, y = p
        return self.rows[y][x] if 0 <= x < 79 and 0 <= y < 21 else ' '

    def is_prompt(self):
        m = self.msg.lower()
        return ('--more--' in self.tty.lower() or 'in what direction' in m or
                'what do you want' in m or '[yn' in m or '(y/n)' in m or
                ('eat' in m and '?' in m))

    def prompt_action(self):
        m = self.msg.lower()
        if '--more--' in self.tty.lower():
            return 'more'
        if 'in what direction' in m:
            d = self.pending_dir
            self.pending_dir = None
            if d:
                return 'move ' + d
            for d, (dx, dy) in DIRS.items():
                if self.level.terrain.get((self.pos[0] + dx, self.pos[1] + dy)) == '+':
                    return 'move ' + d
            return 'wait'  # self-direction cancels an accidental kick
        if 'really attack' in m:
            return 'move se'  # n = no
        if 'eat' in m and ('[yn' in m or '(y/n)' in m):
            # Only explicitly identified safe food; never trust unidentified corpses.
            safe = any(f in m for f in SAFE_FOOD)
            fresh = self.s.get('time', 0) - self.level.corpse_age.get(self.pos, -999) <= 35
            safe = safe or (fresh and any(f + ' corpse' in m for f in SAFE_CORPSES))
            return 'move nw' if safe and self.s.get('hunger', 1) >= 1 else 'move se'
        if 'what do you want to eat' in m:
            if self.food_letter in KEYS:
                a = KEYS[self.food_letter]
                self.food_letter = None
                return a
            return 'more'
        if '[yn' in m or '(y/n)' in m:
            return 'move se'
        if 'what do you want' in m:
            return 'more'
        return None

    def food(self):
        letters = self.obs.get('inventory_letters', [])
        inv = self.obs.get('inventory', [])
        for i, item in enumerate(inv):
            item = str(item).lower()
            if any(f in item for f in SAFE_FOOD) and not ('cursed' in item and 'uncursed' not in item):
                try:
                    letter = letters[i]
                    letter = chr(letter) if isinstance(letter, int) else str(letter)
                except (IndexError, ValueError):
                    continue
                if letter in KEYS:
                    return letter
        return None

    def traversable(self, p, unknown=False):
        x, y = p
        if not (1 <= x < 79 and 0 <= y < 21):
            return False
        ch = self.level.terrain.get(p, ' ')
        if p == self.pos:
            return True
        if ch == ' ':
            return unknown
        return ch not in '|-+^0`~}'

    def neighbors(self, p, monsters=True):
        occupied = {tuple(m['xy']) for m in self.monsters} if monsters else set()
        for d, (dx, dy) in DIRS.items():
            q = (p[0] + dx, p[1] + dy)
            if q in occupied or not self.traversable(q):
                continue
            if self.level.blocked[(p, q)] >= 2:
                continue
            if dx and dy:
                if self.level.terrain.get(p) == '/' or self.level.terrain.get(q) == '/':
                    continue
                if not self.traversable((p[0] + dx, p[1])) and not self.traversable((p[0], p[1] + dy)):
                    continue
            yield q, d

    def paths(self):
        paths = {self.pos: []}
        todo = deque([self.pos])
        while todo:
            p = todo.popleft()
            for q, d in self.neighbors(p):
                if q not in paths:
                    paths[q] = paths[p] + [d]
                    todo.append(q)
        return paths

    def route_action(self, path):
        return 'move ' + path[0] if path else None

    def exploration(self):
        paths = self.paths()
        if self.target and self.steps < self.target_expiry:
            path = paths.get(self.target)
            if path:
                return self.route_action(path)
            self.target = None
        candidates = []
        for p, ch in self.objects:
            if ch == '$' and paths.get(p):
                candidates.append((len(paths[p]) - 25, paths[p]))
        down = [(len(path), p, path) for p, path in paths.items() if self.level.stairs.get(p) == '>']
        if down:
            _, p, path = min(down)
            if not path:
                return 'down'
            if self.mode == 'descend' or self.level.steps > 80 or not candidates:
                candidates.append((len(path) - 12, path))
        if candidates:
            return self.route_action(min(candidates, key=lambda t: t[0])[1])
        # Walk to a frontier and probe the unknown cardinal edge once. Failed edges are remembered.
        frontier = []
        for p, path in paths.items():
            for d, (dx, dy) in list(DIRS.items())[:4]:
                q = (p[0] + dx, p[1] + dy)
                if not (1 <= q[0] < 79 and 0 <= q[1] < 21):
                    continue
                if self.level.terrain.get(q, ' ') == ' ' and self.level.blocked[(p, q)] < 1:
                    # Room interiors do not border blank rock unless dark/unexplored.
                    frontier.append((len(path) + 0.4 * self.level.visits[p], path + [d]))
        if frontier:
            return self.route_action(min(frontier, key=lambda t: t[0])[1])
        doors = []
        for p, path in paths.items():
            for d, (dx, dy) in list(DIRS.items())[:4]:
                q = (p[0] + dx, p[1] + dy)
                if self.level.terrain.get(q) == '+' and self.level.doors[q] < 14:
                    doors.append((len(path) + self.level.doors[q], p, path, d, q))
        if doors:
            _, p, path, d, q = min(doors)
            if path:
                return self.route_action(path)
            if self.level.doors[q] == 0:
                self.level.doors[q] += 1
                return 'move ' + d  # auto-open if supported; confirm from result
            self.level.doors[q] += 1
            self.pending_dir = d
            return 'kick'
        # Cover room walls/dead ends in bounded groups of searches; do not spin at one wall.
        search_sites = []
        for p, path in paths.items():
            walls = 0
            for dx, dy in DIRS.values():
                q = (p[0] + dx, p[1] + dy)
                if self.level.terrain.get(q, ' ') in ' |-':
                    walls += 1
            if walls and self.level.searched[p] < 18:
                search_sites.append((len(path) + self.level.searched[p] * 2 - walls, p, path))
        if search_sites:
            _, p, path = min(search_sites)
            if path:
                return self.route_action(path)
            self.level.searched[p] += 1
            return 'search'
        if down:
            _, p, path = min(down)
            return self.route_action(path) or 'down'
        return None

    def tactical(self):
        nearby = [m for m in self.monsters if dist(self.pos, m['xy']) <= 1]
        hp, maxhp = self.s.get('hp', 1), max(1, self.s.get('hp_max', 1))
        if nearby:
            dangerous = any(m['char'] not in WEAK and m['glyph'] not in self.fight for m in nearby)
            flee = hp < max(5, maxhp * 0.42) or len(nearby) >= 3 or any(m['glyph'] in self.avoid for m in nearby)
            if flee:
                choices = []
                for p, d in self.neighbors(self.pos):
                    nearest = min(dist(p, m['xy']) for m in self.monsters)
                    choices.append((nearest * 20 - self.level.visits[p] * 0.1,
                                    len(list(self.neighbors(p))), d))
                if choices and max(choices)[0] > 15:
                    return 'move ' + max(choices)[2]
            # Adjacent combat is a closed-loop option: one strike, then observe again.
            enemy = min(nearby, key=lambda m: (m['glyph'] in self.avoid, m['char'] not in WEAK))
            dx, dy = enemy['xy'][0] - self.pos[0], enemy['xy'][1] - self.pos[1]
            for d, delta in DIRS.items():
                if delta == (dx, dy):
                    return 'move ' + d
        # Regenerate in safety; check threats and hunger each turn.
        if hp < maxhp * 0.65:
            self.recovering = True
        if hp >= maxhp * 0.92:
            self.recovering = False
        if self.recovering and not any(dist(self.pos, m['xy']) <= 5 for m in self.monsters) and self.s.get('hunger', 1) < 3:
            return 'wait'
        return None

    def advisory_needed(self):
        elapsed = self.steps - self.last_call
        if self.calls >= 12 or elapsed < 18:
            return False
        threat = tuple(sorted((m['glyph'], m['char']) for m in self.monsters if dist(self.pos, m['xy']) <= 3))
        if threat and threat != self.last_danger and any(c not in WEAK for _, c in threat):
            return True
        if self.no_progress > 65 or self.s.get('hunger', 1) >= 3:
            return elapsed >= 65
        return self.steps >= 40 and (self.last_call_level != self.levelkey or elapsed >= 180)

    def consult(self, names):
        budget = self.ctx.budget.remaining()
        if budget.get('system2_calls', 0) <= 1 or budget.get('wall_s', 0) < 210:
            return
        self.calls += 1
        self.last_call = self.steps
        self.last_call_level = self.levelkey
        self.last_danger = tuple(sorted((m['glyph'], m['char']) for m in self.monsters if dist(self.pos, m['xy']) <= 3))
        inv = []
        for letter, item in zip(self.obs.get('inventory_letters', []), self.obs.get('inventory', [])):
            if item:
                inv.append([chr(letter) if isinstance(letter, int) and letter else str(letter), item])
        state = {'stats': self.s, 'message': self.msg, 'monsters': self.monsters, 'pets': self.pets,
                 'inventory': inv, 'recent': list(self.recent), 'mode': self.mode, 'note': self.note,
                 'steps_without_progress': self.no_progress,
                 'stairs': [[list(p), ch] for p, ch in self.level.stairs.items()]}
        maptext = '\n'.join('%02d %s' % (y, r.rstrip()) for y, r in enumerate(self.rows) if r.strip())
        prompt = (
            'You advise an autonomous NetHack explorer. Maximize terminal score through gold, XP and descent. '
            'Code navigates, collects gold, opens/kicks doors, searches, handles prompts, fights adjacent weak monsters '
            'and rests when hurt. You make tactical corrections and choose strategy. All coordinates are [x,y], '
            'zero based; map columns start after the 3-character row label. @ is player, $ gold, > down, '
            '< up, # corridor, + door, |/- walls (brown is open doorway). Pets are separately listed. '
            'Only these commands exist: ' + ', '.join(names) + '. No quaff/wear/wield/zap/pray/escape/pickup. '
            'Inventory prompt letters can only use available command keys h j k l y u b n H J K L Y U B N e s. '
            'move nw sends y/yes; move se sends n/no; more sends Enter. Do not invent keys. '
            'Use actions for an immediate short tactical correction (up to 8); use target for safe route destination. '
            'Avoid unnecessary dangerous combat, especially with poor AC. Prefer fast descent if food is scarce. '
            'Reply ONLY a compact JSON object: {"mode":"explore" or "descend", "target":[x,y] or null, '
            '"actions":[legal strings], "fight":[monster glyph IDs safe to melee], '
            '"avoid":[dangerous glyph IDs], "note":"brief reusable advice"}. '
            'Empty actions delegates to the explorer; never output long reasoning.\n'
            + json.dumps(state, separators=(',', ':')) + '\nMAP\n' + maptext + '\nTTY\n'
            + '\n'.join(lines(self.obs.get('tty', []))[:2])
        )
        try:
            result = self.ctx.think(prompt, session_id=None, timeout_s=min(240.0, budget['wall_s'] - 30))
        except Exception as exc:
            self.ctx.log('adviser_failed', error=type(exc).__name__)
            return
        plan = parse_reply(result.text, names)
        if plan.get('mode') in ('explore', 'descend'):
            self.mode = plan['mode']
        p = plan.get('target')
        if isinstance(p, list) and len(p) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in p):
            if 1 <= p[0] < 79 and 0 <= p[1] < 21:
                self.target, self.target_expiry = tuple(p), self.steps + 70
        for key in ('fight', 'avoid'):
            val = plan.get(key, [])
            if isinstance(val, list):
                getattr(self, key).update(v for v in val if isinstance(v, int))
        self.note = str(plan.get('note', self.note))[:400]
        self.queue.extend(plan['actions'])
        self.ctx.log('adviser_plan', mode=self.mode, target=self.target, actions=list(self.queue), note=self.note)

    def run_episode(self, game):
        ctx = self.ctx
        names = list(game.action_spec.names)
        try:
            obs = ctx.observe()
        except Exception:
            return
        while not ctx.done and self.steps < 7000:
            if not isinstance(obs, dict) or obs.get('done') or obs.get('truncated'):
                break
            self.update(obs)
            prompt_action = self.prompt_action()
            if prompt_action:
                sig = self.msg + self.tty[-180:]
                self.same_prompt = self.same_prompt + 1 if sig == self.last_prompt else 0
                self.last_prompt = sig
                if self.same_prompt > 12:
                    ctx.log('unresolvable_prompt', message=self.msg)
                    break
                action = prompt_action
            else:
                self.same_prompt = 0
                if self.advisory_needed():
                    self.consult(names)
                    if ctx.done:
                        break
                action = None
                if self.queue:
                    action = self.queue.popleft()
                    if action.startswith('run '):
                        action = 'move ' + action.split()[1]
                    if action == 'up' and self.s.get('depth', 1) <= 1:
                        action = None
                    if action == 'kick':
                        action = None  # kick is an atomic directional option owned by code
                hunger = self.s.get('hunger', 1)
                if hunger >= 2 and self.steps - self.eat_retry > 25:
                    letter = self.food()
                    groundfood = self.char(self.pos) == '%' or 'corpse' in self.msg.lower() or 'food ration' in self.msg.lower()
                    if letter or groundfood:
                        self.food_letter = letter
                        self.eat_retry = self.steps
                        action = 'eat'
                if action is None:
                    action = self.tactical() or self.exploration()
                if action is None or self.no_progress > 550:
                    ctx.log('exploration_exhausted', stats=self.s)
                    break
            if action not in names:
                action = 'search' if 'search' in names else game.action_spec.noop
            self.last = (self.levelkey, self.pos, self.s.get('hp', 0), self.s.get('time', 0), action, self.msg)
            if self.steps % 40 == 0:
                ctx.log('controller_state', step=self.steps, stats=self.s, message=self.msg,
                        goal=self.target, mode=self.mode, calls=self.calls, known=len(self.level.terrain))
            self.steps += 1
            try:
                obs = ctx.act(action)
                if obs is None and not ctx.done:
                    obs = ctx.observe()
            except Exception as exc:
                if ctx.done:
                    break
                ctx.log('action_failed', action=action, error=type(exc).__name__)
                self.queue.clear()
                self.last = None
                try:
                    obs = ctx.observe()
                except Exception:
                    break
                if self.steps > 10 and self.still > 10:
                    break
        ctx.memory['controller_summary'] = {'actions': self.steps, 'calls': self.calls,
                                            'levels': len(self.levels), 'note': self.note}


def build(ctx):
    return Harness(ctx)
