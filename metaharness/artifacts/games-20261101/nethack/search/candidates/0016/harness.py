"""Risk-bounded autonomous expeditions with a no-latency coverage controller.

Derived from candidate 0002's autonomous controller. Only ctx observations are
used; the restricted action alphabet is respected even inside menus.
"""
from collections import Counter, deque
import heapq
import json
import re

DIRS = {'n': (0, -1), 'e': (1, 0), 's': (0, 1), 'w': (-1, 0),
        'ne': (1, -1), 'se': (1, 1), 'sw': (-1, 1), 'nw': (-1, -1)}
KEYS = dict(zip('kljhunby', ['move ' + d for d in DIRS]))
KEYS.update({k.upper(): v.replace('move ', 'run ') for k, v in list(KEYS.items())})
KEYS.update({'e': 'eat', 's': 'search', '.': 'wait', '>': 'down', '<': 'up'})
FOOD = ('food ration', 'cram ration', 'lembas', 'pancake', 'apple', 'orange',
        'pear', 'melon', 'banana', 'carrot', 'slime mold', 'fortune cookie',
        'candy bar', 'clove of garlic', 'sprig of wolfsbane')
TERRAIN = '.#<>+|-^{}\\~'
SAFE_CHARS = set('x:rgkdl')


def num(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError, OverflowError):
        return default


def rows(v):
    if isinstance(v, str):
        return v.splitlines()
    if not isinstance(v, (list, tuple)):
        return []
    return [r if isinstance(r, str) else ''.join(chr(num(c)) if 0 < num(c) < 256 else ' ' for c in r)
            for r in v if isinstance(r, (str, list, tuple))]


def distance(a, b):
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))


def parse_reply(text, names):
    obj = {}
    try:
        text = str(text)
        obj = json.JSONDecoder().raw_decode(text[text.index('{'):])[0]
    except (ValueError, TypeError):
        pass
    if not isinstance(obj, dict):
        obj = {}
    answer = {'fight': [], 'avoid': [], 'neutral': [], 'target': None, 'note': '', 'action': None}
    for field in ('fight', 'avoid', 'neutral'):
        v = obj.get(field, [])
        if isinstance(v, list):
            answer[field] = [x for x in v[:24] if type(x) is int and 0 <= x < 6000]
    p = obj.get('target')
    if isinstance(p, list) and len(p) == 2 and all(type(x) is int for x in p):
        if 1 <= p[0] < 79 and 0 <= p[1] < 21:
            answer['target'] = tuple(p)
    a = obj.get('action')
    if isinstance(a, str) and a in names:
        answer['action'] = a
    answer['note'] = str(obj.get('note', ''))[:300]
    return answer


class Level:
    def __init__(self):
        self.terrain = {}
        self.visits = Counter()
        self.searched = Counter()
        self.blocked = set()
        self.probed = Counter()
        self.doors = Counter()
        self.stairs = {}
        self.cooldown = {}
        self.arrival = 0
        self.unique = set()


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.levels = {}
        self.steps = 0
        self.calls = 0
        self.call_errors = 0
        self.action_errors = 0
        self.last_call = -1000
        self.last = None
        self.target = None
        self.pending_dir = None
        self.prompt_direction_used = None
        self.pending_food = None
        self.last_eat = -1000
        self.recovering = False
        self.recent = deque(maxlen=12)
        self.damage = deque(maxlen=12)
        self.fight = set()
        self.avoid = set()
        self.neutral = set()
        self.consulted = set()
        self.note = ''
        self.identity = ''
        self.stale = 0
        self.no_gain = 0
        self.best_progress = None
        self.prompt_repeats = Counter()
        # A declined peaceful-attack confirmation must alter the tactical
        # belief. Otherwise the next world tick retries the same probe.
        self.declined_cells = {}
        self.peak_score = 0
        self.retreat_steps = 0
        self.transient_wait = 0
        self.advice_action = None
        self.stop_reason = None
        # Position changes are not necessarily progress: short cycles can move
        # every turn while revisiting the same room.  Keep temporary edge bans
        # and a compact history so geometry remains recoverable after a break.
        self.position_history = deque(maxlen=48)
        self.cycle_edges = {}
        self.cycle_cooldown = 0
        self.cycle_events = 0
        self.edge_counts = Counter()
        self.coverage_turns = 0
        # A single bounded melee probe can break a corridor sealed by an
        # unfamiliar creature.  It is set only by tactical() after all health
        # and retreat checks, and consumed by guarded().
        self.attack_override = None

    def arr(self, key, p, default=0):
        try:
            return num(self.obs[key][p[1]][p[0]], default)
        except (KeyError, IndexError, TypeError):
            return default

    def update(self, obs):
        self.obs = obs if isinstance(obs, dict) else {}
        # Never carry an attack permission across an observation.  A monster
        # may have moved, a prompt may have appeared, or HP may have changed.
        self.attack_override = None
        s = self.obs.get('stats', {})
        self.s = s if isinstance(s, dict) else {}
        self.pos = (num(self.s.get('x')), num(self.s.get('y')))
        self.key = (num(self.s.get('dungeon_number')), num(self.s.get('dungeon_level', self.s.get('depth', 1))))
        self.level = self.levels.setdefault(self.key, Level())
        self.msg = str(self.obs.get('message', ''))
        rr = rows(self.obs.get('map', self.obs.get('chars', [])))[:21]
        self.map = [r[:79].ljust(79) for r in rr] + [' '*79] * (21-len(rr))
        self.tty = '\n'.join(rows(self.obs.get('tty', self.obs.get('tty_chars', []))))
        self.hp = num(self.s.get('hp'), 1)
        self.maxhp = max(1, num(self.s.get('hp_max'), 1))
        self.turn = num(self.s.get('time'))
        self.hunger = num(self.s.get('hunger'), 1)
        self.score = num(self.s.get('score', self.obs.get('score', 0)))
        self.peak_score = max(self.peak_score, self.score)
        if 'welcome to NetHack' in self.msg:
            self.identity = self.msg
        inv = self.obs.get('inventory', [])
        self.armed = any('weapon in hand' in str(x) or 'wielded' in str(x) for x in inv)
        self.monsters = {}
        self.pets = {}
        self.items = {}
        old_terrain_count = len(self.level.terrain)
        for y, row in enumerate(self.map):
            for x, char in enumerate(row):
                p = (x, y)
                if char == ' ' or p == self.pos:
                    continue
                special = self.arr('specials', p)
                glyph = self.arr('glyphs', p, -1)
                creature = (char.isalpha() or char in ':;&@') and not special & 32
                if creature:
                    m = {'xy': p, 'char': char, 'glyph': glyph, 'color': self.arr('colors', p)}
                    (self.pets if special & 8 else self.monsters)[p] = m
                    # A real creature cannot stand in a closed door. Occupancy also
                    # reveals traversability; do not retain obsolete '+' forever.
                    if self.level.terrain.get(p) == '+':
                        self.level.terrain[p] = '/'
                    else:
                        self.level.terrain.setdefault(p, '.')
                elif char in TERRAIN:
                    t = '/' if (char in '|-' and self.arr('colors', p) == 3) or glyph in (2371, 2372) else char
                    old = self.level.terrain.get(p)
                    self.level.terrain[p] = t
                    if t in '.#/<>':
                        if old != t:
                            self.level.blocked = {e for e in self.level.blocked if e[1] != p}
                    if t in '<>':
                        self.level.stairs[p] = t
                elif char in '0`':
                    self.level.terrain[p] = '`'
                else:
                    self.items[p] = char
                    # Items obscure underlying floor/door/stairs; they are not terrain.
                    self.level.terrain.setdefault(p, '.')
        if self.level.terrain.get(self.pos) in (None, '+', '`'):
            self.level.terrain[self.pos] = '/' if self.level.terrain.get(self.pos) == '+' else '.'
        self.level.visits[self.pos] += 1
        self.level.unique.add(self.pos)
        if len(self.level.terrain) > old_terrain_count:
            # New map information validates the current route and makes old
            # cycle bans safe to release.
            self.cycle_edges.clear()
            self.coverage_turns = 0
        self.position_history.append((self.key, self.pos))
        # Detect periods 2..6 in the observed trajectory.  This runs after an
        # actual observation, so a blocked command or prompt cannot fabricate a
        # cycle.  Ban the traversed edge briefly and force target reselection.
        if self.last and self.steps >= self.cycle_cooldown and len(self.position_history) >= 8:
            hist = list(self.position_history)
            for period in range(2, 7):
                if len(hist) < 2 * period:
                    continue
                a = hist[-period:]
                b = hist[-2 * period:-period]
                if a == b and len({x[1] for x in a}) > 1:
                    oldpos = self.last[1]
                    edge = (oldpos, self.pos)
                    rev = (self.pos, oldpos)
                    self.cycle_edges[edge] = self.steps + 80
                    self.cycle_edges[rev] = self.steps + 80
                    self.target = None
                    self.cycle_events += 1
                    self.cycle_cooldown = self.steps + 12
                    self.ctx.log('cycle_break', period=period, edge=[list(edge[0]), list(edge[1])],
                                 position=list(self.pos), events=self.cycle_events)
                    break
        low = self.msg.lower()
        if 'stairs leading down' in low or 'staircase down' in low:
            self.level.stairs[self.pos] = '>'
        if self.last:
            oldkey, p, hp, turn, action, kind, intended = self.last
            changed = self.key != oldkey or self.pos != p or self.turn != turn
            self.stale = 0 if changed else self.stale+1
            if self.key != oldkey:
                self.target = None
                self.level.arrival = self.turn
                self.retreat_steps = 0
                self.cycle_edges.clear()
                self.position_history.clear()
                self.cycle_cooldown = self.steps + 8
            loss = max(0, hp-self.hp)
            self.damage.append(loss)
            if loss:
                self.target = None
                self.recent.append({'damage': loss, 'hp': self.hp, 'message': self.msg[:180]})
            if oldkey == self.key:
                # Resolve an action transaction from the actual outcome, including
                # the direction response to a kick, before reading an obscured tile.
                if intended is not None:
                    if 'crashes open' in low or 'door opens' in low or 'door shatters' in low:
                        self.level.terrain[intended] = '/'
                        self.level.blocked = {e for e in self.level.blocked if e[1] != intended}
                        self.ctx.log('door_open_verified', xy=intended, message=self.msg)
                    if 'door is closed' in low or 'door is locked' in low or 'door resists' in low:
                        self.level.terrain[intended] = '+'
                    if kind != 'prompt' and action.startswith('move '):
                        dynamic = any(w in low for w in ('in the way', 'swap places', 'really attack'))
                        if self.pos == intended:
                            self.level.blocked.discard((p, intended))
                            if self.level.terrain.get(intended) == '+':
                                self.level.terrain[intended] = '/'
                        elif self.pos == p and dynamic:
                            self.level.blocked.discard((p, intended))
                            self.transient_wait = 1
                        elif self.pos == p and self.turn == turn and not self.prompt_kind() and intended not in self.monsters and intended not in self.pets:
                            self.level.blocked.add((p, intended))
                            self.target = None
                if action == 'search' and kind != 'prompt' and self.turn != turn:
                    for dx, dy in DIRS.values():
                        self.level.searched[(p[0]+dx, p[1]+dy)] += 1
            self.recent.append({'a': action, 'xy': self.pos, 'turn': self.turn, 'hp': self.hp, 'msg': self.msg[:140]})
        # `len(visits)` grows on every turn and therefore hid oscillation in
        # earlier candidates.  Count unique cells instead.
        progress = (self.key, self.score, len(self.level.terrain), len(self.level.unique))
        self.no_gain = self.no_gain+1 if progress == self.best_progress else 0
        self.best_progress = progress
        for cell, expiry in list(self.declined_cells.items()):
            key, p = cell
            if (self.steps >= expiry or key != self.key or
                    p not in self.monsters or distance(p, self.pos) > 1):
                self.declined_cells.pop(cell, None)
        self.threats = {p: m for p, m in self.monsters.items() if m['glyph'] not in self.neutral}

    def prompt_kind(self):
        # Echoed answers are not a second prompt. This matters when 'n' is also SE.
        # Prompts can be present only in tty_chars while message is stale/empty.
        # Keep a short screen projection and recognize prompt text without requiring
        # punctuation at the end (the adaptor may trim the final '?').
        m = self.msg.strip().lower()
        screen = (m + '\n' + self.tty.lower())
        if '--more--' in screen:
            return 'more'
        if ('what direction' in screen or 'in what direction' in screen
                or 'which direction' in screen):
            return 'direction'
        if ('what do you want to eat' in screen
                and not re.search(r'\]\s+[a-z]$', m)):
            return 'eat'
        if (re.search(r'\[(?:yn|ny)[^\]]*\]\s*(?:\([ynq]\))?\s*$', m)
                or re.search(r'\[(?:yn|ny)[^\]]*\]\s*(?:\([ynq]\))?\s*$', screen)):
            return 'yesno'
        if '(end)' in screen or re.search(r'\(\d+ of \d+\)', screen):
            return 'more'
        return None

    def food(self):
        choices = []
        inv = self.obs.get('inventory', [])
        letters = self.obs.get('inventory_letters', [])
        for i, item in enumerate(inv if isinstance(inv, list) else []):
            s = str(item).lower()
            if not any(f in s for f in FOOD) or ('cursed' in s and 'uncursed' not in s):
                continue
            try:
                k = chr(letters[i]) if isinstance(letters[i], int) else str(letters[i])
            except (IndexError, TypeError, ValueError):
                continue
            if k in KEYS:
                choices.append((0 if 'ration' in s or 'lembas' in s else 1, k))
        return min(choices)[1] if choices else None

    def prompt_action(self):
        kind = self.prompt_kind()
        if not kind:
            return None
        sig = (kind, self.msg, self.turn)
        self.prompt_repeats[sig] += 1
        if self.prompt_repeats[sig] > 5:
            self.stop_reason = 'unresolved prompt'
            return None
        if kind == 'more':
            return 'more'
        if kind == 'direction':
            # A kick/open transaction owns its direction. If a malformed or
            # truncated observation lost that field, any cardinal direction still
            # discharges the prompt; waiting leaves the game permanently stuck.
            d = self.pending_dir
            if d not in DIRS:
                for rec in reversed(self.recent):
                    a = rec.get('a') if isinstance(rec, dict) else None
                    if isinstance(a, str) and a.startswith('move ') and a[5:] in DIRS:
                        d = a[5:]
                        break
            if d not in DIRS:
                d = 'n'
            self.prompt_direction_used = d
            self.pending_dir = None
            return 'move '+d
        if kind == 'eat':
            k = self.pending_food or self.food()
            return KEYS.get(k, 'more')
        m = (self.msg + '\n' + self.tty).lower()
        # NetHack's default answer to "Really attack ...?" is no. Remember
        # the exact level and coordinate before sending the vi-key equivalent
        # of `n`; this prevents a prompt deadlock while preserving other
        # creatures of the same species as independent hypotheses.
        if kind == 'yesno' and 'really attack' in m:
            adjacent = [(p, mon) for p, mon in self.monsters.items()
                        if distance(p, self.pos) <= 1]
            for p, mon in adjacent:
                self.declined_cells[(self.key, p)] = self.steps + 24
            self.threats = {p: mon for p, mon in self.monsters.items()
                            if mon['glyph'] not in self.neutral}
            self.ctx.log('decline_attack', glyphs=[mon['glyph'] for _, mon in adjacent],
                         positions=[list(p) for p, _ in adjacent])
        yes = ('continue eating' in m or 'finish eating' in m or
               ('eat' in m and any(f in m for f in FOOD)))
        return 'move nw' if yes else 'move se'

    def passable(self, p):
        return (1 <= p[0] < 79 and 0 <= p[1] < 21 and p in self.level.terrain
                and self.level.terrain[p] not in ' |-=+^{}\\0`~')

    def neighbors(self, p):
        # Expired cycle bans are discarded lazily.  A ban is deliberately
        # temporary: it only redirects the expedition toward novelty and never
        # asserts that a map edge is solid geometry.
        if self.cycle_edges:
            self.cycle_edges = {e: until for e, until in self.cycle_edges.items()
                                if until > self.steps}
        for d, (dx, dy) in DIRS.items():
            q = (p[0]+dx, p[1]+dy)
            if (not self.passable(q) or q in self.monsters or (p, q) in self.level.blocked
                    or self.cycle_edges.get((p, q), 0) > self.steps):
                continue
            if dx and dy:
                if self.level.terrain.get(p) == '/' or self.level.terrain.get(q) == '/':
                    continue
                if not self.passable((p[0]+dx, p[1])) and not self.passable((p[0], p[1]+dy)):
                    continue
            yield q, 'move '+d

    def paths(self):
        """Shortest routes penalize exposure; never step through any creature."""
        cost = {self.pos: 0.0}
        paths = {self.pos: []}
        queue = [(0.0, self.pos)]
        while queue:
            c, p = heapq.heappop(queue)
            if c != cost[p]:
                continue
            for q, a in self.neighbors(p):
                risk = sum(8 if distance(q, t) <= 1 else 2 if distance(q, t) == 2 else 0 for t in self.threats)
                # Prefer routes that expose new cells.  The small visit cost
                # avoids an infinite shortest-path shuttle without forbidding
                # a necessary return through a corridor.
                visit_cost = min(6.0, self.level.visits[q] * 0.35)
                edge_cost = min(4.0, self.edge_counts[(p, q)] * 0.20)
                cycle_cost = 20.0 if self.cycle_edges.get((p, q), 0) > self.steps else 0.0
                nc = c+1+risk+visit_cost+edge_cost+cycle_cost
                if nc < cost.get(q, float('inf')):
                    cost[q] = nc
                    paths[q] = paths[p]+[a]
                    heapq.heappush(queue, (nc, q))
        return paths, cost

    def reserve(self):
        # Recent actual hits raise the reserve immediately. Unknown one-shot attacks
        # are not claimed safe; the threshold is a guard, not a damage oracle.
        return max(4, min(self.maxhp-1, int(self.maxhp*0.40)), 2*max(self.damage, default=0)+1)

    def easy(self, m):
        # These early classes are a prior, qualified by weapon/HP and feedback.
        return m['glyph'] in self.fight or m['char'] in SAFE_CHARS or m['glyph'] == 12

    def retreat(self):
        options = []
        for p, a in self.neighbors(self.pos):
            near = sum(distance(p, t) <= 1 for t in self.threats)
            gap = min((distance(p, t) for t in self.threats), default=10)
            pet = min((distance(p, t) for t in self.pets), default=8)
            options.append(((near, -gap, min(pet, 3), self.level.visits[p]*0.02), a, p))
        if not options:
            return None
        rank, a, p = min(options)
        current = sum(distance(self.pos, t) <= 1 for t in self.threats)
        if rank[0] == 0 or (rank[0] < current and self.hp > self.reserve()):
            self.retreat_steps += 1
            if self.retreat_steps <= 35:
                return a
        return None

    def consult(self, reason, paths=None):
        budget = self.ctx.budget.remaining()
        # A single Astra call can consume most of this tier's virtual wall
        # budget before returning.  Keep this escape hatch effectively cold;
        # deterministic routing has priority for ordinary episodes.
        if (self.calls >= 1 or self.call_errors >= 1 or budget.get('system2_calls', 0) <= 1
                or budget.get('wall_s', 0) < 1500 or self.steps-self.last_call < 20):
            return False
        if paths is None:
            paths, _ = self.paths()
        near = [m for p, m in self.monsters.items() if distance(p, self.pos) <= 5]
        available = {m['glyph'] for m in near}
        self.consulted.update(available)
        self.last_call = self.steps
        self.calls += 1
        state = {'reason': reason, 'stats': self.s, 'identity': self.identity, 'armed': self.armed,
                 'message': self.msg, 'creatures': near, 'pets': list(self.pets.values()),
                 'reserve_hp': self.reserve(), 'note': self.note, 'recent': list(self.recent)[-8:],
                 'inventory': [x for x in self.obs.get('inventory', []) if x],
                 'stairs': [{'xy': p, 'type': t, 'reachable': p in paths} for p, t in self.level.stairs.items()],
                 'safe_moves': [a for p, a in self.neighbors(self.pos) if not any(distance(p, q) <= 1 for q in self.threats)]}
        prompt = (
            'Advise a risk-bounded NetHack expedition using ONLY the supplied observations. '
            'Maximize score and survival. Code handles paths, gold, stairs, doors, food, regeneration, and prompts. '
            'Classify only observed glyph IDs. fight means safe to melee given this equipment; avoid includes '
            'passive damage (acid blobs, floating eyes, molds) and dangerous foes; neutral means confidently '
            'peaceful, immobile harmless creatures (they still block routing). Never classify pets as foes. '
            'No wield, throw, quaff, zap, pray or arbitrary letters exist. Tourist may be unarmed. '
            'Do not suggest impossible inventory use. Newt/grid bug/rat/jackal fights are already automated when safe. '
            'Suggest target [x,y] only when it resolves a navigation impasse. An optional single action will '
            'still be checked against current terrain and danger. A low-HP run may return to preserve its score '
            'if there is no safe continuation. Be selective about fighting; retreat toward pets if useful. '
            'Return ONLY short JSON under 180 tokens: {"fight":[],"avoid":[],"neutral":[],"target":null,'
            '"action":null,"note":"one brief reason"}. '
            'Allowed actions: '+', '.join(self.names)+'. Coordinates zero-based x,y.\n'
            +json.dumps(state, separators=(',', ':'))+'\nMAP\n'
            +'   '+''.join(str(i%10) for i in range(79))+'\n'
            +'\n'.join('%02d %s' % (y, row.rstrip()) for y, row in enumerate(self.map) if row.strip()))
        self.ctx.log('risk_advice_request', reason=reason, stats=self.s, calls=self.calls)
        try:
            result = self.ctx.think(prompt, session_id=None, timeout_s=min(160.0, budget['wall_s']-12))
            reply = parse_reply(result.text, self.names)
        except Exception as exc:
            self.call_errors += 1
            self.ctx.log('risk_advice_failed', error=type(exc).__name__)
            return False
        self.fight.update(set(reply['fight']) & available)
        self.avoid.update(set(reply['avoid']) & available)
        self.neutral.update(set(reply['neutral']) & available)
        self.fight.difference_update(self.avoid | self.neutral)
        self.threats = {p: m for p, m in self.monsters.items() if m['glyph'] not in self.neutral}
        if reply['target'] in paths:
            self.target = ('advice', reply['target'], self.steps+45)
        self.advice_action = reply['action']
        self.note = reply['note']
        self.ctx.log('risk_advice', **reply)
        return True

    def tactical(self):
        self.attack_override = None
        adjacent = [(p, m) for p, m in self.threats.items() if distance(p, self.pos) <= 1]
        close = [(p, m) for p, m in self.threats.items() if distance(p, self.pos) <= 3]
        if adjacent:
            # Prompt/model action queues cannot override this decision.
            low = self.hp <= self.reserve()
            dangerous = any(not self.easy(m) or m['glyph'] in self.avoid or
                            (self.key, p) in self.declined_cells for p, m in adjacent)
            if dangerous:
                unseen = any(m['glyph'] not in self.consulted and not self.easy(m) for _, m in adjacent)
                if unseen and not low:
                    # Mark the observation as assessed locally.  Astra is kept
                    # out of this hot path because its queueing delay can end
                    # an otherwise healthy episode.
                    self.consulted.update(m['glyph'] for _, m in adjacent)
            if low or dangerous or len(adjacent) >= 3 or not self.armed:
                retreat = self.retreat()
                if retreat:
                    return retreat
                # If there is literally no retreat square, a healthy armed
                # character can spend one turn testing a lone unfamiliar foe.
                # This is deliberately narrower than the normal easy whitelist:
                # one target, no recent damage reserve breach, and no known
                # avoidance classification.  The movement guard repeats these
                # checks before posting the attack.
                if (dangerous and len(adjacent) == 1 and self.armed
                        and adjacent[0][1]['glyph'] not in self.avoid
                        and not low and self.hp >= max(2, int(self.maxhp * 0.80))
                        and self.hp >= self.reserve() + 2):
                    self.attack_override = adjacent[0][0]
                    self.ctx.log('unknown_melee_probe', xy=list(self.attack_override),
                                 glyph=adjacent[0][1].get('glyph'), hp=self.hp,
                                 hp_max=self.maxhp)
                else:
                    # An unarmed character may still dispatch a grid bug/newt;
                    # never attack passive blobs merely because all routes are
                    # blocked.
                    trivial = all(self.easy(m) for _, m in adjacent)
                    if low or dangerous or (not self.armed and not trivial):
                        self.stop_reason = 'no safe continuation: threatened health reserve or unsafe melee'
                        return None
            self.retreat_steps = 0
            p, _ = min(adjacent, key=lambda pm: (not self.easy(pm[1]), pm[0]))
            delta = (p[0]-self.pos[0], p[1]-self.pos[1])
            return 'move '+next(d for d in DIRS if DIRS[d] == delta)
        if not close:
            self.retreat_steps = 0
        if self.hp < self.maxhp*0.72:
            self.recovering = True
        if self.hp >= self.maxhp*0.95:
            self.recovering = False
        if self.recovering and not close and self.hunger < 3:
            return 'wait'
        if self.recovering and close:
            retreat = self.retreat()
            if retreat:
                return retreat
            if self.hp <= self.reserve():
                self.stop_reason = 'injured with no safe recovery route'
        return None

    def explore(self):
        paths, costs = self.paths()
        # A known exit is the primary progress target; only short gold detours.
        stairs = [p for p in paths if self.level.stairs.get(p) == '>']
        if self.pos in stairs:
            return 'down'
        goals = []
        for p in stairs:
            goals.append((costs[p]-18, 'stairs', p))
        for p, char in self.items.items():
            if char == '$' and p in paths and p != self.pos:
                goals.append((costs[p]-12, 'gold', p))
        if self.target:
            kind, p, expiry = self.target
            if p not in paths or self.steps >= expiry or p == self.pos:
                self.target = None
            elif kind == 'advice' or (kind == 'frontier' and not goals):
                return self.corridor_action(paths[p])
        if goals:
            _, kind, p = min(goals)
            if paths[p]:
                return self.corridor_action(paths[p])
        # Doors are useful boundaries. A closed door under an occupant was already
        # repaired by update(); kick transactions only target currently empty doors.
        options = []
        for p, path in paths.items():
            if any(distance(p, q) <= 1 for q in self.threats):
                continue
            for d in ('n', 'e', 's', 'w'):
                dx, dy = DIRS[d]
                q = (p[0]+dx, p[1]+dy)
                if not (1 <= q[0] < 79 and 0 <= q[1] < 21):
                    continue
                if (self.level.terrain.get(q) == '+' and q not in self.monsters and q not in self.pets
                        and self.level.doors[q] < 10):
                    options.append((costs[p]+3+self.level.doors[q], 'door', p, d, q))
                elif q not in self.level.terrain and (p, q) not in self.level.blocked and not self.level.probed[(p, q)]:
                    options.append((costs[p]+self.level.visits[p]*0.03+2, 'frontier', p, d, q))
        if options:
            _, kind, p, d, q = min(options)
            if p != self.pos:
                self.target = ('frontier', p, self.steps+80)
                return self.corridor_action(paths[p])
            if kind == 'door':
                self.level.doors[q] += 1
                if self.level.doors[q] == 1:
                    return 'move '+d
                self.pending_dir = d
                return 'kick'
            self.level.probed[(p, q)] += 1
            return 'move '+d
        # If no immediate frontier/door is known, walk toward the least visited
        # reachable cell. Requiring visits < 2 made the parent return while
        # hidden doors and stairs were still undiscovered.
        novel = [(self.level.visits.get(p, 0) + 0.015 * costs[p], p)
                 for p in paths if p != self.pos and self.level.visits.get(p, 0) < 3]
        if novel:
            p = min(novel)[1]
            self.target = ('novelty', p, self.steps + 60)
            self.coverage_turns += 1
            return self.corridor_action(paths[p]) if paths[p] else None
        # Search coverage applies to walls, not player locations. Prefer corridor
        # dead ends; expand to room walls in waves only after cheap exits are used.
        sites = []
        for p, path in paths.items():
            if any(distance(p, q) <= 2 for q in self.threats):
                continue
            t = self.level.terrain.get(p)
            walls = []
            for dx, dy in DIRS.values():
                q = (p[0]+dx, p[1]+dy)
                if self.level.terrain.get(q) in ('|', '-') or (q not in self.level.terrain and t in ('#', '/')):
                    walls.append(q)
            if not walls:
                continue
            coverage = min(self.level.searched[q] for q in walls)
            if coverage < 12:
                exits = sum(1 for _ in self.neighbors(p))
                bonus = 6 if t in ('#', '/') and exits <= 2 else 0
                sites.append((costs[p]+coverage*2-bonus, p))
        if sites:
            p = min(sites)[1]
            return self.corridor_action(paths[p]) if p != self.pos else 'search'
        reachable = [(self.level.visits.get(p, 0) + 0.02 * costs[p], p)
                     for p in paths if p != self.pos]
        if reachable:
            p = min(reachable)[1]
            self.target = ('coverage', p, self.steps + 90)
            return self.corridor_action(paths[p]) if paths[p] else 'search'
        if self.cycle_edges:
            self.cycle_edges.clear()
            for _, a in self.neighbors(self.pos):
                return a
        return 'search' if self.level.searched[self.pos] < 12 else 'wait'

    def guarded(self, action):
        if not isinstance(action, str) or action not in self.names:
            return None
        if action.startswith('move '):
            d = action[5:]
            dx, dy = DIRS[d]
            p = (self.pos[0]+dx, self.pos[1]+dy)
            if not (1 <= p[0] < 79 and 0 <= p[1] < 21) or (self.pos, p) in self.level.blocked:
                return None
            if p in self.monsters:
                m = self.monsters[p]
                allowed_probe = (self.attack_override == p and len(self.threats) == 1
                                 and self.armed and self.hp >= max(2, int(self.maxhp * 0.80))
                                 and self.hp >= self.reserve() + 2
                                 and (self.key, p) not in self.declined_cells)
                if (m['glyph'] in self.neutral or m['glyph'] in self.avoid
                        or (not self.easy(m) and not allowed_probe)
                        or self.hp <= self.reserve()):
                    return None
            # Unknown squares are allowed only for a bounded explorer probe.
            elif p in self.level.terrain and not self.passable(p) and self.level.terrain[p] != '+':
                return None
            return action
        if action == 'up':
            return None  # don't abandon the dungeon or oscillate staircases
        if action in ('more', 'eat', 'kick'):
            return None  # these require controller-owned prompt transactions
        if action == 'down' and self.level.stairs.get(self.pos) != '>':
            return None
        if action.startswith('run '):
            return None
        return action

    def corridor_action(self, path):
        """Use a run only on a long, already mapped straight corridor."""
        if not isinstance(path, list) or len(path) < 4:
            return path[0] if path else None
        dirs = []
        for a in path:
            if not isinstance(a, str) or not a.startswith('move '):
                return path[0]
            d = a[5:]
            if d not in ('n', 'e', 's', 'w'):
                return path[0]
            dirs.append(d)
        if len(set(dirs)) == 1 and ('run ' + dirs[0]) in self.names:
            return 'run ' + dirs[0]
        return path[0]

    def run_episode(self, game):
        self.names = list(game.action_spec.names)
        if not self.names:
            return
        try:
            obs = self.ctx.observe()
        except Exception:
            return
        while not self.ctx.done and self.steps < 6000:
            self.update(obs)
            if self.obs.get('done') or self.obs.get('truncated') or self.hp <= 0:
                break
            if self.ctx.budget.remaining().get('wall_s', 0) <= 1:
                break
            self.prompt_direction_used = None
            action = self.prompt_action()
            kind = 'prompt' if action is not None else 'world'
            intended = None
            if self.stop_reason:
                break
            if kind == 'prompt':
                if self.prompt_kind() == 'direction':
                    d0 = (self.prompt_direction_used if self.prompt_direction_used in DIRS
                          else self.pending_dir if self.pending_dir in DIRS else 'n')
                    dx, dy = DIRS[d0]
                    intended = (self.pos[0]+dx, self.pos[1]+dy)
            else:
                action = self.tactical()
                if self.ctx.done or self.stop_reason:
                    break
                # Run tactical moves through the same final legality gate as
                # advisory moves.  In particular this rechecks the one-turn
                # unknown-melee permission against the current observation.
                if action is not None:
                    action = self.guarded(action)
                if action is None and self.hunger >= 2 and self.steps-self.last_eat > 25:
                    food = self.food()
                    if food and not any(distance(self.pos, p) <= 2 for p in self.threats):
                        self.pending_food = food
                        self.last_eat = self.steps
                        action = 'eat'
                if action is None and self.hunger >= 4 and not self.food():
                    self.stop_reason = 'weak hunger without accessible safe food'
                    break
                if action is None and self.transient_wait:
                    self.transient_wait -= 1
                    if not any(distance(self.pos, p) <= 2 for p in self.threats):
                        action = 'wait'
                if action is None and self.advice_action:
                    action = self.guarded(self.advice_action)
                    self.advice_action = None
                if action is None:
                    action = self.explore()
                if self.ctx.done or self.stop_reason or action is None:
                    break
                if action.startswith('move '):
                    dx, dy = DIRS[action[5:]]
                    intended = (self.pos[0]+dx, self.pos[1]+dy)
                    # Route selection is not permission to walk toward imminent
                    # danger while hurt. An actual tactical retreat remains allowed.
                    if (self.hp <= self.reserve() and any(distance(intended, p) <= 1 for p in self.threats)):
                        self.stop_reason = 'movement would breach health reserve'
                        break
            if action not in self.names:
                self.stop_reason = 'unavailable action'
                break
            self.last = (self.key, self.pos, self.hp, self.turn, action, kind, intended)
            if action.startswith('move '):
                dx, dy = DIRS[action[5:]]
                self.edge_counts[(self.pos, (self.pos[0]+dx, self.pos[1]+dy))] += 1
            self.steps += 1
            if self.steps % 40 == 0:
                self.ctx.log('expedition_state', steps=self.steps, stats=self.s, action=action,
                             calls=self.calls, known=len(self.level.terrain), stale=self.stale,
                             score_peak=self.peak_score, reserve=self.reserve())
            try:
                obs = self.ctx.act(action)
                if obs is None and not self.ctx.done:
                    obs = self.ctx.observe()
            except Exception as exc:
                if self.ctx.done:
                    break
                self.action_errors += 1
                self.ctx.log('expedition_action_failed', action=action, error=type(exc).__name__)
                self.last = None
                self.target = None
                if self.action_errors >= 3:
                    break
                try:
                    obs = self.ctx.observe()
                except Exception:
                    break
            if self.stale >= 18:
                self.stop_reason = 'bounded no-turn-progress stop'
                break
        self.ctx.log('expedition_stop', reason=self.stop_reason or 'episode ended', steps=self.steps,
                     calls=self.calls, score_peak=self.peak_score, stats=getattr(self, 's', {}))
        self.ctx.memory['expedition_summary'] = {'steps': self.steps, 'calls': self.calls,
                                               'reason': self.stop_reason, 'peak_score': self.peak_score}


def build(ctx):
    return Harness(ctx)
