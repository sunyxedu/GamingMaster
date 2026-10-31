"""Observation-only belief search; Astra is the exception and recovery planner."""
from __future__ import annotations

import heapq
import json
import re
from collections import Counter, deque

DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
ACTIONS = {'left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done'}
COLORS = ('red', 'green', 'blue', 'purple', 'yellow', 'grey')
TYPES = ('ball', 'box', 'key', 'door', 'goal')


def decode(obs):
    if isinstance(obs, str):
        try:
            obs = json.loads(obs)
        except (ValueError, TypeError):
            return {}
    return obs if isinstance(obs, dict) else {}


def cells(obs):
    grid = obs.get('grid', [])
    result = {}
    if not isinstance(grid, list):
        return result
    for row in grid:
        for c in (row if isinstance(row, list) else [row]):
            if isinstance(c, dict) and 'x' in c and 'y' in c:
                try:
                    p = (int(c['x']), int(c['y']))
                except (ValueError, TypeError):
                    continue
                result[p] = {k: c[k] for k in ('object', 'color', 'state') if k in c}
    return result


def add(p, d):
    return p[0] + d[0], p[1] + d[1]


def traversable(c):
    return c.get('object') in ('empty', 'floor', 'goal', 'agent') or (
        c.get('object') == 'door' and c.get('state') == 'open')


def signature(c):
    return c.get('object'), c.get('color'), c.get('state')


def parse_json(text):
    text = str(text)
    for i, ch in enumerate(text):
        if ch == '{':
            try:
                value, _ = json.JSONDecoder().raw_decode(text[i:])
                if isinstance(value, dict):
                    return value
            except ValueError:
                pass
    return {}


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.world = {}
        self.pos = (0, 0)
        self.direction = 0
        self.initial_dir = 0
        self.inventory = None
        self.obs = {}
        self.seen_poses = Counter()
        self.history = deque(maxlen=14)
        self.failures = deque(maxlen=8)
        self.repeat = Counter()
        self.revision = 0
        self.last_discovery = 0
        self.steps = 0
        self.contract = None
        self.override = []
        self.session = None
        self.recovery_count = 0
        self.last_recovery = -100
        self.illegal = 0
        self.need_review = None
        self.allowed = ACTIONS
        self.target_override = None

    def transform(self, p, pos=None, direction=None):
        pos = self.pos if pos is None else pos
        d = self.direction if direction is None else direction
        forward, right = DIRS[d], DIRS[(d + 1) % 4]
        return (pos[0] + p[0] * right[0] + p[1] * forward[0],
                pos[1] + p[0] * right[1] + p[1] * forward[1])

    def ingest(self, obs):
        self.obs = decode(obs)
        self.direction = (int(self.obs.get('direction', self.initial_dir + self.direction)) - self.initial_dir) % 4
        self.inventory = self.obs.get('inventory')
        changed = 0
        for p, c in cells(self.obs).items():
            if c.get('object') == 'unknown':
                continue
            q = self.transform(p)
            if p == (0, 0):
                c = {'object': 'empty'}  # The public image's agent cell may encode inventory.
            if signature(c) != signature(self.world.get(q, {})):
                changed += 1
            self.world[q] = c
        if changed:
            self.revision += 1
            self.last_discovery = self.steps
        self.seen_poses[(self.pos, self.direction, self.revision)] += 1

    def compile_mission(self):
        mission = self.obs.get('mission', '').lower().strip().rstrip('.')
        env_id = self.obs.get('environment_id', '')
        # Only compile a deliberately narrow, unambiguous grammar. Novel composition goes to Astra.
        if mission in ('get to the green goal square', 'get to the goal',
                       'use the key to open the door and then get to the goal'):
            return {'kind': 'reach', 'object': 'goal', 'color': None, 'location': None}
        # MultiRoom's wording is a fixed goal-reaching verifier.  Typing it
        # locally avoids a slow semantic call, while the observed-map search
        # still decides which room/door is actually safe.
        if env_id.startswith('MiniGrid-MultiRoom') and 'goal' in mission:
            return {'kind': 'reach', 'object': 'goal', 'color': None, 'location': None}
        if env_id.startswith('MiniGrid-') and 'goal' in mission and not any(
                word in mission for word in ('pick', 'open', 'door', 'key')):
            return {'kind': 'reach', 'object': 'goal', 'color': None, 'location': None}
        match = re.fullmatch(r'(go to|pick up|open) (?:the|a) (?:(red|green|blue|purple|yellow|grey) )?(ball|box|key|door)(?: (on your left|on your right|in front of you|behind you))?', mission)
        if match and (env_id.startswith('BabyAI-') or not env_id):
            verb, color, obj, loc = match.groups()
            return {'kind': {'go to': 'goto', 'pick up': 'pickup', 'open': 'open'}[verb],
                    'object': obj, 'color': color, 'location': loc}
        return None

    def matches(self, p, c):
        spec = self.contract or {}
        if c.get('object') != spec.get('object'):
            return False
        if spec.get('color') and c.get('color') != spec['color']:
            return False
        loc = spec.get('location')
        # Coordinates are frozen to the episode's initial pose, never the current heading.
        if loc == 'on your left' and p[0] >= 0:
            return False
        if loc == 'on your right' and p[0] <= 0:
            return False
        if loc == 'in front of you' and p[1] <= 0:
            return False
        if loc == 'behind you' and p[1] >= 0:
            return False
        return True

    def can_open(self, c):
        if c.get('object') != 'door':
            return False
        if c.get('state') == 'closed':
            return True
        inv = self.inventory or {}
        return c.get('state') == 'locked' and isinstance(inv, dict) and inv.get('object') == 'key' and inv.get('color') == c.get('color')

    def paths(self):
        """Dijkstra on observed (position, heading), including the cost of door toggles."""
        start = (self.pos[0], self.pos[1], self.direction)
        best = {start: (0, [])}
        heap = [(0, start)]
        while heap and len(best) < 12000:
            cost, state = heapq.heappop(heap)
            if cost != best[state][0]:
                continue
            path = best[state][1]
            x, y, d = state
            edges = [((x, y, (d - 1) % 4), ['left']), ((x, y, (d + 1) % 4), ['right'])]
            q = add((x, y), DIRS[d])
            cell = self.world.get(q, {})
            if traversable(cell):
                edges.append(((q[0], q[1], d), ['forward']))
            elif self.can_open(cell):
                edges.append(((q[0], q[1], d), ['toggle', 'forward']))
            for dest, actions in edges:
                nc = cost + len(actions)
                if nc < best.get(dest, (float('inf'), []))[0]:
                    best[dest] = (nc, path + actions)
                    heapq.heappush(heap, (nc, dest))
        return best

    def approach(self, p, paths, final=None):
        choices = []
        for d, vec in enumerate(DIRS):
            pose = (p[0] - vec[0], p[1] - vec[1], d)
            if pose in paths:
                cost, path = paths[pose]
                choices.append((cost + bool(final), path + ([final] if final else [])))
        return min(choices, key=lambda v: v[0]) if choices else None

    def ray_clear(self, a, b):
        # Conservative line-of-sight estimate used only for exploration utility, never movement.
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = max(abs(dx), abs(dy))
        for i in range(1, n):
            q = (round(a[0] + dx * i / n), round(a[1] + dy * i / n))
            c = self.world.get(q, {})
            if c.get('object') == 'wall' or (c.get('object') == 'door' and c.get('state') != 'open'):
                return False
        return True

    def information(self, pose):
        p, d = pose[:2], pose[2]
        gain = 0.0
        spec = self.contract or {}
        for x in range(-3, 4):
            for y in range(0, 7):
                if x == y == 0:
                    continue
                q = self.transform((x, y), p, d)
                if q not in self.world and self.ray_clear(p, q):
                    weight = 1.0 / (1.0 + .2 * (abs(x) + y))
                    loc = spec.get('location')
                    if ((loc == 'behind you' and q[1] < 0) or
                        (loc == 'in front of you' and q[1] > 0) or
                        (loc == 'on your left' and q[0] < 0) or
                        (loc == 'on your right' and q[0] > 0)):
                        weight *= 1.5
                    gain += weight
        return gain

    def choose(self):
        paths = self.paths()
        spec = self.contract or {}
        kind = spec.get('kind')
        candidates = []
        for p, c in self.world.items():
            if not self.matches(p, c):
                continue
            if kind == 'reach':
                for d in range(4):
                    if (*p, d) in paths:
                        candidates.append(paths[(*p, d)])
            else:
                final = {'pickup': 'pickup', 'open': 'toggle'}.get(kind)
                if kind == 'pickup' and self.inventory:
                    continue
                if kind == 'open' and not self.can_open(c):
                    continue
                path = self.approach(p, paths, final)
                if path is not None:
                    candidates.append(path)
        if candidates:
            cost, actions = min(candidates, key=lambda v: v[0])
            if actions:
                return actions[0], 'mission target'
            self.need_review = 'Reached compiled target pose, but environment has not terminated.'
            return None, 'target unconfirmed'

        # Obtain a matching key for an actually observed locked doorway; do not pick random objects.
        if kind == 'reach' and not self.inventory:
            locked = [c for c in self.world.values() if c.get('object') == 'door' and c.get('state') == 'locked']
            key_plans = []
            for p, c in self.world.items():
                if c.get('object') == 'key' and any(c.get('color') == door.get('color') for door in locked):
                    plan = self.approach(p, paths, 'pickup')
                    if plan:
                        key_plans.append(plan)
            if key_plans:
                return min(key_plans, key=lambda v: v[0])[1][0], 'observed key-door dependency'

        # A closed/openable door is an information frontier with a known
        # semantic purpose in MultiRoom.  Prefer reaching its facing pose and
        # toggling it over rotating through several low-yield frontiers.  The
        # route is still emitted one action at a time and rechecked by step().
        if kind == 'reach':
            doors = []
            for p, cell in self.world.items():
                if cell.get('object') != 'door' or not self.can_open(cell):
                    continue
                plan = self.approach(p, paths, 'toggle')
                if plan:
                    doors.append(plan)
            if doors:
                return min(doors, key=lambda v: v[0])[1][0], 'open observed route door'

        # Explore the cheapest useful view. Open doors count as routes and expose new rooms.
        options = []
        for pose, (cost, path) in paths.items():
            if not path:
                continue
            if self.seen_poses[(pose[:2], pose[2], self.revision)]:
                continue
            gain = self.information(pose)
            if gain > .1:
                value = (cost + 1.0) / (1.0 + min(gain, 12.0))
                options.append((value, cost, path))
        if options:
            _, _, path = min(options, key=lambda v: (v[0], v[1]))
            return path[0], 'observe frontier'
        self.need_review = 'No reachable target or informative frontier under current map and inventory.'
        return None, 'no plan'

    def legal(self, action):
        if action not in self.allowed:
            return False
        front = cells(self.obs).get((0, 1), {})
        typ = front.get('object')
        if action == 'forward':
            return traversable(front)
        if action == 'pickup':
            return not self.inventory and typ in ('key', 'ball', 'box')
        if action == 'drop':
            return bool(self.inventory) and typ == 'empty'
        if action == 'toggle':
            return self.can_open(front) or typ == 'box'
        return action in ('left', 'right', 'done')

    def alignment(self, obs, pos, direction):
        score = 0
        for p, c in cells(obs).items():
            if p == (0, 0) or c.get('object') == 'unknown':
                continue
            old = self.world.get(self.transform(p, pos, direction))
            if old:
                # Door state / objects may have changed, so static walls carry more weight.
                weight = 3 if 'wall' in (c.get('object'), old.get('object')) else 1
                score += weight if signature(c) == signature(old) else -weight * 2
        return score

    def step(self, action, why):
        old_obs = self.obs
        old_cells = cells(old_obs)
        old_front = old_cells.get((0, 1), {})
        old_inv = self.inventory
        old_dir = self.direction
        old_pos = self.pos
        error = None
        try:
            result = self.ctx.act(action)
        except Exception as exc:
            if self.ctx.done:
                return False
            self.need_review = 'Action raised ' + type(exc).__name__ + ': ' + str(exc)[:200]
            self.failures.append(self.need_review)
            return True
        new = decode(result if result is not None else self.ctx.observe())
        self.steps += 1
        new_dir = (int(new.get('direction', self.initial_dir + old_dir)) - self.initial_dir) % 4
        if action == 'forward':
            proposed = add(old_pos, DIRS[old_dir])
            stay = self.alignment(new, old_pos, new_dir)
            move = self.alignment(new, proposed, new_dir)
            # With symmetric views, a verified traversable front cell is the motion evidence.
            if traversable(old_front) and move >= stay:
                self.pos = proposed
            else:
                error = 'Forward did not fit predicted displacement; pose retained, map needs review.'
        elif action in ('left', 'right'):
            expected = (old_dir + (-1 if action == 'left' else 1)) % 4
            if new_dir != expected:
                error = 'Turn heading differs from prediction.'
        elif action == 'pickup' and not new.get('inventory'):
            error = 'Pickup did not produce inventory.'
        elif action == 'drop' and new.get('inventory'):
            error = 'Drop did not clear inventory.'
        elif action == 'toggle':
            nf = cells(new).get((0, 1), {})
            if signature(nf) == signature(old_front):
                error = 'Toggle left facing object unchanged.'
        self.ingest(new)
        progress = self.ctx.progress()
        if progress.get('illegal_actions', 0) > self.illegal:
            error = 'ctx.progress reports a newly rejected action.'
        self.illegal = progress.get('illegal_actions', 0)
        if progress.get('since_obs_change', 0) >= 3:
            error = 'ctx.progress reports at least three actions without an observation change.'
        fingerprint = (self.pos, self.direction, tuple(sorted((k, str(v)) for k, v in (self.inventory or {}).items())) if isinstance(self.inventory, dict) else str(self.inventory), self.revision)
        self.repeat[fingerprint] += 1
        if self.repeat[fingerprint] >= 3:
            error = 'Returned to the same pose/inventory/map three times: plan cycle.'
        if self.steps - self.last_discovery > 16:
            error = 'Sixteen actions without any belief-map change.'
        self.history.append({'action': action, 'why': why, 'pose': [*self.pos, self.direction], 'error': error})
        if error and not self.ctx.done:
            self.need_review = error
            self.failures.append(error)
            self.override = []
        self.ctx.log('verified_step', action=action, reason=why, pose=[*self.pos, self.direction],
                     map_cells=len(self.world), revision=self.revision, discrepancy=error, progress=progress)
        return not (self.ctx.done or new.get('terminated') or new.get('truncated'))

    def fallback_contract(self):
        """Recover a usable local controller when Astra is unavailable.

        This parser is deliberately conservative: it only types a single
        verb/object predicate from public mission text.  If a composite
        instruction cannot be typed, returning None still permits safe generic
        frontier exploration rather than ending the episode.
        """
        mission = str(self.obs.get('mission', '')).lower().strip().rstrip('.')
        # Re-run the audited local grammar first (it also preserves qualifiers).
        local = self.compile_mission()
        if local:
            return local
        # Goal wording is stable across the audited MiniGrid goal families.
        if 'goal' in mission and any(w in mission for w in ('get', 'reach', 'traverse', 'use')):
            return {'kind': 'reach', 'object': 'goal', 'color': None, 'location': None}
        colors = '|'.join(COLORS)
        objs = '|'.join(TYPES)
        m = re.search(r'\b(pick up|pickup|go to|reach|open)\b[^.]{0,80}?\b(' +
                      colors + r')\s+(' + objs + r')\b', mission)
        if m:
            verb, color, obj = m.groups()
        else:
            m = re.search(r'\b(pick up|pickup|go to|reach|open)\b[^.]{0,80}?\b(' + objs + r')\b', mission)
            if not m:
                return None
            verb, obj = m.groups()
            color = None
        if m:
            kind = 'pickup' if verb in ('pick up', 'pickup') else ('open' if verb == 'open' else 'goto')
            # A qualifier is accepted only when it is explicit and adjacent to
            # the selected predicate; composite ordering is left to Astra.
            tail = mission[m.end():m.end() + 30]
            loc = next((x for x in ('on your left', 'on your right', 'in front of you', 'behind you') if x in tail), None)
            return {'kind': kind, 'object': obj, 'color': color, 'location': loc}
        return None

    def fallback_after_failure(self, reason):
        contract = self.fallback_contract()
        if contract:
            self.contract = contract
        self.override = []
        self.need_review = None
        self.repeat.clear()
        self.seen_poses.clear()
        self.last_recovery = self.steps
        self.last_discovery = self.steps
        self.ctx.log('deterministic_fallback', reason=reason, contract=self.contract)
        # True means the loop may proceed. With no contract, choose() performs
        # generic safe frontier exploration and never enters unknown cells.
        return True

    def review(self, reason):
        """A failed plan always returns its concrete evidence to Astra before continuing."""
        budget = self.ctx.budget.remaining()
        if budget.get('system2_calls', 0) <= 0 or budget.get('wall_s', 0) < 120:
            return self.fallback_after_failure(reason)
        state = {
            'mission': self.obs.get('mission'), 'environment_id': self.obs.get('environment_id'),
            'frame': 'Initial agent at (0,0), x initial-right, y initial-forward. Heading 0=+y,1=+x,2=-y,3=-x.',
            'pose': [*self.pos, self.direction], 'inventory': self.inventory,
            'contract': self.contract, 'reason': reason, 'failures': list(self.failures),
            'recent': list(self.history), 'progress': self.ctx.progress(),
            'steps_remaining': self.obs.get('remaining_steps'),
            'known_cells': [[*p, c] for p, c in sorted(self.world.items())],
            'observation': self.obs,
        }
        prompt = (
            'You are the exception planner for a partially observed MiniGrid/BabyAI controller. '
            'Use only supplied observations. The executor has stopped on the following evidence. '
            'Diagnose it and return one compact JSON object, no prose. You may return a corrected '
            '"contract": {"kind":"reach|goto|pickup|open", "object":"goal|ball|box|key|door", '
            '"color":null, "location":null}, and/or "actions":[up to 12 action names]. '
            'Location can be "on your left", "on your right", "in front of you", "behind you" '
            'and refers to INITIAL frame. "reach" enters a goal cell; "goto" ends facing an adjacent '
            'object. Do not simplify ordered or relational missions into a wrong single predicate: '
            'for those return a short action plan and "manual":true. '
            'Actions are left/right/forward/pickup/drop/toggle/done. Each will be rechecked and verified. '
            'Unknown and lava are unsafe for forward. No arbitrary wrong-object pickup. '
            'The visible observation explicitly uses front=(0,1); do not reverse y based on raw image conventions. '
            'Use "reset_map":true only if the remembered map is inconsistent; it will reanchor at current pose. '
            'Return at most 180 words.\n' + json.dumps(state, separators=(',', ':')))
        try:
            answer = self.ctx.think(prompt, session_id=None, timeout_s=min(240.0, budget['wall_s'] - 5))
        except Exception as exc:
            self.ctx.log('replan_unavailable', error=type(exc).__name__)
            return self.fallback_after_failure(reason)
        self.recovery_count += 1
        self.last_recovery = self.steps
        data = parse_json(answer.text)
        contract = data.get('contract')
        accepted = False
        if isinstance(contract, dict) and contract.get('kind') in ('reach', 'goto', 'pickup', 'open') and contract.get('object') in TYPES:
            if contract.get('color') in (None, *COLORS) and contract.get('location') in (None, 'on your left', 'on your right', 'in front of you', 'behind you'):
                self.contract = contract
                accepted = True
        if data.get('manual') is True:
            self.contract = None
        raw_actions = data.get('actions', [])
        if isinstance(raw_actions, str):
            raw_actions = re.findall(r'\b(left|right|forward|pickup|drop|toggle|done)\b', raw_actions.lower())
        if isinstance(raw_actions, list):
            self.override = [a for a in raw_actions[:12] if isinstance(a, str) and a in self.allowed]
            accepted = accepted or bool(self.override)
        if data.get('reset_map') is True:
            self.world = {}
            self.ingest(self.obs)
            accepted = True
        self.repeat.clear()
        self.seen_poses.clear()
        self.need_review = None
        self.last_discovery = self.steps
        self.ctx.log('replan_result', parsed=data, recovery=self.recovery_count)
        if accepted and (self.contract or self.override):
            return True
        return self.fallback_after_failure(reason)

    def run_episode(self, game):
        self.allowed = set(game.action_spec.names) & ACTIONS
        obs = decode(self.ctx.observe())
        self.initial_dir = int(obs.get('direction', 0))
        self.ingest(obs)
        self.contract = self.compile_mission()
        if not cells(obs):
            self.need_review = 'Unsupported observation format; request a safe short plan.'
        elif self.contract is None:
            self.need_review = 'Mission does not match the verified simple mission grammar.'
        self.ctx.memory['controller'] = 'verified-belief-search-v1'
        while not self.ctx.done:
            if self.obs.get('terminated') or self.obs.get('truncated'):
                return
            if self.need_review:
                if not self.review(self.need_review):
                    return
            if self.override:
                action = self.override.pop(0)
                why = 'Astra recovery plan'
            elif self.contract:
                action, why = self.choose()
            else:
                # Astra may have failed or declined a composite mission. Keep
                # moving through safe, observed frontiers while preserving the
                # evidence for a later recovery attempt.
                action, why = self.choose()
            if action is None:
                if not self.need_review:
                    self.need_review = 'Planner produced no action.'
                # If no contract/frontier exists and no model budget remains,
                # returning is safer than spinning until the wall deadline.
                if not self.contract and self.ctx.budget.remaining().get('system2_calls', 0) <= 0:
                    return
                continue
            if not self.legal(action):
                self.need_review = 'Precondition veto for ' + action + '; facing=' + str(cells(self.obs).get((0, 1)))
                self.override = []
                continue
            if not self.step(action, why):
                return


def build(ctx):
    return Harness(ctx)
