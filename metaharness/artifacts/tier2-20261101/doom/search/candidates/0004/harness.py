"""Contact cartography: observation-driven servos, batched Jev, asynchronous Astra.

No map files, screenshots, private engine state, or seed-specific routes are used.
The only geometry is the player's actual trajectory and visible label positions.
"""
from __future__ import annotations

import collections
import json
import math
import queue
import threading
from concurrent.futures import CancelledError

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def wrapped(angle):
    return (angle + 180.0) % 360.0 - 180.0


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bearing(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


def cell(p):
    return (math.floor(p[0] / 80.0), math.floor(p[1] / 80.0), math.floor((p[2] + 12.0) / 48.0))


def json_object(text):
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for index, char in enumerate(text[:16000]):
        if char == '{':
            try:
                obj, _ = decoder.raw_decode(text[index:])
                if isinstance(obj, dict):
                    return obj
            except (ValueError, TypeError):
                pass
    return {}


# Engine actor classes, not guesses from the visual appearance of an IWAD.
# Category == Monster is required before this map can assign a threat subtype.
HITSCANNERS = {'Zombieman', 'ShotgunGuy', 'ChaingunGuy', 'WolfensteinSS', 'SpiderMastermind'}
MAJOR = {'Archvile', 'PainElemental', 'Cyberdemon', 'SpiderMastermind', 'Revenant'}
MELEE = {'Demon', 'Spectre', 'LostSoul'}
PICKUPS = {'Key', 'Weapon', 'Health', 'Armor', 'Ammo', 'Powerup', 'PowerUp'}
INTERACTABLE_CATEGORIES = {'exit', 'switch', 'door', 'teleporter', 'teleport'}
INTERACTABLE_NAMES = {'Exit', 'ExitSwitch', 'Switch', 'Door', 'Teleporter', 'TeleportDest'}
WEAPON_NAMES = {'Shotgun': 3, 'SuperShotgun': 3, 'Chaingun': 4,
                'RocketLauncher': 5, 'PlasmaRifle': 6, 'BFG9000': 7,
                'Chainsaw': 1, 'Pistol': 2}


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.stop = threading.Event()
        self.jobs = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=2)
        self.worker = None
        self.pending = None
        self.astra = None
        self.last_jev = -100.0
        self.last_astra = -1000.0
        self.astra_session = None
        self.request_id = 0
        self.policy_version = 0
        self.policy = {'objective': 'explore', 'combat': 'clear_blockers',
                       'hand': 'right', 'focus_id': None,
                       'local_rule': 'Prefer reachable useful pickups; clear immediate threats; explore otherwise.'}
        self.tactic = ('auto', -1.0)
        self.jev_target = (None, -1.0)
        self.jev_goal = (None, -1.0)
        self.plan_disagreements = 0
        self.replan = False
        self.nodes = {}
        self.edges = collections.defaultdict(set)
        self.blocked = {}
        self.use_faces = {}
        self.hazards = collections.defaultdict(float)
        self.landmarks = {}
        self.avoid = {}
        self.suspected_dead = {}
        self.history = collections.deque(maxlen=42)
        self.recent = collections.deque(maxlen=14)
        self.previous = None
        self.previous_node = None
        self.previous_action = 'noop'
        self.previous_reason = 'start'
        self.current = None
        self.target = None
        self.nav = None
        self.nav_path = collections.deque()
        self.contact = None
        self.reverse_until = -1
        self.last_use = -100
        self.last_switch = -100
        self.turn_step = 4.0
        self.turn_sign = -1.0  # right decreases Doom's counterclockwise angle
        self.side = 1
        self.side_since = 0
        self.last_damage = -1000
        self.last_progress = 0
        self.last_log = -100
        self.last_snapshot = -100
        self.last_kills = 0
        self.last_shot = None
        self.fight_start = 0
        self.fight_id = None
        self.bypass_until = -1
        self.failures = 0
        self.last_goal = None
        self.legal = set()

    def observe(self, raw):
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (ValueError, TypeError):
                raw = {}
        if not isinstance(raw, dict):
            raw = {}
        p = raw.get('player') or {}
        if not isinstance(p, dict):
            p = {}
        xyz = tuple(number(p.get(k)) for k in ('x', 'y', 'z'))
        now = number(self.ctx.clock.now())
        tic = int(number(raw.get('tic'), now * 35))
        ammo = raw.get('ammo') if isinstance(raw.get('ammo'), dict) else {}
        weapons = raw.get('weapons') if isinstance(raw.get('weapons'), dict) else {}
        obs = {'p': xyz, 'angle': number(p.get('angle')) % 360, 'tic': tic,
               'now': now, 'health': number(raw.get('health'), 100),
               'armor': number(raw.get('armor')), 'kills': int(number(raw.get('kills'))),
               'items': int(number(raw.get('items'))), 'secrets': int(number(raw.get('secrets'))),
               'damage': number(raw.get('damage_taken')), 'weapon': int(number(raw.get('weapon'), 2)),
               'weapon_ammo': number(raw.get('weapon_ammo')), 'ammo': ammo, 'weapons': weapons,
               'terminal': bool(raw.get('terminal') or raw.get('dead')), 'raw': raw,
               'monsters': [], 'pickups': [], 'projectiles': [], 'labels': [],
               'position_valid': all(k in p for k in ('x', 'y', 'angle'))}
        values = raw.get('visible') or []
        if not isinstance(values, list):
            values = []
        for label in values[:256]:
            if not isinstance(label, dict):
                continue
            world = label.get('world')
            if not isinstance(world, (list, tuple)) or len(world) < 3:
                continue
            pos = tuple(number(v) for v in world[:3])
            name = str(label.get('name', 'unknown'))[:80]
            category = str(label.get('category', 'Unknown'))[:30]
            identifier = str(label.get('id', name + ':' + str(pos)))[:120]
            bb = label.get('bbox')
            bb = [number(v) for v in bb[:4]] if isinstance(bb, (tuple, list)) and len(bb) >= 4 else [0, 0, 0, 0]
            obj = {'id': identifier, 'name': name, 'category': category, 'p': pos,
                   'distance': distance(xyz, pos), 'bearing': bearing(xyz, pos),
                   'dz': pos[2] - xyz[2], 'bbox': bb, 'seen': tic}
            obj['error'] = wrapped(obj['bearing'] - obs['angle'])
            obj['hitscan'] = category == 'Monster' and name in HITSCANNERS
            obj['major'] = category == 'Monster' and name in MAJOR
            obj['melee'] = category == 'Monster' and name in MELEE
            obs['labels'].append(obj)
            if category == 'Monster':
                flat = bb[2] > 0 and bb[3] / bb[2] < 0.40
                if not flat and self.suspected_dead.get(identifier, -1) < tic:
                    obs['monsters'].append(obj)
            elif (category in PICKUPS or category.lower() in INTERACTABLE_CATEGORIES
                  or name in INTERACTABLE_NAMES):
                # Interactables are treated as observed goals, never as proof of a
                # reachable door or of an exit state.  USE is still pulsed by the
                # contact/progress servo and terminal status comes from the engine.
                obs['pickups'].append(obj)
                self.landmarks[identifier] = obj
            elif category.lower() in ('projectile', 'missile'):
                obs['projectiles'].append(obj)
        return obs

    def update(self, o):
        p, tic = o['p'], o['tic']
        key = cell(p)
        previous = self.previous
        if key not in self.nodes:
            self.nodes[key] = {'p': p, 'tries': [0] * 8, 'visits': 0, 'last': tic}
            self.last_progress = tic
        node = self.nodes[key]
        if key != self.previous_node:
            node['visits'] += 1
            if previous and self.previous_node in self.nodes and distance(p, previous['p']) < 100:
                self.edges[self.previous_node].add(key)
                if abs(p[2] - previous['p'][2]) < 25:
                    self.edges[key].add(self.previous_node)
            self.previous_node = key
        node['last'] = tic
        if previous:
            dt = tic - previous['tic']
            da = wrapped(o['angle'] - previous['angle'])
            if 0 < dt <= 6 and self.previous_action in ('turn_left', 'turn_right') and 0.1 < abs(da) < 90:
                step = abs(da) / dt
                self.turn_step = 0.7 * self.turn_step + 0.3 * step
                if self.previous_action == 'turn_right':
                    self.turn_sign = 1.0 if da > 0 else -1.0
                else:
                    self.turn_sign = -1.0 if da > 0 else 1.0
            damage = max(0, previous['health'] - o['health'], o['damage'] - previous['damage'])
            if damage > 0:
                self.last_damage = tic
                self.hazards[key] += min(30, damage)
            if o['kills'] > previous['kills']:
                self.last_progress = tic
                if self.last_shot is not None:
                    # A kill is only circumstantial evidence about the last aimed actor.
                    self.suspected_dead[self.last_shot] = tic + 22
            if o['items'] > previous['items'] or o['weapons'] != previous['weapons']:
                self.last_progress = tic
            if distance(p, previous['p']) > 180:
                self.nav = None
                self.nav_path.clear()
                self.contact = None
                self.replan = True
        self.history.append((tic, p, self.previous_action))
        # A passed-through pickup that is now absent is unavailable, not a proven key inventory.
        visible_ids = {q['id'] for q in o['pickups']}
        for ident, obj in list(self.landmarks.items()):
            if distance(obj['p'], p) < 30 and abs(obj['p'][2] - p[2]) < 30 and ident not in visible_ids:
                self.avoid[ident] = tic + 1050
            if tic - obj['seen'] > 6300:
                del self.landmarks[ident]
        if len(self.landmarks) > 250:
            for ident, _ in sorted(self.landmarks.items(), key=lambda item: item[1]['seen'])[:-250]:
                del self.landmarks[ident]
        self.previous = o
        self.current = o

    def owned(self, o, slot):
        slots = (3, 9) if slot == 3 else (1, 8) if slot == 1 else (slot,)
        return any(bool(o['weapons'].get(str(s), o['weapons'].get(s, False))) for s in slots)

    def ammunition(self, o, slot):
        if slot == 1:
            return 1
        value = number(o['ammo'].get(str(slot), o['ammo'].get(slot)))
        if slot == 3:
            value = max(value, number(o['ammo'].get('9')))
        if slot == (3 if o['weapon'] == 9 else o['weapon']):
            value = max(value, o['weapon_ammo'])
        return value

    def pickup_value(self, obj, o):
        if self.avoid.get(obj['id'], -1) > o['tic']:
            return -1000
        d = distance(o['p'], obj['p'])
        dz = obj['p'][2] - o['p'][2]
        if abs(dz) > 34 or d > 850:
            return -1000  # rendered across an inaccessible ledge is not a route
        cat, name = obj['category'], obj['name']
        cat_lower = cat.lower()
        value = 0
        if cat_lower == 'exit' or name in ('Exit', 'ExitSwitch'):
            value = 180
        elif cat_lower == 'switch' or name == 'Switch':
            value = 115
        elif cat_lower in ('door', 'teleporter', 'teleport') or name in ('Door', 'Teleporter', 'TeleportDest'):
            value = 50
        elif cat == 'Key':
            value = 105
        elif cat == 'Weapon':
            slot = WEAPON_NAMES.get(name)
            value = 100 if slot and not self.owned(o, slot) and slot in (3, 4, 6) else 22
            if slot == 5:
                value = 24
        elif cat == 'Health':
            value = 100 if o['health'] < 40 else 68 if o['health'] < 75 else 25 if o['health'] < 95 else 0
            if 'Bonus' in name or 'Soul' in name or 'Mega' in name:
                value = max(value, 22)
        elif cat == 'Armor':
            value = 60 if o['armor'] < 30 else 30 if o['armor'] < 80 else 5
        elif cat == 'Ammo':
            low = o['weapon_ammo'] < 12
            value = 60 if low else 16
            if 'Shell' in name and not self.owned(o, 3):
                value = 3
            if ('Rocket' in name or 'Cell' in name) and not (self.owned(o, 5) or self.owned(o, 6)):
                value = 3
        else:
            value = 25
        if self.policy['objective'] == 'recover' and cat in ('Health', 'Armor'):
            value += 20
        if self.policy['objective'] == 'arm' and cat in ('Weapon', 'Ammo'):
            value += 20
        if self.policy['objective'] == 'key' and cat == 'Key':
            value += 15
        if self.policy.get('focus_id') == obj['id']:
            value += 10
        return value - d * 0.055 - max(0, o['tic'] - obj['seen'] - 35) * 0.005

    def useful_pickups(self, o):
        ranked = sorted(self.landmarks.values(), key=lambda x: self.pickup_value(x, o), reverse=True)
        return [p for p in ranked if self.pickup_value(p, o) > 10][:5]

    def threat_value(self, m, o):
        value = 100 if m['hitscan'] else 75 if m['major'] else 40
        value += 110 if m['distance'] < 115 else 45 if m['distance'] < 260 else 0
        value -= m['distance'] * 0.055
        value -= abs(m['error']) * 0.15
        if m['id'] == self.fight_id:
            value += 14
        return value

    def choose_enemy(self, o):
        if not o['monsters']:
            return None
        ranked = sorted(o['monsters'], key=lambda m: self.threat_value(m, o), reverse=True)
        ident, until = self.jev_target
        chosen = next((m for m in ranked if m['id'] == ident), None) if until >= o['now'] else None
        close = next((m for m in ranked if m['distance'] < 100), None)
        return close or chosen or ranked[0]

    def turn(self, error):
        tolerance = max(2.3, min(7.0, self.turn_step * 0.56))
        if abs(error) <= tolerance:
            return None
        return 'turn_right' if error * self.turn_sign > 0 else 'turn_left'

    def motion_stalled(self, o, window=10):
        samples = [h for h in self.history if o['tic'] - window <= h[0] < o['tic']]
        if len(samples) < 3 or o['tic'] - samples[0][0] < window - 3:
            return False
        moving = [h for h in samples if any(word in h[2] for word in ('forward', 'backward', 'strafe'))]
        return len(moving) >= len(samples) * 0.6 and distance(samples[0][1], o['p']) < 5

    def weapon_action(self, o, enemy):
        tic = o['tic']
        if tic - self.last_switch < 35:
            return None
        current_slot = 3 if o['weapon'] == 9 else 1 if o['weapon'] == 8 else o['weapon']
        order = [3, 4, 2, 6, 1]
        if enemy and (enemy['major'] or len(o['monsters']) >= 4) and self.ammunition(o, 6) >= 8:
            order = [6, 3, 4, 2, 1]
        elif enemy and enemy['distance'] > 430:
            order = [4, 3, 2, 6, 1]
        desired = next((s for s in order if self.owned(o, s) and self.ammunition(o, s) > 0), None)
        if desired is not None and desired != current_slot:
            action = 'select_weapon_' + str(desired)
            if action in self.legal:
                self.last_switch = tic
                return action
        return None

    def combat(self, o, enemy):
        if not enemy:
            self.fight_id = None
            return None
        tic = o['tic']
        if self.fight_id != enemy['id']:
            self.fight_id = enemy['id']
            self.fight_start = tic
        d = enemy['distance']
        immediate = d < 150 or (enemy['hitscan'] and d < 650) or tic - self.last_damage < 35
        posture = self.tactic[0] if self.tactic[1] >= o['now'] else 'auto'
        if self.bypass_until > tic and not immediate:
            return None
        if tic - self.fight_start > 350 and not immediate:
            self.bypass_until = tic + 70
            return None
        if posture == 'bypass' and not immediate:
            return None
        if d > 850 and not immediate:
            return None
        if o['weapon'] in (5, 7):
            return 'backward'  # do not fire splash / expensive weapons into unknown geometry
        if o['weapon'] in (1, 8) and d > 65:
            return None
        if o['weapon_ammo'] <= 0 and o['weapon'] not in (1, 8):
            return None
        turn = self.turn(enemy['error'])
        if turn:
            return turn
        if self.motion_stalled(o) and tic - self.side_since > 12:
            self.side *= -1
            self.side_since = tic
        elif tic - self.side_since > 65:
            self.side *= -1
            self.side_since = tic
        retreat = d < (140 if enemy['melee'] else 85) or (posture == 'retreat' and immediate)
        if retreat and tic % 18 < 7:
            return 'backward_left' if self.side < 0 else 'backward_right'
        self.last_shot = enemy['id']
        if d > 580 and tic - self.last_damage > 35 and not o['projectiles']:
            return 'fire'
        return 'fire_strafe_left' if self.side < 0 else 'fire_strafe_right'

    def route_to_frontier(self, start, o):
        parents = {start: None}
        todo = collections.deque([start])
        found = None
        while todo and len(parents) < 450:
            k = todo.popleft()
            n = self.nodes.get(k)
            if n is None:
                continue
            if k != start and min(n['tries']) == 0 and self.hazards[k] < 25:
                found = k
                break
            for nxt in sorted(self.edges.get(k, ())):
                if nxt not in parents and self.hazards[nxt] < 70:
                    parents[nxt] = k
                    todo.append(nxt)
        path = []
        while found is not None and found != start:
            path.append(self.nodes[found]['p'])
            found = parents[found]
        return list(reversed(path))

    def explore_goal(self, o):
        p, key = o['p'], cell(o['p'])
        node = self.nodes[key]
        if not self.nav_path and min(node['tries']) >= 1:
            self.nav_path.extend(self.route_to_frontier(key, o))
        while self.nav_path and distance(self.nav_path[0], p) < 28:
            self.nav_path.popleft()
        if self.nav_path:
            return {'p': self.nav_path.popleft(), 'id': None, 'kind': 'backtrack'}
        scores = []
        hand = -1 if self.policy['hand'] == 'right' else 1
        for index in range(8):
            angle = index * 45.0
            rad = math.radians(angle)
            q = (p[0] + math.cos(rad) * 150, p[1] + math.sin(rad) * 150, p[2])
            near = (p[0] + math.cos(rad) * 75, p[1] + math.sin(rad) * 75, p[2])
            near_key, far_key = cell(near), cell(q)
            novelty = 5 * (near_key not in self.nodes) + 3 * (far_key not in self.nodes)
            facing = math.cos(math.radians(wrapped(angle - o['angle'])))
            cost = node['tries'][index] * 3.5 + self.hazards[near_key] * 0.18
            if self.blocked.get((key, index), -1) > o['tic']:
                cost += 25
            tie = 0.15 * math.sin(math.radians(wrapped(angle - o['angle']))) * hand
            scores.append((novelty + facing * 1.8 + tie - cost, index, q))
        _, index, q = max(scores, key=lambda row: row[0])
        node['tries'][index] += 1
        return {'p': q, 'id': None, 'kind': 'probe', 'origin': key, 'direction': index}

    def start_nav(self, o):
        pickups = self.useful_pickups(o)
        chosen = pickups[0] if pickups else None
        preferred, until = self.jev_goal
        if until >= o['now']:
            chosen = next((p for p in pickups if p['id'] == preferred), chosen)
        if chosen:
            nav = {'p': chosen['p'], 'id': chosen['id'], 'kind': 'pickup'}
        else:
            nav = self.explore_goal(o)
        nav.update({'started': o['tic'], 'progress': o['tic'],
                    'best': distance(o['p'], nav['p']), 'origin_p': o['p']})
        self.nav = nav

    def navigation(self, o):
        tic, p = o['tic'], o['p']
        if self.reverse_until > tic:
            return 'backward', 'contact_backoff'
        if self.contact:
            contact = self.contact
            if distance(p, contact['p']) > 24:
                self.contact = None
                self.nav = None
                self.last_progress = tic
            elif tic < contact['until']:
                return 'forward', 'probe_after_use'
            else:
                index = int(round(contact['angle'] / 45)) % 8
                key = cell(p)
                self.blocked[(key, index)] = tic + 1050
                self.nodes[key]['tries'][index] += 1
                if self.nav and self.nav.get('id'):
                    self.avoid[self.nav['id']] = tic + 700
                self.nav = None
                self.nav_path.clear()
                self.contact = None
                self.reverse_until = tic + 5
                self.failures += 1
                return 'backward', 'blocked_face'
        if self.nav:
            d = distance(p, self.nav['p'])
            if d < 24:
                landmark = self.landmarks.get(self.nav.get('id'))
                if landmark and (landmark['category'].lower() in INTERACTABLE_CATEGORIES
                                  or landmark['name'] in INTERACTABLE_NAMES):
                    if tic - self.last_use > 12:
                        self.last_use = tic
                        self.contact = {'p': p, 'angle': o['angle'], 'until': tic + 24}
                        return 'use', 'activate_observed_interactable'
                    return 'forward', 'hold_observed_interactable'
                if self.nav['kind'] == 'pickup':
                    self.avoid[self.nav['id']] = tic + 350
                self.nav = None
            elif tic - self.nav['started'] > 140:
                self.nav = None
            elif d < self.nav['best'] - 7:
                self.nav['best'] = d
                self.nav['progress'] = tic
        if self.nav is None:
            self.start_nav(o)
        nav = self.nav
        angle = bearing(p, nav['p'])
        err = wrapped(angle - o['angle'])
        turn = self.turn(err)
        if turn:
            return turn, 'steer_' + nav['kind']
        stalled = self.motion_stalled(o) or (tic - nav['progress'] > 24 and tic - nav['started'] > 28)
        if stalled:
            face = (cell(p), int(round(o['angle'] / 45)) % 8)
            if self.use_faces.get(face, -1) < tic and tic - self.last_use > 12:
                self.last_use = tic
                self.use_faces[face] = tic + 175
                self.contact = {'p': p, 'angle': o['angle'], 'until': tic + 24}
                return 'use', 'test_contact_for_door_or_switch'
            index = face[1]
            self.blocked[face] = tic + 700
            self.nodes[cell(p)]['tries'][index] += 1
            if nav.get('id'):
                self.avoid[nav['id']] = tic + 700
            self.nav = None
            self.nav_path.clear()
            self.reverse_until = tic + 5
            self.failures += 1
            return 'backward', 'failed_motion'
        if tic - self.last_use > 65:
            self.last_use = tic
            return 'use', 'transit_interaction_probe'
        return 'forward', 'advance_' + nav['kind']

    def control(self, o):
        if not o['position_valid']:
            return 'turn_right', 'missing_proprioception'
        enemy = self.choose_enemy(o)
        self.target = enemy
        switch = self.weapon_action(o, enemy)
        if switch:
            return switch, 'legal_weapon_upgrade'
        action = self.combat(o, enemy)
        if action:
            # Navigation progress clocks must not mistake a fight for a wall collision.
            if self.nav:
                self.nav['progress'] = o['tic']
                self.nav['started'] = o['tic']
            return action, 'combat_' + (self.tactic[0] if self.tactic[1] >= o['now'] else 'fallback')
        if o['tic'] - self.last_damage < 20 and not enemy and self.nav is None:
            # Orient toward a threat instead of blindly firing when hit from behind.
            return 'turn_left' if self.side < 0 else 'turn_right', 'unseen_damage_scan'
        return self.navigation(o)

    def describe_object(self, obj, o):
        d = distance(o['p'], obj['p'])
        err = wrapped(bearing(o['p'], obj['p']) - o['angle'])
        dz = obj['p'][2] - o['p'][2]
        return {'id': obj['id'], 'name': obj['name'], 'category': obj['category'],
                'range': 'contact' if d < 95 else 'near' if d < 230 else 'medium' if d < 550 else 'far',
                'direction': 'centered' if abs(err) < 8 else 'left' if err > 0 else 'right',
                'elevation': 'similar_level' if abs(dz) <= 34 else 'different_level_not_directly_reachable',
                'visibility': 'visible_now' if obj['seen'] == o['tic'] else 'remembered_not_visible',
                'hitscanner': obj.get('hitscan', False), 'major_threat': obj.get('major', False),
                'body_blocker': obj.get('melee', False) and d < 150}

    def jev_packet(self, o):
        self.request_id += 1
        enemies = sorted(o['monsters'], key=lambda m: self.threat_value(m, o), reverse=True)[:5]
        pickups = self.useful_pickups(o)
        state = {'task': 'Classic Doom. Survive and activate the exit; optional kills are secondary. No map geometry is visible.',
                 'rules': 'Coordinates, aiming, ammo legality and collision tests are handled by code. Choose intent, not buttons. A remembered actor is not currently visible. Different-level pickups need an unknown route.',
                 'plan': self.policy.copy(), 'plan_version': self.policy_version,
                 'health': 'critical' if o['health'] < 30 else 'injured' if o['health'] < 65 else 'healthy',
                 'armor': 'unprotected' if o['armor'] < 10 else 'protected',
                 'ammunition': 'empty' if o['weapon_ammo'] <= 0 else 'low' if o['weapon_ammo'] < 10 else 'usable',
                 'weapon': o['weapon'], 'under_recent_damage': o['tic'] - self.last_damage < 35,
                 'projectile_visible': bool(o['projectiles']),
                 'mobility': 'blocked' if self.motion_stalled(o) else 'moving_or_turning',
                 'progress': 'stagnant' if o['tic'] - self.last_progress > 700 else 'recent_progress',
                 'threats': [self.describe_object(m, o) for m in enemies],
                 'goals': [self.describe_object(p, o) for p in pickups],
                 'previous_action': self.previous_action, 'controller_reason': self.previous_reason,
                 'local_rule': self.policy['local_rule']}
        target_options = {'m' + str(i): self.describe_object(m, o) for i, m in enumerate(enemies)}
        target_options['none'] = 'No visible threat warrants targeting; explore or recover instead.'
        goal_options = {'g' + str(i): self.describe_object(p, o) for i, p in enumerate(pickups)}
        goal_options['explore'] = 'Probe untried directions using observed movement and contact memory; do not assume a hidden corridor.'
        goal_options['unknown'] = 'Evidence cannot distinguish a useful safe objective; use the code fallback.'
        # Alternate choice order across requests; no irreversible attack depends on a weak vote.
        if self.request_id % 2 == 0:
            target_options = dict(reversed(list(target_options.items())))
            goal_options = dict(reversed(list(goal_options.items())))
        questions = {
            'posture': Choice('Given threats, health, ammunition and recent damage, choose the immediate tactical posture. Apply local_rule only when consistent with the observed facts.', {
                'engage': 'Aim and shoot a visible immediate threat while strafing; do not charge it.',
                'retreat': 'Create distance from contact threats while maintaining a safe firing opportunity.',
                'bypass': 'Continue exploration or a pickup route rather than fight a distant optional monster.',
                'unknown': 'Facts are insufficient; retain the deterministic safety controller.'}),
            'target': Choice('Which supplied visible threat most urgently needs aim control? Prioritize immediate hitscan or body-blocking danger, not optional kills.', target_options),
            'goal': Choice('Which supplied local objective best supports the plan without chasing a different-level or unavailable pickup? Use explore when no useful reachable pickup is warranted.', goal_options),
            'plan_invalid': Noul('Do the observed threats, resource buckets and recent progress clearly contradict the current plan, requiring a durable strategic revision? Mere uncertainty or one blocked motion is not contradiction.',
                                 {'true': 'Clear mismatch: repeated stagnation, a resource crisis inconsistent with the objective, or an unsafe combat plan.',
                                  'false': 'Plan remains plausible, evidence is uncertain, or code can handle this one local obstacle.'})}
        meta = {'id': self.request_id, 'now': o['now'], 'tic': o['tic'], 'version': self.policy_version,
                'targets': {'m' + str(i): m['id'] for i, m in enumerate(enemies)},
                'goals': {'g' + str(i): p['id'] for i, p in enumerate(pickups)}}
        return state, questions, meta

    def jev_loop(self):
        while not self.stop.is_set() and not self.ctx.done:
            try:
                state, questions, meta = self.jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            if self.stop.is_set() or self.ctx.done:
                break
            try:
                answers = self.ctx.decide(state, questions, timeout_s=2.5)
                result = (meta, answers, None)
            except ModelUnavailable:
                raise
            except (BudgetExceeded, CancelledError):
                return
            except (TimeoutError, RuntimeError, TypeError, ValueError) as exc:
                result = (meta, {}, type(exc).__name__)
            try:
                self.results.put_nowait(result)
            except queue.Full:
                pass

    @staticmethod
    def accepted(answers, key, threshold):
        answer = answers.get(key) if isinstance(answers, dict) else None
        value = getattr(answer, 'value', None)
        probabilities = getattr(answer, 'probabilities', None)
        if not isinstance(value, str) or not isinstance(probabilities, dict):
            return None
        if number(probabilities.get(value)) < threshold:
            return None
        confidence = getattr(answer, 'confidence', None)
        if confidence is not None and number(confidence) < 0.12:
            return None
        return value

    def poll_jev(self, o):
        try:
            meta, answers, error = self.results.get_nowait()
        except queue.Empty:
            return
        self.pending = None
        if error:
            self.ctx.log('jev_failure', error=error, request=meta['id'])
            return
        age = o['now'] - meta['now']
        if meta['version'] != self.policy_version or age > 2.5:
            self.ctx.log('jev_discard', request=meta['id'], age_s=age, reason='old_policy_or_state')
            return
        adopted = {}
        if age <= 0.85 and o['tic'] - meta['tic'] <= 30:
            posture = self.accepted(answers, 'posture', 0.62)
            if posture in ('engage', 'retreat', 'bypass'):
                self.tactic = (posture, o['now'] + 0.7)
                adopted['posture'] = posture
            key = self.accepted(answers, 'target', 0.56)
            ident = meta['targets'].get(key)
            if ident is not None and any(m['id'] == ident for m in o['monsters']):
                self.jev_target = (ident, o['now'] + 0.7)
                adopted['target'] = ident
        goal = self.accepted(answers, 'goal', 0.62)
        ident = meta['goals'].get(goal)
        if goal == 'explore':
            self.jev_goal = (None, -1)
            if self.nav and self.nav.get('kind') == 'pickup' and not self.contact:
                self.nav = None
            adopted['goal'] = 'explore'
        elif ident in self.landmarks and self.pickup_value(self.landmarks[ident], o) > 10:
            self.jev_goal = (ident, o['now'] + 7.0)
            if self.nav and self.nav.get('id') != ident and not self.contact:
                self.nav = None
            adopted['goal'] = ident
        invalid = answers.get('plan_invalid') if isinstance(answers, dict) else None
        if number(getattr(invalid, 'value', 0)) >= 0.85:
            self.plan_disagreements += 1
            if self.plan_disagreements >= 3:
                self.replan = True
        else:
            self.plan_disagreements = max(0, self.plan_disagreements - 1)
        self.ctx.log('jev_bridge', request=meta['id'], age_s=round(age, 3), adopted=adopted,
                     fallback=not bool(adopted), replan=self.replan)

    def launch_astra(self, o):
        # This deliberately includes the local Jev contract, not an inaccessible prompt reference.
        guide = ('Jev is a fast typed classifier with no memory. Code supplies derived English facts, '
                 'candidate labels, resource buckets, and your policy on each request. One ctx.decide '
                 'batch asks posture Choice(engage/retreat/bypass/unknown), target Choice over visible '
                 'actors, goal Choice over observed useful pickups/explore/unknown, and plan_invalid '
                 'Noul. Choice answers have .value string, .probabilities and nullable .confidence; '
                 'Noul .value is probability of yes. Choices require winning probability >= .62 '
                 '(targets .56); tactical answers expire after .85 s and are rechecked against current '
                 'visibility. Jev is weak at arithmetic, raw coordinates, hidden geometry, long plans '
                 'and vague instructions. Your local_rule is appended to the posture instructions and '
                 'included in every state; make it short, conditional, and grounded in named fields. '
                 'It cannot override ammo, collision, aiming or safety code. No generated code is executed.')
        observation = {k: v for k, v in o['raw'].items() if k not in ('visible', 'seed')}
        observation['visible'] = [self.describe_object(m, o) for m in (o['monsters'][:8] + o['pickups'][:10])]
        payload = {'observation': observation, 'recent': list(self.recent),
                   'known_landmarks': [self.describe_object(p, o) for p in self.useful_pickups(o)],
                   'exploration': {'visited_cells': len(self.nodes), 'failed_probes': self.failures,
                                   'stagnant': o['tic'] - self.last_progress > 700,
                                   'current_goal': self.nav, 'hand': self.policy['hand']},
                   'policy': self.policy, 'jev_repeatedly_disagrees': self.plan_disagreements >= 3}
        prompt = ('You are the slow strategic layer of a real-time Doom agent. Never give a button '
                  'or a memorized map route: screen/wall/sector data are unavailable and actors alone '
                  'do not expose doors or exits. The code acts every tic, aims at visible actors, '
                  'learns traversed edges, tries USE on blocked faces, and backtracks to untried '
                  'directions. Your reply may arrive minutes later, so revise durable policy, not '
                  'current aim. Exit is primary, survival prerequisite, optional kills secondary. '
                  'Change exploration handedness or resource objective when observed failures warrant '
                  'it. A focus_id must be one of the supplied known landmark ids, otherwise null. '
                  'No assumptions about the seed, commercial Doom map layout, hidden keys or walls.\n'
                  + guide + '\nReply ONLY with a JSON object of at most 200 words: '
                  '{"objective":"explore|recover|arm|key", "combat":"clear_blockers|avoid|control_hitscanners", '
                  '"hand":"left|right", "focus_id":null, "local_rule":"one conditional rule referencing '
                  'threats, health, ammunition, mobility or progress", "reason":"observed evidence"}. '
                  'Malformed output leaves the previous safe policy intact.\n' + json.dumps(payload, allow_nan=False))
        self.last_astra = o['now']
        self.replan = False
        self.astra = self.ctx.spawn(lambda: self.ctx.think(prompt, session_id=self.astra_session, timeout_s=240.0))
        self.ctx.log('astra_requested', failed_probes=self.failures, visited_cells=len(self.nodes))

    def poll_astra(self, o):
        if self.astra is None or not self.astra.done():
            return
        future = self.astra
        self.astra = None
        try:
            result = future.result()
        except ModelUnavailable:
            raise
        except (BudgetExceeded, CancelledError):
            return
        except (TimeoutError, RuntimeError, TypeError, ValueError) as exc:
            self.ctx.log('astra_failure', error=type(exc).__name__)
            return
        self.astra_session = getattr(result, 'session_id', None)
        proposed = json_object(getattr(result, 'text', ''))
        valid = False
        updated = dict(self.policy)
        for key, options in [('objective', ('explore', 'recover', 'arm', 'key')),
                             ('combat', ('clear_blockers', 'avoid', 'control_hitscanners')),
                             ('hand', ('left', 'right'))]:
            if isinstance(proposed.get(key), str) and proposed[key] in options:
                updated[key] = proposed[key]
                valid = True
        focus = proposed.get('focus_id')
        if focus is None:
            updated['focus_id'] = None
        elif isinstance(focus, (str, int)) and str(focus) in self.landmarks:
            updated['focus_id'] = str(focus)
        local_rule = proposed.get('local_rule')
        if isinstance(local_rule, str) and 4 <= len(local_rule) <= 550:
            updated['local_rule'] = local_rule
            valid = True
        if valid:
            self.policy = updated
            self.policy_version += 1
            self.tactic = ('auto', -1)
            self.jev_target = (None, -1)
            self.jev_goal = (None, -1)
            self.nav = None
            self.nav_path.clear()
            self.plan_disagreements = 0
            self.ctx.log('astra_policy_applied', version=self.policy_version, policy=self.policy,
                         reason=str(proposed.get('reason', ''))[:400])
        else:
            self.ctx.log('astra_invalid_output', fallback='keep_previous_policy')

    def run_episode(self, game):
        ctx = self.ctx
        self.legal = set(game.action_spec.names)
        first = True
        try:
            while not ctx.done:
                raw = ctx.observe()
                o = self.observe(raw)
                if o['terminal']:
                    break
                if self.previous is not None and o['tic'] == self.previous['tic']:
                    ctx.wait_for_tick()
                    continue
                self.update(o)
                if first:
                    # Immediate safe, reversible motion; never wait a minute for Astra.
                    ctx.act('turn_right' if 'turn_right' in self.legal else game.action_spec.noop)
                    self.previous_action = 'turn_right'
                    first = False
                    self.worker = ctx.spawn(self.jev_loop)
                self.poll_astra(o)
                self.poll_jev(o)
                if self.worker is not None and self.worker.done():
                    try:
                        self.worker.result()  # surface backend unavailability, never silently ignore it
                    except ModelUnavailable:
                        raise
                    except (CancelledError, EpisodeOver, BudgetExceeded):
                        pass
                    self.worker = None
                    self.pending = None
                action, reason = self.control(o)
                if action not in self.legal:
                    action = 'forward' if 'forward' in self.legal else game.action_spec.noop
                    reason = 'legal_fallback'
                ctx.act(action)
                if reason != self.previous_reason and (reason.startswith('combat') or reason in ('blocked_face', 'failed_motion', 'test_contact_for_door_or_switch')):
                    ctx.log('control_transition', tic=o['tic'], action=action, reason=reason,
                            target_id=self.target['id'] if self.target else None)
                self.previous_action, self.previous_reason = action, reason
                remaining = ctx.budget.remaining()
                interval = 0.38 if o['monsters'] or o['tic'] - self.last_damage < 35 else 1.2
                if self.worker is not None and self.pending is None and remaining.get('system1_calls', 0) > 3 and o['now'] - self.last_jev >= interval:
                    packet = self.jev_packet(o)
                    try:
                        self.jobs.put_nowait(packet)
                        self.pending = packet[2]
                        self.last_jev = o['now']
                    except queue.Full:
                        pass
                if o['tic'] - self.last_snapshot >= 175:
                    self.last_snapshot = o['tic']
                    snapshot = {'tic': o['tic'], 'health': o['health'], 'armor': o['armor'],
                                'kills': o['kills'], 'items': o['items'], 'weapon': o['weapon'],
                                'ammo': o['weapon_ammo'], 'position': list(o['p']),
                                'nodes': len(self.nodes), 'failed_probes': self.failures,
                                'reason': reason, 'plan_version': self.policy_version}
                    self.recent.append(snapshot)
                    ctx.log('controller_summary', **snapshot)
                due = self.last_astra < -100 or (self.replan and o['now'] - self.last_astra > 110) or o['now'] - self.last_astra > 210
                if self.astra is None and due and remaining.get('system2_calls', 0) > 1 and remaining.get('wall_s', 0) > 100:
                    self.launch_astra(o)
                ctx.wait_for_tick()
        except (EpisodeOver, BudgetExceeded):
            return
        finally:
            self.stop.set()
            ctx.memory['contact_cartography_summary'] = {'nodes': len(self.nodes), 'failed_probes': self.failures,
                                                         'policy_version': self.policy_version,
                                                         'policy': self.policy}


def build(ctx):
    return Harness(ctx)
