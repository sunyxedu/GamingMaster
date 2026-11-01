"""Grounded actions, bounded jobs, and a safe idle lease for Minecraft."""
from __future__ import annotations

import json
import math
import re

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


def number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def name(value):
    if isinstance(value, dict):
        value = value.get('name', value.get('item', ''))
    return str(value or '').removeprefix('minecraft:')


def rows(value):
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def inventory(obs):
    # Never count cursor, window, recipe preview, or container items.
    source = obs.get('inventory', obs.get('inventory_authoritative', {}))
    if isinstance(source, dict) and isinstance(source.get('items'), (dict, list)):
        source = source['items']
    out = {}
    if isinstance(source, dict):
        for key, value in source.items():
            count = value.get('count', 0) if isinstance(value, dict) else value
            if number(count) > 0:
                out[name(key)] = int(number(count))
    elif isinstance(source, list):
        for item in rows(source):
            key = name(item)
            out[key] = out.get(key, 0) + int(number(item.get('count')))
    return out


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.last_action = 'noop'
        self.last_issue = -100.0
        self.attempted = {}
        self.visits = {}
        self.saw_table = False
        self.last_inv = None
        self.last_progress = 0.0
        self.planner = None
        self.planned_at = -1000.0
        self.focus = 'balanced'
        self.adviser = None
        self.decision_at = 0.0
        self.decision_options = []
        self.decision_dimension = None
        self.decision_inventory = None
        self.last_decision = -100.0
        self.furnaces = {}

    def log(self, event, **fields):
        self.ctx.log(event, **fields)

    def post(self, action, now):
        # Every actual operation is followed by a noop tick. This releases the
        # one-shot latch without interrupting the bounded asynchronous job.
        self.ctx.act(action)
        self.last_action = action
        if action != 'noop':
            self.last_issue = now
            self.attempted[action] = now
            if action.startswith('furnace_start '):
                self.furnaces[action.split()[1]] = now
            self.log('grounded_action', action=action, focus=self.focus)

    def options(self, obs, now):
        me = obs.get('self') or {}
        inv = inventory(obs)
        p = me.get('position') or [0, 64, 0]
        if not isinstance(p, (list, tuple)) or len(p) < 3:
            p = [0, 64, 0]
        p = [number(v) for v in p[:3]]
        candidates = rows(obs.get('candidates'))
        blocks = [b for b in candidates if str(b.get('id', '')).startswith('b:')]
        blocks.sort(key=lambda b: number(b.get('distance'), 1000))
        tables = [b for b in blocks if name(b) == 'crafting_table']
        table = next((b for b in tables if b.get('in_reach')), None)
        furnaces = [b for b in blocks if name(b) == 'furnace']
        self.saw_table = self.saw_table or bool(inv.get('crafting_table')) or bool(tables)
        held = name(me.get('held_item'))
        opts = []

        def add(action, purpose, priority=0):
            cooldown = 4 if action.startswith(('attack ', 'equip ', 'look_at ')) else 12
            if now - self.attempted.get(action, -1000) >= cooldown:
                opts.append({'action': action, 'purpose': purpose, 'priority': priority})

        def gather(predicate, purpose, priority=0, count=1):
            eligible = [b for b in blocks if predicate(name(b)) and b.get('harvestable') is not False]
            for b in eligible[:5]:
                before = len(opts)
                add('gather %s %d' % (b['id'], count), purpose, priority)
                if len(opts) > before:
                    return True
            return False

        def craft(item, count, purpose, priority=0, needs_table=False):
            if needs_table and not table:
                if tables:
                    add('navigate ' + tables[0]['id'], 'Approach existing crafting table', priority)
                return
            action = 'craft %s %d' % (item, count)
            if needs_table:
                action += ' ' + table['id']
            add(action, purpose, priority)

        collision = {}
        for cell in obs.get('world', {}).get('local_collision', []):
            if isinstance(cell, list) and len(cell) >= 4:
                collision[tuple(cell[:3])] = cell[3]
        floor = [math.floor(v) for v in p]
        support = []
        for b in blocks:
            bp = b.get('position')
            if not isinstance(bp, list) or len(bp) < 3 or not b.get('in_reach') or not b.get('visible'):
                continue
            dx, dy, dz = (int(number(bp[i])) - floor[i] for i in range(3))
            if dy == -1 and (dx or dz) and name(b) in ('grass_block', 'dirt', 'stone', 'sand', 'cobblestone'):
                if collision.get((dx, 0, dz)) == 'passable' and collision.get((dx, 1, dz)) == 'passable':
                    support.append(b)

        def place(item, purpose, priority):
            if inv.get(item) and support:
                add('place %s %s up' % (item, support[0]['id']), purpose, priority)

        foods = ('bread', 'cooked_beef', 'cooked_porkchop', 'cooked_mutton', 'cooked_chicken',
                 'baked_potato', 'apple', 'carrot', 'sweet_berries', 'beef', 'porkchop', 'mutton')
        food_count = sum(inv.get(k, 0) for k in foods)
        if number(me.get('food'), 20) <= 16:
            food = next((k for k in foods if inv.get(k)), None)
            if food:
                add('eat ' + food, 'Restore hunger before work', 100)

        entities = rows(obs.get('entities', obs.get('nearby_entities', [])))
        enemies = [e for e in entities if e.get('hostile') and e.get('visible') and number(e.get('distance'), 100) < 5]
        if enemies:
            if inv.get('shield'):
                add('shield on', 'Block immediate hostile attack', 99)
            for e in enemies[:1]:
                if number(e.get('distance'), 100) <= 3 and e.get('id') is not None:
                    add('attack ' + str(e['id']), 'Defend against in-range hostile', 98)

        logs = [k for k in inv if k.endswith('_log') or k.endswith('_stem')]
        planks = sum(v for k, v in inv.items() if k.endswith('_planks'))
        wood = sum(inv[k] for k in logs)
        sticks = inv.get('stick', 0)
        pick = any(inv.get(k) for k in ('wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'diamond_pickaxe'))
        stonepick = any(inv.get(k) for k in ('stone_pickaxe', 'iron_pickaxe', 'diamond_pickaxe'))
        cobble = inv.get('cobblestone', 0) + inv.get('cobbled_deepslate', 0)
        if not self.saw_table:
            if planks >= 4:
                craft('crafting_table', 1, 'Earn opening milestone and unlock tools', 95)
            elif logs:
                craft(logs[0].replace('_log', '_planks').replace('_stem', '_planks'), 4,
                      'Convert carried wood into table ingredients', 94)
            else:
                gather(lambda n: n.endswith('_log') or n.endswith('_stem'), 'Collect first logs', 93, 3)
        if self.saw_table and not tables:
            if inv.get('crafting_table'):
                place('crafting_table', 'Place workstation on observed clear ground', 90)
            elif planks >= 4:
                craft('crafting_table', 1, 'Replace missing workstation', 89)
        if wood < 2 and planks < 10:
            gather(lambda n: n.endswith('_log') or n.endswith('_stem'), 'Replenish wood for tools and fuel', 70, 3)
        if logs and planks < 10:
            craft(logs[0].replace('_log', '_planks').replace('_stem', '_planks'), 8, 'Make tool and shield planks', 80)
        if planks >= 2 and sticks < 4:
            craft('stick', 4, 'Prepare tool handles', 79)
        if not pick and planks >= 3 and sticks >= 2:
            craft('wooden_pickaxe', 1, 'Unlock stone mining', 85, True)
        if pick and cobble < (14 if not inv.get('furnace') and not furnaces else 4):
            gather(lambda n: n in ('stone', 'cobblestone', 'deepslate', 'cobbled_deepslate'), 'Collect stone for pick and furnace', 72, 4)
        if not stonepick and cobble >= 3 and sticks >= 2:
            craft('stone_pickaxe', 1, 'Unlock exposed iron ore', 84, True)
        if stonepick and not inv.get('iron_ingot') and inv.get('raw_iron', 0) < 3:
            gather(lambda n: n in ('iron_ore', 'deepslate_iron_ore'), 'Mine exposed iron', 82, 3)
        if cobble >= 8 and not inv.get('furnace') and not furnaces:
            craft('furnace', 1, 'Build real smelting workstation', 81, True)
        if not furnaces:
            place('furnace', 'Place furnace on inspected support', 83)
        if inv.get('raw_iron'):
            for f in furnaces[:1]:
                if now - self.furnaces.get(f['id'], -1000) > 50:
                    fuel = next((k for k in ('coal', 'charcoal') + tuple(k for k in inv if k.endswith('_planks')) if inv.get(k)), None)
                    if fuel:
                        add('furnace_start %s raw_iron %d %s %d' % (f['id'], min(3, inv['raw_iron']), fuel, min(3, inv[fuel])), 'Smelt iron using carried fuel', 87)
        for f in furnaces:
            if f['id'] in self.furnaces and now - self.furnaces[f['id']] >= 12:
                add('furnace_take ' + f['id'], 'Collect completed iron smelting', 88)
        if inv.get('iron_ingot') and planks >= 6 and not inv.get('shield'):
            craft('shield', 1, 'Earn shield milestone and gain defense', 92, True)
        if not any(inv.get(k) for k in ('stone_sword', 'iron_sword')) and cobble >= 2 and sticks:
            craft('stone_sword', 1, 'Prepare hunting and defense weapon', 66, True)
        wool = next((k for k in inv if k.endswith('_wool') and inv[k] >= 3), None)
        if wool and planks >= 3 and not any(k.endswith('_bed') for k in inv):
            craft(wool.replace('_wool', '_bed'), 1, 'Craft bed milestone', 91, True)
        gather_bed = not any(k.endswith('_bed') for k in inv)
        if gather_bed:
            gather(lambda n: n.endswith('_bed'), 'Collect observed bed', 90)
        if inv.get('wheat', 0) >= 3:
            craft('bread', 1, 'Turn wheat into food milestone', 86, True)
        if food_count < 4:
            gather(lambda n: n in ('hay_block', 'melon', 'sweet_berry_bush'), 'Harvest exposed food source', 76)
            if inv.get('hay_block'):
                craft('wheat', 9, 'Unpack hay for bread', 83)
            animals = [e for e in entities if name(e) in ('cow', 'pig', 'sheep', 'chicken') and e.get('visible')]
            animals.sort(key=lambda e: number(e.get('distance'), 100))
            for e in animals[:1]:
                weapon = next((k for k in ('iron_sword', 'stone_sword', 'stone_axe', 'wooden_axe') if inv.get(k)), None)
                if weapon and held != weapon:
                    add('equip ' + weapon, 'Equip hunting weapon', 77)
                elif number(e.get('distance'), 100) <= 3 and e.get('id') is not None:
                    # Attacks may legitimately repeat faster than mining jobs.
                    a = 'attack ' + str(e['id'])
                    if now - self.attempted.get(a, -1000) >= 1:
                        opts.append({'action': a, 'purpose': 'Hunt visible animal for food and possible wool', 'priority': 78})
                elif isinstance(e.get('position'), list) and len(e['position']) >= 3:
                    ep = [number(v) for v in e['position'][:3]]
                    if math.dist(p, ep) < 30:
                        add('goto %.2f %.2f %.2f' % tuple(ep), 'Approach observed food animal by safe path', 65)
        stands = [b for b in candidates if str(b.get('id', '')).startswith('s:') and number(b.get('distance')) > 5]
        stands.sort(key=lambda b: (self.visits.get(b['id'], 0), -min(number(b.get('distance')), 24)))
        for stand in stands[:2]:
            add('navigate ' + stand['id'], 'Explore observed safe stand for missing resources', 10)
        if self.focus == 'food':
            for opt in opts:
                if any(k in opt['purpose'].lower() for k in ('food', 'hunt', 'bread')):
                    opt['priority'] += 10
        elif self.focus == 'iron':
            for opt in opts:
                if 'iron' in opt['purpose'].lower():
                    opt['priority'] += 5
        opts.sort(key=lambda o: o['priority'], reverse=True)
        return opts[:5], inv, enemies

    def run_episode(self, game):
        ctx = self.ctx
        while not ctx.done:
            now = ctx.clock.now()
            obs = ctx.observe()
            if not isinstance(obs, dict):
                self.post('noop', now)
                ctx.wait_for_tick()
                continue
            me = obs.get('self') or {}
            if me.get('dead'):
                self.post('noop' if self.last_action == 'respawn' else 'respawn', now)
                ctx.wait_for_tick()
                continue
            if number(obs.get('observation_age_s')) > 3:
                self.post('cancel' if obs.get('task') and self.last_action != 'cancel' else 'noop', now)
                ctx.wait_for_tick()
                continue
            if obs.get('task') or now - self.last_issue < 1.0:
                self.post('noop', now)
                ctx.wait_for_tick()
                continue
            options, inv, enemies = self.options(obs, now)
            if inv != self.last_inv:
                self.last_inv = dict(inv)
                self.last_progress = now
            if self.planner is not None and self.planner.done():
                try:
                    result = self.planner.result()
                    proposed = str(getattr(result, 'text', '')).strip().lower().strip('` .\n"')
                    if proposed in ('balanced', 'food', 'iron'):
                        self.focus = proposed
                        self.log('astra_focus', focus=proposed)
                except (BudgetExceeded, ModelUnavailable):
                    raise
                except Exception as exc:
                    self.log('astra_fallback', error=type(exc).__name__)
                self.planner = None
            remaining = ctx.budget.remaining()
            if (self.planner is None and now - self.last_progress > 40 and now - self.planned_at > 180
                    and remaining.get('wall_s', 0) > 180 and remaining.get('system2_calls', 0) > 1):
                prompt = ('Choose a Minecraft opening priority from observed facts. Reply ONLY balanced, food, or iron. '
                          'No seed or hidden terrain is available. Code owns recipes and legal actions. '
                          'Prefer food when hunger threatens survival; otherwise iron after wood and table. State: ' +
                          json.dumps({'inventory': inv, 'health': me.get('health'), 'food': me.get('food'),
                                      'available': options, 'focus': self.focus}))
                self.planner = ctx.spawn(lambda p=prompt: ctx.think(p, timeout_s=160.0))
                self.planned_at = now
            chosen = None
            if self.adviser is not None and self.adviser.done():
                try:
                    answers = self.adviser.result()
                    a = answers.get('action') if isinstance(answers, dict) else None
                    hazard = answers.get('unsafe') if isinstance(answers, dict) else None
                    fresh = (now - self.decision_at <= 2.0 and inv == self.decision_inventory
                             and me.get('dimension') == self.decision_dimension)
                    value = getattr(a, 'value', None)
                    probabilities = getattr(a, 'probabilities', {}) or {}
                    if fresh and value in {o['action'] for o in options} and number(probabilities.get(value)) >= .60:
                        chosen = value
                        self.log('jev_selected', action=chosen, probability=probabilities[value])
                    if fresh and hazard is not None and number(getattr(hazard, 'value', 0)) >= .85 and enemies:
                        chosen = 'shield on' if inv.get('shield') else 'noop'
                        self.log('jev_hazard_veto', action=chosen)
                except (BudgetExceeded, ModelUnavailable):
                    raise
                except Exception as exc:
                    self.log('jev_fallback', error=type(exc).__name__)
                self.adviser = None
            if chosen:
                self.post(chosen, now)
                if chosen.startswith('navigate '):
                    ident = chosen.split()[1]
                    self.visits[ident] = self.visits.get(ident, 0) + 1
            elif self.adviser is not None:
                self.post('noop', now)
            elif options:
                # Model absence/uncertainty cannot invalidate a grounded recipe.
                if now - self.last_decision < 3 or remaining.get('system1_calls', 0) <= 2:
                    chosen = options[0]['action']
                    self.post(chosen, now)
                    if chosen.startswith('navigate '):
                        ident = chosen.split()[1]
                        self.visits[ident] = self.visits.get(ident, 0) + 1
                else:
                    state = {'goal': 'Secure opening, food, bed, iron and shield in vanilla survival',
                             'macro_focus': self.focus, 'recommended': options[0]['action'],
                             'inventory': inv, 'health_band': 'low' if number(me.get('health'), 20) < 9 else 'stable',
                             'hunger_band': 'hungry' if number(me.get('food'), 20) < 17 else 'fed',
                             'nearby_threats': [{'name': name(e), 'visible': bool(e.get('visible')),
                                                 'range': 'melee' if number(e.get('distance'), 100) < 3 else 'near'} for e in enemies[:4]],
                             'options': options,
                             'rules': 'All choices are complete commands grounded in current observations. Prefer recommended unless an observed danger makes it inappropriate. No hidden map knowledge.'}
                    questions = {
                        'action': Choice('Select the most useful executable action. Follow recommended unless visible evidence supports a different option.',
                                         {o['action']: o['purpose'] for o in options}),
                        'unsafe': Noul('Does nearby_threats explicitly show an immediate hostile danger that makes resource work unsafe?',
                                       {'true': 'An immediate nearby hostile threatens work.', 'false': 'No immediate hostile evidence; ordinary uncertainty is not danger.'})}
                    self.adviser = ctx.spawn(lambda s=state, q=questions: ctx.decide(s, q, timeout_s=3.0))
                    self.decision_at = now
                    self.last_decision = now
                    self.decision_inventory = dict(inv)
                    self.decision_dimension = me.get('dimension')
                    self.post('noop', now)
            else:
                self.post('noop', now)
            ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
