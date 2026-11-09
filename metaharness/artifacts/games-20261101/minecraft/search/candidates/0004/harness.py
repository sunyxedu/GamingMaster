"""Compiled milestone options, asynchronous tactical judging, and macro advice."""
from __future__ import annotations

import json
import math
from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


def number(value, default=0.0):
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def name(value):
    if isinstance(value, dict):
        value = value.get('name', value.get('id', ''))
    return str(value or '').removeprefix('minecraft:')


def position(obj):
    p = obj.get('position', []) if isinstance(obj, dict) else []
    if isinstance(p, dict):
        p = [p.get('x'), p.get('y'), p.get('z')]
    return [number(x) for x in p] if isinstance(p, (list, tuple)) and len(p) == 3 else None


def inventory(obs):
    raw = obs.get('inventory', obs.get('inventory_authoritative', {}))
    if isinstance(raw, dict) and isinstance(raw.get('items'), (dict, list)):
        raw = raw['items']
    result = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            count = v.get('count', 0) if isinstance(v, dict) else v
            if number(count) > 0:
                result[name(k)] = int(number(count))
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                k = name(item)
                result[k] = result.get(k, 0) + int(number(item.get('count')))
    # Client inventory is usable for action feasibility, never claimed as audited score.
    if not result:
        raw = obs.get('inventory_client', {})
        if isinstance(raw, dict):
            result = {name(k): int(number(v)) for k, v in raw.items() if number(v) > 0}
    return result


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.last = 'noop'
        self.cooldown = {}
        self.visits = {}
        self.last_attack = -100.0
        self.s1 = None
        self.s2 = None
        self.next_plan = 0.0
        self.plan = 'balanced'
        self.plan_version = 0
        self.session = None
        self.furnaces = {}
        self.last_feedback = ''
        self.history = []
        self.ever_table = False
        self.ever_bed = False
        self.failures = 0

    def now(self):
        return self.ctx.clock.now()

    def post(self, action):
        # A noop between repeated one-shots clears the adapter's action latch.
        if action == self.last and action.split()[0] not in ('noop', 'controls', 'attack', 'shield'):
            action = 'noop'
        self.ctx.act(action)
        self.last = action
        if action.startswith('attack '):
            self.last_attack = self.now()
        if action not in ('noop', 'cancel'):
            self.history.append(action)
            self.history = self.history[-10:]
        return action

    def facts(self, obs):
        me = obs.get('self') or {}
        p = position(me) or [0, 64, 0]
        candidates = [b for b in obs.get('candidates', []) if isinstance(b, dict) and b.get('id') and position(b)]
        candidates.sort(key=lambda b: number(b.get('distance'), 999))
        entities = [e for e in obs.get('entities', obs.get('nearby_entities', [])) if isinstance(e, dict)]
        for e in entities:
            if e.get('distance') is None and position(e):
                e = dict(e)
                e['distance'] = math.dist(p, position(e))
        inv = inventory(obs)
        self.ever_table |= inv.get('crafting_table', 0) > 0
        self.ever_bed |= any(k.endswith('_bed') and v for k, v in inv.items())
        return me, p, candidates, entities, inv

    def make_options(self, obs):
        me, p, blocks, entities, inv = self.facts(obs)
        now = self.now()
        options = []
        def add(action, why):
            if self.cooldown.get(action, 0) <= now and action not in [a for a, _ in options]:
                options.append((action, why))
        def block(names):
            return next((b for b in blocks if name(b.get('name')) in names and
                         self.cooldown.get('gather '+b['id']+' 1', 0) <= now), None)
        def gather(b, why, count=1):
            if b:
                add('gather %s %d' % (b['id'], count), why)
        def craft(item, count=1, table=None):
            add('craft %s %d%s' % (item, count, ' '+table['id'] if table else ''), 'Craft '+item+' toward verified material requirements')
        def explore():
            stands = [b for b in blocks if str(b['id']).startswith('s:') and number(b.get('distance')) > 3]
            if stands:
                stands.sort(key=lambda b: (self.visits.get(b['id'], 0), -min(number(b.get('distance')), 20)))
                for b in stands[:2]:
                    add('navigate '+b['id'], 'Explore a listed safe stand not recently visited')
        foods = ['cooked_beef', 'cooked_porkchop', 'cooked_mutton', 'cooked_chicken', 'bread', 'baked_potato', 'apple', 'carrot', 'beef', 'porkchop', 'mutton', 'sweet_berries']
        if number(me.get('food'), 20) <= 16:
            edible = next((k for k in foods if inv.get(k)), None)
            if edible:
                add('eat '+edible, 'Restore hunger before further work')
                return options, 'eat'
        tables = [b for b in blocks if name(b.get('name')) == 'crafting_table']
        table = tables[0] if tables else None
        planks = sum(v for k, v in inv.items() if k.endswith('_planks'))
        logs = [(k, v) for k, v in inv.items() if k.endswith('_log') or k.endswith('_stem')]
        has_pick = any(inv.get(k) for k in ('wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'diamond_pickaxe'))
        ground = [b for b in blocks if b.get('in_reach') and b.get('visible') and
                  name(b.get('name')) in ('grass_block', 'dirt', 'stone', 'sand', 'cobblestone') and
                  abs(position(b)[1] - (math.floor(p[1])-1)) < .1 and
                  math.hypot(position(b)[0]+.5-p[0], position(b)[2]+.5-p[2]) > 1.2]
        def place(item):
            if ground:
                add('place %s %s up' % (item, ground[0]['id']), 'Place a workstation beside the player, not in the player column')
        def wood():
            choices = [b for b in blocks if name(b.get('name')).endswith(('_log', '_stem'))]
            choices.sort(key=lambda b: (not bool(b.get('visible')), number(b.get('distance'), 999)))
            for b in choices[:3]:
                gather(b, 'Collect exposed wood for the opening dependency chain')
            if not options:
                explore()
        if not self.ever_table and not inv.get('crafting_table'):
            if planks >= 4:
                craft('crafting_table')
            elif logs:
                craft(logs[0][0].replace('_log', '_planks').replace('_stem', '_planks'), 4)
            else:
                wood()
            return options, 'logs_and_table'
        if not table:
            if inv.get('crafting_table'):
                place('crafting_table')
            elif planks >= 4:
                craft('crafting_table')
            elif logs:
                craft(logs[0][0].replace('_log', '_planks').replace('_stem', '_planks'), 8)
            else:
                wood()
            if not options:
                explore()
            return options, 'place_table'
        if number(table.get('distance'), 999) > 4:
            add('navigate '+table['id'], 'Approach known crafting workstation')
            return options, 'approach_table'
        if not has_pick:
            if planks < 5 and logs:
                craft(logs[0][0].replace('_log', '_planks').replace('_stem', '_planks'), 8)
            elif planks < 5 and inv.get('stick', 0) < 2:
                wood()
            elif inv.get('stick', 0) < 2:
                craft('stick', 4)
            elif planks >= 3:
                craft('wooden_pickaxe', 1, table)
            else:
                wood()
            return options, 'wooden_pickaxe'
        # Cheap opening milestones are persistent even if later exploration is fatal.
        for color in ('white', 'light_gray', 'gray', 'black', 'brown', 'pink', 'red', 'blue', 'yellow', 'green'):
            if not self.ever_bed and inv.get(color+'_wool', 0) >= 3 and planks >= 3:
                craft(color+'_bed', 1, table)
        if inv.get('hay_block'):
            craft('wheat', 9)
        if inv.get('wheat', 0) >= 3:
            craft('bread', 1, table)
        if options:
            return options, 'opening_crafts'
        if not inv.get('stone_pickaxe') and not inv.get('iron_pickaxe'):
            if inv.get('cobblestone', 0) < 3:
                gather(block({'stone', 'cobblestone'}), 'Mine exposed stone with a verified pickaxe')
            elif inv.get('stick', 0) >= 2:
                craft('stone_pickaxe', 1, table)
            elif planks >= 2:
                craft('stick', 4)
            elif logs:
                craft(logs[0][0].replace('_log', '_planks').replace('_stem', '_planks'), 4)
            else:
                wood()
            if options:
                return options, 'stone_pickaxe'
        if inv.get('iron_ingot', 0) and not inv.get('shield'):
            if planks >= 6:
                craft('shield', 1, table)
            elif logs:
                craft(logs[0][0].replace('_log', '_planks').replace('_stem', '_planks'), 8)
            else:
                wood()
            return options, 'shield'
        furnace = block({'furnace'})
        if inv.get('raw_iron'):
            if not furnace:
                if inv.get('furnace'):
                    place('furnace')
                elif inv.get('cobblestone', 0) >= 8:
                    craft('furnace', 1, table)
                else:
                    gather(block({'stone', 'cobblestone'}), 'Collect stone for an iron-smelting furnace')
            elif number(furnace.get('distance'), 99) > 4:
                add('navigate '+furnace['id'], 'Approach the furnace')
            elif now >= self.furnaces.get(furnace['id'], 0):
                fuel = next((k for k in ('coal', 'charcoal') if inv.get(k)), None)
                fuel = fuel or next((k for k in inv if k.endswith('_planks')), None)
                if fuel:
                    add('furnace_start %s raw_iron %d %s %d' % (furnace['id'], min(inv['raw_iron'], 3), fuel, min(inv[fuel], 3)), 'Load actual iron and fuel; output requires elapsed server time')
                else:
                    wood()
            if options:
                return options, 'smelt_iron'
        for fid, ready in list(self.furnaces.items()):
            if now >= ready and any(b['id'] == fid for b in blocks):
                add('furnace_take '+fid, 'Retrieve iron after the real smelting interval')
        if options:
            return options, 'retrieve_iron'
        need_food = not any(inv.get(k) for k in foods)
        for b in blocks:
            bn = name(b.get('name'))
            if not self.ever_bed and bn.endswith('_bed'):
                gather(b, 'Collect an existing bed for respawn preparation')
                break
        if need_food:
            gather(block({'hay_block'}), 'Collect hay for bread')
        animals = [e for e in entities if name(e.get('name', e.get('type'))) in ('sheep', 'cow', 'pig', 'chicken') and e.get('visible')]
        animals.sort(key=lambda e: (name(e.get('name', e.get('type'))) != 'sheep', number(e.get('distance'), 999)))
        if animals and (need_food or not self.ever_bed):
            e = animals[0]
            if number(e.get('distance'), 999) <= 3 and e.get('id') is not None:
                weapon = next((k for k in ('iron_sword', 'stone_axe', 'stone_sword', 'wooden_axe') if inv.get(k)), None)
                if weapon and name(me.get('held_item')) != weapon:
                    add('equip '+weapon, 'Equip the carried hunting tool')
                elif now-self.last_attack >= 1:
                    add('attack '+str(e['id']), 'Hunt a visible passive animal for food and possibly wool')
            elif position(e) and number(e.get('distance'), 999) <= 30:
                ep = position(e)
                add('goto %.2f %.2f %.2f' % tuple(ep), 'Approach a visible passive animal using bounded safe pathfinding')
        if inv.get('stone_pickaxe') or inv.get('iron_pickaxe'):
            gather(block({'iron_ore', 'deepslate_iron_ore'}), 'Mine exposed iron with a suitable carried pickaxe')
        if not inv.get('coal'):
            gather(block({'coal_ore'}), 'Mine exposed fuel with a carried pickaxe')
        # Pick up observed drops using ordinary movement, not inventory fabrication.
        for e in entities:
            if name(e.get('name', e.get('type'))) in ('item', 'item_stack') and position(e) and number(e.get('distance'), 999) < 6:
                add('goto %.2f %.2f %.2f' % tuple(position(e)), 'Walk to a nearby dropped item')
                break
        if not options:
            explore()
        if not options:
            wood()
        if self.plan == 'iron':
            options.sort(key=lambda x: 'iron' not in x[1].lower())
        elif self.plan == 'food':
            options.sort(key=lambda x: not any(w in x[1].lower() for w in ('food', 'bread', 'hay', 'hunt')))
        elif self.plan == 'bed':
            options.sort(key=lambda x: not any(w in x[1].lower() for w in ('bed', 'wool')))
        return options[:6], 'local_resources'

    def emergency(self, obs):
        me, p, blocks, entities, inv = self.facts(obs)
        if me.get('dead'):
            return 'respawn'
        if number(obs.get('observation_age_s')) > 3:
            return 'cancel' if obs.get('task') else 'noop'
        if me.get('is_in_water') and number(me.get('oxygen'), 300) < 100:
            return 'cancel' if obs.get('task') else 'controls jump'
        threats = [e for e in entities if e.get('hostile') and e.get('visible') and number(e.get('distance'), 999) < 5]
        if not threats and not me.get('is_in_lava'):
            return None
        if obs.get('task'):
            return 'cancel'
        if me.get('is_in_lava'):
            return 'controls forward,jump'
        stands = [b for b in blocks if str(b['id']).startswith('s:') and 3 < number(b.get('distance')) < 15]
        ep = position(threats[0]) if threats else None
        if stands and ep:
            stands.sort(key=lambda b: math.dist(position(b), ep), reverse=True)
            if math.dist(position(stands[0]), ep) > math.dist(p, ep)+2:
                return 'navigate '+stands[0]['id']
        if inv.get('shield'):
            return 'shield on'
        e = threats[0]
        if e.get('id') is not None and number(e.get('distance'), 999) < 3 and self.now()-self.last_attack >= 1:
            return 'attack '+str(e['id'])
        return 'noop'

    def adviser(self, obs, stage, options):
        now = self.now()
        if self.s2 is not None and self.s2.done():
            try:
                response = self.s2.result()
                text = str(response.text)
                start, end = text.find('{'), text.rfind('}')
                parsed = json.loads(text[start:end+1]) if start >= 0 else {}
                mode = parsed.get('priority')
                if mode in ('balanced', 'food', 'bed', 'iron'):
                    self.plan = mode
                    self.plan_version += 1
                    self.ctx.log('macro_priority', priority=mode, version=self.plan_version)
                self.session = getattr(response, 'session_id', None)
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, ValueError, TypeError, AttributeError):
                pass
            self.s2 = None
        if self.s2 is None and now >= self.next_plan and self.ctx.budget.remaining().get('wall_s', 0) > 180 and self.ctx.budget.remaining().get('system2_calls', 0) > 1:
            data = {'stage': stage, 'inventory': inventory(obs), 'self': obs.get('self'),
                    'options': options, 'recent_actions': self.history, 'failures': self.failures,
                    'remaining_wall_s': self.ctx.budget.remaining().get('wall_s')}
            prompt = ('Minecraft survival macro advisor. Code already compiles fully parameterized actions and preserves bounded jobs. '
                      'Do not emit commands or coordinates. Optimize verified milestone progress: logs+table 3, food 3, bed 4, iron 5, shield 5; '
                      'deadline is the supplied remaining wall time. Choose a resource priority appropriate to observed opportunities. '
                      'Reply ONLY JSON {"priority":"balanced|food|bed|iron"}. Your choice reorders feasible local resource actions, '
                      'never overrides prerequisite crafting or immediate survival. State: '+json.dumps(data, default=str))
            session = self.session
            self.s2 = self.ctx.spawn(lambda: self.ctx.think(prompt, session_id=session, timeout_s=160))
            self.next_plan = now+200

    def run_episode(self, game):
        ctx = self.ctx
        while not ctx.done:
            obs = ctx.observe()
            if not isinstance(obs, dict):
                self.post('noop')
                ctx.wait_for_tick()
                continue
            now = self.now()
            feedback = obs.get('action_result', obs.get('feedback'))
            marker = json.dumps(feedback, sort_keys=True, default=str)
            if marker != self.last_feedback:
                self.last_feedback = marker
                if feedback and any(w in marker.lower() for w in ('failed', 'error', 'unreachable', 'timeout', 'cannot')):
                    self.failures += 1
                    if self.history:
                        self.cooldown[self.history[-1]] = now+20
            urgent = self.emergency(obs)
            if urgent:
                self.post(urgent)
                ctx.wait_for_tick()
                continue
            if obs.get('task'):
                self.post('noop')
                ctx.wait_for_tick()
                continue
            options, stage = self.make_options(obs)
            self.adviser(obs, stage, options)
            if not options:
                self.post('noop')
                ctx.wait_for_tick()
                continue
            chosen = None
            if self.s1 is not None:
                future, submitted, seq, version, mapping, episode = self.s1
                if future.done():
                    try:
                        answers = future.result()
                        answer = answers.get('action')
                        safe = answers.get('safe')
                        value = getattr(answer, 'value', None)
                        probabilities = getattr(answer, 'probabilities', {}) or {}
                        selected = mapping.get(value)
                        fresh = now-submitted <= 2 and number(obs.get('seq'))-seq <= 10 and episode == obs.get('episode') and version == self.plan_version
                        if fresh and selected in [x[0] for x in options] and number(probabilities.get(value)) >= .6 and number(getattr(safe, 'value', 0)) >= .65:
                            chosen = selected
                            ctx.log('jev_selected', action=chosen, default=options[0][0], changed=chosen != options[0][0])
                    except (BudgetExceeded, ModelUnavailable):
                        raise
                    except (TimeoutError, RuntimeError, TypeError, AttributeError):
                        pass
                    self.s1 = None
                    if chosen is None:
                        chosen = options[0][0]
                elif now-submitted < 1:
                    self.post('noop')
                    ctx.wait_for_tick()
                    continue
                else:
                    chosen = options[0][0]
            if chosen is None and self.s1 is None and ctx.budget.remaining().get('system1_calls', 0) > 2:
                mapping = {'option_'+str(i): a for i, (a, _) in enumerate(options)}
                state = {'stage': stage, 'macro_priority': self.plan,
                         'health_band': 'low' if number((obs.get('self') or {}).get('health'), 20) < 10 else 'healthy',
                         'threats': [{'kind': name(e.get('name', e.get('type'))), 'close': number(e.get('distance'), 999) < 8, 'visible': bool(e.get('visible'))} for e in obs.get('entities', []) if isinstance(e, dict) and e.get('hostile')][:5],
                         'options': {k: {'action': mapping[k], 'purpose': options[i][1]} for i, k in enumerate(mapping)},
                         'rules': 'Options are compiled from present candidates and materials. Choose progress; do not invent parameters. Safe pathfinding cannot bridge or mine.'}
                criteria = {k: options[i][1] for i, k in enumerate(mapping)}
                criteria['unknown'] = 'No supported option or insufficient information'
                questions = {'action': Choice('Which supplied option best advances stage and macro_priority without approaching an observed threat? Use unknown if none is supported.', criteria),
                             'safe': Noul('Does the supplied state support ordinary resource work without an observed close visible hostile threat?', {'true': 'No close visible hostile threat is reported.', 'false': 'A close visible hostile threat is reported or safety is uncertain.'})}
                self.s1 = (ctx.spawn(lambda s=state, q=questions: ctx.decide(s, q, timeout_s=5.0)), now, number(obs.get('seq')), self.plan_version, mapping, obs.get('episode'))
                self.post('noop')
                ctx.wait_for_tick()
                continue
            chosen = chosen or options[0][0]
            posted = self.post(chosen)
            if posted.startswith('navigate '):
                key = posted.split()[1]
                self.visits[key] = self.visits.get(key, 0)+1
                self.cooldown[posted] = now+12
            elif posted.startswith('furnace_start '):
                self.furnaces[posted.split()[1]] = now+35
            elif posted.startswith('furnace_take '):
                self.furnaces.pop(posted.split()[1], None)
                self.cooldown[posted] = now+5
            ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
