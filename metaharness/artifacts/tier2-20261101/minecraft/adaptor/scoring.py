"""Dragon-survival progress from server inventory audits and client-visible server facts."""
from __future__ import annotations

LOGS = {s + '_log' for s in ('oak', 'spruce', 'birch', 'jungle', 'acacia', 'dark_oak', 'mangrove', 'cherry')}
FOODS = {'bread', 'apple', 'baked_potato', 'carrot', 'golden_carrot', 'potato', 'beef', 'porkchop', 'mutton', 'chicken', 'rabbit', 'cod', 'salmon', 'sweet_berries', 'melon_slice', 'cooked_beef', 'cooked_porkchop', 'cooked_mutton', 'cooked_chicken', 'cooked_rabbit', 'cooked_cod', 'cooked_salmon'}
WEIGHTS = {
    'opening': {'wood_and_workbench': 3, 'food': 3, 'bed': 4},
    'iron_shield': {'iron': 5, 'shield': 5},
    'nether_fortress': {'nether_entry': 8, 'fortress': 6, 'blaze_rod': 6},
    'eyes_stronghold_portal': {'pearl': 4, 'eye': 6, 'stronghold': 5, 'active_end_portal': 5},
    'end_entry': {'end_entry': 10},
    'dragon_kill': {'dragon_kill': 30},
}
ADVANCEMENTS = {
    'minecraft:adventure/sleep_in_bed': 'bed',
    'minecraft:story/smelt_iron': 'iron',
    'minecraft:story/enter_the_nether': 'nether_entry',
    'minecraft:nether/find_fortress': 'fortress',
    'minecraft:nether/obtain_blaze_rod': 'blaze_rod',
    'minecraft:story/follow_ender_eye': 'stronghold',
    'minecraft:story/enter_the_end': 'end_entry',
    'minecraft:end/kill_dragon': 'dragon_kill',
}

class Score:
    def __init__(self):
        self.inventory = {}
        self.maxima = {}
        self.milestones = {}
        self.advancements = {}
        self.samples = 0
        self.deaths = 0
        self.damage_taken = 0.0
        self.inventory_dropped_on_death = 0
        self.death_records = []
        self.last_health = None
        self.crystal_deaths_observed = 0
        self.dragon_health_min = None
        self.nether_portal_observed = False
        self.stronghold_portal_frame_observed = False
        self.saw_logs = False
        self.saw_table = False

    def flag(self, name, elapsed):
        if name not in self.milestones:
            self.milestones[name] = round(float(elapsed), 3)

    def update(self, counts, elapsed):
        self.inventory = {n: int(c) for n, c in counts.items() if c > 0}
        self.samples += 1
        for n, c in self.inventory.items():
            self.maxima[n] = max(self.maxima.get(n, 0), c)
        self.saw_logs |= any(self.inventory.get(n, 0) for n in LOGS)
        self.saw_table |= bool(self.inventory.get('crafting_table'))
        if self.saw_logs and self.saw_table:
            self.flag('wood_and_workbench', elapsed)
        if any(self.inventory.get(n, 0) for n in FOODS):
            self.flag('food', elapsed)
        if any(n.endswith('_bed') for n in self.inventory):
            self.flag('bed', elapsed)
        for item, flag in [('iron_ingot', 'iron'), ('shield', 'shield'), ('blaze_rod', 'blaze_rod'), ('ender_pearl', 'pearl'), ('ender_eye', 'eye')]:
            if self.inventory.get(item):
                self.flag(flag, elapsed)

    def observe(self, snapshot, elapsed):
        player = snapshot.get('self', {})
        health = player.get('health')
        if isinstance(health, (int, float)):
            if self.last_health is not None and health < self.last_health:
                self.damage_taken += self.last_health - health
            self.last_health = health
        for ident in snapshot.get('advancements', []):
            if not ident.startswith('minecraft:recipes/'):
                self.advancements.setdefault(ident, round(elapsed, 3))
            flag = ADVANCEMENTS.get(ident)
            if flag:
                self.flag(flag, elapsed)
        dimension = player.get('dimension')
        if dimension in ('nether', 'the_nether', 'minecraft:the_nether'):
            self.flag('nether_entry', elapsed)
        if dimension in ('end', 'the_end', 'minecraft:the_end'):
            self.flag('end_entry', elapsed)
        for b in snapshot.get('candidates', []):
            if b.get('name') == 'nether_portal' and b.get('visible'):
                self.nether_portal_observed = True
            if b.get('name') == 'end_portal_frame' and b.get('visible'):
                self.stronghold_portal_frame_observed = True
            if b.get('name') == 'end_portal' and b.get('visible') and b.get('distance', 100) <= 8:
                self.flag('active_end_portal', elapsed)
        combat = snapshot.get('combat', {})
        self.crystal_deaths_observed = max(self.crystal_deaths_observed, int(combat.get('crystal_deaths_observed', 0)))
        hp = combat.get('dragon_health_fraction')
        if isinstance(hp, (int, float)) and 0 <= hp <= 1:
            self.dragon_health_min = hp if self.dragon_health_min is None else min(self.dragon_health_min, hp)

    def death(self, snapshot, elapsed):
        self.observe(snapshot, elapsed)
        self.deaths += 1
        dropped = sum(self.inventory.values())
        self.inventory_dropped_on_death += dropped
        self.death_records.append({'elapsed_s': round(elapsed, 3), 'dimension': snapshot.get('self', {}).get('dimension'), 'position': snapshot.get('self', {}).get('position'), 'items_carried_before_death': dict(self.inventory), 'item_count_dropped_estimate': dropped})
        self.inventory = {}
        self.last_health = None

    @property
    def success(self):
        return 'dragon_kill' in self.milestones

    def report(self, elapsed=0, duration=3600, health=20, terminal=None):
        components = {group: float(sum(w for flag, w in weights.items() if flag in self.milestones)) for group, weights in WEIGHTS.items()}
        if self.success:
            components['completion_bonus'] = 100.0 - sum(components.values())
        order = ['wood_and_workbench', 'bed', 'iron', 'shield', 'nether_entry', 'fortress', 'blaze_rod', 'pearl', 'eye', 'stronghold', 'active_end_portal', 'end_entry', 'dragon_kill']
        return {
            'score': float(sum(components.values())), 'components': components,
            'success': self.success,
            'success_evidence': 'minecraft:end/kill_dragon (server advancement)' if self.success else None,
            'milestones': {n: n in self.milestones for weights in WEIGHTS.values() for n in weights},
            'milestone_times_s': dict(self.milestones),
            'furthest_milestone_rank': max([i + 1 for i, n in enumerate(order) if n in self.milestones], default=0),
            'advancements': dict(self.advancements), 'resource_maxima': dict(self.maxima),
            'current_resources': dict(self.inventory), 'verified_inventory_samples': self.samples,
            'deaths': self.deaths, 'damage_taken': round(self.damage_taken, 3),
            'inventory_dropped_on_death_estimate': self.inventory_dropped_on_death,
            'crystal_deaths_observed': self.crystal_deaths_observed,
            'dragon_health_min_fraction_observed': self.dragon_health_min,
            'nether_portal_observed': self.nether_portal_observed,
            'stronghold_portal_frame_observed': self.stronghold_portal_frame_observed,
            'elapsed_s': round(elapsed, 3), 'terminal': terminal,
        }

def parse_inventory(text):
    """Parse just top-level item compounds of vanilla Inventory SNBT.

    Nested component compounds/strings are skipped, never counted as carried items.
    Minecraft 1.21.1 writes lowercase count and id; Slot is a signed byte.
    """
    import re
    compounds=[]; depth=0; quote=None; escape=False; start=None
    for i,c in enumerate(text):
        if quote:
            if escape: escape=False
            elif c=='\\': escape=True
            elif c==quote: quote=None
            continue
        if c in ('"',"'"): quote=c; continue
        if c=='{':
            if depth==0: start=i
            depth+=1
        elif c=='}':
            depth-=1
            if depth==0 and start is not None: compounds.append(text[start:i+1])
    counts={}
    for compound in compounds:
        # Strip nested compounds/lists so item component fields cannot impersonate id/count.
        out=[]; nesting=0; quote=None; escape=False
        for c in compound[1:-1]:
            if quote:
                if nesting==0: out.append(c)
                if escape: escape=False
                elif c=='\\': escape=True
                elif c==quote: quote=None
            elif c in ('"',"'"):
                quote=c
                if nesting==0: out.append(c)
            elif c in '{[': nesting+=1
            elif c in '}]': nesting-=1
            elif nesting==0: out.append(c)
        flat=''.join(out)
        ident=re.search(r'(?:^|,)\s*id:\s*"minecraft:([a-z0-9_]+)"',flat)
        count=re.search(r'(?:^|,)\s*count:\s*(\d+)',flat)
        slot=re.search(r'(?:^|,)\s*Slot:\s*(-?\d+)b',flat)
        if not (ident and count and slot): continue
        s=int(slot[1])
        if 0<=s<=35 or 100<=s<=103 or s==-106:
            counts[ident[1]]=counts.get(ident[1],0)+int(count[1])
    return counts
