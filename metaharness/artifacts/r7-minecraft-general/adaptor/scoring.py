"""Absolute benchmark score from atomic server Inventory NBT snapshots."""
from __future__ import annotations
LOGS = {s + '_log' for s in ('oak','spruce','birch','jungle','acacia','dark_oak','mangrove','cherry')}
FOODS = {'cooked_beef','cooked_porkchop','cooked_mutton','cooked_chicken','cooked_rabbit','cooked_cod','cooked_salmon','baked_potato'}
WEIGHTS = {'crafting_table':5,'wooden_pickaxe':5,'stone_pickaxe':10,'stone_axe':5,'furnace':5,'torch':5,'cooked_food':5,'iron_ingot':5,'shield':5}
class Score:
    def __init__(self):
        self.maxima = dict(logs=0, cobblestone=0, coal_charcoal=0, raw_iron=0)
        self.milestones = {}
        self.inventory = {}
        self.samples = 0
    def update(self, counts, elapsed):
        self.inventory = dict(counts)
        self.samples += 1
        values = {'logs':sum(counts.get(n,0) for n in LOGS), 'cobblestone':counts.get('cobblestone',0), 'coal_charcoal':counts.get('coal',0)+counts.get('charcoal',0), 'raw_iron':counts.get('raw_iron',0)}
        for key, val in values.items(): self.maxima[key] = max(self.maxima[key],val)
        for name in WEIGHTS:
            present = any(counts.get(n,0)>0 for n in FOODS) if name=='cooked_food' else counts.get(name,0)>0
            if present and name not in self.milestones: self.milestones[name] = round(elapsed,4)
    def report(self, elapsed, duration, health, terminal):
        m=self.maxima
        resources=min(m['logs'],10)+.5*min(m['cobblestone'],20)+min(m['coal_charcoal'],5)+min(m['raw_iron'],5)
        equipment=sum(WEIGHTS[k] for k in self.milestones)
        survival=10*min(max(elapsed,0),duration)/duration
        final_health=10*min(max(health,0),20)/20 if terminal=='time_limit' else 0
        death=-20 if terminal=='death' else 0
        components=dict(resources=resources,equipment=equipment,survival=survival,final_health=final_health,death=death)
        return {'score':float(max(0,min(100,sum(components.values())))), 'components':components,'progress_points':resources+equipment,'resource_maxima':dict(m),'milestone_times':dict(self.milestones),'verified_samples':self.samples}

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
