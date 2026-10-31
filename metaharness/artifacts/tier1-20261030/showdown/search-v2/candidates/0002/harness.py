"""Astra compiles guarded matchup policies; code executes against fresh observations.

Only public observation data is used. Damage estimates are explicitly approximate:
they guide a legal fallback, not a simulator or a claim about hidden sets.
"""
from __future__ import annotations
import json
import math
import re
from collections import Counter


def ident(x):
    return re.sub(r'[^a-z0-9]', '', str(x).lower())


def species(p):
    return str(p.get('details', p.get('ident', ''))).split(',')[0].split(':')[-1].strip()


def health(p):
    s = str(p.get('condition', '100/100'))
    if 'fnt' in s:
        return 0., 1.
    m = re.match(r'(\d+)(?:/(\d+))?', s)
    if not m:
        return 1., 1.
    return float(m[1]), float(m[2] or 100)


def fraction(p):
    h, m = health(p)
    return h / max(1., m)


def status(p):
    return p.get('status') or next((x for x in str(p.get('condition', '')).split() if x in ('brn','par','slp','frz','psn','tox')), '')


def stage(n):
    n = max(-6, min(6, float(n or 0)))
    return (2+n)/2 if n >= 0 else 2/(2-n)


# Attack type -> super effective / resisted / immune. Gen 9 ordinary type chart.
_CHART = {
 'Normal': ('', 'Rock Steel', 'Ghost'),
 'Fire': ('Grass Ice Bug Steel','Fire Water Rock Dragon',''),
 'Water': ('Fire Ground Rock','Water Grass Dragon',''),
 'Electric': ('Water Flying','Electric Grass Dragon','Ground'),
 'Grass': ('Water Ground Rock','Fire Grass Poison Flying Bug Dragon Steel',''),
 'Ice': ('Grass Ground Flying Dragon','Fire Water Ice Steel',''),
 'Fighting': ('Normal Ice Rock Dark Steel','Poison Flying Psychic Bug Fairy','Ghost'),
 'Poison': ('Grass Fairy','Poison Ground Rock Ghost','Steel'),
 'Ground': ('Fire Electric Poison Rock Steel','Grass Bug','Flying'),
 'Flying': ('Grass Fighting Bug','Electric Rock Steel',''),
 'Psychic': ('Fighting Poison','Psychic Steel','Dark'),
 'Bug': ('Grass Psychic Dark','Fire Fighting Poison Flying Ghost Steel Fairy',''),
 'Rock': ('Fire Ice Flying Bug','Fighting Ground Steel',''),
 'Ghost': ('Psychic Ghost','Dark','Normal'),
 'Dragon': ('Dragon','Steel','Fairy'),
 'Dark': ('Psychic Ghost','Fighting Dark Fairy',''),
 'Steel': ('Ice Rock Fairy','Fire Water Electric Steel',''),
 'Fairy': ('Fighting Dragon Dark','Fire Poison Steel',''),
}


def effectiveness(typ, types):
    strong, weak, immune = [x.split() for x in _CHART.get(typ, ('','',''))]
    mult = 1.
    for t in types:
        mult *= 0 if t in immune else 2 if t in strong else .5 if t in weak else 1
    return mult


def unpack(text):
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for m in re.finditer(r'\{', text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
            if isinstance(obj, dict) and ('rules' in obj or 'action' in obj):
                return obj
        except (ValueError, TypeError):
            pass
    return {}


class Position:
    def __init__(self, obs):
        self.obs = obs
        self.req = obs.get('request') or {}
        self.team = self.req.get('side', {}).get('pokemon', [])
        self.pub = obs.get('public_summary') or {}
        self.pubs = self.pub.get('pokemon', {})
        self.me = next((p for p in self.team if p.get('active')), self.team[0] if self.team else {})
        enemy_id = (obs.get('active') or {}).get('p2', '')
        self.foe = self.pubs.get(enemy_id) or next((p for p in obs.get('opponent_revealed', []) if p.get('ident') == enemy_id), {})
        self.moves = ((self.req.get('active') or [{}])[0] or {}).get('moves', [])
        self.legal = [a for a in obs.get('legal_actions', []) if isinstance(a, str) and a != 'noop']
        self.dex = obs.get('move_dex') or {}
        self.sdex = obs.get('species_dex') or {}

    def public(self, p):
        return self.pubs.get(p.get('ident'), p)

    def boosts(self, p):
        # Switching clears stages; public bench caches can retain old values.
        if p in self.team and not p.get('active'):
            return {}
        return self.public(p).get('boosts') or {}

    def data(self, p):
        name = species(p)
        return self.sdex.get(name) or next((v for k,v in self.sdex.items() if ident(k) == ident(name)), {})

    def types(self, p, tera=False):
        t = p.get('terastallized') or self.public(p).get('teraTypeRevealed') or self.public(p).get('tera_type') or self.public(p).get('terastallized')
        if tera:
            t = p.get('teraType') or ((self.req.get('active') or [{}])[0] or {}).get('canTerastallize')
        if isinstance(t, str) and t and t != 'Stellar':
            return [t]
        return self.data(p).get('types') or []

    def ability(self, p):
        return ident(p.get('ability') or self.public(p).get('ability') or p.get('baseAbility', ''))

    def item(self, p):
        return ident(p.get('item') or self.public(p).get('item', ''))

    def stat(self, p, key, ignore_boost=False):
        val = (p.get('stats') or {}).get(key)
        if val is None:
            base = self.data(p).get('base_stats', {}).get(key, 90)
            lv = self.level(p)
            # Neutral nature, 31 IV, 85 EV is only a range-center prior.
            val = math.floor((2*base+31+21)*lv/100)+5
        if not ignore_boost:
            val *= stage(self.boosts(p).get(key, 0))
        ability, item = self.ability(p), self.item(p)
        if key == 'atk':
            if ability in ('hugepower','purepower'): val *= 2
            if item == 'choiceband': val *= 1.5
        if key == 'spa' and item == 'choicespecs': val *= 1.5
        if key == 'spd' and item == 'assaultvest': val *= 1.5
        if key == 'spe':
            if item == 'choicescarf': val *= 1.5
            if status(p) == 'par' and ability != 'quickfeet': val *= .5
            side = 'p1' if p in self.team else 'p2'
            conds = self.pub.get('side_conditions', {}).get(side, {})
            if 'tailwind' in {ident(x) for x in conds}: val *= 2
        return max(1., val)

    def level(self, p):
        m = re.search(r'\bL(\d+)', p.get('details', ''))
        return int(m[1]) if m else 100

    def maxhp(self, p):
        if p in self.team: return health(p)[1]
        base = self.data(p).get('base_stats', {}).get('hp', 90)
        lv = self.level(p)
        return math.floor((2*base+52)*lv/100)+lv+10

    def move(self, name):
        mid = ident(name)
        return self.dex.get(mid) or next((v for k,v in self.dex.items() if ident(k) == mid), {})

    def damage(self, p, target, name, tera=False):
        d = self.move(name)
        if not d or d.get('category') == 'Status': return 0.
        mid = ident(name)
        typ = d.get('type', 'Normal')
        cat = d.get('category', 'Physical')
        bp = float(d.get('base_power', d.get('basePower', 0)) or 0)
        pt = self.types(p, tera)
        original = self.data(p).get('types', [])
        tt = self.types(target)
        terad = tera or bool(p.get('terastallized') or self.public(p).get('teraTypeRevealed') or self.public(p).get('tera_type'))
        if mid == 'terablast' and terad and pt:
            typ = pt[0]
            cat = 'Physical' if self.stat(p,'atk') > self.stat(p,'spa') else 'Special'
        if mid in ('seismictoss','nightshade'):
            return self.level(p)/self.maxhp(target) if effectiveness(typ,tt) else 0.
        if mid in ('superfang','naturesmadness','ruination'): return fraction(target)*.5
        if mid in ('grassknot','lowkick','heavyslam','heatcrash'): bp = 80  # unknown weight
        if mid == 'electroball': bp = 80
        if mid == 'gyroball': bp = min(150, 25*self.stat(target,'spe')/self.stat(p,'spe')+1)
        if mid in ('eruption','waterspout','dragonenergy'): bp = 150*fraction(p)
        if mid in ('flail','reversal'): bp = 120 if fraction(p)<.2 else 60
        if mid in ('storedpower','powertrip'): bp = 20+20*sum(max(0,v) for v in self.boosts(p).values())
        if mid == 'knockoff' and self.item(target): bp *= 1.5
        if mid == 'facade' and status(p): bp *= 2
        if mid == 'hex' and status(target): bp *= 2
        if mid == 'acrobatics' and not self.item(p): bp *= 2
        if mid in ('bulletseed','rockblast','iciclespear','pinmissile','tailslap'): bp *= 4 if self.item(p)=='loadeddice' or self.ability(p)=='skilllink' else 3
        if mid in ('populationbomb',): bp *= 8
        if mid in ('surgingstrikes','tripleaxel'): bp = 75 if mid=='surgingstrikes' else 100
        eff = effectiveness(typ, tt)
        ab = self.ability(target)
        ignores = self.ability(p) in ('moldbreaker','teravolt','turboblaze')
        if not ignores:
            if (typ=='Ground' and ab=='levitate') or (typ=='Water' and ab in ('waterabsorb','stormdrain','dryskin')) or (typ=='Fire' and ab in ('flashfire','wellbakedbody')) or (typ=='Electric' and ab in ('voltabsorb','lightningrod','motordrive')) or (typ=='Grass' and ab=='sapsipper'): return 0.
            if ab == 'wonderguard' and eff<=1: return 0.
            if ab == 'thickfat' and typ in ('Fire','Ice'): eff *= .5
            if ab in ('multiscale','shadowshield') and fraction(target)>.99: eff *= .5
            if ab in ('filter','solidrock','prismarmor') and eff>1: eff *= .75
            if ab == 'purifyingsalt' and typ=='Ghost': eff *= .5
        if self.item(target)=='airballoon' and typ=='Ground': return 0.
        atk_key = 'atk' if cat=='Physical' else 'spa'
        def_key = 'def' if cat=='Physical' or mid in ('psyshock','psystrike','secretsword') else 'spd'
        atk = self.stat(p, 'def' if mid=='bodypress' else atk_key, ab=='unaware' and not ignores)
        if mid=='foulplay': atk = self.stat(target,'atk')
        defense = self.stat(target, def_key, self.ability(p)=='unaware')
        stab = (2 if typ in pt and typ in original else 1.5) if terad and typ in pt else 1.5 if typ in original else 1
        if self.ability(p)=='adaptability' and stab>1: stab += .5
        mod = stab*eff
        if status(p)=='brn' and cat=='Physical' and mid!='facade' and self.ability(p)!='guts': mod *= .5
        if self.ability(p)=='guts' and status(p) and cat=='Physical': mod *= 1.5
        if self.ability(p)=='technician' and bp<=60: mod *= 1.5
        if self.item(p)=='lifeorb': mod *= 1.3
        weather = ident(self.pub.get('weather',''))
        if weather in ('raindance','rain','primordialsea'): mod *= 1.5 if typ=='Water' else .5 if typ=='Fire' else 1
        if weather in ('sunnyday','sun','desolateland'): mod *= 1.5 if typ=='Fire' else .5 if typ=='Water' else 1
        side = 'p1' if target in self.team else 'p2'
        screens = {ident(x) for x in self.pub.get('side_conditions',{}).get(side,{})}
        if 'auroraveil' in screens or ('reflect' if cat=='Physical' else 'lightscreen') in screens: mod *= .5
        dmg = (((2*self.level(p)/5+2)*bp*atk/defense)/50+2)*mod*.925
        # Charge moves without an explicit bypass have half their per-turn value.
        if mid in ('meteorbeam','electroshot','solarbeam','solarblade','skyattack','fly','dig','phantomforce','shadowforce') and self.item(p)!='powerherb':
            bypass = mid in ('solarbeam','solarblade') and weather in ('sun','sunnyday') or mid=='electroshot' and weather in ('rain','raindance')
            if not bypass: dmg *= .5
        return max(0., dmg/self.maxhp(target))

    def hazard(self, p):
        if self.item(p)=='heavydutyboots' or self.ability(p)=='magicguard': return 0.
        cs = self.pub.get('side_conditions',{}).get('p1',{})
        keys = {ident(k):v for k,v in cs.items()}
        damage = .125*effectiveness('Rock',self.types(p)) if 'stealthrock' in keys else 0.
        if 'spikes' in keys and 'Flying' not in self.types(p) and self.ability(p)!='levitate': damage += .17
        return damage

    def move_action(self, name, tera=False):
        for i,m in enumerate(self.moves,1):
            if ident(m.get('id') or m.get('move')) == ident(name):
                a = 'move '+str(i)+(' terastallize' if tera else '')
                if a in self.legal: return a
        return None

    def switch_action(self, name):
        for i,p in enumerate(self.team,1):
            if ident(species(p))==ident(name) or ident(p.get('ident',''))==ident(name):
                a='switch '+str(i)
                if a in self.legal: return a
        return None

    def attack_ranking(self, p=None):
        p = p or self.me
        names = [m.get('id') or m.get('move') for m in self.moves] if p is self.me else p.get('moves',[])
        out=[]
        for name in names:
            if p is self.me and not self.move_action(name): continue
            d = self.move(name)
            dmg = self.damage(p,self.foe,name)
            if not dmg: continue
            acc = d.get('accuracy',100)
            acc = 1. if acc is True else float(acc or 100)/100
            v = min(dmg, fraction(self.foe)+.12)*acc
            priority = d.get('priority',0)
            faster = self.stat(p,'spe') > self.stat(self.foe,'spe')
            if any(ident(k)=='trickroom' for k in self.pub.get('field',{})): faster = not faster
            if dmg*.91 >= fraction(self.foe): v += .6 if priority>0 or faster else .25
            if priority<0: v -= .08
            if ident(name) in ('uturn','voltswitch','flipturn'): v += .045
            if ident(name) in ('gigadrain','drainingkiss','drainpunch','oblivionwing'): v += min(1-fraction(p),dmg*.4)*.3
            if ident(name) in ('explosion','selfdestruct','mistyexplosion','finalgambit'): v -= .5*fraction(p)
            if ident(name) in ('hyperbeam','gigaimpact','blastburn','frenzyplant','hydrocannon'): v -= .13
            out.append((v,name,dmg))
        return sorted(out, reverse=True)

    def threat(self, p):
        # Known move metadata may be absent: use a transparent generic STAB prior.
        known = [self.damage(self.foe,p,m) for m in self.foe.get('moves',[]) if self.move(m)]
        rough=[]
        for typ in self.types(self.foe):
            eff=effectiveness(typ,self.types(p))
            ab=self.ability(p)
            if (typ=='Ground' and ab=='levitate') or (typ=='Water' and ab in ('waterabsorb','stormdrain','dryskin')) or (typ=='Fire' and ab in ('flashfire','wellbakedbody')) or (typ=='Electric' and ab in ('voltabsorb','lightningrod','motordrive')) or (typ=='Grass' and ab=='sapsipper'): eff=0
            ratio=max(self.stat(self.foe,'atk')/self.stat(p,'def'),self.stat(self.foe,'spa')/self.stat(p,'spd'))
            rough.append(((2*self.level(self.foe)/5+2)*90*ratio/50+2)*1.5*eff/self.maxhp(p))
        return max(known+rough+[.1])

    def fallback(self, history):
        attacks=self.attack_ranking()
        attack = self.move_action(attacks[0][1]) if attacks else None
        forced=not any(a.startswith('move ') for a in self.legal)
        switches=[]
        for i,p in enumerate(self.team,1):
            a='switch '+str(i)
            if a not in self.legal: continue
            hp=fraction(p)
            if hp<=0: # Revival Blessing
                switches.append((.4+sum(self.stat(p,k) for k in ('atk','spa','spe'))/1800,a)); continue
            h=self.hazard(p)
            damage=self.threat(p)
            moves=self.attack_ranking(p)
            offense=moves[0][0] if moves else 0
            v=offense-.85*damage+.12*hp-1.3*h
            if not forced: v -= damage+.16
            if hp<=h+damage and not forced: v-=.7
            if hp<=h: v-=2
            if history and history[-1].get('action','').startswith('switch'): v-=.25
            switches.append((v,a))
        switches.sort(reverse=True)
        if forced: return switches[0][1] if switches else (self.legal[0] if self.legal else 'noop')
        # Basic recovery and anti-setup logic only; strategic status/setup is Astra's job.
        hp=fraction(self.me); threat=self.threat(self.me)
        heals={'recover','roost','softboiled','slackoff','milkdrink','shoreup','moonlight','synthesis','morningsun','healorder','strengthsap'}
        for m in self.moves:
            name=ident(m.get('id') or m.get('move'))
            a=self.move_action(name)
            if a and name in heals and hp<.52 and threat<.38 and (hp>threat or self.stat(self.me,'spe')>self.stat(self.foe,'spe')):
                if not history or history[-1].get('move') != name: return a
            if a and name in ('haze','clearsmog') and max(self.boosts(self.foe).values(), default=0)>=2 and max(self.boosts(self.me).values(), default=0)<=0: return a
        attack_value=attacks[0][0] if attacks else -.15
        if switches and switches[0][0] > attack_value-.75*threat+.18:
            return switches[0][1]
        if attack: return attack
        choices=[a for a in self.legal if 'terastallize' not in a]
        # Do not repeat the same ineffective status move indefinitely.
        if history and len(choices)>1:
            choices=[a for a in choices if a!=history[-1].get('action')] or choices
        return choices[0] if choices else self.legal[0] if self.legal else 'noop'


SYSTEM = '''You are a Gen 9 Random Battle expert controlling p1 against a strong search opponent.
Compile a small, conditional policy for the CURRENT OPPONENT SPECIES. Code re-evaluates every
rule each decision and remaps move/species names to current legal slots. Rules are ordered,
first legal match wins. They can cover all six own Pokemon, forced replacements, and repeated
attacks across many turns. This is not an open-loop action sequence.
Use exact own stats/items, hazards, public boosts, PP, speed, abilities, observed damage and
priority. Opponent hidden sets are uncertain. Preserve crucial checks, but make progress.
Defensive Tera, setup, recovery and switching need concrete payoff. Don't repeat setup past
its target stage, heal at full HP, attack immunities, or switch-cycle. A slow switch-in takes
an entry hit then another hit before it attacks. Prefer KO lines and robust attacks.
Burn only weakens physical attacks. Rest requires surviving the hit on the healing turn
and two asleep turns; switch out of unsafe sleep. Compare the whole team's offensive
options before committing a special attacker to a special wall. Use defensive Tera when
it preserves a key member or enables a concrete winning line; do not hoard it indefinitely.
The code damage estimates are APPROXIMATE and can miss mechanics: use protocol over them.
Reply ONLY compact JSON, <=1000 tokens:
{"action":"one exact currently legal action", "plan":"brief win condition and uncertainty",
 "rules":[{"self":"own species or *", "move":"move name or best", "switch":"own species",
 "hp_min":0,"hp_max":1,"foe_hp_min":0,"foe_hp_max":1,
 "boost":"atk", "boost_min":-6,"boost_max":6,
 "foe_boost":"spa","foe_boost_min":-6,"foe_boost_max":6,
 "status":"any|none|brn|par|slp|frz|psn|tox", "foe_status":"any|none|...",
 "forced":false,"tera":false,"uses":99}]}
Omit unused fields. Exactly ONE of move/switch per rule. self must name a current own species
unless a forced replacement rule. HP thresholds are fractions [0,1]. boost is a stat name.
forced:true only matches forced replacement; forced:false only matches normal move requests;
omitted matches either. uses caps executions for this policy. Tera true spends only while
available; a rule requesting Tera cannot match after it has been used. Use a later normal
attack rule for the post-Tera line. Names, not slots, make policies safe after switches.
Include 6-12 rules: conditional switch/heal/setup before unconditional attacks. For an
unfavorable active member, a named switch rule is important. A forced:true switch rule can
choose the best healthy replacement. Unknown/disabled moves are skipped. A move:best rule
uses code's current damaging-move ranking (not recovery/setup). Use this when coverage varies.
Policies expire on opponent Tera or contradictions. Don't invent future opponent species.
'''


class Harness:
    def __init__(self, ctx):
        self.ctx=ctx
        self.policies={}
        self.history=[]
        self.calls=0
        self.last_call_at=-10
        self.failures=[]
        self.replans=Counter()
        self.uses=Counter()
        self.last_thought=''
        self.fail_streak=0

    def key(self, pos):
        return ident(species(pos.foe))+':'+','.join(pos.types(pos.foe))

    def prompt(self, pos, reason):
        # Fresh sessions prevent quadratic replay of long, duplicated observations.
        view={k:pos.obs.get(k) for k in ('turn','request','active','opponent_revealed','public_summary','last_error')}
        view['recent_events']=pos.obs.get('recent_events',[])[-65:]
        view['legal_actions']=pos.legal
        view['move_dex']=pos.dex
        view['species_dex']=pos.sdex
        view['approximate_attack_ranking']=[{'move':m,'foe_maxhp_fraction':round(d,3)} for _,m,d in pos.attack_ranking()]
        view['approximate_team_attacks']={species(p):[{'move':m,'damage':round(d,2)} for _,m,d in pos.attack_ranking(p)[:2]] for p in pos.team if fraction(p)>0}
        view['approximate_switch_threat']={species(p):round(pos.threat(p),3) for p in pos.team if fraction(p)>0}
        return ('Reason for policy compilation: '+reason+'\nPrevious strategic note: '+self.last_thought+
                '\nRecent action outcomes: '+json.dumps(self.history[-6:],separators=(',',':'))+
                '\nPolicy failures: '+json.dumps(self.failures[-5:],separators=(',',':'))+
                '\nSTATE:\n'+json.dumps(view,separators=(',',':'),default=str))

    def ask(self, pos, reason):
        ctx=self.ctx
        left=ctx.budget.remaining()
        # Always leave calls and wall time for finishing under a cached/fallback policy.
        if self.calls>=12 or left.get('system2_calls',0)<=1 or left.get('wall_s',0)<350:
            return None
        self.calls+=1
        self.last_call_at=len(self.history)
        key=self.key(pos)
        self.replans[key]+=1
        try:
            result=ctx.think(self.prompt(pos,reason), system=SYSTEM, session_id=None,
                             timeout_s=min(210.,left['wall_s']-130.))
            obj=unpack(result.text)
        except Exception as e:
            self.failures.append({'model_error':type(e).__name__})
            return None
        rules=obj.get('rules',[])
        if isinstance(rules,list):
            clean=[r for r in rules[:18] if isinstance(r,dict)]
            self.policies[key]=clean
            for i in range(len(clean)): self.uses[(key,i)]=0
        self.last_thought=str(obj.get('plan',''))[:900]
        ctx.memory['strategy']=self.last_thought
        action=obj.get('action')
        if action in pos.legal: return action
        # Plain exact legal action is useful when the policy JSON failed.
        for line in str(result.text).splitlines():
            a=line.strip().strip('`" ')
            if a in pos.legal: return a
        return None

    def rule_action(self, pos):
        key=self.key(pos)
        forced=not any(a.startswith('move ') for a in pos.legal)
        for i,r in enumerate(self.policies.get(key,[])):
            try:
                if r.get('self','*')!='*' and ident(r['self'])!=ident(species(pos.me)): continue
                if 'forced' in r and bool(r['forced']) != forced: continue
                if not float(r.get('hp_min',0))<=fraction(pos.me)<=float(r.get('hp_max',1)): continue
                if not float(r.get('foe_hp_min',0))<=fraction(pos.foe)<=float(r.get('foe_hp_max',1)): continue
                if self.uses[(key,i)]>=int(r.get('uses',99)): continue
                if 'boost' in r:
                    b=pos.boosts(pos.me).get(r['boost'],0)
                    if not float(r.get('boost_min',-6))<=b<=float(r.get('boost_max',6)): continue
                if 'foe_boost' in r:
                    b=pos.boosts(pos.foe).get(r['foe_boost'],0)
                    if not float(r.get('foe_boost_min',-6))<=b<=float(r.get('foe_boost_max',6)): continue
                if r.get('status','any') not in ('any',status(pos.me) or 'none'): continue
                if r.get('foe_status','any') not in ('any',status(pos.foe) or 'none'): continue
                if r.get('switch'):
                    a=pos.switch_action(r['switch'])
                    if a:
                        target=pos.team[int(a.split()[1])-1]
                        if fraction(target)<=pos.hazard(target) and fraction(target)>0: continue
                elif r.get('move'):
                    move=r['move']
                    if ident(move)=='best':
                        ranks=pos.attack_ranking()
                        move=ranks[0][1] if ranks else ''
                    if ident(move)=='rest' and status(pos.me)=='slp': continue
                    a=pos.move_action(move,bool(r.get('tera',False)))
                    # Observed immunities must override the cached instruction.
                    if a and pos.move(move).get('category')!='Status' and pos.damage(pos.me,pos.foe,move,bool(r.get('tera')))<=0: continue
                else: continue
                if a:
                    self.uses[(key,i)]+=1
                    return a, i
            except (TypeError,ValueError,IndexError,KeyError):
                continue
        return None,None

    def incoming_hit(self, before, after):
        """Extract incoming move damage before recovery, including switch-ins."""
        def actor(x):
            return re.sub(r'^(p[12])[a-z]:', r'\1:', str(x))
        hp={p.get('ident'):health(p)[0] for p in before.team}
        maxhp={p.get('ident'):health(p)[1] for p in before.team}
        losses=Counter()
        hit_moves={}
        source=''
        last_move=''
        for line in after.obs.get('recent_events',[]):
            fields=line.split('|')
            if len(fields)<3: continue
            event=fields[1]
            if event=='move':
                source=fields[2]
                last_move=fields[3] if len(fields)>3 else ''
            if event in ('switch','drag') and len(fields)>4:
                who=actor(fields[2]);hp[who]=health({'condition':fields[4]})[0]
            if event in ('-heal','-damage') and len(fields)>3:
                who=actor(fields[2]);newhp=health({'condition':fields[3]})[0]
                oldhp=hp.get(who,newhp)
                if event=='-damage' and source.startswith('p2') and who.startswith('p1') and '[from]' not in line:
                    losses[who]+=max(0,oldhp-newhp)/max(1,maxhp.get(who,1))
                    hit_moves[who]=last_move
                hp[who]=newhp
        who=after.me.get('ident')
        return float(losses.get(who,0)),hit_moves.get(who,'')

    def verify(self, before, after, action, progress_before, rule):
        reasons=[]
        progress=self.ctx.progress()
        if after.obs.get('last_error') or progress.get('illegal_actions',0)>progress_before.get('illegal_actions',0):
            reasons.append('rejected choice '+str(after.obs.get('last_error')))
        if progress.get('since_obs_change',0)>=2: reasons.append('observation stalled')
        if progress.get('since_score_change',0)>=5: reasons.append('five actions without score change')
        same_me=species(before.me)==species(after.me)
        same_foe=species(before.foe)==species(after.foe)
        mine=fraction(before.me)-fraction(after.me) if same_me else None
        theirs=fraction(before.foe)-fraction(after.foe) if same_foe else None
        events=after.obs.get('recent_events',[])
        eventtext='\n'.join(events)
        incoming,incoming_move=self.incoming_hit(before,after)
        if incoming>.43 or (mine is not None and mine>.43):
            reasons.append('critical incoming damage '+str(round(incoming,2)))
        if same_foe and 0<fraction(after.me)<=incoming*1.12:
            ranks=after.attack_ranking()
            safe_ko=False
            if ranks and status(after.me) not in ('slp','frz','par'):
                _,name,damage=ranks[0]
                priority=after.move(name).get('priority',0)
                faster=after.stat(after.me,'spe')>after.stat(after.foe,'spe')
                if any(ident(k)=='trickroom' for k in after.pub.get('field',{})): faster=not faster
                safe_ko=damage*.9>=fraction(after.foe) and (priority>0 or (priority==0 and faster))
            if not safe_ko: reasons.append('critical: another observed hit could KO; reassess switch/Tera')
        if any('|-immune|p2' in e or '|-fail|p1' in e for e in events): reasons.append('move failed or foe immune')
        if '|-terastallize|p2' in eventtext: reasons.append('opponent changed defensive type')
        if max(after.boosts(after.foe).values(),default=0)>=2 and max(before.boosts(before.foe).values(),default=0)<2: reasons.append('opponent reached dangerous boosts')
        move=''
        if action.startswith('move '):
            idx=int(action.split()[1])-1
            if idx<len(before.moves): move=ident(before.moves[idx].get('id') or before.moves[idx].get('move'))
        row={'action':action,'self':species(before.me),'foe':species(before.foe),'move':move,
             'self_loss':round(mine,3) if mine is not None else None,
             'foe_loss':round(theirs,3) if theirs is not None else None,
             'incoming_damage':round(incoming,3),'incoming_move':incoming_move}
        self.history.append(row)
        if len(self.history)>=3:
            last=self.history[-3:]
            if all(x['foe']==row['foe'] and x['self']==row['self'] and x['move']==move and x['foe_loss'] is not None and x['foe_loss']<=.02 for x in last):
                reasons.append('three repeated actions failed to make HP progress')
        if len(self.history)>=4 and all(x['action'].startswith('switch') for x in self.history[-4:]): reasons.append('switch cycle')
        if reasons:
            self.failures.append({'at':len(self.history),'reason':reasons,'events':events[-20:]})
            # A contradiction revokes the whole matching policy, not just a positional command.
            self.policies.pop(self.key(before),None)
        return '; '.join(reasons)

    def run_episode(self, game):
        ctx=self.ctx
        obs=ctx.observe()
        reason='initial matchup'
        steps=0
        while not ctx.done and steps<850:
            if not isinstance(obs,dict) or obs.get('done'): return
            pos=Position(obs)
            if not pos.legal: return
            key=self.key(pos)
            action=None
            # Unseen matchups and evidence of failure get strategic compilation, with a
            # cooldown so repeated surprises cannot consume the episode's whole budget.
            need=key not in self.policies
            urgent=bool(reason) and any(x in reason for x in ('critical','rejected','immune','dangerous boosts'))
            if need and (urgent or len(self.history)-self.last_call_at>=2 or self.calls==0) and self.replans[key]<3:
                action=self.ask(pos,reason or 'new opponent or invalidated policy')
                reason=''
                if ctx.done: return
            rule=None
            if not action: action,rule=self.rule_action(pos)
            source='policy' if action else 'fallback'
            if not action: action=pos.fallback(self.history)
            if action not in pos.legal: action=pos.legal[0]
            progress=ctx.progress()
            ctx.log('policy_choice',action=action,source=source,matchup=key,rule=rule)
            try:
                nxt=ctx.act(action)
            except Exception as e:
                if ctx.done: return
                self.fail_streak+=1
                self.failures.append({'act_error':type(e).__name__})
                self.policies.pop(key,None)
                reason='action exception; use updated request'
                if self.fail_streak>=5: return
                obs=ctx.observe()
                continue
            if ctx.done: return
            self.fail_streak=0
            steps+=1
            if not isinstance(nxt,dict): nxt=ctx.observe()
            after=Position(nxt)
            reason=self.verify(pos,after,action,progress,rule)
            obs=nxt


def build(ctx):
    return Harness(ctx)
