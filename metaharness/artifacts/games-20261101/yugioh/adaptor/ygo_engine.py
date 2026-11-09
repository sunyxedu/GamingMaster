"""Isolated native duel driver and player-view boundary; never imported by policy."""
import ctypes as C
import hashlib
import json
import random
import struct
from collections import Counter
from pathlib import Path
from ygo_protocol import Reader,Prompt,PROMPTS,POSITIONS
ROOT=Path(__file__).resolve().parent
R=ROOT/'runtime'
LOCATIONS={1:'deck',2:'hand',4:'monsters',8:'spells_traps',16:'graveyard',32:'banished',64:'extra'}
PHASES={1:'Draw',2:'Standby',4:'Main 1',8:'Battle start',16:'Battle step',32:'Damage',64:'Damage calculation',128:'Battle end',256:'Main 2',512:'End'}

def derive(seed,domain):
    return hashlib.sha256(f'yugioh-benchmark-v1:{seed}:{domain}'.encode()).digest()
def rng(seed,domain):return random.Random(int.from_bytes(derive(seed,domain),'little'))

class Native:
    def __init__(self,seed):
        self.lib=C.CDLL(str(R/'libbridge.so'))
        v=C.c_void_p;u=C.c_uint32;b=C.c_uint8;p=C.POINTER(u)
        specs={'create':(v,[C.c_char_p,C.c_char_p,C.POINTER(C.c_uint64)]),'error':(C.c_char_p,[v]),
               'destroy':(None,[v]),'card':(None,[v,u,b,u,u,u]),'start':(None,[v]),'process':(C.c_int,[v]),
               'message':(v,[v,p]),'response':(None,[v,v,u]),'count':(u,[v,b,u]),'query':(v,[v,b,u,u,p])}
        for name,(ret,args) in specs.items():
            f=getattr(self.lib,'ygo_'+name);f.restype=ret;f.argtypes=args;setattr(self,name,f)
        self.handle=self.create(str(R/'cards.tsv').encode(),str(R/'vendor/scripts').encode(),(C.c_uint64*4)(*struct.unpack('<4Q',seed)))
        try:self.check()
        except Exception:
            self.close();raise
    def check(self):
        error=self.error(self.handle)
        if error:raise RuntimeError(error.decode(errors='replace'))
    def buffer(self,fn,*args):
        size=C.c_uint32();ptr=fn(self.handle,*args,C.byref(size))
        if size.value>10000000:raise RuntimeError('Engine buffer exceeds bound')
        return C.string_at(ptr,size.value) if size.value else b''
    def close(self):
        if self.handle:self.destroy(self.handle);self.handle=None

class Duel:
    def __init__(self,seed,round_index,agent_deck,opponent_deck,seat,limits,fixture=None):
        self.seed=seed;self.round_index=round_index;self.agent=seat
        self.decks=[agent_deck,opponent_deck];self.limits=limits
        self.cards=json.loads((R/'cards.json').read_text());self.catalog={k:{x:v for x,v in c.items() if x!='effects'} for k,c in self.cards.items()}
        self.strings={};self.victories={}
        for line in (R/'strings.conf').read_text().splitlines():
            fields=line.split(maxsplit=2)
            if len(fields)==3 and fields[0] in ('!system','!victory'):
                (self.strings if fields[0]=='!system' else self.victories)[int(fields[1],0)]=fields[2]
        self.native=Native(derive(seed,f'round-{round_index}:engine'))
        self.opponent_rng=rng(seed,f'round-{round_index}:opponent')
        # Independent policy stream retained verbatim for the historical greedy baseline.
        # WindBot gets its own seeded C# Random from the same domain.
        self.opponent_policy='windbot-fixed-pool-v3';self.bot=None;self.hint_ids=[0,0];self.battle=None;self.chain_epoch=0
        self.lp=[8000,8000];self.turn=0;self.turn_player=0;self.phase=0;self.decisions=0;self.agent_decisions=0
        self.hints=['',''];self.events=[];self.chain=[];self.pending=None;self.result=None;self.invalid_actions=0;self.last_error=None
        self.coverage=Counter();self.event_coverage=Counter();self.activated_cards=Counter();self.trace=hashlib.sha256();self.query_cache={}
        recipes=json.loads((ROOT/'decks.json').read_text())['decks']
        if fixture:
            for item in fixture:self.native.card(self.native.handle,*item)
        else:
            for player,deck in [(seat,agent_deck),(1-seat,opponent_deck)]:
                codes=recipes[deck].copy();rng(seed,f'round-{round_index}:shuffle-player-{player}').shuffle(codes)
                for code in codes:self.native.card(self.native.handle,code,player,1,0,8)
        self.native.check();self.native.start(self.native.handle)
        self.advance()
    def identity(self,code):
        c=self.cards.get(str(code));return {'code':code,'name':c['name']} if c else {'name':'unknown card'}
    def description(self,n):
        if n<10000:return self.strings.get(n,f'choice {n}')
        code,idx=n>>4,n&15;c=self.cards.get(str(code))
        return (c['effects'][idx] or c['text']) if c else f'effect {n}'
    def query(self,p,loc,seq):
        key=p,loc,seq
        if key in self.query_cache:return self.query_cache[key]
        data=self.native.buffer(self.native.query,p,loc,seq)
        if not data:return None
        r=Reader(data);out={}
        while r.pos<len(data):
            n=r.u16();sub=Reader(r.read(n));flag=sub.u32();out[flag]=sub.read(n-4)
            if flag==0x80000000:break
        r.end();self.query_cache[key]=out;return out
    @staticmethod
    def qnum(q,flag,default=0,signed=False):
        return int.from_bytes(q[flag],'little',signed=signed) if flag in q else default
    def reference(self,code,loc,viewer,force=False):
        p,l,s,pos=loc
        out={'player':p,'zone':LOCATIONS.get(l,str(l))}
        if l not in (1,2,64) or p==viewer:out['slot']=s
        known=force or (p==viewer and l!=1) or l==16 or (l in (4,8,32) and pos&5)
        if not known and l in (4,8,32) and pos==0:
            q=self.query(p,l,s)
            known=bool(q and (self.qnum(q,2)&5 or self.qnum(q,0x100000)))
        if known and code:out.update(self.identity(code))
        else:out['name']='unknown card'
        return out
    def board_card(self,p,loc,seq,viewer):
        q=self.query(p,loc,seq)
        if not q:return None
        code=self.qnum(q,1);pos=self.qnum(q,2)
        out=self.reference(code,(p,loc,seq,pos),viewer)
        out['position']=POSITIONS.get(pos,'face-up' if pos&5 else 'face-down')
        if 'code' in out:
            for key,flag in [('type',8),('level',16),('attack',256),('defense',512)]:out[key]=self.qnum(q,flag,signed=key in ('attack','defense'))
            if 0x4000 in q:
                locref=Reader(q[0x4000]).loc()
                if locref[1]:out['equipped_to']=self.reference(0,locref,viewer)
            if 0x8000 in q:
                r=Reader(q[0x8000]);out['targets']=[self.reference(0,r.loc(),viewer) for _ in range(r.u32())]
            if 0x20000 in q:
                r=Reader(q[0x20000]);out['counters']=[{'type':(v:=r.u32())&65535,'count':v>>16} for _ in range(r.u32())]
        return out
    def snapshot(self,viewer=None):
        if viewer is None:viewer=self.agent
        players=[]
        for p in (0,1):
            out={'player':p,'role':'you' if p==viewer else 'opponent','lp':self.lp[p]}
            for loc,label in LOCATIONS.items():
                count=self.native.count(self.native.handle,p,loc);out[label+'_count']=count
                if loc in (1,64) or (loc==2 and p!=viewer):continue
                slots=range(7 if loc==4 else 8 if loc==8 else count)
                out[label]=[c for s in slots if (c:=self.board_card(p,loc,s,viewer)) is not None]
            players.append(out)
        return players
    def event(self,msg):
        self.events.append(msg);self.events=self.events[-40:]
    def consume(self,data):
        r=Reader(data);msg=r.u8();self.event_coverage[msg]+=1
        if msg in PROMPTS:
            if self.pending:raise RuntimeError('Multiple prompts before response')
            self.coverage[msg]+=1
            self.pending=Prompt(data,self,f'r{self.round_index+1}p{self.agent_decisions+1}')
            return
        if msg==1:raise RuntimeError('Engine rejected a validated response (MSG_RETRY)')
        if msg==5:
            winner=r.u8();reason=r.u8()
            self.finish('native',winner,self.victories.get(reason,{1:'LP reached zero',2:'deck-out'}.get(reason,f'native reason {reason}')),reason)
        elif msg==2:
            kind=r.u8();p=r.u8();value=r.u64()
            if kind==3:self.hints[p]=self.description(value);self.hint_ids[p]=value
            elif kind in (1,2,4,5,10) and p==self.agent:self.event({'event':'hint','text':self.description(value) if kind!=10 else self.identity(value)})
        elif msg==40:
            self.turn_player=r.u8();self.turn+=1;self.event({'event':'new_turn','player':self.turn_player,'turn':self.turn})
        elif msg==41:
            self.phase=r.u16();self.battle=None;self.event({'event':'phase','phase':PHASES.get(self.phase,str(self.phase))})
        elif msg in (91,92,94,100):
            p=r.u8();amount=r.u32()
            self.lp[p]=max(0,amount if msg==94 else self.lp[p]+(amount if msg==92 else -amount))
            self.event({'event':{91:'damage',92:'recover',94:'LP update',100:'LP cost'}[msg],'player':p,'amount':amount,'lp':self.lp[p]})
        elif msg==90:
            p=r.u8();n=r.u32();cards=[]
            for _ in range(n):
                code=r.u32();pos=r.u32()
                if p==self.agent or pos&5:cards.append(self.identity(code))
            self.event({'event':'draw','player':p,'count':n,'known_cards':cards})
        elif msg==50:
            code=r.u32();src=r.loc();dest=r.loc();reason=r.u32()
            known=src[0]==self.agent or dest[0]==self.agent or src[1]==16 or dest[1]==16 or (src[1] in (4,8,32) and src[3]&5) or (dest[1] in (4,8,32) and dest[3]&5)
            # Use event-time positions, never query a later card in the same slot.
            event={'event':'move','from':{'player':src[0],'zone':LOCATIONS.get(src[1],str(src[1]))},'to':{'player':dest[0],'zone':LOCATIONS.get(dest[1],str(dest[1]))}}
            if known:event['card']=self.identity(code)
            self.event(event)
        elif msg==53:
            code=r.u32();p=r.u8();loc=r.u8();seq=r.u8();old=r.u8();new=r.u8()
            self.event({'event':'position','card':self.reference(code,(p,loc,seq,new),self.agent,force=bool(old&5)),'position':POSITIONS.get(new,str(new))})
        elif msg in (60,62,64,54):
            code=r.u32();loc=r.loc()
            self.event({'event':{60:'summon',62:'special summon',64:'flip summon',54:'set'}[msg],'card':self.reference(code,loc,self.agent)})
        elif msg==70:
            code=r.u32();loc=r.loc();p=r.u8();l=r.u8();s=r.u32();desc=r.u64();link=r.u32();self.activated_cards[code]+=1
            event={'event':'chain activation','link':link,'card':self.reference(code,loc,self.agent,force=True),'effect':self.description(desc)}
            self.chain.append(event);self.event(event.copy())
        elif msg in (71,72,73,75,76):
            link=r.u8() if len(data)==2 else r.u32()
            self.event({'event':{71:'chain added',72:'chain resolving',73:'chain resolved',75:'chain negated',76:'chain disabled'}[msg],'link':link})
        elif msg==74:self.chain=[];self.chain_epoch+=1;self.event({'event':'chain ended'})
        elif msg==83:
            targets=[self.reference(0,r.loc(),self.agent) for _ in range(r.u32())]
            self.event({'event':'targets','targets':targets})
            if self.chain:self.chain[-1]['targets']=targets
        elif msg==110:
            self.battle=(r.loc(),r.loc());self.event({'event':'attack','attacker':self.reference(0,self.battle[0],self.agent),'target':self.reference(0,self.battle[1],self.agent)})
        elif msg in (30,31,42):
            p=r.u8();n=r.u32();cards=[]
            for _ in range(n):
                code=r.u32();loc=(r.u8(),r.u8(),r.u32(),0)
                if p==self.agent:cards.append(self.reference(code,loc,self.agent,force=True))
            if cards:self.event({'event':'revealed cards','cards':cards})
        elif msg in (32,33,39):
            p=r.u8();self.event({'event':'shuffle','player':p,'zone':{32:'deck',33:'hand',39:'extra'}[msg]})
            r.read(len(data)-r.pos) # never expose shuffled codes/order
        elif msg in (112,114):
            self.battle=None;r.read(len(data)-r.pos)
        elif msg in (3,54,55,56,61,63,65,80,81,93,95,96,97,101,102,111,113,120,160,165,170,190):
            # These are animation/relationship notifications; authoritative board queries
            # supply position, stats, equip targets, card targets and counters.
            r.read(len(data)-r.pos)
        else:raise RuntimeError(f'Unsupported engine event {msg}')
        r.end()
    def finish(self,kind,winner=None,reason='',reason_code=None):
        w=0 if winner not in (0,1) else 1 if winner==self.agent else -1
        margin=max(-1,min(1,(self.lp[self.agent]-self.lp[1-self.agent])/8000))
        self.result={'kind':kind,'native_result':('win' if w==1 else 'loss' if w==-1 else 'draw') if kind=='native' else None,
                     'outcome':('win' if w==1 else 'loss' if w==-1 else 'draw') if kind in ('native','forfeit') else 'truncated',
                     'winner_seat':winner,'reason':reason,'reason_code':reason_code,'agent_lp':self.lp[self.agent],
                     'opponent_lp':self.lp[1-self.agent],'turns':self.turn,'decisions':self.decisions,'agent_decisions':self.agent_decisions,
                     'score':w+0.1*margin if kind in ('native','forfeit') else 0.0,'win':int(w==1),
                     'truncated':kind=='truncation','forfeit':kind=='forfeit','technical_error':False,
                     'agent_deck':self.decks[0],'opponent_deck':self.decks[1],'agent_first':self.agent==0,
                     'invalid_actions':self.invalid_actions}
        self.pending=None
    def advance(self):
        while not self.result:
            status=self.native.process(self.native.handle);self.native.check()
            data=self.native.buffer(self.native.message);self.trace.update(data);self.query_cache={}
            r=Reader(data)
            while r.pos<len(data):self.consume(r.read(r.u32()))
            if self.result:break
            if self.turn>=self.limits['max_turns'] or self.decisions>=self.limits['max_decisions']:
                self.finish('truncation',reason='benchmark turn/decision cap');break
            if self.pending:
                prompt=self.pending
                # Both players use the same visibility filter and response validator.
                if prompt.player==self.agent and not (prompt.mode=='choice' and len(prompt.responses)==1):
                    self.agent_decisions+=1
                    return
                response=prompt.responses[0] if prompt.mode=='choice' and len(prompt.responses)==1 else self.opponent_response(prompt)
                self.respond(response)
            elif status==0:raise RuntimeError('Engine ended without native winner')
            elif status==1:raise RuntimeError('Engine awaiting an unsupported/missing prompt')

    def opponent_response(self,prompt):
        from windbot import WindBot
        if self.bot is None:self.bot=WindBot(self)
        return self.bot.response(self,prompt)

    def greedy_response(self,prompt):
        """Choose a legal response with a small, explainable tactical policy.

        The policy only sees ``Prompt``'s player-filtered candidates and the public
        board.  It prefers profitable attacks, removal/interaction, useful summons,
        and conservative sets.  Every branch returns one of the already validated
        native responses, so an unusual card prompt still remains legal.
        """
        # Stable score helpers.  Prompt card records intentionally omit hidden data.
        def atk(card):
            try:return int(card.get('attack',0) or 0)
            except (TypeError,ValueError):return 0
        def label(i):return str(prompt.options[i].get('label','')).lower()
        def text(i):
            o=prompt.options[i]
            return ' '.join(str(o.get(k,'')) for k in ('label','effect','hint')).lower()
        viewed=self.snapshot(prompt.player)
        own=viewed[prompt.player]
        opp=viewed[1-prompt.player]
        own_monsters=[c for c in own.get('monsters',[]) if c.get('code')]
        opp_monsters=[c for c in opp.get('monsters',[]) if c.get('code')]

        def best(items):
            """Return a deterministic, independently seeded tie break."""
            if not items:
                return None
            top=max(v for v,_ in items)
            return self.opponent_rng.choice([i for v,i in items if v==top])

        if prompt.mode=='choice':
            if not prompt.options:
                return prompt.responses[0]
            # Battle command: attack directly or trade only when the attacker wins.
            if prompt.kind==10:
                attacks=[]
                for i,o in enumerate(prompt.options):
                    if o.get('label')!='attack':
                        continue
                    c=o.get('card') or {}
                    direct=bool(o.get('direct_attack_available'))
                    value=100000 if direct else atk(c)
                    if opp_monsters:
                        strongest=max(atk(x) for x in opp_monsters)
                        if not direct and atk(c)<=strongest:
                            value-=50000  # avoid losing a monster in an unfavourable battle
                    attacks.append((value,i))
                if attacks and max(v for v,_ in attacks)>0:
                    return prompt.responses[best(attacks)]
            # Main command: summon the strongest available monster, then activate
            # removal/negation, then set a monster or back row.  Ending the phase is
            # preferred to an empty or low-value activation.
            if prompt.kind==11:
                summon=[(atk(o.get('card') or {}),i) for i,o in enumerate(prompt.options)
                        if 'summon' in label(i) and 'change position' not in label(i)]
                if summon:
                    return prompt.responses[best(summon)]
                useful=[]
                ahead=self.lp[prompt.player] >= self.lp[1-prompt.player]
                for i in range(len(prompt.options)):
                    t=text(i)
                    if any(k in t for k in ('destroy','remove','banish','negate','return','bounce')):
                        useful.append((5 if ahead else 3,i))
                    elif 'activate' in label(i):
                        useful.append((1,i))
                    elif 'set' in label(i):
                        useful.append((0,i))
                if useful:
                    return prompt.responses[best(useful)]
            # Effect/chain decisions: use interaction when it can answer a visible
            # threat; decline optional low-impact chains to preserve resources.
            if prompt.kind in (12,13):
                hint=(prompt.hint or '').lower()
                if any(k in hint for k in ('destroy','remove','banish','negate','return','damage','summon')):
                    return prompt.responses[0]
                return prompt.responses[1] if len(prompt.responses)>1 else prompt.responses[0]
            if prompt.kind==16:
                for i in range(len(prompt.options)):
                    t=text(i)
                    if any(k in t for k in ('destroy','remove','banish','negate','return')):
                        return prompt.responses[i]
                # Optional chain response is the final decline choice.
                return prompt.responses[-1]
            if prompt.kind==14:
                ranked=[]
                for i in range(len(prompt.options)):
                    t=text(i); score=0
                    if any(k in t for k in ('destroy','remove','banish','negate')):score+=5
                    if any(k in t for k in ('draw','summon','attack')):score+=2
                    ranked.append((score,i))
                return prompt.responses[best(ranked)]
            # Position/yes-no and any future choice prompts: first legal answer is
            # deterministic; the policy never fabricates a native response.
            return prompt.responses[0]

        if prompt.mode=='subset':
            n=len(prompt.candidates)
            if not n:
                return prompt.responses[0] if prompt.responses else b''
            # Target selections: remove the highest attack opposing cards first.
            opposing=[i for i,c in enumerate(prompt.candidates) if c.get('player')==1-prompt.player]
            own_indices=[i for i,c in enumerate(prompt.candidates) if c.get('player')==prompt.player]
            if opposing:
                ordered=sorted(opposing,key=lambda i:atk(prompt.candidates[i]),reverse=True)
            else:
                # Tribute/material costs: spend the smallest bodies that satisfy the
                # weighted minimum, preserving the strongest board card.
                ordered=sorted(own_indices,key=lambda i:atk(prompt.candidates[i]))
            chosen=[];weight=0
            for i in ordered:
                if len(chosen)>=prompt.maximum:break
                chosen.append(i);weight += prompt.weights[i] if prompt.kind==20 else 1
                if weight>=prompt.minimum:break
            if weight < prompt.minimum:
                # A legal cancellation can be preferable to an invalid native reply.
                if prompt.responses:return prompt.responses[0]
                chosen=list(range(min(prompt.maximum,n)))
            from ygo_protocol import subset
            return subset(sorted(chosen))

        if prompt.mode=='order':
            # Preserve the engine's presented order; random permutation is needlessly
            # disruptive and can turn a profitable chain into a bad one.
            return bytes(range(len(prompt.candidates)))
        return prompt.responses[0] if prompt.responses else b''
    def respond(self,response):
        self.trace.update(b'R'+response);self.decisions+=1
        self.native.response(self.native.handle,response,len(response));self.pending=None
    def step(self,action):
        if self.result:return
        try:response=self.pending.response(action)
        except (ValueError,TypeError,OverflowError) as e:
            self.invalid_actions+=1;self.last_error=str(e);return
        self.last_error=None;self.respond(response);self.advance()
    def observation(self):
        return {'schema_version':1,'your_deck':self.decks[0],'round':self.round_index+1,'agent_seat':self.agent,'turn_player':self.turn_player,
                'turn':self.turn,'phase':PHASES.get(self.phase,str(self.phase)),'players':self.snapshot(),
                'chain':self.chain,'recent_events':self.events,'prompt':self.pending.public() if self.pending else None,
                'card_catalog':self.catalog,'last_action_error':self.last_error,'duel_result':self.public_result()}
    def public_result(self):
        if not self.result:return None
        return {k:v for k,v in self.result.items() if k not in ('agent_deck','opponent_deck')}
    def diagnostics(self):
        return {'activated_cards':dict(self.activated_cards),'prompt_coverage':dict(self.coverage),'event_coverage':dict(self.event_coverage),'trace_sha256':self.trace.hexdigest(),
                'agent_deck':self.decks[0],'opponent_deck':self.decks[1],'agent_first':self.agent==0,
                'opponent_policy':self.opponent_policy}
    def close(self):
        if self.bot:self.bot.close();self.bot=None
        self.native.close()
