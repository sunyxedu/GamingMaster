"""Bounded EDOPro API 11 prompt decoder. All native response bytes remain private."""
import itertools
import math
import struct

class Reader:
    def __init__(self,data): self.data=data; self.pos=0
    def read(self,n):
        if n<0 or self.pos+n>len(self.data): raise ValueError('Truncated engine message')
        out=self.data[self.pos:self.pos+n];self.pos+=n;return out
    def num(self,fmt): return struct.unpack('<'+fmt,self.read(struct.calcsize('<'+fmt)))[0]
    def u8(self):return self.num('B')
    def u16(self):return self.num('H')
    def u32(self):return self.num('I')
    def u64(self):return self.num('Q')
    def loc(self):return (self.u8(),self.u8(),self.u32(),self.u32())
    def end(self):
        if self.pos!=len(self.data): raise ValueError(f'Extra message bytes: {len(self.data)-self.pos}')

def integer(n):return struct.pack('<i',n)
def subset(indices):return struct.pack('<II',0,len(indices))+b''.join(struct.pack('<I',i) for i in indices)

PROMPTS={10,11,12,13,14,15,16,18,19,20,21,24,25,26}
KINDS={10:'battle_command',11:'main_command',12:'activate_effect',13:'yes_no',14:'option',15:'select_cards',16:'chain_response',18:'select_zone',19:'select_position',20:'tributes',21:'sort_chain',24:'disable_zones',25:'sort_cards',26:'select_unselect'}
POSITIONS={1:'face-up attack',2:'face-down attack',4:'face-up defense',8:'face-down defense'}

class Prompt:
    def __init__(self,data,engine,token):
        r=Reader(data);self.kind=r.u8();self.player=r.u8();self.token=token
        self.options=[];self.responses=[];self.candidates=[];self.mode='choice'
        self.minimum=0;self.maximum=0;self.weights=[];self.cancel=False
        self.hint=engine.hints[self.player]
        self.hint_id=getattr(engine,"hint_ids",[0,0])[self.player]
        self.description_ids=[];self.question_id=0
        def description(value):
            self.description_ids.append(value);return engine.description(value)
        engine.hints[self.player]=''
        if hasattr(engine,'hint_ids'):engine.hint_ids[self.player]=0
        card=lambda code,loc:engine.reference(code,loc,self.player)
        def candidate(wide=True,full=False):
            code=r.u32()
            if full:loc=r.loc()
            else:loc=(r.u8(),r.u8(),r.u32() if wide else r.u8(),0)
            return card(code,loc)
        def add(label,response,**extra):
            self.options.append({'label':label,**extra});self.responses.append(response)
        if self.kind==11:
            for command,label in enumerate(['normal summon','special summon','change position / flip summon','set monster','set spell/trap','activate']):
                for i in range(r.u32()):
                    c=candidate(command!=2)
                    ex={}
                    if command==5:ex={'effect':description(r.u64())};r.u8()
                    add(label,integer((i<<16)|command),card=c,**ex)
            if r.u8():add('enter Battle Phase',integer(6))
            if r.u8():add('end turn',integer(7))
            if r.u8():add('shuffle own hand',integer(8))
        elif self.kind==10:
            for i in range(r.u32()):
                c=candidate();desc=description(r.u64());r.u8()
                add('activate',integer(i<<16),card=c,effect=desc)
            for i in range(r.u32()):
                c=candidate(False);direct=bool(r.u8())
                add('attack',integer((i<<16)|1),card=c,direct_attack_available=direct)
            if r.u8():add('enter Main Phase 2',integer(2))
            if r.u8():add('end turn',integer(3))
        elif self.kind in (12,13):
            ex={}
            if self.kind==12:ex['card']=candidate(full=True)
            self.question_id=r.u64();self.hint=engine.description(self.question_id) or self.hint
            add('yes',integer(1),**ex);add('no',integer(0),**ex)
        elif self.kind==14:
            for i in range(r.u8()):add(description(r.u64()),integer(i))
        elif self.kind==16:
            r.u8();forced=r.u8();r.u32();r.u32()
            for i in range(r.u32()):
                c=candidate(full=True);desc=description(r.u64());r.u8()
                add('chain activation',integer(i),card=c,effect=desc)
            if not forced:add('decline chain',integer(-1))
        elif self.kind in (15,20):
            self.mode='subset';self.cancel=bool(r.u8());self.minimum=r.u32();self.maximum=r.u32()
            for i in range(r.u32()):
                self.candidates.append(candidate(full=self.kind==15))
                self.weights.append(r.u8() if self.kind==20 else 1)
            if self.cancel:add('cancel / select none',integer(-1))
        elif self.kind==26:
            finish=r.u8();cancel=r.u8();self.minimum=r.u32();self.maximum=r.u32()
            i=0
            for verb in ['select','unselect']:
                for _ in range(r.u32()):
                    add(verb,struct.pack('<ii',1,i),card=candidate(full=True));i+=1
            if finish or cancel:add('finish selection' if finish else 'cancel',integer(-1))
        elif self.kind in (18,24):
            count=r.u8();mask=r.u32();zones=[]
            for relative in range(2):
                for loc,offset,n in [(4,0,7),(8,8,8)]:
                    for seq in range(n):
                        if not mask & (1<<(relative*16+offset+seq)):
                            zones.append((self.player^relative,loc,seq))
            for comb in itertools.combinations(zones,count):
                add('place in '+', '.join(f'player {p} {"monster" if l==4 else "spell/trap"} zone {s}' for p,l,s in comb),bytes(v for z in comb for v in z))
        elif self.kind==19:
            code=r.u32();mask=r.u8()
            for pos,label in POSITIONS.items():
                if mask&pos:add(label,integer(pos),card=engine.identity(code))
        elif self.kind in (21,25):
            self.mode='order'
            for _ in range(r.u32()):
                code=r.u32();loc=(r.u8(),r.u32(),r.u32(),0)
                self.candidates.append(card(code,loc))
            add('keep original order',b'\xff')
        else:raise ValueError(f'Unsupported prompt {self.kind}')
        r.end()
        if not self.responses and not self.candidates:raise ValueError('Empty legal prompt')
        self._engine=engine

    def public(self):
        out={'token':self.token,'kind':KINDS[self.kind],'player':self.player,'hint':self.hint,
             'choices':[{'action':f'choose {self.token} {i}',**o} for i,o in enumerate(self.options)]}
        if self.mode!='choice':
            out.update(mode=self.mode,candidates=[{'index':i,**c} for i,c in enumerate(self.candidates)])
            if self.mode=='subset':out.update(minimum=self.minimum,maximum=self.maximum,weights=self.weights,
                constraint='sum(weights) >= minimum and count <= maximum' if self.kind==20 else 'minimum <= count <= maximum',
                grammar=f'select {self.token} i,j,... (use - for empty)')
            else:out['grammar']=f'order {self.token} i,j,... (all indices, first chain link / deck position first)'
        return out

    def response(self,action):
        if not isinstance(action,str):raise ValueError('Action must be a string')
        parts=action.split()
        if len(parts)!=3 or parts[1]!=self.token:raise ValueError('Expected current prompt token and arguments')
        verb,_,arg=parts
        if verb=='choose':
            i=int(arg)
            if i<0 or i>=len(self.responses):raise ValueError('Choice out of range')
            return self.responses[i]
        if verb not in ('select','order') or (verb=='select')!=(self.mode=='subset') or self.mode=='choice':
            raise ValueError('Wrong action for this prompt')
        indices=[] if arg=='-' else [int(i) for i in arg.split(',')]
        if len(set(indices))!=len(indices) or any(i<0 or i>=len(self.candidates) for i in indices):raise ValueError('Invalid candidate indices')
        if self.mode=='order':
            if len(indices)!=len(self.candidates):raise ValueError('Order requires all candidates')
            # Wire format maps original index -> chosen position.
            return bytes(indices.index(i) for i in range(len(indices)))
        if len(indices)>self.maximum or (sum(self.weights[i] for i in indices) if self.kind==20 else len(indices))<self.minimum:
            raise ValueError('Selection violates count / tribute constraints')
        return subset(indices)

    def random_response(self,rng):
        if self.mode=='choice':return rng.choice(self.responses)
        if self.mode=='order':
            values=list(range(len(self.candidates)));rng.shuffle(values)
            return bytes(values)
        n=len(self.candidates)
        if self.kind==20:
            combos=[c for k in range(min(n,self.maximum)+1) for c in itertools.combinations(range(n),k) if sum(self.weights[i] for i in c)>=self.minimum]
            options=[subset(c) for c in combos]+self.responses
            return rng.choice(options)
        counts=[(k,math.comb(n,k)) for k in range(self.minimum,min(n,self.maximum)+1)]
        total=sum(c for _,c in counts)+len(self.responses)
        chosen=rng.randrange(total)
        if chosen<len(self.responses):return self.responses[chosen]
        chosen-=len(self.responses)
        for k,c in counts:
            if chosen<c:return subset(sorted(rng.sample(range(n),k)))
            chosen-=c
        raise AssertionError('subset sampler')
