"""Integration validation against the staged real engine. Outputs are private."""
import itertools,json,math,random,time,hashlib,subprocess,sys,os,signal,ctypes as C,tempfile
from collections import Counter,defaultdict
from pathlib import Path
from game import make_game,SEEDS
from scenarios import schedule,DECKS
from ygo_engine import Duel,ROOT,Native,derive
from ygo_protocol import integer
OUT=ROOT/'runtime/verification';OUT.mkdir(parents=True,exist_ok=True)
LIMITS={'max_turns':200,'max_decisions':10000}
RECIPES=json.loads((ROOT/'decks.json').read_text())
IDS={c['name']:int(code) for code,c in RECIPES['cards'].items()}

def random_action(obs,r):
    p=obs['prompt'];choices=p['choices'];mode=p.get('mode','choice');token=p['token']
    if mode=='choice':return r.choice(choices)['action']
    n=len(p['candidates'])
    if mode=='order':
        indices=list(range(n));r.shuffle(indices);return f'order {token} '+','.join(map(str,indices))
    if p['kind']=='tributes':
        combos=[c for k in range(min(n,p['maximum'])+1) for c in itertools.combinations(range(n),k) if sum(p['weights'][i] for i in c)>=p['minimum']]
        actions=[f'select {token} '+(','.join(map(str,c)) or '-') for c in combos]+[o['action'] for o in choices]
        return r.choice(actions)
    counts=[(k,math.comb(n,k)) for k in range(p['minimum'],min(n,p['maximum'])+1)]
    idx=r.randrange(sum(c for _,c in counts)+len(choices))
    if idx<len(choices):return choices[idx]['action']
    idx-=len(choices)
    for k,c in counts:
        if idx<c:return f'select {token} '+(','.join(map(str,sorted(r.sample(range(n),k)))) or '-')
        idx-=c

def aggressive_action(obs,r):
    p=obs['prompt'];own=obs['agent_seat'];mine=obs['players'][own];enemy=obs['players'][1-own];choices=p['choices'];token=p['token']
    cat=obs['card_catalog']
    if p.get('mode')=='subset':
        candidates=p['candidates']
        def value(c):return cat.get(str(c.get('code')),{}).get('attack',0)
        target=p.get('hint','').lower()
        friendly=any(w in target for w in ('special summon','add to your hand','face-up'))
        pool=sorted(candidates,key=lambda c: ((c.get('player')==own)==friendly,value(c)),reverse=True)
        if p['kind']=='tributes':pool=sorted(candidates,key=value)
        chosen=[];weight=0
        for c in pool:
            if weight>=p['minimum']:break
            chosen.append(c['index']);weight+=p['weights'][c['index']]
        return f'select {token} '+(','.join(map(str,chosen)) or '-')
    if p['kind']=='main_command':
        summons=[c for c in choices if c['label']=='normal summon']
        if summons:return max(summons,key=lambda c:cat[str(c['card']['code'])]['attack'])['action']
        for label in ['change position / flip summon','activate','set spell/trap','enter Battle Phase','set monster','end turn']:
            found=[c for c in choices if c['label']==label]
            if label=='change position / flip summon':
                found=[c for c in found if any(m['slot']==c['card']['slot'] and m['position']!='face-up attack' for m in mine['monsters'])]
            if found:return found[0]['action']
    if p['kind']=='battle_command':
        found=[c for c in choices if c['label']=='attack']
        if found:return max(found,key=lambda c:cat[str(c['card']['code'])]['attack'])['action']
        for label in ['enter Main Phase 2','end turn']:
            for c in choices:
                if c['label']==label:return c['action']
    if p['kind']=='chain_response':
        # Conservative baseline; decline optional activation windows.
        for c in choices:
            if c['label']=='decline chain':return c['action']
    if p['kind']=='select_position':
        for c in choices:
            if c['label']=='face-up attack':return c['action']
    return random_action(obs,r)

def assert_view(obs):
    json.dumps(obs,allow_nan=False);assert len(json.dumps(obs).encode())<200000
    a=obs['agent_seat'];opp=obs['players'][1-a]
    assert 'hand' not in opp and 'deck' not in opp and 'seed' not in obs
    for c in opp['monsters']+opp['spells_traps']+opp['banished']:
        if c['position'].startswith('face-down'):assert 'code' not in c and 'attack' not in c,c
    for e in obs['recent_events']:
        if e['event']=='draw' and e['player']!=a:assert not e['known_cards']

def play_native(seed,a,b,seat,policy='random'):
    d=Duel(seed,0,a,b,seat,LIMITS);r=random.Random(8192+seed)
    actions=[]
    try:
        for _ in range(10000):
            obs=d.observation();assert_view(obs)
            # Validate the symmetric board boundary for the opponent, too.
            other=d.snapshot(1-seat)
            assert 'hand' not in other[seat]
            if d.result:break
            action=(random_action if policy=='random' else aggressive_action)(obs,r);actions.append(action)
            d.step(action);assert d.last_error is None,d.last_error
        assert d.result and d.result['kind']=='native',d.result
        return {'seed':seed,'policy':policy,**d.result,**d.diagnostics(),'actions_sha256':hashlib.sha256(json.dumps(actions).encode()).hexdigest()}
    finally:d.close()

def card(name,p,loc,seq=0,pos=8):return (IDS[name],p,loc,seq,pos)
def fixture(extra,seed=4567,own_deck='Monster Reborn',opp_deck='Monster Reborn'):
    cards=[card(own_deck,0,1) for _ in range(40)]+[card(opp_deck,1,1) for _ in range(40)]+extra
    return Duel(seed,0,'normal_pressure','normal_pressure',0,LIMITS,fixture=cards)
def find_choice(d,label,name=None):
    p=d.pending.public()
    for c in p['choices']:
        if c['label']==label and (name is None or c.get('card',{}).get('name')==name):return c['action']
    raise AssertionError((label,name,p))
def submit(d,a):d.step(a);assert d.last_error is None,d.last_error

def rules():
    checked=[]
    d=fixture([card('Blue-Eyes White Dragon',0,4,0,1)])
    try:
        o=d.observation();assert o['turn']==1 and len(o['players'][0]['hand'])==5
        assert not any('Battle Phase' in c['label'] for c in o['prompt']['choices'])
        checked.append('first player: five cards, no first-turn draw or Battle Phase')
        for _ in range(100):
            if d.result:break
            p=d.pending.public()
            labels={c['label']:c['action'] for c in p['choices']}
            a=next((labels[l] for l in ['attack','enter Battle Phase','end turn','decline chain','enter Main Phase 2'] if l in labels),None)
            if a is None and p.get('mode')=='subset':a=random_action(d.observation(),random.Random(18))
            assert a,p
            submit(d,a)
        assert d.result['reason_code']==1 and d.result['agent_lp']==8000 and d.result['opponent_lp']==0,d.result
        assert d.result['score']==1.1
        checked.append('three native 3000-ATK direct attacks, LP-zero winner, terminal margin')
    finally:d.close()
    d=fixture([card('Mystical Elf',0,4,0,1),card('Mystical Elf',0,4,1,1),card('Blue-Eyes White Dragon',0,2)],own_deck='Mystical Elf')
    try:
        submit(d,find_choice(d,'normal summon','Blue-Eyes White Dragon'))
        assert d.pending.kind in (20,26),d.pending.public()
        for _ in range(20):
            obs=d.observation()
            if any(c.get('name')=='Blue-Eyes White Dragon' for c in obs['players'][0]['monsters']):break
            p=obs['prompt']
            if p.get('mode')=='subset':submit(d,f"select {p['token']} 0,1")
            else:submit(d,p['choices'][0]['action'])
        obs=d.observation();assert any(c.get('name')=='Blue-Eyes White Dragon' for c in obs['players'][0]['monsters'])
        assert sum(c['name']=='Mystical Elf' for c in obs['players'][0]['graveyard'])==2
        checked.append('native two-Tribute summon and exact tribute costs')
    finally:d.close()
    d=fixture([card('Man-Eater Bug',0,4,0,8),card('Alexandrite Dragon',1,4,0,1)])
    try:
        submit(d,find_choice(d,'change position / flip summon','Man-Eater Bug'))
        for _ in range(20):
            o=d.observation()
            if o['players'][1]['graveyard_count']:break
            p=o['prompt']
            if p.get('mode')=='subset':
                i=next(c['index'] for c in p['candidates'] if c.get('name')=='Alexandrite Dragon')
                submit(d,f"select {p['token']} {i}")
            else:submit(d,random_action(o,random.Random(12)))
        assert any(c['name']=='Alexandrite Dragon' for c in d.snapshot()[1]['graveyard'])
        checked.append('native Flip Summon, mandatory Man-Eater Bug trigger and target destruction')
    finally:d.close()
    d=fixture([card('Blue-Eyes White Dragon',0,16,0,1),card('Monster Reborn',0,2)])
    try:
        submit(d,find_choice(d,'activate','Monster Reborn'))
        for _ in range(30):
            o=d.observation()
            if any(c.get('name')=='Blue-Eyes White Dragon' for c in o['players'][0]['monsters']):break
            p=o['prompt']
            if p.get('mode')=='subset':
                i=next(c['index'] for c in p['candidates'] if c.get('name')=='Blue-Eyes White Dragon')
                submit(d,f"select {p['token']} {i}")
            else:
                decline=[c for c in p['choices'] if c['label']=='decline chain']
                submit(d,decline[0]['action'] if decline else random_action(o,random.Random(12)))
        assert any(c.get('name')=='Blue-Eyes White Dragon' for c in d.snapshot()[0]['monsters'])
        checked.append('Monster Reborn graveyard targeting, position/zone choice and native special summon')
    finally:d.close()
    # Native winner semantics also cover alternate-win reasons and true draws;
    # injected native message tests ensure LP never overrides winner mapping.
    d=fixture([])
    try:
        d.lp=[10,8000];d.consume(bytes([5,0,16]));assert d.result['outcome']=='win' and d.result['score']>0
        d.result=None;d.consume(bytes([5,2,1]));assert d.result['native_result']=='draw'
        checked.append('winner message overrides LP (alternate-condition fixture), native draw mapping')
    finally:d.close()
    # Changing an opponent face-down identity must not change the player view.
    x=fixture([card('Mystical Elf',1,4,0,8)])
    y=fixture([card('Blue-Eyes White Dragon',1,4,0,8)])
    try:
        assert x.observation()==y.observation()
        checked.append('hidden opponent identity substitution leaves full observation unchanged')
    finally:x.close();y.close()
    return checked

def public_episode(seed):
    g=make_game();r=random.Random(12);size=0
    try:
        obs=g.reset(seed);initial=obs
        assert g.info()['valid_trial'],g.info()
        for _ in range(10000):
            assert_view(obs);size=max(size,len(json.dumps(obs).encode()))
            if g.done:break
            obs=g.step(random_action(obs,r))
            assert g.info()['valid_trial'],g.info()
        assert g.done and len(g.info()['results'])==3
        return {'initial':initial,'info':g.info(),'score':g.score,'breakdown':g.score_breakdown(),'max_observation_bytes':size}
    finally:g.close();g.close()

def failures():
    checks=[];g=make_game()
    try:
        o=g.reset(3301);token=o['prompt']['token'];before=g.info()['current']['trace_sha256']
        for bad in [None,{},[],123,'choose old 0','choose '+token+' -1','select '+token+' 0,0','__not_an_action__','choose']:
            g.step(bad);assert not g.done and g.observe()['prompt']['token']==token
        assert g.info()['current']['trace_sha256']==before
        g.step('noop');assert g.info()['current']['trace_sha256']==before
        checks.append('invalid, malformed, duplicate and stale actions do not reach engine; noop waits')
        g.deadline=time.monotonic()-1;g.observe()
        assert g.info()['results'][0]['forfeit'] and g.info()['results'][0]['native_result'] is None
        checks.append('monotonic response deadline produces identifiable forfeit')
        g.reset(3301);g.proc.kill();g.proc.wait();g.step('choose nonsense 0')
        assert g.done and not g.info()['valid_trial'] and g.score==0
        checks.append('native worker crash invalidates trial and releases process')
    finally:g.close()
    g=make_game(max_turns=1)
    try:
        g.reset(3301);assert g.done and g.score==0
        assert all(r['truncated'] and r['native_result'] is None for r in g.info()['results'])
        checks.append('turn cap is zero-score truncation, not a draw')
    finally:g.close()
    g=make_game(max_decisions=1)
    try:
        o=g.reset(3301)
        for _ in range(10):
            if g.done:break
            o=g.step(random_action(o,random.Random(123)))
        assert g.done and g.score==0 and all(r['truncated'] for r in g.info()['results'])
        checks.append('decision cap is separately reported truncation')
    finally:g.close()
    g=make_game(worker_timeout_s=0.1)
    try:
        # Give startup its ordinary budget, then stop the process at a pending prompt.
        g.worker_timeout=30;g.reset(3301);g.worker_timeout=0.1
        os.kill(g.proc.pid,signal.SIGSTOP);g.step('choose invalid 0')
        assert g.done and not g.info()['valid_trial'] and 'watchdog' in g.info()['invalid_reason']
        checks.append('native processing watchdog invalidates and kills a stalled worker')
    finally:g.close()
    n=Native(derive(1,'missing-script-test'))
    try:
        with tempfile.TemporaryDirectory(dir=OUT) as empty:
            handle=n.create(str(ROOT/'runtime/cards.tsv').encode(),empty.encode(),(C.c_uint64*4)(1,2,3,4))
            try:assert b'Missing script' in n.error(handle)
            finally:n.destroy(handle)
        checks.append('missing required scripts fail explicitly at native startup')
    finally:n.close()
    from ygo_protocol import Reader
    try:Reader(b'\x01').u32()
    except ValueError:checks.append('truncated native payload is rejected by bounded parser')
    else:raise AssertionError('unbounded parser')
    # Deliberately bypass adaptor validation in an isolated direct fixture.
    d=fixture([])
    try:
        d.respond(integer(99))
        try:d.advance()
        except RuntimeError as e:assert 'MSG_RETRY' in str(e)
        else:raise AssertionError('native retry was not detected')
        checks.append('native MSG_RETRY is a technical error')
    finally:d.close()
    return checks

def main():
    start=time.monotonic();report={}
    for split in ('train','heldout','validation'):
        strata=[s for seed in SEEDS[split] for s in schedule(seed)]
        assert len(strata)==36 and set(Counter(strata).values())=={2}
    report['schedule']='Each twelve-seed split covers all 18 ordered deck/seat strata twice'
    report['rules']=rules();report['failure_semantics']=failures()
    rows=[];cov=Counter();activated=Counter();chain_checks={'multi_link_chains':0,'max_depth':0}
    consume=Duel.consume
    def checked_consume(self,data):
        stack=getattr(self,'_test_chain_stack',[])
        if data[0]==70:
            stack.append(int.from_bytes(data[-4:],'little'))
            chain_checks['max_depth']=max(chain_checks['max_depth'],len(stack))
            if len(stack)==2:chain_checks['multi_link_chains']+=1
        elif data[0]==72:
            link=data[1]
            assert stack and stack.pop()==link,('chain resolution order',stack,link)
        elif data[0]==74:assert not stack
        self._test_chain_stack=stack
        return consume(self,data)
    Duel.consume=checked_consume
    for policy in ['random','aggressive']:
        for i,(a,b,seat) in enumerate(itertools.product(DECKS,DECKS,(0,1))):
            row=play_native(5100+i,a,b,seat,policy);rows.append(row);cov.update(row['prompt_coverage']);activated.update(row['activated_cards'])
            print(policy,i,row['outcome'],row['turns'],flush=True)
    Duel.consume=consume
    assert chain_checks['multi_link_chains']>0
    report['chain_resolution']=chain_checks
    effect_ids={int(code) for code,c in RECIPES['cards'].items() if not c['type']&0x10}
    missing=effect_ids-set(activated)
    assert not missing,[(code,RECIPES['cards'][str(code)]['name']) for code in missing]
    report['prompt_coverage']=dict(cov);report['activated_pool_cards']={RECIPES['cards'][str(c)]['name']:n for c,n in activated.items()}
    report['baselines']=rows
    report['baseline_summary']={policy:{'duels':len(sub:=[r for r in rows if r['policy']==policy]),'completion_rate':1.0,'win_rate':sum(r['win'] for r in sub)/len(sub),'mean_score':sum(r['score'] for r in sub)/len(sub),'mean_turns':sum(r['turns'] for r in sub)/len(sub),'errors':0} for policy in ('random','aggressive')}
    first=public_episode(3301);second=public_episode(3301)
    assert first==second,'Public episode replay differs'
    # A fresh process also checks native trace/terminal determinism.
    script="from verify_suite import play_native; import json; print(json.dumps(play_native(5100,'normal_pressure','normal_pressure',0,'random'),sort_keys=True))"
    replay=json.loads(subprocess.check_output([sys.executable,'-c',script],cwd=ROOT,text=True))
    assert json.loads(json.dumps(rows[0]))==replay
    report['determinism']='Identical full public 3-duel replay and fresh-process native trace/terminal result'
    report['public_episode']={k:v for k,v in first.items() if k!='initial'}
    report['max_observation_bytes']=first['max_observation_bytes'];report['seconds']=time.monotonic()-start
    report['status']='PASS'
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k in ('status','baseline_summary','prompt_coverage','activated_pool_cards','max_observation_bytes','seconds')},indent=2))
if __name__=='__main__':main()
