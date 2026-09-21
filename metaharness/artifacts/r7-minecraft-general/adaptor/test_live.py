"""Real-server checks; no mocked game. Natural play never uses setup commands."""
import json, pathlib, random, time
from game import make_game
ROOT=pathlib.Path(__file__).resolve().parent

def perform(g, action, timeout=30):
    g.tick('noop');g.tick(action)
    feedback=g.observe()['action_feedback'] or {}
    ident=feedback.get('job_id')
    if ident is None:
        print('REJECT',action,feedback,flush=True);return feedback
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        o=g.observe();result=o.get('last_result')
        if result and result.get('id')==ident:
            print('ACTION',action,json.dumps(result),'INVENTORY',o['inventory'],'SCORE',round(g.score,3),flush=True)
            return result
        if g.done:raise AssertionError(g.info())
        time.sleep(.1)
    raise AssertionError('job did not finish: '+action)

def natural():
    with make_game() as g:
        g.reset(0)
        o=g.observe();assert not o['inventory'];assert o['self']['health']==20
        print('START',json.dumps(o),flush=True)
        logs=[c for c in o['candidates'] if c.get('name','').endswith('_log')]
        assert logs,'no observed logs'
        success=False
        for c in logs[:4]:
            r=perform(g,'gather '+c['id'])
            if r.get('status')=='succeeded':success=True;harvested_id=c['id'];break
        assert success,'could not harvest a natural log'
        o=g.observe();wood=next(k for k in o['inventory'] if k.endswith('_log'))
        planks=wood[:-4]+'_planks'
        assert perform(g,f'craft {planks} 4')['status']=='succeeded'
        assert g.observe()['inventory'].get(planks)==4
        assert perform(g,'craft crafting_table 1')['status']=='succeeded'
        assert g.observe()['progress']['milestone_times'].get('crafting_table') is not None
        placed=None
        refs=[c for c in g.observe()['candidates'] if c.get('name') in ('grass_block','dirt','stone') and 2<c['distance']<4]
        for ref in refs[:6]:
            result=perform(g,'place crafting_table '+ref['id']+' up')
            if result.get('status')=='succeeded':placed=[ref['position'][0],ref['position'][1]+1,ref['position'][2]];break
        assert placed is not None,'could not place table on an observed nearby surface'
        # Random valid/bad actions test bounded rejection, observation JSON and live tick counter.
        rng=random.Random(42)
        for action in rng.choices(['noop','look_at 0 70 0','equip absent_item','craft stone_pickaxe 1','wait 0.1','gather b:999,999,999'],k=5):perform(g,action)
        outcome={'natural':g.info(),'observation_bytes':len(json.dumps(g.observe()))}
        (ROOT/'runtime/natural-test.json').write_text(json.dumps(outcome,indent=2))
        start=g._scenario['spawn'];g.reset(0)
        assert g.observe()['inventory']=={}
        assert g.observe()['progress']['milestone_times']=={}
        assert g.observe()['self']['position']==start
        assert any(c['id']==harvested_id for c in g.observe()['candidates'])
        assert not any(c.get('name')=='crafting_table' and c['position']==placed for c in g.observe()['candidates'])
        print('CLEAN RESET VERIFIED',flush=True)
    return outcome

if __name__=='__main__':natural()
