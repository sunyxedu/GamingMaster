"""Bounded error feedback and real cancellation on a disposable vanilla fixture."""
import json,pathlib,threading,time
from game import make_game
from scoring import Score
from test_live import perform
results={}
with make_game() as g:
    g.reset(0);g._active=False;g._monitor_stop.set();g._rpc('deactivate')
    g._commands(['fill -8 99 -8 8 99 8 grass_block','fill -8 100 -8 8 107 8 air',
                 'fill 3 99 -8 3 99 8 lava','setblock -1 100 1 oak_log','setblock 2 100 -1 stone',
                 'tp HarnessBot 0.5 100 0.5','clear HarnessBot','give HarnessBot dirt 2304'])
    g._score=Score();g._scenario={'fixture':'edge cases, not benchmark'};g._snapshot=g._rpc('activate');g._start=time.monotonic();g._active=True;g._last_action=None
    g._monitor_stop=threading.Event();threading.Thread(target=g._monitor,args=(g._monitor_stop,),daemon=True).start()
    def fail(action,error):
        r=perform(g,action);assert r.get('status')=='failed' and error in r.get('error',''),r;results[action]=r
    fail('gather b:99999,100,99999','unknown_or_stale_candidate')
    fail('craft wooden_pickaxe 1','missing_materials_or_table')
    fail('equip nonexistent_item','missing_item')
    stone=next(c for c in g.observe()['candidates'] if c.get('name')=='stone' and c['position']==[2,100,-1])
    fail('gather '+stone['id'],'wrong_tool')
    # Start a real hand dig then interrupt it before its vanilla completion time.
    log=next(c for c in g.observe()['candidates'] if c.get('name')=='oak_log')
    g.tick('noop');g.tick('gather '+log['id']);time.sleep(.25);g.tick('cancel')
    deadline=time.monotonic()+7
    while g.observe()['task'] and not g.done and time.monotonic()<deadline:time.sleep(.1)
    assert not g.done,g.info()
    assert g.observe()['last_result']['status']=='cancelled'
    assert any(c['id']==log['id'] for c in g.observe()['candidates'])
    results['cancel_dig']=g.observe()['last_result']
    fail('gather '+log['id'],'no_drop_or_inventory_full')
    assert not g.observe()['inventory'].get('oak_log')
    # The observed far stand exists, but crossing the lava strip is forbidden.
    points=[c for c in g.observe()['candidates'] if c['kind']=='stand' and c['position'][0]>=4]
    if points:
        r=perform(g,'navigate '+points[0]['id']);assert r.get('status')=='failed',r;results['no_path']=r
    assert g.info()['valid_trial'],g.info()
pathlib.Path('runtime/edge-test.json').write_text(json.dumps(results,indent=2))
print('EDGE TEST PASS',json.dumps(results),flush=True)
