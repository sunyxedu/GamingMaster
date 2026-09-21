"""Cancel a real multi-execution crafting transaction and check it cannot resume."""
import json,pathlib,threading,time
from game import make_game
from scoring import Score
with make_game() as g:
    g.reset(0);g._active=False;g._monitor_stop.set();g._rpc('deactivate')
    g._commands(['give HarnessBot oak_log 16'])
    g._inventory_barrier()
    g._score=Score();g._scenario={'fixture':'inventory cancellation, not benchmark'};g._snapshot=g._rpc('activate');g._start=time.monotonic();g._active=True
    g._monitor_stop=threading.Event();threading.Thread(target=g._monitor,args=(g._monitor_stop,),daemon=True).start()
    g.tick('craft oak_planks 64');time.sleep(.15);g.tick('cancel')
    deadline=time.monotonic()+7
    while g.observe()['task'] and not g.done and time.monotonic()<deadline:time.sleep(.1)
    assert not g.done,g.info()
    assert g.observe()['task'] is None and g.observe()['last_result']['status']=='cancelled',g.observe()
    time.sleep(.5);before=g.observe()['inventory'];time.sleep(1);after=g.observe()['inventory']
    assert before==after,(before,after)
    assert after.get('oak_log',0)>0,'entire cancelled batch unexpectedly completed'
    result={'inventory_after_cancel':after,'result':g.observe()['last_result'],'valid_trial':g.info()['valid_trial']}
    pathlib.Path('runtime/cancel-inventory-test.json').write_text(json.dumps(result,indent=2))
    print('INVENTORY CANCEL PASS',json.dumps(result),flush=True)
