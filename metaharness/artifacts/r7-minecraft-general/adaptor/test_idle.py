"""Full, unshortened 300-second natural-world idle trial."""
from game import make_game
import json,time,pathlib
with make_game() as g:
    g.reset(1)
    print('START',json.dumps(g.info()),flush=True)
    previous=-1
    while not g.done:
        g.tick('noop')
        elapsed=int(g.info()['elapsed_s'])
        if elapsed//30!=previous:
            previous=elapsed//30;print('IDLE',elapsed,g.score,g.observe()['self']['health'],flush=True)
        time.sleep(g.tick_s)
    result=g.info()
    pathlib.Path('runtime/idle-test.json').write_text(json.dumps(result,indent=2))
    print('FINISHED',json.dumps(result),flush=True)
    assert result['valid_trial'],result
    assert result['terminal']=='time_limit',result
    assert result['elapsed_s']==300,result
    assert g.score==20,g.score
    frozen=g.score;time.sleep(1);assert g.score==frozen
