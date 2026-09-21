"""Real process failure classification and owned-server EOF cleanup."""
import json,pathlib,time
from game import make_game
results=[]
with make_game() as g:
    g.reset(2)
    first=g.observe();g.tick({'episode':'previous-episode','seq':first['seq'],'action':'wait 1'})
    assert g.observe()['action_feedback']['error']=='stale_episode_or_observation'
    owned=g._bridge;owned.terminate()
    deadline=time.monotonic()+10
    while not g.done and time.monotonic()<deadline:time.sleep(.1)
    assert g.done and not g.info()['valid_trial'] and g.info()['terminal']=='infrastructure_failure',g.info()
    assert g.observe()['progress']['components']['final_health']==0
    results.append({'lost_bridge':g.info()})
with make_game() as g:
    g.reset(3)
    server=g._server;server.stdin.close()  # Controller EOF must shut Java down.
    deadline=time.monotonic()+30
    while server.poll() is None and time.monotonic()<deadline:time.sleep(.1)
    assert server.poll() is not None,'server supervisor did not handle EOF'
    results.append({'server_eof':g.info(),'supervisor_exit':server.returncode})
pathlib.Path('runtime/lifecycle-test.json').write_text(json.dumps(results,indent=2))
print('LIFECYCLE PASS',json.dumps(results),flush=True)
