"""Disposable real vanilla test fixture. Admin setup happens BEFORE its test clock.
This is not a benchmark episode, nor is its score a natural-world result.
"""
import json,pathlib,time,threading
from game import make_game
from scoring import Score
from test_live import perform

def fixture():
    with make_game() as g:
        g.reset(0)
        g._active=False;g._monitor_stop.set();g._rpc('deactivate')
        g._commands(['difficulty peaceful','gamerule doMobSpawning false',
          'fill -8 99 -8 12 99 8 minecraft:grass_block','fill -8 100 -8 12 107 8 minecraft:air',
          'fill 2 100 2 2 104 2 minecraft:oak_log','fill -3 100 -3 -3 102 3 minecraft:stone',
          'setblock 4 100 0 minecraft:stone','setblock 5 100 0 minecraft:stone','setblock 5 101 0 minecraft:stone',
          'tp HarnessBot 0.5 100 0.5','clear HarnessBot',
          'summon minecraft:cow 7 100 4 {NoAI:1b,Tags:["HarnessFixtureCow"]}','difficulty normal',
          'give HarnessBot raw_iron 2','give HarnessBot coal 3','give HarnessBot beef 1',
          'effect give HarnessBot minecraft:hunger 3 150 true'])
        time.sleep(3.5)
        g._commands(['effect clear HarnessBot','difficulty normal','gamerule doMobSpawning true'])
        g._score=Score();g._scenario={'fixture':'disposable mechanics verification, not benchmark'};g._snapshot=g._rpc('activate');g._start=time.monotonic();g._active=True;g._last_action=None
        g._monitor_stop=threading.Event();threading.Thread(target=g._monitor,args=(g._monitor_stop,),daemon=True).start()
        def current(name):
            deadline=time.monotonic()+3
            while time.monotonic()<deadline:
                found=[c for c in g.observe()['candidates'] if c.get('name')==name]
                if found:return found
                time.sleep(.2)
            raise AssertionError('no candidate '+name)
        def act(text):
            result=perform(g,text)
            assert result.get('status')=='succeeded',(text,result,g.info())
            return result
        def craft(n,c=1):
            tables=[b for b in g.observe()['candidates'] if b.get('name')=='crafting_table']
            suffix=' '+tables[0]['id'] if tables else ''
            act(f'craft {n} {c}'+suffix)
        # Movement + real cancellation while a long wait is pending.
        stands=[c for c in g.observe()['candidates'] if c['kind']=='stand']
        act('navigate '+stands[0]['id'])
        for x in range(2,7):
            points=[c for c in g.observe()['candidates'] if c['kind']=='stand' and c['position'][0]==x and c['position'][2]==0]
            assert points,(x,g.observe()['candidates'])
            act('navigate '+points[0]['id'])
            assert abs(g.observe()['self']['position'][1]-points[0]['position'][1])<.2
        points=[c for c in g.observe()['candidates'] if c['kind']=='stand']
        act('navigate '+min(points,key=lambda c:c['position'][0])['id'])
        g.tick('wait 5');time.sleep(.1);g.tick('cancel');time.sleep(.4)
        deadline=time.monotonic()+8
        while g.observe()['task'] and not g.done and time.monotonic()<deadline:time.sleep(.1)
        assert not g.done,g.info()
        assert g.observe()['last_result']['status']=='cancelled'
        for _ in range(5):
            candidates=current('oak_log');candidates.sort(key=lambda c:c['position'][1])
            act('gather '+candidates[0]['id'])
        craft('oak_planks',20);craft('stick',8);craft('crafting_table')
        refs=[c for c in current('grass_block') if c['position'][1]==99 and 2<c['distance']<4]
        assert refs
        act('place crafting_table '+refs[0]['id']+' up')
        craft('wooden_pickaxe')
        assert g.observe()['inventory'].get('wooden_pickaxe')==1
        # Approach the wall using only observed stand candidates.
        stands=[c for c in g.observe()['candidates'] if c['kind']=='stand']
        act('navigate '+min(stands,key=lambda c:c['position'][0])['id'])
        # Harvest stone wall, never floor supporting the player.
        for _ in range(3):
            wall=[c for c in current('stone') if c['position'][0]==-3 and c['position'][1]>=100]
            wall.sort(key=lambda c:(c['position'][1],c['distance']))
            act('gather '+wall[0]['id'])
        craft('stone_pickaxe')
        for _ in range(11):
            wall=[c for c in current('stone') if c['position'][0]==-3 and c['position'][1]>=100]
            wall.sort(key=lambda c:(c['position'][1],c['distance']))
            act('gather '+wall[0]['id'])
        craft('stone_axe');craft('furnace');craft('torch',4)
        refs=[c for c in current('grass_block') if c['position'][1]==99 and 2<c['distance']<4]
        # Try a reference with free top; occupied placement is a bounded failure.
        placed=False
        for ref in refs:
            r=perform(g,'place furnace '+ref['id']+' up')
            if r.get('status')=='succeeded':placed=True;break
        assert placed
        f=current('furnace')[0]['id']
        act(f'furnace_start {f} raw_iron 2 coal 1')
        time.sleep(21)
        f=current('furnace')[0]['id'];act('furnace_take '+f)
        assert g.observe()['inventory'].get('iron_ingot')==2
        # Fuel left in the furnace is real state; insert another unit explicitly.
        act(f'furnace_start {f} beef 1 coal 1');time.sleep(11)
        f=current('furnace')[0]['id'];act('furnace_take '+f)
        assert g.observe()['inventory'].get('cooked_beef')==1
        craft('shield')
        act('shield on');act('shield off');act('shield on');act('shield off');act('eat cooked_beef')
        for _ in range(4):
            cow=next(e for e in g.observe()['entities'] if e['name']=='cow' and e['position'][1]>95)
            if cow['distance']<=2.8:break
            stands=[c for c in g.observe()['candidates'] if c['kind']=='stand']
            target=min(stands,key=lambda c:sum((a-b)**2 for a,b in zip(c['position'],cow['position'])))
            act('navigate '+target['id'])
        act('look_at '+' '.join(map(str,cow['position'])))
        act('attack '+str(cow['id']))
        g._commands(['data get entity @e[tag=HarnessFixtureCow,limit=1] Health'])
        import re
        health_lines=[line for line in g._server_lines if 'Cow has the following entity data:' in line]
        assert health_lines and float(re.search(r'data: ([0-9.]+)f',health_lines[-1])[1])<10,health_lines

        assert g.observe()['progress']['milestone_times'].get('cooked_food') is not None
        assert g.observe()['progress']['components']['equipment']==50
        result={'fixture':g.info(),'inventory':g.observe()['inventory']}
        pathlib.Path('runtime/fixture-test.json').write_text(json.dumps(result,indent=2))
        print('FIXTURE SUCCESS',json.dumps(result),flush=True)
        # Administrative terminal stimulus is solely a test, not a policy action.
        g._console('kill HarnessBot')
        deadline=time.monotonic()+4
        while not g.done and time.monotonic()<deadline:time.sleep(.1)
        assert g.done and g.info()['terminal']=='death',g.info()
        time.sleep(1);assert g.observe()['self']['health']==0
        result['death']=g.info();assert result['death']['progress']['components']['death']==-20
        pathlib.Path('runtime/fixture-test.json').write_text(json.dumps(result,indent=2))
        print('DEATH NO RESPAWN VERIFIED',g.score,flush=True)

if __name__=='__main__':fixture()
