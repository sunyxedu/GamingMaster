"""End-to-end early-game play using only public legal actions; no privileged grants."""
import json
import math
from game import make_game, ROOT


def run():
    g = make_game()
    trace = []
    try:
        o = g.reset(9002)
        ore = {n: next(r[0] for r in o['resources'] if r[1] == n) for n in ['iron-ore', 'copper-ore', 'coal', 'stone']}
        def act(a):
            nonlocal o
            assert not g.done, 'Ran out of turns'
            o = g.step(a)
            assert not g.invalid_reason, g.info()
            assert all(v['ok'] for v in o['results']), (a, o['results'], o['stock'])
            trace.append({'action': a, 'turn': o['turn'], 'score': g.score})
            print(o['turn'], a, flush=True)
            return o
        act(f'mine_resource {ore["coal"]} 15; place stone-furnace 0 0')
        furnace = o['results'][1]['id']
        act(f'insert {furnace} fuel coal 15; mine_resource {ore["iron-ore"]} 15')
        # Mine and smelt 60 iron plates, then 30 copper plates for native unlocks.
        for i in range(4):
            more = f'; mine_resource {ore["iron-ore"]} 15' if i < 3 else ''
            act(f'insert {furnace} source iron-ore 15' + more)
        while o['production'].get('iron-plate', 0) < 60:
            act('wait')
        act(f'extract {furnace} result iron-plate 100; mine_resource {ore["copper-ore"]} 15')
        act(f'insert {furnace} source copper-ore 15; mine_resource {ore["copper-ore"]} 15')
        act(f'insert {furnace} source copper-ore 15; mine_resource {ore["stone"]} 10')
        while o['production'].get('copper-plate', 0) < 30:
            act('wait')
        act(f'extract {furnace} result copper-plate 100; mine_resource {ore["coal"]} 15')
        assert 'steam-power' in o['research']['completed'] and 'electronics' in o['research']['completed']
        # More iron to pay all real power/lab construction costs.
        act(f'insert {furnace} fuel coal 15; mine_resource {ore["iron-ore"]} 15')
        for i in range(8):
            more = f'; mine_resource {ore["iron-ore"]} 15' if i < 7 else ''
            extract = f'; extract {furnace} result iron-plate 80' if i else ''
            act(f'insert {furnace} source iron-ore 15' + extract + more)
        while o['production'].get('iron-plate', 0) < 180:
            act(f'extract {furnace} result iron-plate 100')
        act(f'extract {furnace} result iron-plate 100; mine_tree {o["trees"][0][0]}')
        def craft(item, amount):
            need = amount - o['stock'].get(item, 0)
            if need <= 0:
                return
            r = next((r for r in o['enabled_recipes'] if r['name'] == item), None)
            assert r, (item, o['stock'], o['research'], o['available_research'])
            output = next(p['amount'] for p in r['products'] if p['name'] == item)
            count = math.ceil(need / output)
            while any(o['stock'].get(i['name'], 0) < i['amount'] * count for i in r['ingredients']):
                for ingredient in r['ingredients']:
                    assert ingredient['type'] == 'item'
                    craft(ingredient['name'], ingredient['amount'] * count)
            act(f'handcraft {item} {count}')
            while o['queue']:
                act('wait')
        for item, n in [('lab', 1), ('boiler', 1), ('steam-engine', 1), ('offshore-pump', 1), ('small-electric-pole', 8), ('pipe', 20), ('automation-science-pack', 5)]:
            craft(item, n)
        assert 'automation-science-pack' in o['research']['completed']
        assert o['production'].get('automation-science-pack', 0) == 0
        assert o['manual_output']['automation-science-pack'] == 5
        # Report paid bootstrap, native and mirrored trigger unlocks, and physical stock.
        report = {'turn': o['turn'], 'charged_ticks': o['charged_ticks'], 'score': g.score,
                  'stock': o['stock'], 'research': o['research'], 'production': o['production'],
                  'manual_output': o['manual_output'], 'trace': trace}
        (ROOT / 'runtime' / 'legal-play.json').write_text(json.dumps(report, indent=2))
        print('Legal bootstrap passed:', o['turn'], 'turns;', g.score, 'score;', o['research']['completed'])
    finally:
        g.close()

if __name__ == '__main__':
    run()
