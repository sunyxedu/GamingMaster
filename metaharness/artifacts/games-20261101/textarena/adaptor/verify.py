"""Offline acceptance and strength checks; artifacts go only under runtime/."""
from __future__ import annotations
import socket

def deny_network(*args, **kwargs):
    raise AssertionError("Network use forbidden during acceptance checks")
socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
socket.create_connection = deny_network
socket.getaddrinfo = deny_network

import json
import random
import re
import time
from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache
import itertools
import game

LINES = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))

def won(board, who):
    return any(all(board[i] == who for i in line) for line in LINES)

@lru_cache(None)
def negamax(board, who):
    if won(board, 1-who): return -1
    if None not in board: return 0
    return max(-negamax(board[:i]+(who,)+board[i+1:],1-who) for i in range(9) if board[i] is None)

def perfect(obs):
    b=tuple(obs['public_state']['board']); who=obs['player_id']
    return f'[{max((i for i in range(9) if b[i] is None), key=lambda i:-negamax(b[:i]+(who,)+b[i+1:],1-who))}]'

def tactical(obs):
    b=obs['public_state']['board']; who=obs['player_id']
    for player in (who,1-who):
        for i in range(9):
            if b[i] is None:
                c=list(b);c[i]=player
                if won(c,player):return f'[{i}]'
    return next(f'[{i}]' for i in [4,0,2,6,8,1,3,5,7] if b[i] is None)

CODES=list(itertools.permutations(range(1,7),4))
def feedback(a,b):
    black=sum(x==y for x,y in zip(a,b));return black,len(set(a)&set(b))-black

def code_solver(obs):
    # Only permitted, player-visible messages; no environment or RNG access.
    hist=[]
    for m in re.finditer(r'Submitted \[([1-6 ]+)\]\. Feedback: (\d+) black peg\(s\), (\d+) white',obs['observation_text']):
        hist.append((tuple(map(int,m[1].split())),(int(m[2]),int(m[3]))))
    candidates=[c for c in CODES if all(feedback(c,g)==f for g,f in hist)]
    assert candidates
    # Minimize expected number of candidates remaining; prefer a possible solution.
    if len(candidates)==1:c=candidates[0]
    else:
        def cost(g):
            buckets=Counter(feedback(g,c) for c in candidates)
            return sum(n*n for n in buckets.values())
        c=min(candidates,key=cost)
    return '['+' '.join(map(str,c))+']'

report={'network':'socket connection and DNS APIs blocked before importing game', 'checks':[], 'strength':{},'timings':{}}
times=[];sizes=[]

def episode(seed,policy):
    g=game.make_game(); before=random.getstate(); start=time.monotonic();o=g.reset(seed);times.append(time.monotonic()-start)
    assert before==random.getstate()
    trace=[o]; actions=[]
    assert g.observe()==g.observe()
    while not g.done:
        a=policy(o);actions.append(a); start=time.monotonic();o=g.step(a);times.append(time.monotonic()-start)
        assert before==random.getstate()
        sizes.append(len(json.dumps(o,allow_nan=False).encode()))
        trace.append(o)
        assert g.info()['valid_trial'],g.info()
        assert 0<=g.score<=1
        if 'board' in o['public_state']:
            native=g._env.state.game_state['board']
            expected=[None if x=='' else (0 if x=='O' else 1) for row in native for x in row]
            assert expected==o['public_state']['board']
    info=g.info();assert g.step(None)==o
    g.close();g.close()
    return trace,actions,info

# Full traces repeated on fresh instances for all declared scenarios, including hidden code feedback.
for split,seeds in game.SEEDS.items():
    for seed in seeds:
        first=episode(seed,lambda o:o['legal_actions'][0])
        second=episode(seed,lambda o:o['legal_actions'][0])
        assert first==second,(split,seed)
report['checks'].append('All 26 declared seeds: repeat full action/observation/outcome traces exactly')

# Interleaved instances do not share environment RNG or player histories.
a=game.make_game();b=game.make_game();oa=a.reset(11);ob=b.reset(13)
expected={11:episode(11,lambda o:o['legal_actions'][0])[0],13:episode(13,lambda o:o['legal_actions'][0])[0]}
for seed,g,o in [(11,a,oa),(13,b,ob)]:assert o==expected[seed][0]
idx={11:1,13:1}
while not(a.done and b.done):
    for seed,g in [(11,a),(13,b)]:
        if not g.done:
            o=g.step(g.observe()['legal_actions'][0]);assert o==expected[seed][idx[seed]];idx[seed]+=1
for g in [a,b]:g.close()
report['checks'].append('Interleaved instances and host Python RNG preserved')

# Both seats, natural native draws and defeats, with independent player scripts.
for name,policy in [('first_legal',lambda o:o['legal_actions'][0]),('win_block_center',tactical),('perfect_negamax',perfect)]:
    outcomes=[]
    for seed in range(400,440,2):
        _,_,info=episode(seed,policy);outcomes.append(info['normalized_score'])
    assert 1.0 not in outcomes
    if name=='perfect_negamax': assert set(outcomes)=={0.5}
    report['strength'][name]={'episodes':len(outcomes),'baseline_wins':outcomes.count(0.0),'draws':outcomes.count(0.5),'baseline_losses':outcomes.count(1.0)}

# Random action play plus a competent observation-only puzzle solver: headroom audit.
for name in ['random_legal','competent']:
    by_game=defaultdict(list)
    for seed in game.SEEDS['train']+game.SEEDS['validation']+game.SEEDS['heldout']:
        rng=random.Random(seed)
        policy=(lambda o:rng.choice(o['legal_actions'])) if name=='random_legal' else (lambda o:perfect(o) if o['game_id']=='TicTacToe-v0' else code_solver(o))
        _,_,info=episode(seed,policy);by_game[info['game_id']].append(info['normalized_score'])
    means={k:sum(v)/len(v) for k,v in by_game.items()}
    report['strength'][name]={'per_game_scores':dict(by_game),'per_game_means':means,'equal_weight_mean':sum(means.values())/2}
    assert report['strength'][name]['equal_weight_mean']<1
    if name == 'competent':
        assert report['strength'][name]['equal_weight_mean'] < .75

# Native scoring boundary fixtures: deliberately use evaluator-only knowledge for checks,
# never as a measured player. Test natural winning/losing native action sequences.
for seed in [10,11]:
    g=game.make_game();g.reset(seed)
    for bad in [None, {'move':0}, '__bad__']:g.step(bad)
    assert g.done and g.score==0 and g.info()['termination']=='wrapper_invalid_forfeit'
    assert g.info()['valid_trial'];g.close()
    g=game.make_game();o=g.reset(seed);time.sleep(.05);assert g.observe()==o
    for _ in range(32):g.step('noop')
    assert g.done and g.score==0 and g.info()['wrapper_truncated'];g.close()
report['checks'].append('Malformed actions, invalid forfeits, noop truncations, idle stability and terminal idempotence')

g=game.make_game();g.reset(11);g.step('[1 2 3 4]')
before=g.info()['move_count'];g.step('[1 2 3 4]')
assert g.info()['invalid_actions']==1 and g.info()['move_count']==before;g.close()
g=game.make_game();g.reset(10)
occupied=next(i for i,v in enumerate(g.observe()['public_state']['board']) if v is not None)
before=g.info()['move_count'];g.step(f'[{occupied}]')
assert g.info()['invalid_actions']==1 and g.info()['move_count']==before;g.close()
report['checks'].append('Repeated Mastermind guess and occupied TicTacToe cell rejected without native advance')

# Mastermind real solve, exact zero partial reward, and maximum partial reward.
g=game.make_game();g.reset(11)
secret=tuple(g._env.state.game_state['secret_code'])
g.step('['+' '.join(map(str,secret))+']');assert g.done and g.score==1;g.close()
# Every valid distinct-digit guess overlaps by at least two digits, so native legal
# failure reward cannot be zero: lower bound .25, upper failed reward .75 (3 black).
# Normalization still uses declared engine bounds [0,1], never batch extrema.
g=game.make_game();g.reset(11)
secret=tuple(g._env.state.game_state['secret_code']);worst=min(CODES,key=lambda c:sum((x==y) for x,y in zip(c,secret))*.5+len(set(c)&set(secret))*.5)
others=[c for c in CODES if c not in [worst,secret]][:2]
for c in others+[worst]:g.step('['+' '.join(map(str,c))+']')
assert g.done and g.score==.25 and g.info()['raw_reward']==.25;g.close()
report['checks'].append('Natural Mastermind solve and lower achievable legal partial reward (.25); fixed normalization [0,1]')

# Native two-player victory fixture verifies learner reward=1 mapping (unreachable against perfect baseline).
g=game.make_game();g.reset(12)
for n in [0,1,3,4,6]:g._apply_native(f'[{n}]',actor=g._current)
assert g.done and g.score==1 and g.info()['raw_reward']==1;g.close()
report['checks'].append('Native TicTacToe win fixture and independent perfect-play draws / scripted losses')

# Infrastructure exception must be invalid, not a zero-scored bad-play episode.
g=game.make_game();g.reset(11)
def crash(action):raise RuntimeError('injected engine failure')
g._env.step=crash;g.step('[1 2 3 4]')
assert g.done and not g.info()['valid_trial'] and g.info()['termination']=='infrastructure_failure';g.close()
report['checks'].append('Injected engine failure marked invalid_trial; no fallback game or score')
report['timings']={'max_reset_or_step_s':max(times),'mean_reset_or_step_s':sum(times)/len(times),'max_observation_bytes':max(sizes)}
Path('runtime').mkdir(exist_ok=True)
Path('runtime/acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
