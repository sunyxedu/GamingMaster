"""Reproducible, stratified native-engine opponent comparisons (private results)."""
import hashlib
import inspect
import json
import random
import time
from collections import Counter,defaultdict
from game import SEEDS
from scenarios import schedule
from ygo_engine import Duel,ROOT
from verify_suite import aggressive_action,random_action,assert_view,LIMITS

class PreviousGreedyDuel(Duel):
    opponent_response=Duel.greedy_response


def play(seed,round_index,stratum,agent,opponent):
    cls=Duel if opponent=='windbot-fixed-pool-v3' else PreviousGreedyDuel
    d=cls(seed,round_index,*stratum,LIMITS)
    rand=random.Random(seed*8192+round_index)
    start=time.monotonic()
    try:
        while not d.result:
            obs=d.observation();assert_view(obs)
            if agent=='greedy-tactical-v1':
                response=d.greedy_response(d.pending)
                d.respond(response);d.advance()
            else:
                action=(aggressive_action if agent=='aggressive' else random_action)(obs,rand)
                d.step(action)
                assert d.last_error is None,d.last_error
        assert d.result['kind']=='native',d.result
        return {'seed':seed,'round_index':round_index,'agent_policy':agent,'opponent_policy':opponent,
                **d.result,'coverage':d.diagnostics()['prompt_coverage'],'seconds':time.monotonic()-start}
    finally:d.close()


def summarize(rows):
    return {'duels':len(rows),'opponent_wins':sum(r['native_result']=='loss' for r in rows),
            'opponent_win_rate':sum(r['native_result']=='loss' for r in rows)/len(rows),
            'agent_win_rate':sum(r['win'] for r in rows)/len(rows),
            'draws':sum(r['native_result']=='draw' for r in rows),
            'mean_agent_score':sum(r['score'] for r in rows)/len(rows),
            'mean_duel_seconds':sum(r['seconds'] for r in rows)/len(rows),'errors':0,'truncations':0}


def main():
    started=time.monotonic();rows=[]
    pairs=[('greedy-tactical-v1','windbot-fixed-pool-v3',['train','validation','heldout']),
           ('aggressive','windbot-fixed-pool-v3',['heldout']),('random','windbot-fixed-pool-v3',['heldout']),
           ('aggressive','greedy-tactical-v1',['heldout'])]
    for agent,opponent,groups in pairs:
        for group in groups:
            for seed in SEEDS[group]:
                for index,stratum in enumerate(schedule(seed)):
                    row=play(seed,index,stratum,agent,opponent);row['split']=group;rows.append(row)
                print(agent,'vs',opponent,group,seed,'complete',flush=True)
    summaries={}
    for row in rows:
        key=f"{row['agent_policy']} vs {row['opponent_policy']} / {row['split']}"
        summaries.setdefault(key,[]).append(row)
    strata=defaultdict(list)
    for row in rows:
        if row['agent_policy']=='greedy-tactical-v1':
            key=f"{row['agent_deck']} / {row['opponent_deck']} / greedy {'first' if row['agent_first'] else 'second'}"
            strata[key].append(row)
    report={'status':'PASS','seconds':time.monotonic()-started,'manifest_version':3,
            'greedy_source_sha256':hashlib.sha256(inspect.getsource(Duel.greedy_response).encode()).hexdigest(),
            'summary':{k:summarize(v) for k,v in summaries.items()},
            'strata_vs_greedy':{k:summarize(v) for k,v in sorted(strata.items())},'matches':rows}
    (ROOT/'runtime/verification/opponent-v3.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary'],indent=2),flush=True)
if __name__=='__main__':main()
