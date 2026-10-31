"""Run Foul Play (p2) versus the exact legacy bridge heuristic (p1)."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
import game


def episode(seed):
    start = time.monotonic()
    g = game.make_game()
    latencies = []
    try:
        g.reset(seed)
        largest = 0
        for _ in range(1000):
            obs = g.observe()
            largest = max(largest, len(json.dumps(obs, allow_nan=False).encode()))
            if g.done:
                break
            # Benchmark-only RPC uses the unchanged previous heuristic on p1's
            # own view. This is not an action available to the evaluated agent.
            step_start = time.monotonic()
            g._rpc({'op': 'baseline'})
            latencies.append(time.monotonic() - step_start)
        return {'seed': seed, **g.info(), 'seconds': round(time.monotonic()-start, 3),
                'largest_observation_bytes': largest, 'completed': g.done,
                'step_seconds': latencies}
    finally:
        g.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=110)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--output', default='runtime/opponent-benchmark-bounded.json')
    ap.add_argument('--start', type=int, default=0)
    args = ap.parse_args()
    results = []
    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(episode, 1000003+i*7919) for i in range(args.start,args.start+args.count)]
        for fut in as_completed(futures):
            row = fut.result()
            results.append(row)
            out.write_text(json.dumps(sorted(results,key=lambda r:r['seed']), indent=2)+'\n')
            print(json.dumps({'n':len(results),'seed':row['seed'],'valid':row['valid_trial'],
                              'outcome_p1':row.get('outcome'),'reason':row['termination_reason'],
                              'error':row['invalid_reason']}),flush=True)
    valid = [r for r in results if r['valid_trial'] and r['completed']]
    print(json.dumps({'valid':len(valid),'foulplay_wins':sum(r['outcome']==-1 for r in valid),
                      'losses':sum(r['outcome']==1 for r in valid),'ties':sum(r['outcome']==0 for r in valid)}))
