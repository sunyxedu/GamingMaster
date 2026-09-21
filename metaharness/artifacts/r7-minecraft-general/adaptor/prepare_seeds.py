#!/usr/bin/env python3
"""Prequalify a declared seed list before evaluating ANY policy; never remap rejections."""
import argparse,json,hashlib,uuid
from game import make_game,seed_config,InadmissibleSeed
p=argparse.ArgumentParser(description=__doc__);p.add_argument('seeds',type=int,nargs='+');args=p.parse_args()
with make_game() as g:
    g._artifacts()
    for seed in args.seeds:
        scenario=seed_config(seed);key=hashlib.sha256(json.dumps(scenario,sort_keys=True).encode()).hexdigest()[:24]
        g._episode='prequalify-'+uuid.uuid4().hex
        try:
            record=g._prepare(scenario,g.run/'baselines'/key)
            print(json.dumps({'seed':seed,'world_seed':scenario['world_seed'],'admitted':True,'time_of_day':scenario['time_of_day'],'spawn':record['spawn'],'baseline':key}),flush=True)
        except InadmissibleSeed as exc:print(json.dumps({'seed':seed,'world_seed':scenario['world_seed'],'admitted':False,'reason':str(exc)}),flush=True)
