"""Evaluator-private schedule: twelve seeds x three duels covers 18 strata twice."""
from ygo_engine import rng
SEEDS={'train':[1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112],
       'heldout':[2201,2202,2203,2204,2205,2206,2207,2208,2209,2210,2211,2212],
       'dev':[3301,3302],
       'validation':[4401,4402,4403,4404,4405,4406,4407,4408,4409,4410,4411,4412]}
DECKS=['normal_pressure','flip_control','tribute_pressure']
def schedule(seed):
    for group,seeds in SEEDS.items():
        if seed in seeds:
            strata=[(a,b,s) for a in DECKS for b in DECKS for s in (0,1)]
            rng(0,'schedule:'+group).shuffle(strata)
            index=seeds.index(seed)
            # Preserve v2's first six scenarios; add an independently permuted block.
            if index>=6:
                rng(0,'schedule:'+group+':block-2').shuffle(strata)
                index-=6
            return strata[index*3:index*3+3]
    r=rng(seed,'scenario-selection')
    return [(r.choice(DECKS),r.choice(DECKS),r.randrange(2)) for _ in range(3)]
