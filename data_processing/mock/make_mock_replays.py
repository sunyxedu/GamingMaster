import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--H', type=int, default=12)
    ap.add_argument('--W', type=int, default=12)
    ap.add_argument('--C', type=int, default=18)
    ap.add_argument('--G', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.n):
        grid = [[[ [random.random() for _ in range(args.W)] for _ in range(args.H) ] for _ in range(args.C)] for _ in range(args.T)]
        globals_seq = [[random.random() for _ in range(args.G)] for _ in range(args.T)]
        legal = [ [ [ [True for _ in range(args.W)] for _ in range(args.H) ] ] for _ in range(args.T) ]
        K=4
        count = [random.randint(0,1) for _ in range(args.T)]
        src_y = [ [random.randint(0,args.H-1) for _ in range(K)] for _ in range(args.T) ]
        src_x = [ [random.randint(0,args.W-1) for _ in range(K)] for _ in range(args.T) ]
        dir_id = [ [random.randint(0,3) for _ in range(K)] for _ in range(args.T) ]
        ratio_id = [ [random.randint(0,1) for _ in range(K)] for _ in range(args.T) ]
        value = [random.random() for _ in range(args.T)]
        enemy = [ [ [ [0.0 for _ in range(args.W)] for _ in range(args.H) ] ] for _ in range(args.T) ]
        nextv = [ [ [ [0.0 for _ in range(args.W)] for _ in range(args.H) ] ] for _ in range(args.T) ]
        obj = dict(grid_seq=grid, globals_seq=globals_seq,
                   legal_source_mask_seq=legal, max_actions_per_tick=K,
                   count=count, src_y=src_y, src_x=src_x, dir_id=dir_id, ratio_id=ratio_id,
                   value=value, enemy_heatmap=enemy, next_visibility=nextv)
        with open(os.path.join(args.out, f"{i:06d}.json"), "w") as f: json.dump(obj, f)
    print("done", len(os.listdir(args.out)))
if __name__ == "__main__": main()
