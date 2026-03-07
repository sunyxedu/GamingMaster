"""Check direction distribution in training data."""
import json
import os
import collections
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--max_files", type=int, default=50)
    args = p.parse_args()

    c = collections.Counter()
    for f in sorted(os.listdir(args.data_dir))[:args.max_files]:
        if not f.endswith(".json"):
            continue
        obj = json.load(open(os.path.join(args.data_dir, f)))
        for dirs, cnt in zip(obj["dir_id"], obj["count"]):
            if cnt > 0:
                c[dirs[0]] += 1

    labels = {0: "down", 1: "right", 2: "up", 3: "left"}
    total = sum(c.values())
    print(f"Total moves: {total}")
    for k in sorted(c.keys()):
        pct = c[k] / total * 100
        print(f"  {k} ({labels.get(k, '?'):>5}): {c[k]:>6}  ({pct:.1f}%)")

if __name__ == "__main__":
    main()
