"""A stand-in for env/hok5v5_worker.py that needs no SDK, gamecore or licence.

It speaks the worker protocol of env/hok5v5.py (length-prefixed pickles on stdin /
stdout: an init dict answered by the metadata, then ``(command, argument)`` requests
answered by ``("ok", payload)`` / ``("error", text)``) and plays a scripted match:

  * every controlled camp receives one frame per ``feature`` request, numbered from 1;
  * every third frame is a prediction frame (five hero rows of 5,630 features, a fresh
    token) that must be answered by ``action`` before the camp's next ``feature``;
  * the frame numbered ``max_frame_num`` is the game-over frame (five hero rows with
    their terminal rewards, no prediction);
  * ``action`` returns, per hero, the argmax of each categorical head, all-legal masks
    and the submitted head probabilities.

``HOK5V5_FAKE_OBSERVATION_SIZE`` (environment) makes the metadata report another
observation size — the ABI check of the interface.
"""

import hashlib
import os
import pickle
import struct
import sys

HEADS = (13, 25, 42, 42, 39, 1)
POLICY = sum(HEADS)
OBS = 5630
HEROES = 5
PREDICT_EVERY = 3


def read(stream):
    head = stream.read(8)
    if len(head) < 8:
        raise EOFError
    return pickle.loads(stream.read(struct.unpack("<Q", head)[0]))


def write(stream, obj):
    data = pickle.dumps(obj, protocol=4)
    stream.write(struct.pack("<Q", len(data)) + data)
    stream.flush()


def hero_rows(agent, frame_no, reward):
    rows = []
    for hero in range(HEROES):
        value = agent + hero / 10.0 + frame_no / 1000.0
        rows.append({"hero_runtime_id": 10 * agent + hero + 1, "camp_id": agent + 1,
                     "features": [value] * OBS, "reward": reward,
                     "hero_rid_reward": {10 * agent + hero + 1: reward}})
    return rows


def main():
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    init = read(stdin)
    with open(init["config"], "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    write(stdout, ("ok", {
        "action_head_sizes": HEADS,
        "observation_size": int(os.environ.get("HOK5V5_FAKE_OBSERVATION_SIZE", OBS)),
        "heroes_per_camp": HEROES,
        "config_sha256": digest,
        "interface_sha256": None,
    }))
    max_frame = int(init["max_frame_num"])
    frames, pending, controlled, token = {}, {}, (), 0
    while True:
        try:
            command, argument = read(stdin)
        except EOFError:
            return 0
        try:
            if command == "reset":
                if argument["camp"]["mode"] != "5v5" or len(argument["camp"]["heroes"]) != 2:
                    raise ValueError("reset: not a 5v5 camp")
                controlled = tuple(i for i, common in enumerate(argument["use_common_ai"])
                                   if not common)
                frames, pending = {a: 0 for a in controlled}, {}
                write(stdout, ("ok", {"reset": True}))
            elif command == "feature":
                agent = int(argument)
                if agent not in controlled:
                    raise ValueError(f"agent {agent} is not controlled")
                if agent in pending:
                    raise RuntimeError(f"agent {agent} has an unanswered prediction frame")
                frames[agent] += 1
                frame_no = frames[agent]
                gameover = frame_no >= max_frame
                predict = not gameover and frame_no % PREDICT_EVERY == 0
                payload = {"continue_process": predict, "token": None, "features": [],
                           "state": {"frame_no": frame_no, "gameover": gameover}}
                if predict:
                    token += 1
                    pending[agent] = token
                    payload["token"] = token
                    payload["features"] = hero_rows(agent, frame_no, 0.0)
                elif gameover:
                    payload["features"] = hero_rows(agent, frame_no, 1.0 if agent == 0 else -1.0)
                write(stdout, ("ok", payload))
            elif command == "action":
                agent = int(argument["agent_id"])
                if pending.get(agent) != argument["token"]:
                    raise RuntimeError("action: stale token")
                probabilities = argument["probabilities"]
                if tuple(probabilities.shape) != (HEROES, POLICY):
                    raise ValueError(f"action: probabilities {probabilities.shape}")
                del pending[agent]
                rows = []
                for hero in range(HEROES):
                    row, offset, heads = probabilities[hero], 0, []
                    for width in HEADS:
                        heads.append([float(v) for v in row[offset:offset + width]])
                        offset += width
                    rows.append({
                        "actions": [max(range(len(h)), key=h.__getitem__) for h in heads[:-1]],
                        "legal_action": [[1.0] * width for width in HEADS[:-1]],
                        "sub_actions": [1.0] * (len(HEADS) - 1),
                        "final_prob_list": heads,
                        "is_train": True,
                    })
                write(stdout, ("ok", rows))
            elif command == "end":
                frames, pending, controlled = {}, {}, ()
                write(stdout, ("ok", {"end": True}))
            elif command == "close":
                write(stdout, ("ok", {"close": True}))
                return 0
            else:
                raise ValueError(f"unknown command {command!r}")
        except Exception as exc:        # reported to the interface, the worker lives on
            write(stdout, ("error", f"{type(exc).__name__}: {exc}"))


if __name__ == "__main__":
    sys.exit(main())
