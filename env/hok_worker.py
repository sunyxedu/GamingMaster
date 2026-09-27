"""One Honor of Kings 1v1 game of Tencent's research environment (hok_env SDK
``hok.hok1v1.HoK1v1``), driven by env/vec_hok.py over stdin / stdout.

Runs under the SDK's own interpreter (Python 3.6-3.9, the compiled ``interface``
module), NOT the repo's: a standalone script without torch or repo imports.  Messages
are pickled (protocol 4) with an 8-byte little-endian length prefix:

    ← {"gamecore": "host:port", "aiserver_ip": ip, "ports": [p0, p1],
       "runtime_id": str, "req_timeout_ms": int}                      → ("ok", {})
    ← ("reset", {"camp": camp_config, "eval": bool})                  → ("ok", frame)
    ← ("step", [action_p0, action_p1])  (6 ints each)                 → ("ok", frame)
    ← ("end", None)   the game is over: release it in the gamecore     → ("ok", {})
    ← ("close", None) end the process                                  → ("ok", {})
    any failure                                                       → ("error", traceback)

frame = {"players": [per player: features [725] f32, legal [172] f32, sub [12, 6] f32,
         reward (the SDK's component list; [-1] = the weighted sum), done],
         "frame_no": int, "done": bool, "camps": [camp of player 0, 1],
         "loss_camp": camp whose crystal fell (-1: none yet / timeout)}
"""

import os
import pickle
import struct
import sys
import traceback

import numpy as np

CRYSTAL = 24          # organ type of the base crystal (aiarena/1v1/actor/actor.py)
FEATURES = 725


def _read(stream):
    head = stream.read(8)
    if len(head) < 8:
        return None
    (n,) = struct.unpack("<Q", head)
    return pickle.loads(stream.read(n))


def _write(stream, obj):
    data = pickle.dumps(obj, protocol=4)
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)
    stream.flush()


def _frame(env, state):
    players = []
    for i in range(env.PLAYER_NUM):
        s = state[i]
        feat = np.asarray(s["observation"], dtype=np.float32).reshape(-1)
        if feat.shape[0] != FEATURES:
            raise ValueError("player %d features have %d values, expected %d"
                             % (i, feat.shape[0], FEATURES))
        sub = s["sub_action_mask"]
        sub = np.stack([np.asarray(sub[b], dtype=np.float32) for b in range(12)])
        players.append({"features": feat,
                        "legal": np.asarray(s["legal_action"], dtype=np.float32).reshape(-1),
                        "sub": sub,
                        "reward": [float(x) for x in s["reward"]],
                        "done": bool(s["done"])})
    done = any(p["done"] for p in players)
    loss_camp = -1
    camps = [env.player_camp.get(env.player_list[i], -1) for i in range(env.PLAYER_NUM)]
    if done:
        req = next(r for r in env.cur_req_pb if r is not None)
        for organ in req.organ_list:
            if organ.type == CRYSTAL and organ.hp <= 0:
                loss_camp = organ.camp
    return {"players": players, "frame_no": int(env.cur_frame_no), "done": done,
            "camps": camps, "loss_camp": loss_camp}


def main():
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    # the SDK logs to stdout: keep the protocol stream private
    sys.stdout = sys.stderr
    init = _read(stdin)
    try:
        from hok.common.gamecore_client import GamecoreClient
        from hok.hok1v1 import HoK1v1
        from hok.hok1v1.env1v1 import interface_default_config
        from hok.hok1v1.hero_config import get_default_hero_config
        import hok.hok1v1.lib.interface as interface

        lib = interface.Interface()
        lib.Init(interface_default_config)
        launcher = GamecoreClient(server_addr=init["gamecore"],
                                  gamecore_req_timeout=init.get("req_timeout_ms", 30000),
                                  default_hero_config=get_default_hero_config())
        addrs = ["tcp://0.0.0.0:%d" % p for p in init["ports"]]
        env = HoK1v1(init["runtime_id"], launcher, lib, addrs,
                     aiserver_ip=init.get("aiserver_ip", "127.0.0.1"))
        _write(stdout, ("ok", {"pid": os.getpid()}))
    except Exception:
        _write(stdout, ("error", traceback.format_exc()))
        return
    while True:
        msg = _read(stdin)
        if msg is None:
            break
        cmd, arg = msg
        try:
            if cmd == "reset":
                _, _, _, state = env.reset(arg["camp"], use_common_ai=[False, False],
                                           eval=bool(arg.get("eval", False)))
                _write(stdout, ("ok", _frame(env, state)))
            elif cmd == "step":
                _, _, _, state = env.step([tuple(int(x) for x in a) for a in arg])
                _write(stdout, ("ok", _frame(env, state)))
            elif cmd == "end":
                env.close_game()
                _write(stdout, ("ok", {}))
            elif cmd == "close":
                try:
                    env.close_game()
                except Exception:
                    pass
                _write(stdout, ("ok", {}))
                break
            else:
                raise ValueError("unknown command %r" % (cmd,))
        except Exception:
            _write(stdout, ("error", traceback.format_exc()))


if __name__ == "__main__":
    main()
