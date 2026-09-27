"""A stand-in for env/hok_worker.py (the hok_env SDK + licensed gamecore are cluster-only):
the same stdin/stdout protocol, random frames, a game of ``5 + (port % 3)`` steps that the
camp-1 hero wins when the first port is even (else camp 2)."""

import pickle
import struct
import sys

import numpy as np


def _read(stream):
    head = stream.read(8)
    if len(head) < 8:
        return None
    (n,) = struct.unpack("<Q", head)
    return pickle.loads(stream.read(n))


def _write(stream, obj):
    data = pickle.dumps(obj, protocol=4)
    stream.write(struct.pack("<Q", len(data)) + data)
    stream.flush()


def main():
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    init = _read(stdin)
    port = init["ports"][0]
    length = 5 + port % 3
    loss_camp = 2 if port % 2 == 0 else 1
    rng = np.random.default_rng(port)
    state = {"t": 0}
    _write(stdout, ("ok", {"pid": 0}))

    def frame(done):
        players = []
        for _ in range(2):
            legal = (rng.random(172) < 0.6).astype(np.float32)
            legal[[2, 12, 28, 44, 60]] = 1.0          # Move and bin 0 of every direction head
            legal[76:172].reshape(12, 8)[:, 0] = 1.0  # target "none" for every button
            sub = (rng.random((12, 6)) < 0.5).astype(np.float32)
            sub[:, 0] = 1.0
            players.append({"features": rng.random(705).astype(np.float32), "legal": legal,
                            "sub": sub, "reward": [0.0] * 8 + [0.01], "done": done})
        return {"players": players, "frame_no": 3 * state["t"], "done": done, "camps": [1, 2],
                "loss_camp": loss_camp if done else -1}

    while True:
        msg = _read(stdin)
        if msg is None:
            break
        cmd, arg = msg
        if cmd == "reset":
            state["t"] = 0
            _write(stdout, ("ok", frame(False)))
        elif cmd == "step":
            assert len(arg) == 2 and all(len(a) == 6 for a in arg), arg
            state["t"] += 1
            _write(stdout, ("ok", frame(state["t"] >= length)))
        elif cmd in ("end", "close"):
            _write(stdout, ("ok", {}))
            if cmd == "close":
                break


if __name__ == "__main__":
    main()
