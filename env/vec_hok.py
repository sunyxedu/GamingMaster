"""Honor of Kings 1v1 as a VecGameEnv (env/vec_game.py) — Tencent's Honor of Kings AI
Open Research Environment: the licensed gamecore (Windows simulator under Wine, launched
by its Linux gamecore server) and the hok_env SDK, one SDK worker process per game
(env/hok_worker.py, under the SDK's own Python 3.8).

Simultaneous: ``rows_per_env = n_seats = 2``, row ``(env, seat)`` is that hero's own
feature view, one frame per decision (every 3 game frames, the SDK's predict frequency).
Observation ``features`` [725] and the legal dict of model/games/hok.py (``button``,
``move_x`` / ``move_z`` / ``skill_x`` / ``skill_z``, the per-button ``target`` [12, 8] and
``sub`` [12, 6] tables).  Heroes: each episode draws both from ``heroes`` (default: every
1v1 hero the gamecore ships a scene for).

Reward per seat and step: ``shaping`` × the SDK's shaped reward (its weighted sum:
gold, experience, hp, tower hp, kills, deaths, last hits) + ``outcome`` × (±1 when a
crystal falls, to the winner / loser).  ``max_steps`` truncates an episode (a soft
terminal worth 0: the shaped rewards already scored the lead).

Needs the research environment (``hok_root``, default ``$HOK_ROOT`` or ``~/hok``):
``venv38`` (the SDK installed), ``wine-root`` (the SDK's bundled Wine debs, extracted),
``hok_env`` (the SDK repo, for the gamecore server's conf rewriter) and a gamecore
``gamecore_<version>/gamecore`` with ``core_assets/license.dat``.  ``gamecore="host:port"``
uses a running gamecore server instead of starting one.
"""

from __future__ import annotations

import fcntl
import glob
import os
import pickle
import shutil
import socket
import struct
import subprocess
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from env.vec_game import StepResult, VecEnvBase, VecObs

FEATURES = 725
BUTTONS = 12
DIRS = 16
TARGETS = 8
#: legal_action slices (the SDK's 172-long vector)
_LEGAL = {"button": (0, 12), "move_x": (12, 28), "move_z": (28, 44), "skill_x": (44, 60),
          "skill_z": (60, 76), "target": (76, 172)}
_HEADS = ("button", "move_x", "move_z", "skill_x", "skill_z", "target")
#: hero name → id (hok_env hok/common/camp.py HERO_DICT, the 1v1 subset)
HERO_IDS = {
    "buzhihuowu": 157, "diaochan": 141, "direnjie": 133, "ganjiangmoye": 182, "gongsunli": 199,
    "guanyu": 140, "hanxin": 150, "houyi": 169, "huamulan": 154, "jvyoujing": 163, "kai": 193,
    "libai": 131, "luban": 112, "luna": 146, "lvbu": 123, "makeboluo": 132, "miyue": 121,
    "peiqinhu": 502, "shangguanwaner": 513, "zhongkui": 175,
}
_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hok_worker.py")
_LAUNCHER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scripts", "hok", "sgame_simulator.sh")


def resolve_root(root: Optional[str] = None) -> str:
    return os.path.expanduser(root or os.environ.get("HOK_ROOT", "~/hok"))


def gamecore_dir(root: str) -> str:
    """The newest ``gamecore_<version>/gamecore`` under ``root`` (license included)."""
    found = sorted(glob.glob(os.path.join(root, "gamecore_*", "gamecore")))
    if not found:
        raise FileNotFoundError(f"no gamecore_<version>/gamecore under {root}")
    gc = found[-1]
    if not os.path.isfile(os.path.join(gc, "core_assets", "license.dat")):
        raise FileNotFoundError(f"{gc}/core_assets/license.dat missing (the research licence)")
    return gc


def _wait_port(host: str, port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"gamecore server did not open {host}:{port} in {timeout:.0f} s")


class GamecoreServer:
    """The gamecore server on this node, simulators under the portable Wine.

    The gamecore is copied once per node into ``local`` (fast local disk: a simulator
    reads its assets at every launch); concurrent copies are serialised by a lock."""

    def __init__(self, root: Optional[str] = None, port: int = 23432,
                 local: Optional[str] = None, log: Optional[str] = None):
        self.root = resolve_root(root)
        self.port = int(port)
        src = gamecore_dir(self.root)
        local = local or f"/tmp/{os.environ.get('USER', 'user')}-hok"
        os.makedirs(local, exist_ok=True)
        self.gamecore = os.path.join(local, os.path.basename(os.path.dirname(src)))
        with open(os.path.join(local, ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not os.path.isfile(os.path.join(self.gamecore, ".complete")):
                shutil.rmtree(self.gamecore, ignore_errors=True)
                shutil.copytree(src, self.gamecore, symlinks=True)
                os.chmod(os.path.join(self.gamecore, "gamecore-server-linux-amd64"), 0o755)
                open(os.path.join(self.gamecore, ".complete"), "w").close()
        bindir = os.path.join(local, "bin")
        os.makedirs(bindir, exist_ok=True)
        for name in ("sgame_simulator_remote_zmq", "sgame_simulator_repeated_zmq"):
            dst = os.path.join(bindir, name)
            if not os.path.exists(dst):
                shutil.copy(_LAUNCHER, dst)
                os.chmod(dst, 0o755)
        env = dict(os.environ, GAMECORE_PATH=self.gamecore, HOK=self.root,
                   WINE_PREFIX_ROOT=os.path.join(local, "wine"))
        self._log = open(log or os.path.join(local, f"gamecore-server-{self.port}.log"), "ab")
        self.proc = subprocess.Popen(
            [os.path.join(self.gamecore, "gamecore-server-linux-amd64"), "server",
             f"--server-address=:{self.port}",
             "--simulator-remote-bin", os.path.join(bindir, "sgame_simulator_remote_zmq"),
             "--simulator-repeat-bin", os.path.join(bindir, "sgame_simulator_repeated_zmq")],
            cwd=self.gamecore, env=env, stdout=self._log, stderr=subprocess.STDOUT)
        _wait_port("127.0.0.1", self.port, 60.0)

    @property
    def address(self) -> str:
        return f"127.0.0.1:{self.port}"

    def heroes(self) -> List[str]:
        """1v1 heroes with a scene file in this gamecore."""
        scenes = {os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(self.gamecore, "scene", "*.abs"))}
        return sorted(h for h in HERO_IDS if h in scenes)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._log.close()


class _Worker:
    """One SDK worker process (one game)."""

    def __init__(self, cmd: Sequence[str], init: Dict[str, Any], log):
        self.proc = subprocess.Popen(list(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=log)
        self.send(init)
        self.recv()

    def send(self, obj) -> None:
        data = pickle.dumps(obj, protocol=4)
        self.proc.stdin.write(struct.pack("<Q", len(data)) + data)
        self.proc.stdin.flush()

    def recv(self):
        head = self.proc.stdout.read(8)
        if len(head) < 8:
            raise RuntimeError(f"hok worker {self.proc.pid} exited ({self.proc.poll()})")
        (n,) = struct.unpack("<Q", head)
        status, payload = pickle.loads(self.proc.stdout.read(n))
        if status != "ok":
            raise RuntimeError(f"hok worker {self.proc.pid}: {payload}")
        return payload

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send(("close", None))
                self.recv()
            except Exception:
                pass
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class HokVecEnv(VecEnvBase):
    """N Honor of Kings 1v1 games (see the module docstring)."""

    game = "hok"
    n_seats = 2
    rows_per_env = 2
    interleaved = False
    ACTION_KEYS = {"action_button": "button_idx", "action_move_x": "move_x_idx",
                   "action_move_z": "move_z_idx", "action_skill_x": "skill_x_idx",
                   "action_skill_z": "skill_z_idx", "action_target": "target_idx"}

    def __init__(self, n_envs: int = 1, *, autoreset: bool = False, seed: Optional[int] = None,
                 gamecore: Optional[str] = None, hok_root: Optional[str] = None,
                 heroes: Optional[Sequence[str]] = None, max_steps: Optional[int] = None,
                 shaping: float = 1.0, outcome: float = 1.0, port_base: int = 35300,
                 gamecore_port: int = 23432, runtime_prefix: str = "gm",
                 worker_cmd: Optional[Sequence[str]] = None, eval_mode: bool = False,
                 log: Optional[str] = None):
        super().__init__(n_envs, autoreset, seed)
        self.root = resolve_root(hok_root)
        self.shaping, self.outcome = float(shaping), float(outcome)
        self.max_steps = None if max_steps is None else int(max_steps)
        self.eval_mode = bool(eval_mode)
        self._server = None
        if gamecore is None and worker_cmd is None:
            self._server = GamecoreServer(self.root, gamecore_port)
            gamecore = self._server.address
        self._gamecore = gamecore or "127.0.0.1:0"
        pool = list(heroes) if heroes else (self._server.heroes() if self._server else sorted(HERO_IDS))
        unknown = [h for h in pool if h not in HERO_IDS]
        if unknown:
            raise ValueError(f"unknown 1v1 heroes {unknown}; have {sorted(HERO_IDS)}")
        self.heroes = pool
        self._cmd = list(worker_cmd) if worker_cmd else [os.path.join(self.root, "venv38", "bin", "python"), _WORKER]
        self._port_base, self._prefix = int(port_base), runtime_prefix
        self._log = open(log, "ab") if log else subprocess.DEVNULL
        self._workers: List[_Worker] = []
        self._spawn(n_envs)
        self._frames: List[Optional[Dict]] = [None] * n_envs
        self._steps = np.zeros(n_envs, dtype=np.int64)
        self._finished = np.ones(n_envs, dtype=bool)
        self._new = np.zeros(n_envs, dtype=bool)
        self._camp_heroes: List[List[str]] = [[] for _ in range(n_envs)]

    # -- processes -------------------------------------------------------------

    def _spawn(self, n: int) -> None:
        while len(self._workers) < n:
            i = len(self._workers)
            init = {"gamecore": self._gamecore, "aiserver_ip": "127.0.0.1",
                    "ports": [self._port_base + 2 * i, self._port_base + 2 * i + 1],
                    "runtime_id": f"{self._prefix}-{self._port_base}-{i}", "req_timeout_ms": 30000}
            self._workers.append(_Worker(self._cmd, init, self._log))
        while len(self._workers) > n:
            self._workers.pop().close()

    def close(self) -> None:
        for w in self._workers:
            w.close()
        self._workers = []
        if self._server is not None:
            self._server.close()
            self._server = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- episodes ------------------------------------------------------------------

    @property
    def finished(self) -> torch.Tensor:
        return torch.from_numpy(self._finished.copy())

    def _camp(self, seed: int):
        """(the two hero names, the SDK camp config) of an episode seed."""
        rng = np.random.default_rng(seed)
        pick = [self.heroes[int(i)] for i in rng.integers(0, len(self.heroes), size=2)]
        return pick, {"mode": "1v1", "heroes": [[{"hero_id": HERO_IDS[pick[0]]}],
                                                [{"hero_id": HERO_IDS[pick[1]]}]]}

    def reset(self, n: Optional[int] = None, *, seed: Optional[int] = None,
              env_ids: Optional[Sequence[int]] = None) -> VecObs:
        if n is not None and n != self._n:
            if env_ids is not None:
                raise ValueError("reset: give n (resize) or env_ids, not both")
            self._resize(int(n))
        ids = self._env_ids(env_ids)
        seeds = self._episode_seeds(len(ids), seed)
        for e, s in zip(ids.tolist(), seeds):
            if not self._finished[e] and self._frames[e] is not None:
                self._workers[e].send(("end", None))
                self._workers[e].recv()
            pick, camp = self._camp(s)
            self._camp_heroes[e] = pick
            self._workers[e].send(("reset", {"camp": camp, "eval": self.eval_mode}))
        for e in ids.tolist():
            self._frames[e] = self._workers[e].recv()
            self._steps[e] = 0
            self._finished[e] = False
            self._new[e] = True
        return self.observe()

    def _resize(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"n_envs must be >= 1, got {n}")
        self._spawn(n)
        grow = n - len(self._frames)
        if grow > 0:
            self._frames += [None] * grow
            self._camp_heroes += [[] for _ in range(grow)]
            self._steps = np.concatenate([self._steps, np.zeros(grow, dtype=np.int64)])
            self._finished = np.concatenate([self._finished, np.ones(grow, dtype=bool)])
            self._new = np.concatenate([self._new, np.zeros(grow, dtype=bool)])
        else:
            self._frames, self._camp_heroes = self._frames[:n], self._camp_heroes[:n]
            self._steps, self._finished, self._new = self._steps[:n], self._finished[:n], self._new[:n]
        self._n = n

    # -- observation ----------------------------------------------------------------------

    def obs_schema(self):
        return {"features": ((FEATURES,), torch.float32)}

    def observe(self) -> VecObs:
        R = self.num_rows
        feats = np.zeros((R, FEATURES), dtype=np.float32)
        legal = np.zeros((R, 172), dtype=np.float32)
        sub = np.zeros((R, BUTTONS, 6), dtype=np.float32)
        for e, fr in enumerate(self._frames):
            if fr is None:
                continue
            for seat, p in enumerate(fr["players"]):
                r = 2 * e + seat
                feats[r], legal[r], sub[r] = p["features"], p["legal"], p["sub"]
        lg = {}
        for k, (a, b) in _LEGAL.items():
            m = torch.from_numpy(legal[:, a:b] > 0.5)
            lg[k] = m.reshape(R, BUTTONS, TARGETS) if k == "target" else m
        lg["sub"] = torch.from_numpy(sub > 0.5)
        active = torch.from_numpy(np.repeat(~self._finished, 2))
        seats = torch.arange(R) % 2
        new = torch.from_numpy(np.repeat(self._new, 2))
        return VecObs(
            obs={"features": torch.from_numpy(feats)}, legal=lg, to_play=seats,
            env_id=self._row_env(), stream_id=torch.arange(R),
            step_idx=torch.from_numpy(np.repeat(self._steps, 2)), new_episode=new,
            active=active, forced=torch.zeros(R, dtype=torch.bool),
            stream_seats=torch.nn.functional.one_hot(seats, 2).bool())

    # -- stepping ------------------------------------------------------------------------

    def step(self, actions: Mapping[str, torch.Tensor]) -> StepResult:
        acts = self.replay_actions(actions)
        keys = [f"{h}_idx" for h in _HEADS]
        self._require(acts, keys)
        a = torch.stack([acts[k].reshape(-1).long().cpu() for k in keys], dim=1).numpy()
        live = [e for e in range(self._n) if not self._finished[e]]
        for e in live:
            self._workers[e].send(("step", [a[2 * e].tolist(), a[2 * e + 1].tolist()]))
        rewards = torch.zeros(self._n, 2)
        done = torch.zeros(self._n, dtype=torch.bool)
        terminated = torch.zeros(self._n, dtype=torch.bool)
        truncated = torch.zeros(self._n, dtype=torch.bool)
        winner = torch.full((self._n,), -1, dtype=torch.long)
        self._new[:] = False
        for e in live:
            fr = self._workers[e].recv()
            self._frames[e] = fr
            self._steps[e] += 1
            for seat, p in enumerate(fr["players"]):
                rewards[e, seat] = self.shaping * (p["reward"][-1] if p["reward"] else 0.0)
            if fr["done"]:
                terminated[e] = True
                if fr["loss_camp"] != -1:
                    win = [c != fr["loss_camp"] for c in fr["camps"]]
                    for seat in range(2):
                        rewards[e, seat] += self.outcome * (1.0 if win[seat] else -1.0)
                    winner[e] = int(np.argmax(win))
            elif self.max_steps is not None and self._steps[e] >= self.max_steps:
                truncated[e] = True
            if terminated[e] or truncated[e]:
                done[e] = True
                self._workers[e].send(("end", None))
                self._workers[e].recv()
                self._finished[e] = True
        if self.autoreset and bool(done.any()):
            self.reset(env_ids=[int(e) for e in torch.nonzero(done).flatten()])
        info = {"terminated": terminated, "truncated": truncated, "winner": winner,
                "applied": {k: torch.from_numpy(a[:, j]) for j, k in enumerate(keys)}}
        return self.observe(), rewards, done, info

    def random_actions(self, obs: VecObs, generator: Optional[torch.Generator] = None
                       ) -> Dict[str, torch.Tensor]:
        g = self._generator(generator)
        R = obs.n_rows
        lg = obs.legal

        def pick(mask: torch.Tensor) -> torch.Tensor:
            w = mask.float() + 1e-9 * (~mask.any(-1, keepdim=True)).float()
            return torch.multinomial(w, 1, generator=g).squeeze(-1)

        button = pick(lg["button"])
        out = {"button_idx": button}
        for k in ("move_x", "move_z", "skill_x", "skill_z"):
            out[f"{k}_idx"] = pick(lg[k])
        out["target_idx"] = pick(lg["target"][torch.arange(R), button])
        return out
