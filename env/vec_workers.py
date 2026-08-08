"""Persistent env-worker pool for the vectorized rollout collector.

The vectorized collector (collect_selfplay_episodes_vec) removed the
batch-1 inference bottleneck, which exposed the next one: N envs stepped
serially in the rank process cost N * ~5-10ms of pure Python per lockstep
timestep. This pool spreads the envs over W persistent worker processes,
stepping them in parallel while the rank process keeps env slot 0 local
(replay recording needs get_state_snapshot/_player_stats access).

Design notes:
- ``spawn`` start method: rank processes hold CUDA contexts, fork is
  unsafe. Workers import only env code and torch CPU tensors.
- Workers build their own envs (and load the map pool themselves, in
  parallel) from a plain kwargs/attrs description — env objects never
  cross the pipe.
- Per timestep each worker receives its slots' actions and returns
  pre-stacked observation tensors per seat plus per-env scalars; the
  parent concatenates a fixed slot order, so batch rows (and therefore
  step-cache rows) stay aligned across steps.
- The protocol mirrors the collector's needs exactly: step results carry
  (rewards, done, slim info, valid_directions per seat computed AFTER the
  step — same order as the sequential path) and a truncation_score RPC
  covers budget-exhausted games.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

_INFO_KEYS = ("turn", "illegal", "winner", "reward_breakdown")


def _obs_to_numpy(obs_pair):
    """Torch tensors must not cross the pipe: torch's ForkingPickler ships
    storages by file-descriptor sharing, and at 64 envs x ~12 tensors per
    step that exhausts the process fd budget within seconds
    ("received 0 items of ancdata"). Numpy arrays pickle by value."""
    return {pid: {k: (v.numpy() if isinstance(v, torch.Tensor) else v)
                  for k, v in obs_pair[pid].items()}
            for pid in (0, 1)}


def _obs_to_torch(obs_pair):
    import numpy as np
    return {pid: {k: (torch.from_numpy(v) if isinstance(v, np.ndarray)
                      else v)
                  for k, v in obs_pair[pid].items()}
            for pid in (0, 1)}


def _build_env(env_kwargs: Dict[str, Any], env_attrs: Dict[str, Any],
               map_pool_src: Optional[Tuple[str, str]]):
    from env.generals_env import GeneralsEnv

    env = GeneralsEnv(**env_kwargs)
    for k, v in env_attrs.items():
        setattr(env, k, v)
    if map_pool_src is not None:
        env.load_map_pool(*map_pool_src)
    return env


def _slim_info(info: Dict[str, Any]) -> Dict[str, Any]:
    return {k: info[k] for k in _INFO_KEYS if k in info}


def _worker_main(conn, slots: Sequence[int], env_kwargs: Dict[str, Any],
                 env_attrs: Dict[str, Any],
                 map_pool_src: Optional[Tuple[str, str]]) -> None:
    torch.set_num_threads(1)
    envs = {s: None for s in slots}
    first = _build_env(env_kwargs, env_attrs, map_pool_src)
    envs[slots[0]] = first
    for s in slots[1:]:
        e = _build_env(env_kwargs, env_attrs, None)
        e.map_pool = first.map_pool  # share the parsed pool within a worker
        envs[s] = e

    while True:
        msg = conn.recv()
        cmd = msg[0]
        if cmd == "reset":
            out = {}
            for s in slots:
                out[s] = _obs_to_numpy(envs[s].reset())
            conn.send(out)
        elif cmd == "step":
            # msg[1]: {slot: (a0, a1, src0, src1)} — src is the sampled
            # source_idx for the post-step valid-direction lookup, present
            # even when the env action is None (pass / no legal move).
            out = []
            for s, (a0, a1, src0, src1) in msg[1].items():
                new_obs, rewards, done, info = envs[s].step(a0, a1)
                vd = (envs[s].get_valid_directions(src0),
                      envs[s].get_valid_directions(src1))
                # rewards is a {pid: float} dict — keep pid indexing
                out.append((s, _obs_to_numpy(new_obs),
                            (rewards[0], rewards[1]), done,
                            _slim_info(info), vd))
            conn.send(out)
        elif cmd == "trunc":
            s = msg[1]
            conn.send((envs[s].truncation_score(0),
                       envs[s].truncation_score(1)))
        elif cmd == "close":
            conn.close()
            return


class EnvWorkerPool:
    """W persistent processes owning env slots ``1..n_envs-1``.

    Slot 0 stays in the parent (the collector records replays from it).
    The parent-facing API is slot-oriented; all pipe traffic is batched
    per worker per timestep.
    """

    def __init__(self, n_envs: int, n_workers: int,
                 env_kwargs: Dict[str, Any], env_attrs: Dict[str, Any],
                 map_pool_src: Optional[Tuple[str, str]] = None):
        if n_envs < 2:
            raise ValueError("EnvWorkerPool needs n_envs >= 2 "
                             "(slot 0 lives in the parent)")
        self.n_envs = n_envs
        remote = list(range(1, n_envs))
        n_workers = max(1, min(n_workers, len(remote)))
        self._slot_groups: List[List[int]] = [
            remote[i::n_workers] for i in range(n_workers)]
        self._slot_to_worker: Dict[int, int] = {
            s: w for w, group in enumerate(self._slot_groups) for s in group}
        ctx = mp.get_context("spawn")
        self._conns = []
        self._procs = []
        for group in self._slot_groups:
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_worker_main,
                args=(child_conn, group, env_kwargs, env_attrs,
                      map_pool_src),
                daemon=True,
            )
            p.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._procs.append(p)

    def reset_all(self) -> Dict[int, Dict[int, Dict[str, torch.Tensor]]]:
        """Reset every remote env; returns {slot: obs_pair}."""
        for c in self._conns:
            c.send(("reset",))
        out: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        for c in self._conns:
            out.update({s: _obs_to_torch(pair)
                        for s, pair in c.recv().items()})
        return out

    def step(
        self,
        actions: Dict[int, Tuple[Optional[dict], Optional[dict], int, int]],
    ) -> Dict[int, tuple]:
        """Step the given slots. ``actions[slot] = (a0, a1, src0, src1)``.

        Returns {slot: (obs_pair, rewards, done, info, (vd0, vd1))}.
        """
        per_worker: Dict[int, Dict[int, tuple]] = {}
        for s, payload in actions.items():
            per_worker.setdefault(self._slot_to_worker[s], {})[s] = payload
        for w, batch in per_worker.items():
            self._conns[w].send(("step", batch))
        out: Dict[int, tuple] = {}
        for w in per_worker:
            for s, obs, rewards, done, info, vd in self._conns[w].recv():
                out[s] = (_obs_to_torch(obs), rewards, done, info, vd)
        return out

    def truncation_scores(self, slot: int) -> Tuple[float, float]:
        w = self._slot_to_worker[slot]
        self._conns[w].send(("trunc", slot))
        return self._conns[w].recv()

    def close(self) -> None:
        for c in self._conns:
            try:
                c.send(("close",))
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
