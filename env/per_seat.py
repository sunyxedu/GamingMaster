"""Per-seat streams of an alternating game: the stream layout its IL data is trained in.

The records of the two-player perfect-information board games (chess, chess variants,
shogi, xiangqi, go, othello, gomoku) hold both movers' frames, each from its mover's
view; IL serves every record as TWO streams, one per mover
(``data_processing.games.dataset.SEAT_SPLIT_GAMES``): a stream is one player's own
decision frames, in one view throughout.  The simulators of these games emit the
interleaved layout (one row per game, the view switching with the mover every frame),
so a network stepped on their rows would carry, in its temporal state, frames it never
saw in a stream during IL — the other mover's, mirrored / rotated / colour-swapped.

:class:`PerSeatStreams` wraps such an env and serves the layout of the IL streams:
``n_seats`` rows per env, row ``env * n_seats + seat`` holding that seat's own decision
frames only — the layout of the hidden-information gmrules games (env/vec_gmsim.py,
per-seat), with the same conventions:

  * a row is ``active`` only when its seat is to act (one row per env and step);
  * ``step_idx`` counts the row's own frames (frame k of seat s's stream is the k-th
    frame s decided on, forced frames included — othello's forced passes are frames
    of the passing seat, as in its records);
  * ``new_episode`` flags every row of an env on the step its episode started; a
    seat that acts later starts its stream at ``step_idx`` 0;
  * ``stream_seats`` = the row's seat; ``to_play`` = the row's seat;
  * rows are not in lockstep (``lockstep = False``: the trainer's ``--actor stream``).

Rewards, ``done`` and the per-env ``info`` entries are the wrapped env's; ``info``'s
per-row entries (``acted``, ``applied``) are spread over the per-seat rows.  The frames
themselves are the wrapped env's, bit for bit: frame k of row (e, s) is the wrapped
env's k-th frame of env e with seat s to move, i.e. the IL seat stream of the same
game (``dataset.seat_stream``).

``make_vec_env`` applies the wrapper to the SEAT_SPLIT_GAMES by default
(``per_seat=False`` keeps the simulator's interleaved rows).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from env.vec_game import StepResult, VecObs

#: the wrapper's own attributes; every other attribute is the wrapped env's
_OWN = frozenset({"env", "n_seats", "rows_per_env", "_frames", "_cur"})


class PerSeatStreams:
    """An interleaved alternating env served as per-seat rows (module docstring)."""

    interleaved = False
    lockstep = False
    per_seat = True

    def __init__(self, env: Any):
        if not getattr(env, "interleaved", False) or int(env.rows_per_env) != 1:
            raise ValueError(f"{getattr(env, 'game', env)!r}: per-seat streams wrap an "
                             f"interleaved env (one row per game, one mover per frame)")
        object.__setattr__(self, "env", env)
        object.__setattr__(self, "n_seats", int(env.n_seats))
        object.__setattr__(self, "rows_per_env", int(env.n_seats))
        object.__setattr__(self, "_frames", torch.zeros(env.num_envs, self.n_seats,
                                                       dtype=torch.long))
        object.__setattr__(self, "_cur", env.observe())

    # every attribute the wrapper does not define is the wrapped env's (game, autoreset,
    # obs_schema, replay_actions, trajectory, deal_group, ...), reads and writes alike
    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "env"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    # the VecGameEnv protocol's members, spelled out (a runtime protocol check looks
    # attributes up statically, past __getattr__)
    @property
    def game(self) -> str:
        return self.env.game

    @property
    def autoreset(self) -> bool:
        return bool(self.env.autoreset)

    def obs_schema(self):
        return self.env.obs_schema()

    def replay_actions(self, actions: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.env.replay_actions(actions)

    # -- layout ------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return int(self.env.num_envs)

    @property
    def num_rows(self) -> int:
        return self.num_envs * self.n_seats

    @property
    def finished(self) -> torch.Tensor:
        return self.env.finished

    def _mover_rows(self, o: VecObs) -> torch.Tensor:
        """[N] long: the per-seat row of each env's current mover."""
        S = self.n_seats
        return torch.arange(o.n_rows, dtype=torch.long) * S + o.to_play.long().clamp(0, S - 1)

    def _view(self, o: VecObs) -> VecObs:
        """The wrapped env's frames (one row per env) on the per-seat rows."""
        N, S = o.n_rows, self.n_seats
        R = N * S
        act = o.active.bool()
        dst = self._mover_rows(o)[act]

        def spread(t: torch.Tensor) -> torch.Tensor:
            out = t.new_zeros((R,) + tuple(t.shape[1:]))
            out[dst] = t[act]
            return out

        rows = torch.arange(R, dtype=torch.long)
        seat = rows % S
        return VecObs(
            obs={k: spread(v) for k, v in o.obs.items()},
            legal={k: spread(v) for k, v in o.legal.items()},
            to_play=seat.clone(),
            env_id=rows // S,
            stream_id=rows,
            step_idx=self._frames.reshape(R).clone(),
            new_episode=o.new_episode.bool()[rows // S],
            active=spread(act),
            forced=spread(o.forced.bool() & act),
            stream_seats=torch.nn.functional.one_hot(seat, S).bool(),
            privileged=(None if o.privileged is None
                        else {k: spread(v) for k, v in o.privileged.items()}),
            aux={k: spread(v) for k, v in o.aux.items()},
        )

    def _adopt(self, o: VecObs, restarted: Optional[torch.Tensor] = None) -> VecObs:
        """Take the wrapped env's observation ``o``; ``restarted`` [N] bool: envs whose
        frame counts start over."""
        N = o.n_rows
        if self._frames.shape[0] != N:
            self._frames = torch.zeros(N, self.n_seats, dtype=torch.long)
        elif restarted is not None:
            self._frames[restarted.bool()] = 0
        self._cur = o
        return self._view(o)

    # -- episodes ------------------------------------------------------------

    def _ids_mask(self, n: int, env_ids: Optional[Sequence[int]]) -> torch.Tensor:
        mask = torch.zeros(n, dtype=torch.bool)
        if env_ids is None:
            mask[:] = True
        else:
            mask[torch.as_tensor(list(env_ids), dtype=torch.long)] = True
        return mask

    def reset(self, n: Optional[int] = None, *, seed: Optional[int] = None,
              env_ids: Optional[Sequence[int]] = None) -> VecObs:
        o = self.env.reset(n, seed=seed, env_ids=env_ids)
        return self._adopt(o, self._ids_mask(o.n_rows, env_ids))

    def reset_from(self, prefixes: Sequence[Any], *, env_ids: Optional[Sequence[int]] = None,
                   seed: Optional[int] = None) -> VecObs:
        """The wrapped env's ``reset_from`` (gmrules trajectories, (seat, id) events):
        every seat's frame count starts at the decisions it made in its prefix."""
        o = self.env.reset_from(prefixes, env_ids=env_ids, seed=seed)
        mask = self._ids_mask(o.n_rows, env_ids)
        view = self._adopt(o, mask)
        for e, pre in zip(mask.nonzero().squeeze(1).tolist(), prefixes):
            seats = np.asarray(pre, dtype=np.int64).reshape(-1, 2)[:, 0]
            for s in range(self.n_seats):
                self._frames[e, s] = int((seats == s).sum())
        return self._view(self._cur) if len(prefixes) else view

    def observe(self) -> VecObs:
        return self._adopt(self.env.observe())

    def step(self, actions: Mapping[str, torch.Tensor]) -> StepResult:
        """One action per per-seat row; each env takes its mover's row."""
        cur = self._cur
        N, R = cur.n_rows, cur.n_rows * self.n_seats
        src = self._mover_rows(cur)
        inner: Dict[str, torch.Tensor] = {}
        for k, v in self.env.replay_actions(actions).items():
            if v.shape[0] != R:
                raise ValueError(f"{self.env.game}: action {k!r} has {v.shape[0]} rows, "
                                 f"expected {R} (one per seat stream)")
            inner[k] = v[src.to(v.device)]
        acted_env = cur.active.bool()
        nxt, rew, done, info = self.env.step(inner)
        self._frames[acted_env, cur.to_play.long().clamp(0, self.n_seats - 1)[acted_env]] += 1
        out = dict(info)

        def spread(t: torch.Tensor) -> torch.Tensor:
            full = t.new_zeros((R,) + tuple(t.shape[1:]))
            full[src[acted_env]] = t[acted_env]
            return full

        if torch.is_tensor(info.get("acted")) and info["acted"].shape[0] == N:
            out["acted"] = spread(info["acted"])
        if isinstance(info.get("applied"), Mapping):
            out["applied"] = {k: (spread(v) if torch.is_tensor(v) and v.dim() >= 1
                                  and v.shape[0] == N else v)
                              for k, v in info["applied"].items()}
        # an env that restarted on this step (autoreset) starts its streams over
        return self._adopt(nxt, nxt.new_episode.bool()), rew, done, out

    # -- actions / state -------------------------------------------------------

    def random_actions(self, obs: VecObs,
                       generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """A uniformly random legal action per row of ``obs`` (its env's mover's; the
        rows of a ``VecObs.select`` subset keep their order)."""
        a = self.env.random_actions(self._cur, generator=generator)
        e = obs.env_id.long()
        return {k: v[e.to(v.device)] for k, v in a.items()}

    def state_dict(self) -> Dict[str, Any]:
        return {**self.env.state_dict(), "per_seat_frames": self._frames.clone()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        state = dict(state)
        frames = state.pop("per_seat_frames", None)
        self.env.load_state_dict(state)
        self._adopt(self.env.observe())
        if frames is not None and tuple(frames.shape) == tuple(self._frames.shape):
            self._frames.copy_(torch.as_tensor(frames, dtype=torch.long))
