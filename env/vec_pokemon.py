"""Pokémon Showdown battles as a :class:`~env.vec_game.VecGameEnv`: N battles run by
Showdown's own simulator (``env/pokemon_worker.js``: a few Node processes hosting the
battles, the Showdown checkout of the vocabulary's commit), observed through the converter's
parser (gmsim ``pokemon_live_frame`` = data_processing/rs/gmpokemon ``live_frame``): a
request's frame is the frame the IL converter writes for that request once the battle's log
is complete (tests/test_pokemon_sim.py converts simulated battles' logs and compares).

Rows / streams (per-seat, env/vec_gmsim.py's convention): row ``env * 2 + side``, active
while the side has a request that is not ``wait`` (both sides act on the same step at a turn;
one side on a forced switch; both at team preview).  Frames are the public view — the
spectator-log information the IL data has (a side's own moves / items appear once revealed);
the request's exact legality is the ``legal`` dict (model.games.pokemon.legal_from_request:
usable moves, trapping, tera, the switchable slots).

    obs     species_ids [12] long, move_ids [4] long, poke_feats [12, 18] f32,
            slot_mask [12] bool, field_seq [22] f32, slot_moves [12, 4] long,
            slot_item / slot_ability / slot_tera_type [12] long, slot_no_item [12] bool,
            format_id [] long   (converter v5)
    legal   action_type [2] bool, move_id [V] bool, switch [6] bool, tera [] bool,
            active_idx [] long (state), move_ids [4] long (state)
    action  action_type [R] (0 move / 1 switch), move_id [R], switch_idx [R], tera [R]
    privileged  oracle [R, 204] f32: both teams as they really are — every Pokémon of the
            row's own side, then of the foe's, from the sides' requests (:func:`team_oracle`, ``env.side_teams``);
            critic-only (train_rl_joint --privileged_critic pokemon), never the policy's input

Formats: ``gen9randombattle`` (Showdown's generator deals the teams; no team preview: an own
Pokémon enters the frame's slots when it first appears, so switching to an unrevealed one
picks the next in the request's order) and team formats with a pool of packed teams
(``teams``: a JSONL of {"packed": ...} — scripts/pokemon_teams.py builds validated pools,
e.g. metamon's high-ladder gen9ou set), both sides drawing a team per battle.  Reward: +1 /
−1 to the winner / loser when the battle ends (the IL value target), 0 on a truncation
(``max_turns``) or a tie.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from env.vec_game import StepResult, VecEnvBase, VecObs

SIDES = 2
TEAM = 6
FEAT = 18
FIELD = 22
MOVES = 4
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "env", "pokemon_worker.js")

OBS_SPEC = {  # name → (shape, dtype, bytes dtype)
    "species_ids": ((2 * TEAM,), torch.long, np.int64),
    "move_ids": ((MOVES,), torch.long, np.int64),
    "poke_feats": ((2 * TEAM, FEAT), torch.float32, np.float32),
    "slot_mask": ((2 * TEAM,), torch.bool, np.uint8),
    "field_seq": ((FIELD,), torch.float32, np.float32),
    "slot_moves": ((2 * TEAM, MOVES), torch.long, np.int64),
    "slot_item": ((2 * TEAM,), torch.long, np.int64),
    "slot_no_item": ((2 * TEAM,), torch.bool, np.uint8),
    "slot_ability": ((2 * TEAM,), torch.long, np.int64),
    "slot_tera_type": ((2 * TEAM,), torch.long, np.int64),
    "format_id": ((), torch.long, None),
}


def _phase(req: Mapping[str, Any]) -> str:
    if req.get("teamPreview"):
        return "preview"
    if any(bool(x) for x in (req.get("forceSwitch") or [])):
        return "forced"
    return "turn"


def _nick(ident: str) -> str:
    return ident.split(": ", 1)[1] if ": " in ident else ident


STATUSES = ("brn", "par", "slp", "frz", "psn", "tox")
MON_ORACLE = 17
ORACLE_DIM = SIDES * TEAM * MON_ORACLE


def mon_oracle(p: Mapping[str, Any]) -> List[float]:
    """One Pokémon of a side's request (``side.pokemon[k]``, Showdown's switch request data)
    as the critic's features: HP fraction, fainted, active, level / 100, max HP / 500, the
    five stats / 500, the status (one-hot), terastallized."""
    cond = str(p.get("condition") or "0 fnt").split()
    fainted = "fnt" in cond[1:] or cond[0] == "0"
    cur, _, mx = cond[0].partition("/")
    try:
        hp, max_hp = float(cur), float(mx or 0.0)
    except ValueError:
        hp, max_hp = 0.0, 0.0
    level = 100
    for part in str(p.get("details") or "").split(", ")[1:]:
        if part[:1] == "L" and part[1:].isdigit():
            level = int(part[1:])
    stats = p.get("stats") or {}
    status = cond[1] if len(cond) > 1 and cond[1] != "fnt" else ""
    return ([0.0 if fainted or max_hp <= 0 else hp / max_hp, float(fainted), float(bool(p.get("active"))),
             level / 100.0, max_hp / 500.0]
            + [float(stats.get(k, 0) or 0) / 500.0 for k in ("atk", "def", "spa", "spd", "spe")]
            + [float(status == x) for x in STATUSES] + [float(bool(p.get("terastallized")))])


def team_oracle(own: Optional[Sequence[Mapping[str, Any]]],
                foe: Optional[Sequence[Mapping[str, Any]]]) -> np.ndarray:
    """[ORACLE_DIM]: the row's own side's Pokémon (request order), then the foe's, each side
    padded to six; a side not known yet is zeros."""
    out = np.zeros((SIDES, TEAM, MON_ORACLE), dtype=np.float32)
    for k, team in enumerate((own, foe)):
        for j, p in enumerate(list(team or [])[:TEAM]):
            out[k, j] = mon_oracle(p)
    return out.reshape(-1)


def _base_species(details: str) -> str:
    return details.split(",")[0].strip()


def slot_map(request: Mapping[str, Any], slots: Sequence[Tuple[str, str]]) -> Dict[str, int]:
    """Request ``ident`` → the frame's slot: by nickname (the converter's slots are keyed by
    the log's nicknames), else by species (team preview's ``|poke|`` slots have no nickname
    yet), else — a Pokémon the log has not shown (random battles) — the next free slots in
    the request's order (where the converter will file it when it appears)."""
    by_nick = {n: i for i, (n, _) in enumerate(slots) if n}
    taken = set()
    out: Dict[str, int] = {}
    pokemon = (request.get("side") or {}).get("pokemon", [])
    later = []
    for p in pokemon:
        ident = p.get("ident", "")
        k = by_nick.get(_nick(ident))
        if k is None:
            sp = _base_species(p.get("details", ""))
            k = next((i for i, (n, s) in enumerate(slots) if i not in taken and not n and s
                      and (s == sp or s.rstrip("-*") == sp.split("-")[0] or sp.startswith(s.rstrip("*")))), None)
        if k is None:
            later.append(ident)
            continue
        taken.add(k)
        out[ident] = k
    free = [i for i in range(TEAM) if i not in taken and i not in out.values()]
    for ident, k in zip(later, free):
        out[ident] = k
    return out


def _auto_choice(req: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The choice the env makes itself: Revival Blessing's pick of a fainted Pokémon (a
    ``forceSwitch`` whose active is ``reviving``; the log writes no ``|switch|``, so the IL
    data has no frame for it) — the first fainted Pokémon in the request's order."""
    if not req or req.get("wait") or not any(bool(x) for x in (req.get("forceSwitch") or [])):
        return None
    pokemon = (req.get("side") or {}).get("pokemon", [])
    if not any(p.get("active") and p.get("reviving") for p in pokemon):
        return None
    for j, p in enumerate(pokemon):
        if str(p.get("condition", "")).strip().endswith("fnt"):
            return f"switch {j + 1}"
    return None


def _decides(req: Optional[Mapping[str, Any]]) -> bool:
    """The request is a policy decision (not waiting, not answered by the env itself)."""
    return bool(req) and not req.get("wait") and _auto_choice(req) is None


def _trapped_if_maybe(req: Mapping[str, Any]) -> Mapping[str, Any]:
    """A request whose active Pokémon is ``maybeTrapped`` (an unrevealed trapping ability of
    the foe is possible) as trapped: Showdown refuses the switch if it is — the env keeps to
    choices that cannot be refused."""
    act = (req.get("active") or [None])[0]
    if not act or not act.get("maybeTrapped") or act.get("trapped"):
        return req
    return {**req, "active": [{**act, "trapped": True}] + list(req["active"][1:])}


class _Worker:
    def __init__(self, showdown_dir: Optional[str]):
        env = dict(os.environ)
        if showdown_dir:
            env["GM_SHOWDOWN_DIR"] = showdown_dir
        self.proc = subprocess.Popen(["node", WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, env=env, bufsize=1)

    def call(self, cmd: Mapping[str, Any]) -> Dict[str, Any]:
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"pokemon worker died: {self.proc.stderr.read()[-2000:]}")
        return json.loads(line)

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.terminate()


class PokemonVecEnv(VecEnvBase):
    """N Pokémon Showdown battles (see the module docstring)."""

    ACTION_KEYS = {"action_type": "action_type", "move_id": "move_id", "switch_idx": "switch_idx",
                   "tera": "tera"}
    rows_per_env = SIDES
    lockstep = False
    n_seats = SIDES

    def __init__(self, n_envs: int = 1, *, format: str = "gen9randombattle", teams: Optional[str] = None,
                 autoreset: bool = False, seed: Optional[int] = 0, max_turns: int = 300,
                 workers: Optional[int] = None, showdown_dir: Optional[str] = None):
        super().__init__(n_envs, autoreset, seed)
        from data_processing.games.pokemon import FORMATS
        from model.games.pokemon import PokemonConfig

        self.format = "".join(c for c in format.lower() if c.isalnum())
        if self.format not in FORMATS:
            raise ValueError(f"pokemon: format {format!r} is not one of {[f for f in FORMATS if f]}")
        self.random_teams = "random" in self.format
        self.team_pool: List[str] = []
        if not self.random_teams:
            if not teams:
                raise ValueError(f"pokemon: {self.format} needs a team pool (teams=<jsonl of packed teams>)")
            with open(teams) as fh:
                self.team_pool = [json.loads(l)["packed"] for l in fh if l.strip()]
            if not self.team_pool:
                raise ValueError(f"pokemon: no team in {teams}")
        self.max_turns = int(max_turns)
        self.cfg = PokemonConfig()                      # the exact vocabulary's id scheme
        n_workers = int(workers or os.environ.get("GM_POKEMON_WORKERS", "1"))
        self._workers = [_Worker(showdown_dir) for _ in range(max(1, n_workers))]
        self._team_rng = random.Random(0 if seed is None else seed)
        self._alloc(self._n)
        self.reset()

    # ------------------------------------------------------------------

    def _alloc(self, n: int) -> None:
        self._n = n
        self.logs: List[List[str]] = [[] for _ in range(n)]
        self.reqs: List[List[Optional[dict]]] = [[None, None] for _ in range(n)]
        #: each side's latest team (its request's ``side.pokemon``; the privileged oracle) —
        #: not ``teams``: that name is the team-game protocol (env/league.py: seat → team id)
        self.side_teams: List[List[Optional[list]]] = [[None, None] for _ in range(n)]
        self.done = np.ones(n, dtype=bool)
        self.frames = np.zeros((n, SIDES), dtype=np.int64)
        self.fresh = np.zeros(n, dtype=bool)
        self.turns = np.zeros(n, dtype=np.int64)
        self.steps = np.zeros(n, dtype=np.int64)
        self._ids = list(range(n))
        self._next_id = n
        self._slots: List[List[Dict[str, int]]] = [[{}, {}] for _ in range(n)]
        self._cache: Dict[Tuple[int, int], Any] = {}

    def _worker(self, i: int) -> _Worker:
        return self._workers[i % len(self._workers)]

    def _start(self, ids: Sequence[int], seeds: Sequence[int]) -> None:
        by_worker: Dict[int, List[dict]] = {}
        for i, sd in zip(ids, seeds):
            bid = self._next_id
            self._next_id += 1
            self._ids[i] = bid
            teams = None if self.random_teams else [self._team_rng.choice(self.team_pool) for _ in range(2)]
            s = int(sd) & 0xFFFFFFFF
            cmd = {"op": "start", "id": bid, "format": self.format,
                   "seed": [s & 0xFFFF, (s >> 16) & 0xFFFF, (s * 7 + 3) & 0xFFFF, (s * 13 + 5) & 0xFFFF], "teams": teams}
            by_worker.setdefault(i % len(self._workers), []).append((i, cmd))
            self.logs[i] = []
            self.side_teams[i] = [None, None]
            self.done[i] = False
            self.frames[i] = 0
            self.fresh[i] = True
            self.turns[i] = 0
            self.steps[i] = 0
        for w, items in by_worker.items():
            res = self._workers[w].call({"op": "batch", "cmds": [c for _, c in items]})["results"]
            for (i, _), r in zip(items, res):
                self._absorb(i, r)
        self._auto_resolve()

    def _auto_resolve(self) -> None:
        """Answer the requests the env answers itself (:func:`_auto_choice`) until every
        running battle waits for a policy decision or has ended."""
        for _ in range(64):
            pending: Dict[int, List[Tuple[int, dict]]] = {}
            for i in range(self._n):
                if self.done[i] or self._ended.get(i, (False, None))[0]:
                    continue
                auto = [_auto_choice(r) for r in self.reqs[i]]
                if any(a is not None for a in auto):
                    pending.setdefault(i % len(self._workers), []).append(
                        (i, {"op": "choose", "id": self._ids[i], "choices": auto}))
            if not pending:
                return
            for w, items in pending.items():
                res = self._workers[w].call({"op": "batch", "cmds": [c for _, c in items]})["results"]
                for (i, _), r in zip(items, res):
                    self._absorb(i, r)
                    if r.get("error"):
                        raise RuntimeError(f"pokemon battle {self._ids[i]}: {r['error']}")
        raise RuntimeError("pokemon: automatic choices do not settle")

    def _absorb(self, i: int, r: Mapping[str, Any]) -> None:
        if r.get("error") and "refused" not in r["error"] and "without a request" not in r["error"]:
            raise RuntimeError(f"pokemon battle {self._ids[i]}: {r['error']}")
        self.last_error = r.get("error")
        self.logs[i].extend(r.get("log", []))
        self.reqs[i] = list(r.get("requests") or [None, None])
        for side, req in enumerate(self.reqs[i]):
            team = ((req or {}).get("side") or {}).get("pokemon")
            if team:
                self.side_teams[i][side] = team
        self.turns[i] = int(r.get("turn") or 0)
        self._ended = getattr(self, "_ended", {})
        self._ended[i] = (bool(r.get("ended")), r.get("winner"))
        self._cache = {k: v for k, v in self._cache.items() if k[0] != i}

    @property
    def finished(self) -> torch.Tensor:
        return torch.from_numpy(self.done.copy())

    def reset(self, n: Optional[int] = None, *, seed: Optional[int] = None,
              env_ids: Optional[Sequence[int]] = None) -> VecObs:
        if seed is not None:
            self._seed_rngs(seed)
            self._team_rng.seed(seed)
        if n is not None and n != self._n:
            if env_ids is not None:
                raise ValueError("resize (n) and env_ids are exclusive")
            self._end_all()
            self._alloc(int(n))
        ids = self._env_ids(env_ids)
        self._start(ids.tolist(), self._episode_seeds(len(ids), seed))
        return self.observe()

    def _end_all(self):
        for i in range(self._n):
            try:
                self._worker(i).call({"op": "end", "id": self._ids[i]})
            except Exception:
                pass

    def close(self) -> None:
        for w in self._workers:
            w.close()

    def obs_schema(self) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        return {k: (shape, dt) for k, (shape, dt, _) in OBS_SPEC.items()}

    # ------------------------------------------------------------------

    def _row_frame(self, i: int, side: int):
        key = (i, side, len(self.logs[i]))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        from gmsim import _native

        req = self.reqs[i][side]
        d = _native.pokemon_live_frame("\n".join(self.logs[i]), side, _phase(req), self.format)
        slots = slot_map(req, d["slots"])
        self._slots[i][side] = slots
        self._cache[key] = (d, slots)
        return d, slots

    def observe(self) -> VecObs:
        from model.games.pokemon import legal_from_request

        n, R = self._n, self._n * SIDES
        obs = {k: torch.zeros((R,) + shape, dtype=dt) for k, (shape, dt, _) in OBS_SPEC.items()}
        V = int(self.cfg.pokemon_move_vocab)
        legal = {"action_type": torch.zeros(R, 2, dtype=torch.bool), "move_id": torch.zeros(R, V, dtype=torch.bool),
                 "switch": torch.zeros(R, TEAM, dtype=torch.bool), "tera": torch.zeros(R, dtype=torch.bool),
                 "active_idx": torch.zeros(R, dtype=torch.long), "move_ids": torch.zeros(R, MOVES, dtype=torch.long)}
        active = torch.zeros(R, dtype=torch.bool)
        forced = torch.zeros(R, dtype=torch.bool)
        oracle = np.zeros((R, ORACLE_DIM), dtype=np.float32)
        for i in range(n):
            if self.done[i]:
                continue
            for side in range(SIDES):
                req = self.reqs[i][side]
                if not _decides(req):
                    continue
                row = i * SIDES + side
                oracle[row] = team_oracle(self.side_teams[i][side], self.side_teams[i][1 - side])
                d, slots = self._row_frame(i, side)
                for k, (shape, dt, npdt) in OBS_SPEC.items():
                    if npdt is None:
                        obs[k][row] = int(d[k])
                    else:
                        obs[k][row] = torch.from_numpy(np.frombuffer(d[k], dtype=npdt).reshape(shape).copy()).to(dt)
                lg = legal_from_request(_trapped_if_maybe(req), slots, self.cfg, move_ids=obs["move_ids"][row])
                for k in legal:
                    legal[k][row] = lg[k][0]
                active[row] = True
                n_choices = int(lg["move_id"][0].sum()) * int(lg["action_type"][0, 0]) + int(lg["switch"][0].sum())
                forced[row] = n_choices == 1
        rows = torch.arange(R, dtype=torch.long)
        return VecObs(
            obs=obs, legal=legal, to_play=rows % SIDES, env_id=rows // SIDES, stream_id=rows,
            step_idx=torch.from_numpy(self.frames.reshape(-1).copy()),
            new_episode=torch.from_numpy(np.repeat(self.fresh, SIDES)),
            active=active, forced=forced,
            stream_seats=torch.nn.functional.one_hot(rows % SIDES, SIDES).bool(),
            privileged={"oracle": torch.from_numpy(oracle)},
        )

    @property
    def privileged_dim(self) -> int:
        """The width of ``VecObs.privileged["oracle"]`` (train_rl_joint --privileged_critic)."""
        return ORACLE_DIM

    # ------------------------------------------------------------------

    def _choice(self, i: int, side: int, a: Mapping[str, int]) -> str:
        """The Showdown choice of a (type, move id, switch slot, tera) action."""
        from data_processing.games.pokemon import VOCAB

        req = self.reqs[i][side]
        slots = self._slots[i][side]
        pokemon = (req.get("side") or {}).get("pokemon", [])
        pos_of = {k: j + 1 for j, p in enumerate(pokemon) for ident, k in slots.items() if p.get("ident") == ident}
        ph = _phase(req)
        if ph == "preview":
            return f"team {pos_of[int(a['switch_idx'])]}"
        if ph == "forced" or int(a["action_type"]) == 1:
            return f"switch {pos_of[int(a['switch_idx'])]}"
        act = (req.get("active") or [{}])[0]
        want = int(a["move_id"])
        usable = [(j, m) for j, m in enumerate(act.get("moves", [])) if not m.get("disabled")]
        j = next((j for j, m in usable if VOCAB.move_id(m.get("id") or m.get("move", "")) == want), None)
        if j is None:        # an UNK id: the first usable move the vocabulary does not know
            j = next((j for j, m in usable if VOCAB.move_id(m.get("id") or m.get("move", "")) >= self.cfg.pokemon_move_vocab
                      or VOCAB.move_id(m.get("id") or m.get("move", "")) == 1), usable[0][0] if usable else 0)
        tera = " terastallize" if int(a.get("tera", 0)) and act.get("canTerastallize") else ""
        return f"move {j + 1}{tera}"

    def step(self, actions: Mapping[str, torch.Tensor]) -> StepResult:
        acts = self.replay_actions(actions)
        self._require(acts, ("action_type", "move_id", "switch_idx"))
        cols = {k: acts[k].detach().cpu().reshape(-1).long().tolist() for k in ("action_type", "move_id", "switch_idx")}
        cols["tera"] = acts["tera"].detach().cpu().reshape(-1).long().tolist() if "tera" in acts else [0] * (self._n * SIDES)
        n = self._n
        rewards = torch.zeros(n, SIDES)
        done = torch.zeros(n, dtype=torch.bool)
        terminated = torch.zeros(n, dtype=torch.bool)
        truncated = torch.zeros(n, dtype=torch.bool)
        winner = torch.full((n,), -1, dtype=torch.long)
        episode_len = torch.zeros(n, dtype=torch.long)
        acted = torch.zeros(n * SIDES, dtype=torch.bool)
        by_worker: Dict[int, List[Tuple[int, dict]]] = {}
        for i in range(n):
            if self.done[i]:
                continue
            choices = [None, None]
            for side in range(SIDES):
                req = self.reqs[i][side]
                if not _decides(req):
                    continue
                row = i * SIDES + side
                choices[side] = self._choice(i, side, {k: cols[k][row] for k in cols})
                acted[row] = True
                self.frames[i, side] += 1
            self.fresh[i] = False
            self.steps[i] += 1
            by_worker.setdefault(i % len(self._workers), []).append((i, {"op": "choose", "id": self._ids[i], "choices": choices}))
        for w, items in by_worker.items():
            res = self._workers[w].call({"op": "batch", "cmds": [c for _, c in items]})["results"]
            for (i, _), r in zip(items, res):
                self._absorb(i, r)
                if r.get("error"):
                    raise RuntimeError(f"pokemon battle {self._ids[i]}: {r['error']}")
        self._auto_resolve()
        ended = []
        for i in range(n):
            if self.done[i]:
                continue
            fin, win = self._ended.get(i, (False, None))
            if not fin and self.turns[i] > self.max_turns:
                done[i] = truncated[i] = True
            elif fin:
                done[i] = terminated[i] = True
                if win is not None:
                    winner[i] = int(win)
                    rewards[i, int(win)] = 1.0
                    rewards[i, 1 - int(win)] = -1.0
            if done[i]:
                episode_len[i] = int(self.steps[i])
                self.done[i] = True
                ended.append(i)
        for i in ended:
            self._worker(i).call({"op": "end", "id": self._ids[i]})
        if self.autoreset and ended:
            self._start(ended, self._episode_seeds(len(ended), None))
        info: Dict[str, Any] = {"terminated": terminated, "truncated": truncated, "winner": winner,
                                "episode_len": episode_len, "acted": acted, "illegal": torch.zeros(n, dtype=torch.bool),
                                "applied": {k: torch.tensor(cols[k]) for k in ("action_type", "move_id", "switch_idx", "tera")}}
        return self.observe(), rewards, done, info

    def random_actions(self, obs: VecObs, generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """A uniformly random legal action per row of ``obs``."""
        gen = self._generator(generator)
        L = obs.legal
        R = L["action_type"].shape[0]
        t_mask = L["action_type"].float()
        t_mask[t_mask.sum(-1) == 0, 1] = 1.0
        t = torch.multinomial(t_mask, 1, generator=gen).squeeze(1)
        mv = L["move_id"].float()
        mv[mv.sum(-1) == 0, 0] = 1.0
        sw = L["switch"].float()
        sw[sw.sum(-1) == 0, 0] = 1.0
        return {"action_type": t, "move_id": torch.multinomial(mv, 1, generator=gen).squeeze(1),
                "switch_idx": torch.multinomial(sw, 1, generator=gen).squeeze(1),
                "tera": (torch.rand(R, generator=gen) < 0.1).long() & L["tera"].long()}
