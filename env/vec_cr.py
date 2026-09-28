"""Clash Royale as a :class:`~env.vec_game.VecGameEnv`: N battles of two seats in RoyaleSim
(github.com/RoyaleGym/RoyaleSim, MIT: a deterministic integer battle engine, its movement
rules measured on recorded battles, the live 2026 client's card table — Supercell's data,
research use), observed through crdata's own frame builder (gmsim's ``CrFrames`` =
``crdata::frames::SideFrames``) and expanded by data_processing/games/clash_royale/expand.py —
so an RL frame IS the IL frame of the same public record: the plays both sides made, their
cells and times, the elixir.  The replays carry no unit positions, and neither do the frames
(the engine keeps them to itself: its state decides the plays' legality and the result).

Rows ``env * 2 + seat`` (seat 0 = the engine's Blue = crdata's team, low y; seat 1 = Red =
its opponent); both act on every step (simultaneous, like the IL frames: one every
``step_ticks`` = 10 ticks = 0.5 s).  A decision is wait, or a deck token (card_slot) at a
cell of the own frame (place_idx = y · 18 + x, own back line y = 0) — the decoder's
``card_slot`` / ``place_idx`` / ``waited``.  Differences from the IL frames, both about the
own hand: the player knows it here, so the deck tokens' P(in hand) / P(next) are the engine's
hand (the IL belief is a posterior over it given the plays), and no card is legal before the
engine opens deploys (tick 90).

Reward at the end: +1 the winner, −1 the loser, 0 a draw (the IL value target).  Legal:
``card`` (a card in hand the frame's elixir pays for), ``placement`` (own half; the whole
arena for a spell) — expand.py's; a play the engine refuses (a cell outside the deploy
territory, a building that does not fit) is a wait, counted in ``info["refused"]``.

Decks: ``decks`` (lists of 8 catalog keys, e.g. from the IL data: scripts/cr_deck_pool.py) or
a small built-in meta set; decks with cards the engine cannot place yet (Mirror, Miner,
Goblin Drill) or evolutions / heroes (played as the base card here) are the caller's to
leave out.  The engine: ``import royalesim`` (built from a RoyaleSim checkout with maturin,
data/derived generated — its README, "Install").
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from env.vec_game import StepResult, VecEnvBase, VecObs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG = os.path.join(ROOT, "data_processing", "games", "clash_royale", "catalog")
GRID_W, GRID_H = 18, 32
STEP_TICKS = 10
RECENT = 8
#: the engine opens deploys at this tick (its TOO_EARLY before)
FIRST_DEPLOY_TICK = 90
#: crdata convert's shard meta (the expansion's parameters)
SHARD_META = {"arena_h": 32, "arena_w": 18, "arena_channels": 16, "hand_size": 4, "card_tokens": "deck",
              "deck_size": 8, "card_feature_dim": 9, "cr_global_features": 6, "step_ticks": STEP_TICKS,
              "recent": RECENT, "tau_ticks": 60.0}
#: meta decks for a start (catalog keys; all simulable)
DEFAULT_DECKS = (
    ("hog-rider", "musketeer", "cannon", "fireball", "the-log", "ice-spirit", "skeletons", "ice-golem"),
    ("golem", "night-witch", "baby-dragon", "lightning", "tornado", "mega-minion", "lumberjack", "zap"),
    ("x-bow", "tesla", "archers", "knight", "fireball", "the-log", "skeletons", "ice-spirit"),
    ("pekka", "battle-ram", "bandit", "electro-wizard", "royal-ghost", "poison", "zap", "magic-archer"),
    ("royal-giant", "fisherman", "hunter", "electro-spirit", "the-log", "fireball", "royal-ghost", "skeletons"),
    ("lava-hound", "balloon", "mega-minion", "minions", "skeleton-dragons", "arrows", "fireball", "tombstone"),
)
SCALARS = ("tick", "time_s", "phase", "overtime", "elixir", "opp_elixir")
RAW_KEYS = ("tick", "time_s", "phase", "overtime", "elixir", "opp_elixir", "deck_ids", "deck_form", "deck_kind",
            "deck_cost", "deck_in_hand", "deck_next", "recent_side", "recent_card", "recent_x", "recent_y", "recent_age")


class ClashRoyaleVecEnv(VecEnvBase):
    """N Clash Royale battles of two seats (see the module docstring)."""

    ACTION_KEYS = {"card_slot": "card_slot", "place_idx": "place_idx", "waited": "waited"}
    game = "clash_royale"
    n_seats = 2
    rows_per_env = 2
    interleaved = False

    def __init__(self, n_envs: int = 1, *, autoreset: bool = False, seed: Optional[int] = 0,
                 decks: Optional[Sequence[Sequence[str]]] = None, catalog: str = CATALOG,
                 max_ticks: int = 6000):
        super().__init__(n_envs, autoreset, seed)
        import royalesim
        from gmsim import _native

        from data_processing.games.clash_royale import expand as cr_expand

        self._rs, self._native, self._expand = royalesim, _native, cr_expand.expand
        self.catalog, self.max_ticks = catalog, int(max_ticks)
        with open(os.path.join(catalog, "cards.json")) as f:
            entries = json.load(f)["cards"]
        self._id_of_key = {e["key"]: int(e["id"]) for e in entries}
        self._name_of_id = {int(e["id"]): e["name"] for e in entries}
        self._cost_of_id = {cid: cost for cid, _, cost, _ in _native.cr_catalog_cards(catalog)}
        probe = royalesim.Battle(None, [[0, 1, 2], [0, 1, 2]])
        self._sim_cat = [c[0] for c in json.loads(probe.catalogue_json())]
        sim_idx = {n: i for i, n in enumerate(self._sim_cat)}
        self._sim_of_id = {cid: sim_idx[n] for cid, n in self._name_of_id.items() if n in sim_idx}
        self._id_of_sim = {s: c for c, s in self._sim_of_id.items()}
        self.decks = [tuple(d) for d in (decks or DEFAULT_DECKS)]
        for d in self.decks:
            bad = [k for k in d if self._id_of_key.get(k) not in self._sim_of_id]
            if len(d) != 8 or bad:
                raise ValueError(f"clash_royale deck {d}: 8 simulable cards needed (not simulable: {bad})")
        self._battles = [royalesim.Battle(None, [[0, 1, 2], [0, 1, 2]]) for _ in range(self._n)]
        self._frames: List[List[Any]] = [[None, None] for _ in range(self._n)]
        self._raw: List[List[Dict[str, list]]] = [[{}, {}] for _ in range(self._n)]
        self._obs: List[List[Dict[str, torch.Tensor]]] = [[{}, {}] for _ in range(self._n)]
        self._done = np.zeros(self._n, dtype=bool)
        self._new = np.ones(self._n, dtype=bool)
        self._steps = np.zeros(self._n, dtype=np.int64)
        self._deck_of = [[None, None] for _ in range(self._n)]
        self._decks_of: List[Any] = [None] * self._n
        self._plays: List[List[tuple]] = [[] for _ in range(self._n)]
        self._key_of_id = {v: k for k, v in self._id_of_key.items()}
        self.reset()

    # ------------------------------------------------------------------

    def reset(self, n: Optional[int] = None, *, seed: Optional[int] = None,
              env_ids: Optional[Sequence[int]] = None) -> VecObs:
        if seed is not None:
            self._seed_rngs(seed)
        if n is not None and n != self._n:
            raise ValueError("clash_royale: the number of battles is fixed at construction")
        ids = self._env_ids(env_ids)
        for i, s in zip(ids.tolist(), self._episode_seeds(len(ids), seed)):
            self._start(i, int(s))
        return self.observe()

    def _start(self, i: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        decks = [self.decks[int(rng.integers(len(self.decks)))] for _ in range(2)]
        ids = [[self._id_of_key[k] for k in d] for d in decks]
        self._battles[i].reset(seed, [[self._sim_of_id[c] for c in d] for d in ids], 1, 0, None, None, [])
        self._decks_of[i] = decks
        self._plays[i] = []
        for seat in range(2):
            self._frames[i][seat] = self._native.CrFrames(self.catalog, ids[seat], [0] * 8, seat, STEP_TICKS, RECENT)
            self._raw[i][seat] = {k: [] for k in RAW_KEYS + ("recent_cost",)}
            self._deck_of[i][seat] = ids[seat]
        self._done[i] = False
        self._new[i] = True
        self._steps[i] = 0
        self._refresh(i)

    def _state(self, i: int) -> Dict[str, Any]:
        return json.loads(self._battles[i].state_json())

    def _refresh(self, i: int) -> None:
        """Both seats' observation at the battle's current tick."""
        st = self._state(i)
        tick = int(st["tick"])
        for seat in range(2):
            fb = self._frames[i][seat]
            # u8 columns cross as bytes; one row: the per-frame scalars unwrapped
            row = {k: (list(v) if isinstance(v, (bytes, bytearray)) else v) for k, v in fb.frame(tick).items()}
            for k in SCALARS:
                row[k] = row[k][0]
            # the own hand: the engine's (the player sees it)
            hand = {self._id_of_sim[c] for c in st["players"][seat]["hand"]}
            nxt = self._id_of_sim.get(st["players"][seat]["next_card"])
            tok = fb.tok_ids
            row["deck_in_hand"] = [float(c in hand) for c in tok]
            row["deck_next"] = [float(c == nxt) for c in tok]
            row["recent_cost"] = [float(self._cost_of_id.get(c, 0.0)) for c in row["recent_card"]]
            raw = self._raw[i][seat]
            for k, v in row.items():
                raw[k].append(v)
            arr = {k: np.asarray(v) for k, v in raw.items()}
            obs = self._expand(arr, SHARD_META)
            last = {k: v[-1] for k, v in obs.items()}
            if tick < FIRST_DEPLOY_TICK:
                last["legal_card"] = torch.zeros_like(last["legal_card"])
            self._obs[i][seat] = last

    def obs_schema(self):
        o = self._obs[0][0]
        return {k: (tuple(v.shape), v.dtype) for k, v in o.items() if k not in ("legal_card", "legal_place")}

    def legal_schema(self):
        o = self._obs[0][0]
        return {"card": (tuple(o["legal_card"].shape), torch.bool), "placement": (tuple(o["legal_place"].shape), torch.bool)}

    # ------------------------------------------------------------------

    def observe(self) -> VecObs:
        n = self._n
        rows = [self._obs[i][s] for i in range(n) for s in range(2)]
        keys = [k for k in rows[0] if k not in ("legal_card", "legal_place")]
        obs = {k: torch.stack([r[k] for r in rows]) for k in keys}
        legal = {"card": torch.stack([r["legal_card"] for r in rows]).bool(),
                 "placement": torch.stack([r["legal_place"] for r in rows]).bool()}
        active = torch.from_numpy(np.repeat(~self._done, 2))
        legal["card"] &= active[:, None]
        R = 2 * n
        return VecObs(obs=obs, legal=legal, to_play=torch.arange(R) % 2, env_id=torch.arange(R) // 2,
                      stream_id=torch.arange(R), step_idx=torch.from_numpy(np.repeat(self._steps, 2)),
                      new_episode=torch.from_numpy(np.repeat(self._new, 2)), active=active,
                      forced=torch.zeros(R, dtype=torch.bool),
                      stream_seats=torch.nn.functional.one_hot(torch.arange(R) % 2, 2).bool())

    def step(self, actions: Mapping[str, torch.Tensor]) -> StepResult:
        acts = self.replay_actions(actions)
        self._require(acts, ("card_slot", "place_idx", "waited"))
        n, S = self._n, self._rs.SUBTILE
        card_slot = acts["card_slot"].reshape(-1).cpu().numpy().astype(np.int64)
        place = acts["place_idx"].reshape(-1).cpu().numpy().astype(np.int64)
        waited = acts["waited"].reshape(-1).cpu().numpy().astype(bool)
        rewards = np.zeros((n, 2), dtype=np.float32)
        done = np.zeros(n, dtype=bool)
        refused = np.zeros(2 * n, dtype=bool)
        played = np.zeros(2 * n, dtype=bool)
        was_active = ~self._done
        for i in np.flatnonzero(was_active).tolist():
            st = self._state(i)
            cmds, who = [], []
            for seat in range(2):
                r = 2 * i + seat
                legal_card = self._obs[i][seat]["legal_card"]
                k = int(card_slot[r])
                if waited[r] or not (0 <= k < 8) or not bool(legal_card[k]):
                    continue
                card = self._frames[i][seat].tok_ids[k]
                hand = st["players"][seat]["hand"]
                slot = hand.index(self._sim_of_id[card]) if self._sim_of_id[card] in hand else -1
                if slot < 0:
                    refused[r] = True
                    continue
                px, py = int(place[r]) % GRID_W, int(place[r]) // GRID_W
                gx, gy = (px, py) if seat == 0 else (GRID_W - 1 - px, GRID_H - 1 - py)
                cmds.append((seat, slot, gx * S + S // 2, gy * S + S // 2))
                who.append((seat, card))
            outs = self._battles[i].step(cmds, STEP_TICKS)
            for (seat, card), o in zip(who, outs):
                if self._rs.DEPLOY_REASONS[o[1]] != "OK":
                    refused[2 * i + seat] = True
                    continue
                played[2 * i + seat] = True
                tick, x, y = int(o[2]), int(o[3]), int(o[4])
                gx, gy = min(max(x // S, 0), GRID_W - 1), min(max(y // S, 0), GRID_H - 1)
                for view in range(2):
                    self._frames[i][view].observe_play(seat, card, 0, tick, gx, gy)
                self._plays[i].append(("team" if seat == 0 else "opponent", self._key_of_id[card], tick, gx, gy))
            self._steps[i] += 1
            self._new[i] = False
            st = self._state(i)
            if st["game_over"] or int(st["tick"]) >= self.max_ticks:
                w = int(st["winner"])
                if w in (0, 1):
                    rewards[i, w], rewards[i, 1 - w] = 1.0, -1.0
                done[i] = True
                if self.autoreset:
                    self._start(i, int(self._episode_seeds(1, None)[0]))
                else:
                    self._done[i] = True
            else:
                self._refresh(i)
        info: Dict[str, Any] = {
            "terminated": torch.from_numpy(done.copy()),
            "truncated": torch.zeros(n, dtype=torch.bool),
            "winner": torch.from_numpy(np.where(done, rewards.argmax(1), -1).astype(np.int64)),
            "episode_len": torch.from_numpy(np.where(done, self._steps, 0)),
            "acted": torch.from_numpy(np.repeat(was_active, 2)),
            "applied": {k: v for k, v in acts.items()},
            "illegal": torch.zeros(n, dtype=torch.bool),
            "refused": torch.from_numpy(refused),
            "played": torch.from_numpy(played),
        }
        return self.observe(), torch.from_numpy(rewards), torch.from_numpy(done), info

    def export_replay(self, i: int) -> str:
        """Battle ``i`` so far as a RoyaleAPI replay payload (``crdata convert``'s input: the
        decks, the plays with their ticks and team-frame cells, the result)."""
        st = self._state(i)
        crowns = [p["crowns"] for p in st["players"]]
        w = int(st["winner"]) if st["game_over"] else -1
        result = {0: "victory", 1: "defeat"}.get(w, "draw")
        deck = lambda d: [{"card_key": k, "level": 11, "name": k} for k in d]
        events = [{"kind": "play_card", "side": side, "card_key": key, "card_play_number": 1, "replay_tick_20hz": tick,
                   "time_seconds": tick / 20.0, "form_at_play": "unknown", "deck_card_key_candidates": [key],
                   "ability_source_candidates": [], "coordinates": {"grid_cell_floor": {"x": gx, "y": gy},
                                                                   "inside_18x32_arena": True}, "source_index": j}
                  for j, (side, key, tick, gx, gy) in enumerate(self._plays[i])]
        tower = {"card_key": "tower-princess", "level": 11, "name": "Tower Princess"}
        dur = int(st["tick"]) / 20.0
        return json.dumps({"battle": {"battle_type": "pathOfLegend", "game_mode": "Ranked", "result": result,
                                      "team": {"crowns": crowns[0], "players": [{"deck": deck(self._decks_of[i][0]),
                                                                                "tower_card": tower}]},
                                      "opponent": {"crowns": crowns[1], "players": [{"deck": deck(self._decks_of[i][1]),
                                                                                    "tower_card": tower}]}},
                           "events": events, "replay": {"duration": {"timeline_seconds": dur, "display_seconds": dur}},
                           "schema_version": "royaleapi-battle-replay.v1", "source": {"site": "RoyaleSim"}})

    def random_actions(self, obs: VecObs, generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        gen = self._generator(generator)
        R = obs.legal["card"].shape[0]
        card = obs.legal["card"].float()
        can = card.sum(-1) > 0
        card[~can, 0] = 1.0
        slot = torch.multinomial(card, 1, generator=gen).squeeze(1)
        place = torch.multinomial(obs.legal["placement"].float() + 1e-9, 1, generator=gen).squeeze(1)
        waited = ~can | (torch.rand(R, generator=gen) < 0.8)
        return {"card_slot": slot, "place_idx": place, "waited": waited}
