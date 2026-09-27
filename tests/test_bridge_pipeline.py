"""Contract bridge end to end through every layer.  The gmrules bridge rules are the
converter's AND the simulator's code (OpenSpiel parity: tests/test_bridge_reference.py):

    real records (WBridge5 lines, Ben WBC boards, PBN, BBO LIN) ─► gmdata bridge convert
      (replay + verification of every record) ─► gm-shards ─► CandidateEncoder /
      CandidateDecoder ─► train_multi_game (with othello) ─► gmsim env
      ─► train_rl_joint (bridge + othello, 1 iteration)

- registration everywhere (model registry, LABEL_KEYS, META_ARCH_KEYS, GAME_SPECS,
  VEC_ENV_REGISTRY, teams);
- the converter on the real sample fixtures: verdicts (corrupt Ben boards rejected,
  claims adjudicated), shard schema / meta = the simulator's FrameSpec;
- golden: converter frames == the rules' replay frames of each record; ENV frames ==
  converter frames on the real records (the env plays each record's board and deal,
  scripted by config, stepped with the recorded calls and cards); the env's own
  random episodes, written as BBO LIN, convert to exactly the frames it emitted;
- invariance: the simulator battery (causality; privacy on prefixes: by the end of a
  deal every card is public); the converter's causality (records cut after every
  decision) and privacy (the cards a seat has not seen re-dealt by an independent
  Python re-dealer, void suits respected) on real records;
- the IL contract on real bridge frames (il_loss = Σ il_head_terms, soft targets);
- train_multi_game (bridge + othello, 2 steps), train_rl_joint (1 iteration): the
  declarer's stream carries dummy's plays, per-seat rewards, μ = π_old;
- the fetch script is valid bash.
"""

from __future__ import annotations

import collections
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import gmsim
import numpy as np
import pytest
import torch

from data_processing.games.dataset import GameShardDataset, collate_game_sequences, load_gm_samples
from env.vec_game import VEC_ENV_REGISTRY, has_simulator, make_vec_env
from env.vec_gmsim import GmSimVecEnv
from model.games import build_game_network
from model.games.gmrules import CANDIDATE_META_ARCH, candidate_games
from tests import invariance

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "data_processing"
FIX = ROOT / "tests" / "fixtures" / "bridge"
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
CPU = torch.device("cpu")
TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0,
            core_type="transformer", core_layers=1)
CORE = ["d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
        "rd_prelude_layers=1", "rd_recurrent_layers=1", "rd_coda_layers=1", "rd_mean_iters=2",
        "rd_backprop_iters=1", "moe_experts=4"]
SOURCES = {"openspiel": ("openspiel", FIX / "openspiel_test_sample.txt"),
           "jack": ("jack", FIX / "ben_wbc_sample.txt"),
           "pbn": ("pbn", FIX / "sample.pbn"),
           "bba": ("pbn", FIX / "bba_sample.pbn"),
           "lin": ("lin", FIX)}
SEATS = "NESW"
SUITS = "CDHS"
RANKS = "23456789TJQKA"
#: the standard duplicate rotation: board 1..16 → (dealer, (N/S vul, E/W vul))
BOARDS = {b: ((b - 1) % 4, v) for b, v in zip(range(1, 17), [
    (0, 0), (1, 0), (0, 1), (1, 1), (1, 0), (0, 1), (1, 1), (0, 0),
    (0, 1), (1, 1), (0, 0), (1, 0), (1, 1), (0, 0), (1, 0), (0, 1)])}
G_FOR_DUMMY = 3         # globals: the frame plays dummy's card (bridge.rs G_FOR_DUMMY)
A_CARD = 1              # action features: the candidate is a card (bridge.rs A_CARD)


@pytest.fixture(scope="module")
def gmdata():
    if not os.path.exists(CARGO):
        pytest.skip("cargo not installed")
    b = subprocess.run([CARGO, "build", "--release", "-q", "-p", "gmdata"], cwd=WS,
                       capture_output=True, text=True)
    if b.returncode != 0:
        pytest.skip(f"cargo build failed: {b.stderr[-500:]}")
    return str(WS / "target" / "release" / "gmdata")


def _run(*args) -> dict:
    res = subprocess.run([str(a) for a in args], capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def shards(gmdata, tmp_path_factory):
    """The four real sample fixtures converted (with their trajectories)."""
    root = tmp_path_factory.mktemp("bridge")
    stats = {name: _run(gmdata, "bridge", "convert", "--format", fmt, "--input", inp,
                        "--out", root / "all" / name, "--store-events", "--shard-size", "64")
             for name, (fmt, inp) in SOURCES.items()}
    return root, stats


def _samples(root: Path, name: str) -> List[dict]:
    return load_gm_samples(str(root / "all" / name))


def _by_record(samples: List[dict]) -> Dict[str, Dict[int, dict]]:
    out: Dict[str, Dict[int, dict]] = collections.defaultdict(dict)
    for s in samples:
        out[s["meta"]["record"]][s["meta"]["seat"]] = s
    return out


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def test_registration_everywhere():
    from training.train_il_game import LABEL_KEYS, META_ARCH_KEYS
    from training.train_rl_joint import GAME_SPECS, CandidateSpec

    assert "bridge" in candidate_games() and "bridge" in gmsim.games()
    assert LABEL_KEYS["bridge"] == () and META_ARCH_KEYS["bridge"] == CANDIDATE_META_ARCH
    assert isinstance(GAME_SPECS["bridge"], CandidateSpec)
    assert has_simulator("bridge") and "bridge" in VEC_ENV_REGISTRY
    info = gmsim.game_info("bridge")
    assert info["teams"] == [0, 1, 0, 1]
    assert json.loads(info["config"]) == {"board": 0, "reward": "imp"}
    spec = info["spec"]
    assert (spec["schema"], spec["layout"], spec["n_seats"], spec["hidden_info"]) == \
        ("candidate.v1", "per_seat", 4, True)
    assert spec["meta"] == {"max_entities": 91, "entity_features": 40, "entity_types": 0,
                            "global_features": 66, "max_actions": 38, "action_features": 48}
    net = build_game_network("bridge", **TINY)
    assert (net.cfg.cand_max_entities, net.cfg.cand_max_actions) == (91, 38)
    assert net.obs_encoder.n_tokens == 91 + 1 + 38
    env = make_vec_env("bridge", 2, seed=0)
    assert isinstance(env, GmSimVecEnv)
    assert (env.rows_per_env, env.lockstep, env.teams, env.action_keys) == \
        (4, False, (0, 1, 0, 1), ("action_idx",))


# ---------------------------------------------------------------------------
# converter on the real samples
# ---------------------------------------------------------------------------

def test_converter_verdicts_on_the_real_samples(shards):
    _, st = shards
    os_, jack, pbn, bba, lin = st["openspiel"], st["jack"], st["pbn"], st["bba"], st["lin"]
    # WBridge5 (OpenSpiel dataset): every deal verified
    assert (os_["records"], os_["converted"], os_["illegal"], os_["returns_mismatch"]) == \
        (50, 50, 0, 0)
    # Ben WBC: two boards open the play from the wrong hand (data errors) — rejected;
    # every recorded result of the others agrees with the replay
    assert (jack["records"], jack["converted"], jack["illegal"], jack["returns_mismatch"]) == \
        (47, 45, 2, 0)
    assert all("is illegal for seat" in e for e in jack["errors"])
    # PBN: a whole deal, an auction without the play, a passed-out deal, a claim
    assert (pbn["records"], pbn["converted"], pbn["adjudicated"]) == (4, 4, 1)
    # Ben's BBA bidding set: auctions with the deal's result (no play), all adjudicated
    assert (bba["records"], bba["converted"], bba["adjudicated"], bba["illegal"]) == (20, 20, 20, 0)
    # LIN: a hand record + a 3-board vugraph file (one claim, one passed out)
    assert (lin["records"], lin["converted"], lin["adjudicated"]) == (4, 4, 1)


def test_shards_schema_and_meta(shards):
    root, st = shards
    ds = GameShardDataset(str(root / "all"), game="bridge")      # one shard family
    assert len(ds) == sum(s["samples"] for s in st.values())
    spec = gmsim.game_info("bridge")["spec"]
    m = ds.shard_meta
    assert m["schema"] == "candidate.v1" and m["layout"] == "per_seat" and m["n_seats"] == 4
    assert m["config"] == {"board": 0, "reward": "imp"}
    for k, v in spec["meta"].items():
        assert m[k] == v, k
    for s in _samples(root, "openspiel")[:40]:
        assert set(s["obs"]) == {"entities", "entity_mask", "globals", "actions", "action_mask"}
        assert set(s["tgt"]) == {"action_idx", "value", "value_mask"}
        assert s["obs"]["entities"].dtype == torch.float32               # bit-packed on disk
        idx = s["tgt"]["action_idx"]
        assert bool(s["obs"]["action_mask"].gather(1, idx.unsqueeze(1)).all())
        assert bool(s["tgt"]["value_mask"].all())
    # every record: one stream per seat; dummy only calls, the declarer plays its own
    # 13 cards and dummy's 13, each defender 13 (a card frame: its candidates are cards)
    for rec, seats in _by_record(_samples(root, "openspiel")).items():
        assert sorted(seats) == [0, 1, 2, 3]
        card_frames = sorted(int((s["obs"]["actions"][:, 0, A_CARD] == 1).sum())
                             for s in seats.values())
        passed_out = len(seats[0]["meta"]["events"]) == 1 + 52 + 4
        assert card_frames == ([0, 0, 0, 0] if passed_out else [0, 13, 13, 26]), rec


def test_claims_and_unfinished_records(shards):
    root, _ = shards
    lin = _by_record(_samples(root, "lin"))
    claimed = lin["vugraph_sample.lin#2"]
    for seat, s in claimed.items():
        m = s["meta"]
        # 2S doubled by S, 7 tricks claimed after five: one down, nobody vulnerable —
        # N/S -100 = -3 IMPs; the claim is every frame's value target
        assert m["adjudicated"] is True and m["claim"] == 7 and m["contract"] == "2SX by S"
        assert m["outcome"] == pytest.approx([-3 / 24, 3 / 24, -3 / 24, 3 / 24])
        assert m["outcome"][0] == pytest.approx(imps(-100) / 24)
        assert bool(s["tgt"]["value_mask"].all())
        value = s["tgt"]["value"].tolist()
        assert value == pytest.approx([m["outcome"][seat]] * len(value))
    pbn = _by_record(_samples(root, "pbn"))
    auction_only = pbn["sample.pbn:34"]
    for s in auction_only.values():
        assert not bool(s["tgt"]["value_mask"].any()) and s["meta"].get("adjudicated") is None
    claimed_pbn = pbn["sample.pbn:51"]
    whole = pbn["sample.pbn:3"]
    for seat in range(4):
        # same deal, auction and result: the claim keeps the finished deal's value
        assert claimed_pbn[seat]["meta"]["outcome"] == whole[seat]["meta"]["outcome"]
        assert claimed_pbn[seat]["meta"]["adjudicated"] is True


def test_bidding_sets_take_the_recorded_result(shards):
    """Ben's BBA set: the auction frames' value is the recorded result (board 1: 3NT
    by E making 10, everybody vulnerable, E/W +630 = 12 IMPs); the systems are kept."""
    root, _ = shards
    recs = _by_record(_samples(root, "bba"))
    assert len(recs) == 20
    first = recs["bba_sample.pbn:3"]
    for seat, s in first.items():
        m = s["meta"]
        assert m["adjudicated"] is True and m["contract"] == "3NT by E" and m["tricks"] == 10
        assert m["systems"] == ["Sayc - Standard American Yellow Card"] * 2
        sign = 1 if seat % 2 == 1 else -1
        assert m["outcome"][seat] == pytest.approx(sign * imps(630) / 24)
        assert bool(s["tgt"]["value_mask"].all())
        assert not bool((s["obs"]["actions"][:, 0, A_CARD] == 1).any())      # calls only


def imps(score: int) -> int:
    """The WBF IMP scale (the IMPs of a score against 0), written out independently."""
    up = [20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500, 600, 750, 900, 1100, 1300,
          1500, 1750, 2000, 2250, 2500, 3000, 3500, 4000]
    return sum(abs(score) >= t for t in up) * (1 if score >= 0 else -1)


# ---------------------------------------------------------------------------
# golden: converter == rules == simulator
# ---------------------------------------------------------------------------

def test_converter_frames_are_the_rules_frames(shards):
    """Every sample == the replay stream of its record (gmrules encode_frame)."""
    root, _ = shards
    n = 0
    for name in SOURCES:
        for s in _samples(root, name):
            r = gmsim.replay("bridge", None, s["meta"]["events"])
            idx = dict(r["streams"])[s["meta"]["seat"]]
            for k in r["obs"]:
                assert np.array_equal(s["obs"][k].numpy(), r["obs"][k][idx]), (name, k)
            assert s["tgt"]["action_idx"].tolist() == r["action_idx"][idx].tolist()
            if r["terminal"]:
                assert s["tgt"]["value"].tolist() == [r["outcome"][s["meta"]["seat"]]] * len(idx)
            n += 1
    assert n > 400


def _play_record(events: np.ndarray) -> Tuple[Dict[int, List[dict]], np.ndarray, bool]:
    """Play a record's deal in the gmsim env (its board and deal scripted by config),
    stepping the recorded calls and cards through the policy interface (candidate
    indices); returns the frames every row saw, the rewards and whether it ended."""
    board = int(events[0][1]) + 1
    deal = [int(c) for _, c in events[1:53]]
    env = GmSimVecEnv("bridge", 1, seed=0, board=board, deal=deal)
    obs = env.reset()
    seen: Dict[int, List[dict]] = {r: [] for r in range(4)}
    rew = torch.zeros(1, 4)
    done = torch.zeros(1, dtype=torch.bool)
    for seat, action in events[53:]:
        rows = obs.active.nonzero().squeeze(1).tolist()
        assert rows == [int(seat)], (rows, seat)                 # the rules' acting seat
        r = rows[0]
        assert int(obs.to_play[r]) == seat
        seen[r].append({k: v[r].clone() for k, v in obs.obs.items()})
        idx = env.legal_ids(r).index(int(action))
        acts = torch.zeros(4, dtype=torch.long)
        acts[r] = idx
        obs, rew, done, info = env.step({"action_idx": acts})
        assert int(info["applied"]["action_idx"][r]) == idx
    return seen, rew[0], bool(done[0])


def test_env_frames_are_the_converter_frames_on_real_records(shards):
    """Golden IL = RL on REAL records: the env, playing each record's deal with the
    recorded calls and cards, emits exactly the converter's frames (and pays the
    converter's value targets)."""
    root, _ = shards
    n_records = n_frames = 0
    for name in SOURCES:
        for rec, seats in _by_record(_samples(root, name)).items():
            events = gmsim.events_array(next(iter(seats.values()))["meta"]["events"])
            seen, rew, ended = _play_record(events)
            for seat, s in seats.items():
                frames = seen[seat]
                assert len(frames) == s["obs"]["entities"].shape[0], (rec, seat)
                for k in s["obs"]:
                    ours = torch.stack([f[k] for f in frames])
                    assert ours.dtype == s["obs"][k].dtype and torch.equal(ours, s["obs"][k]), \
                        (rec, seat, k)
                n_frames += len(frames)
                if ended:
                    assert float(rew[seat]) == pytest.approx(float(s["tgt"]["value"][0]))
            n_records += 1
    assert n_records == 50 + 45 + 4 + 20 + 4 and n_frames > 5000


def _lin_hand(cards: Sequence[int]) -> str:
    return "".join(su + "".join(RANKS[c // 4] for c in sorted(cards, reverse=True) if c % 4 == s)
                   for su, s in (("S", 3), ("H", 2), ("D", 1), ("C", 0)))


def _lin_call(a: int) -> str:
    if a == 52:
        return "p"
    if a == 53:
        return "d"
    if a == 54:
        return "r"
    b = a - 55
    return f"{b // 5 + 1}{'CDHSN'[b % 5]}"


def _lin_board(events: np.ndarray, n: int) -> str:
    """A gmrules bridge trajectory (default config) as one BBO LIN board."""
    dealer, (ns, ew) = BOARDS[int(events[0][1]) + 1]
    hands: List[List[int]] = [[], [], [], []]
    for k, (_, c) in enumerate(events[1:53]):
        hands[k % 4].append(int(c))
    md = "SWNE".index(SEATS[dealer]) + 1                     # LIN dealer digit 1=S 2=W 3=N 4=E
    parts = ",".join(_lin_hand(hands[SEATS.index(p)]) for p in "SWNE")
    sv = {(0, 0): "o", (1, 0): "n", (0, 1): "e", (1, 1): "b"}[(ns, ew)]
    out = [f"qx|o{n}|md|{md}{parts}|sv|{sv}|"]
    for _, a in events[53:]:
        a = int(a)
        out.append(f"pc|{SUITS[a % 4]}{RANKS[a // 4]}|" if a < 52 else f"mb|{_lin_call(a)}|")
    return "".join(out)


def test_simulator_episodes_convert_to_the_emitted_frames(gmdata, tmp_path):
    """Golden IL = RL on the env's own play: random episodes (every board) written
    as a BBO LIN file and converted — the shards hold exactly the frames the env
    emitted."""
    n_envs = 12
    env = GmSimVecEnv("bridge", n_envs, seed=0)
    obs = env.reset(seed=11)
    gen = torch.Generator().manual_seed(0)
    seen = {r: [] for r in range(obs.n_rows)}
    rewards = torch.zeros(n_envs, 4)
    while bool(obs.active.any()):
        for r in obs.active.nonzero().squeeze(1).tolist():
            seen[r].append({k: v[r].clone() for k, v in obs.obs.items()})
        obs, rew, done, _ = env.step(env.random_actions(obs, generator=gen))
        rewards += rew
    boards = [_lin_board(env.trajectory(e), e + 1) for e in range(n_envs)]
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "sim.lin").write_text("\n".join(boards) + "\n")
    st = _run(gmdata, "bridge", "convert", "--format", "lin", "--input", tmp_path / "in",
              "--out", tmp_path / "out")
    assert st["converted"] == n_envs and st["illegal"] == 0
    recs = _by_record(load_gm_samples(str(tmp_path / "out")))
    boards = {int(env.trajectory(e)[0][1]) + 1 for e in range(n_envs)}
    assert len(boards) > 4                                   # several dealers / vulnerabilities
    for e in range(n_envs):
        for seat, s in recs[f"sim.lin#{e + 1}"].items():
            frames = seen[e * 4 + seat]
            for k in s["obs"]:
                assert torch.equal(torch.stack([f[k] for f in frames]), s["obs"][k]), (e, seat, k)
            assert float(s["tgt"]["value"][0]) == pytest.approx(float(rewards[e, seat]))


# ---------------------------------------------------------------------------
# invariance
# ---------------------------------------------------------------------------

def test_simulator_invariance_battery():
    """Causality on random deals of every board; privacy on prefixes (a finished
    deal has no hidden card left) cut in the auction, at the opening lead and in
    the play, through the rules' own perturbation."""
    st = invariance.check_sim_battery("bridge", None, n_games=4, seed=3)
    assert st["decisions"] > 150
    n = 0
    for i in range(12):
        ev = gmsim.random_trajectory("bridge", None, 100 + i)
        r = gmsim.replay("bridge", None, ev)
        cards = np.flatnonzero(r["action"] < 52)
        cuts = [int(r["event"][min(3, r["n"] - 1)])]                       # in the auction
        if len(cards):
            cuts += [int(r["event"][cards[0]]), int(r["event"][cards[len(cards) // 2]])]
        for cut in cuts:
            n += invariance.check_sim_privacy("bridge", None, ev[:cut + 1], seeds=(i, i + 50))
    assert n > 60
    for cfg in (None, {"board": 9, "reward": "score"}):
        assert "privacy_checks" in gmsim.check_game("bridge", cfg, 30, 1)


def _convert_lines(gmdata: str, tmp: Path, lines: List[str], fmt: str = "openspiel",
                   group: int = 1) -> Dict[str, List[dict]]:
    """Convert many records in one gmdata call: {record text: its samples}
    (``group`` source lines per record, e.g. 4 for Jack boards)."""
    d = tmp / f"batch{abs(hash(tuple(lines)))}"
    if d.exists():
        shutil.rmtree(d)
    (d / "in").mkdir(parents=True)
    ext = {"openspiel": "txt", "jack": "txt"}[fmt]
    recs = ["\n".join(lines[i:i + group]) for i in range(0, len(lines), group)]
    (d / "in" / f"r.{ext}").write_text("\n".join(recs) + "\n")
    _run(gmdata, "bridge", "convert", "--format", fmt, "--input", d / "in", "--out", d / "out")
    by_line: Dict[int, List[dict]] = collections.defaultdict(list)
    for s in load_gm_samples(str(d / "out")):
        by_line[int(s["meta"]["record"].split(":")[1])].append(s)
    return {rec: by_line.get(1 + i * group, []) for i, rec in enumerate(recs)}


def _truncations(ids: List[int]) -> List[str]:
    """An OpenSpiel record cut after every action (the deal kept)."""
    return [" ".join(map(str, ids[:52 + k])) for k in range(1, len(ids) - 52)]


def test_converter_causality_on_real_records(gmdata, tmp_path):
    lines = FIX.joinpath("openspiel_test_sample.txt").read_text().splitlines()
    picks = [ln for ln in lines if len(ln.split()) > 56][:3]
    picks.append(next(ln for ln in lines if len(ln.split()) == 56))           # passed out
    batch = []
    for ln in picks:
        batch += [ln] + _truncations([int(x) for x in ln.split()])
    cache = _convert_lines(gmdata, tmp_path, batch)
    n = 0
    for ln in picks:
        n += invariance.check_converter_causality(lambda recs: cache[recs[0]], ln,
                                                  _truncations([int(x) for x in ln.split()]))
    assert n > 300
    # a Ben board (dealer W, everybody vulnerable) cut in its play line (after the
    # opening lead, a trick, all but one card): same frames, 4 streams each
    jack = FIX.joinpath("ben_wbc_sample.txt").read_text().splitlines()[:4]
    cuts = [jack[:3] + [jack[3][:2 * k]] for k in (1, 13, 51)]
    flat = jack + [line for c in cuts for line in c]
    cache = _convert_lines(gmdata, tmp_path, flat, fmt="jack", group=4)
    assert invariance.check_converter_causality(lambda recs: cache[recs[0]], "\n".join(jack),
                                                ["\n".join(c) for c in cuts]) == 3 * 4


def _redeal(ids: List[int], cut: int, seat: int, n: int, seed: int) -> List[str]:
    """Independent re-dealer (not gmrules' perturb_hidden): versions of the OpenSpiel
    record ``ids`` cut after ``cut`` actions in which the cards ``seat`` has not seen
    by then — the unplayed cards of the other hands, dummy's excepted once the
    opening lead is on the table — are re-dealt among those hands (the suits a hand
    showed out of respected, played cards unchanged): every recorded action stays
    legal."""
    deal, acts = ids[:52], ids[52:52 + cut]
    holder = {c: k % 4 for k, c in enumerate(deal)}
    played, void = set(), set()
    led = None
    for a in acts:
        if a >= 52:
            continue
        if len(played) % 4 == 0:
            led = a % 4
        elif a % 4 != led:
            void.add((holder[a], led))
        played.add(a)
    hidden = [p for p in range(4) if p != seat]
    if played:
        leader = holder[next(a for a in acts if a < 52)]
        dummy = (leader + 1) % 4                          # declarer = leader's RHO
        hidden = [p for p in hidden if p != dummy]
    pool = [c for c in range(52) if holder[c] in hidden and c not in played]
    need = collections.Counter(holder[c] for c in pool)
    rng = random.Random(seed)
    out: List[str] = []
    for _ in range(200):
        if len(out) >= n:
            break
        cards = pool[:]
        rng.shuffle(cards)
        cards.sort(key=lambda c: sum((p, c % 4) not in void for p in hidden))
        cap = dict(need)
        new = dict(holder)
        ok = True
        for c in cards:
            opts = [p for p in hidden if cap.get(p, 0) > 0 and (p, c % 4) not in void]
            if not opts:
                ok = False
                break
            p = rng.choices(opts, weights=[cap[p] for p in opts])[0]
            new[c] = p
            cap[p] -= 1
        if not ok or new == holder:
            continue
        hands = [sorted(c for c in range(52) if new[c] == p) for p in range(4)]
        line = " ".join(map(str, [hands[k % 4][k // 4] for k in range(52)] + acts))
        if line not in out:
            out.append(line)
    return out


def test_converter_privacy_on_real_records(gmdata, tmp_path):
    lines = FIX.joinpath("openspiel_test_sample.txt").read_text().splitlines()
    cases = []
    for i, ln in enumerate([ln for ln in lines if len(ln.split()) > 56][:4]):
        ids = [int(x) for x in ln.split()]
        n_calls = next(k for k, a in enumerate(ids[52:]) if a < 52)
        for cut in (n_calls, n_calls + 1, n_calls + 9, n_calls + 30):   # auction end,
            prefix = " ".join(map(str, ids[:52 + cut]))                # lead, play
            for seat in range(4):
                cases.append((prefix, seat, _redeal(ids, cut, seat, 3, 7 * i + seat)))
    batch = [c[0] for c in cases] + [p for c in cases for p in c[2]]
    cache = _convert_lines(gmdata, tmp_path, list(dict.fromkeys(batch)))
    n = 0
    for prefix, seat, perturbed in cases:
        assert perturbed, (prefix[:40], seat)
        n += invariance.check_converter_privacy(lambda recs: cache[recs[0]], prefix, perturbed,
                                                seat)
    assert n >= 3 * len(cases)
    # negative control: re-dealing a card the seat DOES see changes its frames
    ids = [int(x) for x in lines[0].split()]
    swap = ids[:]
    a = next(k for k in range(52) if k % 4 == 0)                     # a card of N
    b = next(k for k in range(52) if k % 4 == 1)                     # a card of E
    swap[a], swap[b] = swap[b], swap[a]
    cut = next(k for k, x in enumerate(ids[52:]) if x < 52)          # auction only
    base, leak = " ".join(map(str, ids[:52 + cut])), " ".join(map(str, swap[:52 + cut]))
    cache = _convert_lines(gmdata, tmp_path, [base, leak])
    with pytest.raises(AssertionError, match="entities"):
        invariance.check_converter_privacy(lambda recs: cache[recs[0]], base, [leak], 0)


# ---------------------------------------------------------------------------
# IL contract on real bridge frames
# ---------------------------------------------------------------------------

def test_il_contract_on_real_frames(shards):
    from training.train_il_game import batch_loss

    root, _ = shards
    torch.manual_seed(0)
    net = build_game_network("bridge", **TINY).train()
    b = collate_game_sequences(_samples(root, "jack")[:6])
    obs, tgt, mask = b["obs"], b["tgt"], b["meta"]["frame_mask"]
    pred = net(obs)
    assert pred["action_logits"].shape[-1] == 38
    dec = net.action_decoder
    il = dec.il_loss(pred, tgt, mask)
    t = dec.il_head_terms(pred, tgt, mask)["action"]
    assert torch.allclose(il["loss"], (t["ce"] * t["weight"]).sum() / t["norm"])
    assert float(t["weight"].sum()) == float(mask.sum())               # every frame labelled
    mb = {"obs": obs, "tgt": tgt, "frame_mask": mask, "labels": None, "legal": None}
    hard, stats, _ = batch_loss(net, mb, CPU, 0.2, 0.0, False)
    assert torch.isfinite(hard) and 0.0 <= stats["action_acc"] <= 1.0
    # a one-hot teacher over the candidates == the hard label; a genuine one differs
    A = pred["action_logits"].shape[-1]
    onehot = torch.nn.functional.one_hot(tgt["action_idx"].clamp(min=0), A).float()
    onehot = onehot * mask.unsqueeze(-1)
    same, st1, _ = batch_loss(net, {**mb, "tgt": {**tgt, "action_probs": onehot}}, CPU, 0.2,
                              0.0, False)
    assert torch.allclose(hard, same, atol=1e-5) and "action_teacher_kl" in st1
    legal = obs["action_mask"].float()
    soft = legal / legal.sum(-1, keepdim=True).clamp(min=1)
    diff, _, _ = batch_loss(net, {**mb, "tgt": {**tgt, "action_probs": soft}}, CPU, 0.2, 0.0,
                            False)
    assert not torch.allclose(diff, hard)
    diff.backward()
    assert net.action_decoder.pointer.net[0].weight.grad is not None


# ---------------------------------------------------------------------------
# training: IL with othello, RL with othello
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stage1(gmdata, shards, tmp_path_factory):
    from training import train_multi_game

    root, _ = shards
    out = tmp_path_factory.mktemp("bridge_stage1")
    subprocess.run([gmdata, "othello", "convert", "--input", str(ROOT / "tests" / "fixtures" /
                                                                 "othello"),
                    "--out", str(out / "othello")], check=True, capture_output=True)
    torch.manual_seed(0)
    res = train_multi_game.main([
        "--data", f"bridge={root / 'all'}", "--data", f"othello={out / 'othello'}",
        "--val", f"bridge={root / 'all' / 'openspiel'}", "--eval_every", "2",
        "--eval_batches", "1", "--steps", "2", "--batch_size", "4", "--warmup_steps", "1",
        "--device", "cpu", "--save_dir", str(out / "ck"), "--override", *CORE])
    return out, root, res


def test_train_multi_game_with_an_existing_game(stage1):
    out, _, res = stage1
    ck = torch.load(out / "ck" / "multi_final.pt", map_location="cpu", weights_only=False)
    assert ck["games"] == ["bridge", "othello"]
    cfg = ck["configs"]["bridge"]
    assert (cfg["cand_max_entities"], cfg["cand_max_actions"], cfg["cand_global_features"]) == \
        (91, 38, 66)
    assert sum(ck["mixture"].values()) == pytest.approx(1.0)
    for p in res["model"].parameters():
        assert torch.isfinite(p).all()


def test_train_rl_joint_bridge(stage1):
    from training import train_rl_joint
    from training.train_rl_joint import Diag, JointRLTrainer, parse_args

    out, root, _ = stage1
    args = ["--init", str(out / "ck" / "multi_final.pt"), "--rl", "bridge", "--rl", "othello",
            "--data", f"bridge={root / 'all' / 'openspiel'}", "--replay_batch_size", "2",
            "--tokens_per_iter", "100", "--envs", "bridge=2", "--envs", "othello=2",
            "--mirror_prob", "bridge=0.5", "--mirror_prob", "othello=0.5",
            "--snapshot_every", "1", "--epochs", "1", "--minibatches", "2",
            "--snapshot_dir", "memory", "--device", "cpu", "--seed", "3"]
    res = train_rl_joint.main(args + ["--iterations", "1", "--save_dir", str(out / "rl")])
    rec = res["history"][-1]
    assert rec["dropped"] == 0 and rec["nonfinite"] == 0
    for g in ("bridge", "othello"):
        row = rec["games"][g]
        assert row["tokens"] >= 1 and row["episodes"] > 0
        assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
        for k in ("entropy", "kl", "value_loss", "adv_std"):
            assert np.isfinite(row[k]), (g, k)
    assert (out / "rl" / "rl_final.pt").exists()
    tr = JointRLTrainer(parse_args(args + ["--save_dir", str(out / "rl2")]))
    diag = Diag(tr.games, tr.n_moe)
    tr._rollouts(0, {"bridge": 100, "othello": 40}, diag)
    streams = tr.rl["bridge"].streams
    assert streams
    dummy_frames = 0
    for s in streams:
        # a per-seat stream: one seat, its reward and done on its LAST frame (also
        # dummy's, whose last call comes long before the deal ends)
        seat = int(s.seat[0])
        assert bool((s.seat == seat).all())
        assert bool(s.done[-1]) and not bool(s.done[:-1].any())
        assert float(s.rewards[:-1].abs().sum()) == 0.0
        dummy_frames += int((s.obs["globals"][:, G_FOR_DUMMY] == 1).sum())
    assert dummy_frames > 0                  # the declarer's stream carries dummy's plays
    tr._recompute("bridge")
    for s in tr.rl["bridge"].streams:
        assert torch.allclose(s.lp_old, s.mu, atol=1e-4)


def test_evaluation_network_team_vs_random_team():
    """play_episodes (env/league.py): the network plays one partnership, a random
    policy the other; the winner is a team (its lowest seat) and the score counts it."""
    from env.league import League, NetworkPolicy, RandomPolicy, play_episodes

    torch.manual_seed(0)
    net = build_game_network("bridge", **TINY).eval()
    env = GmSimVecEnv("bridge", 8, seed=0)
    # random actions of a row subset (what a RandomPolicy opponent gets) are aligned
    # with the subset and equal the full call's on those rows
    obs = env.reset(seed=4)
    rows = obs.active.nonzero().squeeze(1)
    full = env.random_actions(obs, generator=torch.Generator().manual_seed(9))
    sub = env.random_actions(obs.select(rows), generator=torch.Generator().manual_seed(9))
    assert torch.equal(sub["action_idx"], full["action_idx"][rows])
    league = League(env, {"random": RandomPolicy(env, seed=1)}, mirror_prob=0.0, seed=2)
    res = play_episodes(env, league, NetworkPolicy(net))
    for row in res["learner_seats"].tolist():
        assert row in ([True, False, True, False], [False, True, False, True])
    assert set(res["winner"].tolist()) <= {-1, 0, 1}
    for w, seats, score in zip(res["winner"].tolist(), res["learner_seats"].tolist(),
                               res["score"].tolist()):
        assert score == (0.5 if w < 0 else float(seats[w]))
    assert bool((res["episode_len"] >= 4).all())


def test_fetch_script_is_valid_bash():
    script = ROOT / "scripts" / "fetch_bridge.sh"
    assert script.exists() and os.access(script, os.X_OK)
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
