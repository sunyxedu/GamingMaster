"""Heads-up no-limit hold'em end to end, on real ACPC records:

    real PHH hands (tests/fixtures/hunl/acpc_sample.phhs: 2009-2017, 2009 rescaled)
      ─► gmdata hunl convert (every hand replayed and verified by the gmrules rules,
         recorded payoffs checked) ─► gm-shards (candidate layout + exact sizes)
      ─► candidate adapter ─► train_multi_game (with kuhn_poker, 2 steps)
      ─► gmsim env ─► train_rl_joint (1 iteration, with kuhn_poker)

- shards: schema, dtypes, shard meta = the simulator's spec, exact-size targets;
- golden IL = RL: converter frames == the rules' replay frames of each record; the
  RL env started from every real record's prefix (GmSimVecEnv.reset_from) shows
  exactly the converter's frame of each real decision; the env's own episodes,
  written as PHH and converted, give exactly the frames the env emitted;
- hidden information: converter causality (hands cut after every action) and
  privacy (the opponent's hole cards re-dealt, showdown lines included) with
  tests/invariance.py, the simulator battery on structured random hands;
- the IL contract of the adapter on real batches;
- the trained checkpoint playing through scripts/slumbot_client.py (against the local
  fake of the Slumbot API, tests/test_slumbot_client.py);
- scripts/fetch_hunl.sh end to end offline (SKIP_FETCH=1): splits, HQ, subsample.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import gmsim
import numpy as np
import pytest
import torch

from data_processing.games.dataset import GameShardDataset, collate_game_sequences, load_gm_samples
from env.vec_gmsim import GmSimVecEnv
from tests import invariance

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 (the project's tomli dependency)
    import tomli as tomllib  # type: ignore[no-redef]
from tests.test_hunl_reference import VARIANTS, our_hand

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "data_processing"
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
FIXTURE = ROOT / "tests" / "fixtures" / "hunl" / "acpc_sample.phhs"
CFG = VARIANTS["acpc"]
N_HANDS = len(re.findall(r"(?m)^\[\d+\]$", FIXTURE.read_text()))
CPU = torch.device("cpu")
TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0, core_type="transformer",
            core_layers=1)
CORE = ["d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
        "rd_prelude_layers=1", "rd_recurrent_layers=1", "rd_coda_layers=1", "rd_mean_iters=2",
        "rd_backprop_iters=1", "moe_experts=4"]
RANKS, SUITS = "23456789TJQKA", "cdhs"


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
def data(gmdata, tmp_path_factory):
    root = tmp_path_factory.mktemp("hunl")
    st = _run(gmdata, "hunl", "convert", "--input", FIXTURE, "--rescale", "--out", root / "train",
              "--shard-size", "64", "--store-events")
    assert st["records"] == st["converted"] == N_HANDS
    assert st["illegal"] == st["returns_mismatch"] == st["unparseable"] == 0
    assert st["filters"]["rescaled"] > 0            # the 2009 hands (1 / 2 / 400)
    return root, st


# ---------------------------------------------------------------------------
# PHH helpers
# ---------------------------------------------------------------------------

def card(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


def read_hands(path: Path) -> List[Dict]:
    d = tomllib.loads(path.read_text())
    return [d[k] for k in sorted(d, key=int)]


def phh_text(hands: List[Dict]) -> str:
    keys = ("variant", "antes", "blinds_or_straddles", "min_bet", "starting_stacks", "actions",
            "hand", "players", "_results")
    out = []
    for i, h in enumerate(hands):
        out.append(f"[{i + 1}]")
        out.extend(f"{k} = {json.dumps(h[k])}" for k in keys if k in h)
        out.append("")
    return "\n".join(out)


def phh_actions(events, cfg) -> List[str]:
    """A gmrules trajectory as PHH actions (street amounts)."""
    cards = [i for s, i in events if s < 0]
    acts = [f"d dh p1 {card(cards[0])}{card(cards[1])}",
            f"d dh p2 {card(cards[2])}{card(cards[3])}"]
    spent, base, n_chance, board = [cfg["big_blind"], cfg["small_blind"]], 0, 0, []
    for s, i in events:
        if s < 0:
            n_chance += 1
            if n_chance > 4:
                board.append(i)
                if n_chance in (7, 8, 9):
                    acts.append("d db " + "".join(card(c) for c in board))
                    board, base = [], max(spent)
        elif i == 0:
            acts.append(f"p{s + 1} f")
        elif i == 1:
            spent[s] = min(max(spent), cfg["stack"])
            acts.append(f"p{s + 1} cc")
        else:
            spent[s] = i
            acts.append(f"p{s + 1} cbr {i - base}")
    return acts


def phh_hand(events, cfg, hand: int, returns=None) -> Dict:
    h = {"variant": "NT", "antes": [0, 0],
         "blinds_or_straddles": [cfg["small_blind"], cfg["big_blind"]],
         "min_bet": cfg["big_blind"], "starting_stacks": [cfg["stack"]] * 2,
         "actions": phh_actions(events, cfg), "hand": hand, "players": ["a", "b"]}
    if returns is not None:
        h["_results"] = [int(round(x)) for x in returns]
    return h


def _convert(gmdata, tmp: Path, hands: List[Dict], *extra) -> List[dict]:
    """gmdata hunl convert as a function of PHH hands (--allow-truncated)."""
    d = tmp / f"c{abs(hash(json.dumps(hands, sort_keys=True)))}"
    if d.exists():
        shutil.rmtree(d)
    (d / "raw").mkdir(parents=True)
    (d / "raw" / "h.phhs").write_text(phh_text(hands))
    _run(gmdata, "hunl", "convert", "--input", d / "raw", "--out", d / "out", "--rescale",
         "--allow-truncated", "--no-dedup", *extra)
    return load_gm_samples(str(d / "out"))


# ---------------------------------------------------------------------------
# shards
# ---------------------------------------------------------------------------

def test_shards_schema_and_meta(data):
    from model.games.gmrules import CANDIDATE_META_ARCH

    root, st = data
    ds = GameShardDataset(str(root / "train"), game="hunl")
    assert len(ds) == st["samples"] and ds.total_frames() == st["frames"]
    spec = gmsim.game_info("hunl", CFG)["spec"]
    m = ds.shard_meta
    assert m["schema"] == "candidate_abstract.v1" and m["layout"] == "per_seat"
    assert m["config"] == CFG and m["n_seats"] == 2
    for k in CANDIDATE_META_ARCH:
        assert m[k] == spec["meta"][k], k
    assert m["candidates"][:4] == ["fold", "check_call", "all_in", "min_raise"]
    n_raise = 0
    for s in load_gm_samples(str(root / "train")):
        assert set(s["obs"]) == {"entities", "entity_mask", "entity_type", "globals", "actions",
                                 "action_mask"}
        assert set(s["tgt"]) == {"action_idx", "raise_to", "raise_frac", "value"}
        assert s["obs"]["entities"].dtype == torch.float32
        assert s["obs"]["entity_type"].dtype == torch.int32
        idx = s["tgt"]["action_idx"]
        assert bool(s["obs"]["action_mask"].gather(1, idx.unsqueeze(1)).all())
        chosen = s["obs"]["actions"][torch.arange(len(idx)), idx]           # [T, Fa]
        is_raise = chosen[:, 3] == 1
        assert torch.equal(is_raise, s["tgt"]["raise_to"] > 0)
        assert bool((s["tgt"]["raise_frac"][is_raise] > 0).all())
        assert bool((s["tgt"]["raise_frac"][~is_raise] == 0).all())
        n_raise += int(is_raise.sum())
        seat = s["meta"]["seat"]
        value = s["meta"]["returns"][seat] / CFG["stack"]
        assert s["tgt"]["value"].tolist() == pytest.approx([value] * len(idx))
        # own hole cards only: two hole-card entities (type 1) in every frame
        assert bool(((s["obs"]["entity_type"] == 1).sum(1) == 2).all())
    assert n_raise > 100


def test_converter_frames_are_the_rules_frames(data):
    """Golden: every sample == the rules' replay stream of its record."""
    root, _ = data
    for s in load_gm_samples(str(root / "train")):
        r = gmsim.replay("hunl", CFG, s["meta"]["events"])
        idx = dict(r["streams"])[s["meta"]["seat"]]
        for k in r["obs"]:
            assert np.array_equal(s["obs"][k].numpy(), r["obs"][k][idx]), k
        for k in ("action_idx", "raise_to", "raise_frac"):
            assert np.array_equal(s["tgt"][k].numpy(), r["target"][k][idx]), k
        assert r["outcome"][s["meta"]["seat"]] == pytest.approx(float(s["tgt"]["value"][0]))


def test_env_frames_equal_converter_frames_on_real_records(data):
    """Golden IL = RL on REAL records: the gmsim env started from each record's
    prefix up to every real decision shows exactly the converter's frame of that
    decision (obs, legal mask, stream position)."""
    root, _ = data
    samples = load_gm_samples(str(root / "train"))
    jobs = []
    for si, s in enumerate(samples):
        ev = s["meta"]["events"]
        r = gmsim.replay("hunl", CFG, ev)
        for t, d in enumerate(dict(r["streams"])[s["meta"]["seat"]]):
            jobs.append((ev[:int(r["event"][d])], s["meta"]["seat"], si, t))
    env = GmSimVecEnv("hunl", len(jobs), config=CFG)
    obs = env.reset_from([j[0] for j in jobs], seed=0)
    for e, (_, seat, si, t) in enumerate(jobs):
        row = 2 * e + seat
        assert bool(obs.active[row]) and int(obs.step_idx[row]) == t
        assert not bool(obs.active[2 * e + 1 - seat])
        for k, v in obs.obs.items():
            assert torch.equal(v[row], samples[si]["obs"][k][t]), (si, t, k)
        assert torch.equal(obs.legal["action"][row], samples[si]["obs"]["action_mask"][t])
    assert len(jobs) == sum(len(s["tgt"]["action_idx"]) for s in samples) > 1000


def test_simulator_frames_are_the_converter_frames(gmdata, tmp_path):
    """Golden IL = RL: play the gmsim env (random legal candidates), write its episodes
    as PHH hands, convert them with gmdata — the shards' frames are exactly the frames
    the env emitted."""
    n = 24
    env = GmSimVecEnv("hunl", n, config=CFG)
    obs = env.reset(seed=11)
    gen = torch.Generator().manual_seed(0)
    seen = {r: [] for r in range(2 * n)}
    while bool(obs.active.any()):
        for r in obs.active.nonzero().squeeze(1).tolist():
            seen[r].append({k: v[r].clone() for k, v in obs.obs.items()})
        mask = obs.legal["action"].float()
        idx = torch.multinomial(mask + (mask.sum(1, keepdim=True) == 0).float(), 1, generator=gen)
        obs, _, _, _ = env.step({"action_idx": idx.squeeze(1)})
    hands = []
    for e in range(n):
        ev = env.trajectory(e).tolist()
        hands.append(phh_hand(ev, CFG, e, gmsim.replay("hunl", CFG, ev)["returns"]))
    samples = _convert(gmdata, tmp_path, hands)
    assert sum(len(v) > 0 for v in seen.values()) == len(samples)
    for s in samples:
        frames = seen[2 * int(s["meta"]["hand"]) + s["meta"]["seat"]]
        for k in s["obs"]:
            ours = torch.stack([f[k] for f in frames])
            assert ours.dtype == s["obs"][k].dtype and torch.equal(ours, s["obs"][k]), k


# ---------------------------------------------------------------------------
# hidden information
# ---------------------------------------------------------------------------

def _is_bet(a: str) -> bool:
    return a.startswith("p") and not a.split()[1] == "sm"


def _truncations(h: Dict) -> List[Dict]:
    """The hand cut right after every betting action but the last (payoffs unknown)."""
    bets = [i for i, a in enumerate(h["actions"]) if _is_bet(a)]
    out = []
    for i in bets[:-1]:
        t = {k: v for k, v in h.items() if k != "_results"}
        t["actions"] = h["actions"][:i + 1]
        out.append(t)
    return out


def _redeals(h: Dict, seat: int, rng: random.Random, n: int = 3) -> List[Dict]:
    """The hand with the OTHER seat's hole cards re-dealt (its showdown line too)."""
    other = f"p{2 - seat}"
    used = set()
    for a in h["actions"]:
        tok = a.split()
        if tok[0] == "d" and not (tok[1] == "dh" and tok[2] == other):
            used.update(tok[-1][i:i + 2] for i in range(0, len(tok[-1]), 2))
    pool = [r + s for r in RANKS for s in SUITS if r + s not in used]
    out = []
    for _ in range(n):
        new = "".join(rng.sample(pool, 2))
        acts = []
        for a in h["actions"]:
            tok = a.split()
            if tok[:3] == ["d", "dh", other]:
                a = f"d dh {other} {new}"
            elif tok[0] == other and tok[1] == "sm" and len(tok) == 3:
                a = f"{other} sm {new}"
            acts.append(a)
        out.append({**{k: v for k, v in h.items() if k != "_results"}, "actions": acts})
    return out


def test_converter_causality_and_privacy(gmdata, tmp_path):
    """On real hands: a hand cut after any action converts to prefixes of the full
    hand's streams; re-dealing the opponent's hole cards (hidden until the showdown,
    after every decision) leaves every frame and label of the seat unchanged."""
    hands = read_hands(FIXTURE)
    rng = random.Random(0)
    pick = [h for h in hands if any(" sm " in a + " " for a in h["actions"])][::7][:6]
    pick += sorted(hands, key=lambda h: -len(h["actions"]))[:3]
    pick += [h for h in hands if abs(h["_results"][0]) == h["starting_stacks"][0]][:3]

    def convert(recs):
        return _convert(gmdata, tmp_path, recs)

    n_caus = n_priv = 0
    for h in pick:
        n_caus += invariance.check_converter_causality(convert, h, _truncations(h))
        seats = {s["meta"]["seat"] for s in convert([h])}
        for seat in seats:
            n_priv += invariance.check_converter_privacy(convert, h, _redeals(h, seat, rng), seat)
    assert n_caus > 30 and n_priv > 30, (n_caus, n_priv)


def test_simulator_invariance_battery():
    """Causality and privacy of the rules' frames on structured random hands (folds,
    all-ins, every street) of three configs, plus the gmrules conformance battery."""
    rng = random.Random(3)
    for key in ("acpc", "acpc2009", "short"):
        cfg = VARIANTS[key]
        st = {"decisions": 0, "privacy": 0}
        for _ in range(12):
            ev = our_hand(cfg, rng)
            st["decisions"] += invariance.check_sim_causality("hunl", cfg, ev)
            st["privacy"] += invariance.check_sim_privacy("hunl", cfg, ev, seeds=(1, 2))
        assert st["decisions"] > 20 and st["privacy"] > 20, (key, st)
    assert "privacy_checks" in gmsim.check_game("hunl", CFG, 60, 1)
    assert "privacy_checks" in gmsim.check_game("hunl", VARIANTS["short"], 200, 2)


def test_privacy_negative_control():
    """The comparison does see the seat's OWN cards: changing them changes its frames."""
    ev = our_hand(CFG, random.Random(1))
    mine = [list(e) for e in ev]
    mine[0][1] = next(c for c in range(52) if c not in {i for s, i in ev if s < 0})
    a = gmsim.replay("hunl", CFG, ev)
    b = gmsim.replay("hunl", CFG, mine)
    seat0 = [{"entities": r["obs"]["entities"][r["seat"] == 0]} for r in (a, b)]
    with pytest.raises(AssertionError, match="entities"):
        invariance.assert_frames_prefix(seat0[0], seat0[1], 1, "leak")


# ---------------------------------------------------------------------------
# model: IL contract on real batches
# ---------------------------------------------------------------------------

def test_il_contract_on_real_batches(data):
    from model.games import build_game_network
    from training.train_il_game import (
        META_ARCH_KEYS,
        apply_meta_arch_defaults,
        batch_loss,
        check_meta_arch,
    )

    root, _ = data
    meta = GameShardDataset(str(root / "train"), game="hunl").shard_meta
    kw = apply_meta_arch_defaults("hunl", dict(TINY), meta)
    assert kw["cand_max_actions"] == 15 and kw["cand_entity_types"] == 5
    torch.manual_seed(0)
    net = build_game_network("hunl", **kw).train()
    check_meta_arch("hunl", net.cfg, meta)
    assert set(META_ARCH_KEYS["hunl"]) <= set(meta)
    b = collate_game_sequences(load_gm_samples(str(root / "train"))[:8])
    mask = b["meta"]["frame_mask"]
    pred = net(b["obs"])
    dec = net.action_decoder
    il = dec.il_loss(pred, b["tgt"], mask)
    t = dec.il_head_terms(pred, b["tgt"], mask)["action"]
    assert torch.allclose(il["loss"], (t["ce"] * t["weight"]).sum() / t["norm"])
    assert torch.equal(t["weight"].bool(), mask)                 # every real frame is labelled
    mb = {"obs": b["obs"], "tgt": b["tgt"], "frame_mask": mask, "labels": None, "legal": None}
    hard, stats, _ = batch_loss(net, mb, CPU, 0.2, 0.0, False)
    assert torch.isfinite(hard)
    onehot = torch.zeros(*pred["action_logits"].shape)
    onehot.scatter_(-1, b["tgt"]["action_idx"].clamp(min=0).unsqueeze(-1), 1.0)
    mb_soft = {**mb, "tgt": {**b["tgt"], "action_probs": onehot}}
    soft, _, _ = batch_loss(net, mb_soft, CPU, 0.2, 0.0, False)
    assert torch.allclose(hard, soft, atol=1e-5)
    hard.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stage1(gmdata, data, tmp_path_factory):
    from training import train_multi_game

    root, _ = data
    out = tmp_path_factory.mktemp("hunl_stage1")
    _run(gmdata, "kuhn-poker", "generate", "--out", out / "kraw" / "h.log", "--hands", "200",
         "--seed", "1")
    _run(gmdata, "kuhn-poker", "convert", "--input", out / "kraw", "--out", out / "kuhn")
    torch.manual_seed(0)
    res = train_multi_game.main([
        "--data", f"hunl={root / 'train'}", "--data", f"kuhn_poker={out / 'kuhn'}",
        "--val", f"hunl={root / 'train'}", "--eval_every", "2", "--eval_batches", "1",
        "--steps", "3", "--batch_size", "4", "--warmup_steps", "1", "--device", "cpu",
        "--save_dir", str(out / "ck"), "--override", *CORE])
    return out, root, res


def test_train_multi_game_with_kuhn(stage1):
    out, _, res = stage1
    ck = torch.load(out / "ck" / "multi_final.pt", map_location="cpu", weights_only=False)
    assert ck["games"] == ["hunl", "kuhn_poker"]
    assert ck["configs"]["hunl"]["cand_max_actions"] == 15
    assert ck["configs"]["hunl"]["cand_global_features"] == 48
    for p in res["model"].parameters():
        assert torch.isfinite(p).all()


def test_train_rl_joint_with_kuhn(stage1):
    from training import train_rl_joint

    out, root, _ = stage1
    args = ["--init", str(out / "ck" / "multi_final.pt"), "--rl", "hunl", "--rl", "kuhn_poker",
            "--data", f"hunl={root / 'train'}", "--replay_batch_size", "4",
            "--tokens_per_iter", "120", "--envs", "hunl=6", "--envs", "kuhn_poker=4",
            "--snapshot_every", "1", "--epochs", "1", "--minibatches", "2",
            "--snapshot_dir", "memory", "--device", "cpu", "--seed", "3", "--iterations", "1",
            "--save_dir", str(out / "rl")]
    res = train_rl_joint.main(args)
    rec = res["history"][-1]
    assert rec["dropped"] == 0 and rec["nonfinite"] == 0
    row = rec["games"]["hunl"]
    assert row["tokens"] >= 1 and row["episodes"] > 0
    assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
    for k in ("entropy", "kl", "value_loss", "adv_std"):
        assert np.isfinite(row[k]), k
    assert (out / "rl" / "rl_final.pt").exists()


def test_slumbot_client_plays_the_trained_model(stage1):
    """scripts/slumbot_client.py's ModelPolicy (the checkpoint's hunl network on the
    hand's frames of its seat) against the local fake of the Slumbot API."""
    from scripts import slumbot_client as sc
    from tests.test_slumbot_client import FakeSlumbot

    out, _, _ = stage1
    policy = sc.ModelPolicy(str(out / "ck" / "multi_final.pt"), temperature=1.0, seed=0)
    greedy = sc.ModelPolicy(str(out / "ck" / "multi_final.pt"), temperature=0.0)
    fake = FakeSlumbot(seed=4)
    token = None
    for i in range(16):
        token, h = sc.play_hand(fake.post, token, policy if i % 2 else greedy)
        assert h.replayed == h.winnings
    assert len(fake.client_actions) >= 16


def test_fetch_script_offline(gmdata, tmp_path):
    """scripts/fetch_hunl.sh end to end without the network (SKIP_FETCH=1 on logs laid
    out like the fetch's, <RAW>/<year>/...): train / val splits that partition the
    hands, the HQ and subsample filters, the stats logs, idempotent re-runs."""
    script = ROOT / "scripts" / "fetch_hunl.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    raw = tmp_path / "raw" / "2017" / "logs"
    raw.mkdir(parents=True)
    shutil.copy(FIXTURE, raw / "sample.phhs")
    env = {**os.environ, "SKIP_FETCH": "1", "RAW": str(tmp_path / "raw"), "YEARS": "2017",
           "GMDATA": gmdata, "VAL_PER_MILLE": "200"}

    def run(out, **kw):
        res = subprocess.run(["bash", str(script), str(out)], env={**env, **kw},
                             capture_output=True, text=True, check=True)
        logs = {s: json.loads((out / "logs" / f"convert_{s}_2017.json").read_text())
                for s in ("train", "val")}
        return res.stdout, logs

    _, logs = run(tmp_path / "all", HQ="0")
    tr, va = logs["train"], logs["val"]
    assert tr["converted"] + va["converted"] == N_HANDS and va["converted"] > 10
    assert tr["illegal"] == va["illegal"] == tr["returns_mismatch"] == 0
    for split, st in (("train", tr), ("val", va)):
        ds = GameShardDataset(str(tmp_path / "all" / split), game="hunl")
        assert ds.total_frames() == st["frames"]
    stdout, again = run(tmp_path / "all", HQ="0")
    assert "have train 2017" in stdout and again == logs
    # winning bots only, a uniform half of the train hands
    _, hq = run(tmp_path / "hq", HQ="1", KEEP_PER_MILLE="500")
    assert hq["train"]["hq"]["kept"] > 0 and hq["train"]["samples"] < tr["samples"]
    assert 0 < hq["train"]["filters"]["subsampled"] < tr["converted"]
    assert hq["val"]["filters"]["subsampled"] == 0
