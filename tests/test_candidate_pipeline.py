"""The toy game end to end through every layer, and the invariance tests of its
converter:

    gmrules rules ─► gmdata kuhn-poker convert (synthetic hand logs) ─► gm-shards
      ─► CandidateEncoder / CandidateDecoder ─► train_multi_game (2 steps)
      ─► gmsim env ─► train_rl_joint (1 iteration)

- the shards: schema, dtypes, shard meta = the simulator's FrameSpec, --hq filter;
- golden: converter frames == replay (rules) frames of each record, and
  simulator frames == converter frames of the simulator's own episodes (IL = RL);
- converter causality (truncated hands) and privacy (re-dealt opponent cards) with
  tests/invariance.py, plus the simulator-level battery;
- RL: per-seat streams get their terminal reward on the seat's last frame even
  when another seat ended the hand; --actor fixed refused for per-seat games;
  goofspiel (simultaneous) and pig (interleaved, perfect information) from
  `gmdata rules random` shards, jointly with kuhn;
- the mixture frame cap / game weights of train_multi_game.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List

import gmsim
import numpy as np
import pytest
import torch

from data_processing.games.dataset import GameShardDataset, load_gm_samples
from env.vec_candidate import CandidateVecEnv
from tests import invariance

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "data_processing"
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
CORE = ["d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
        "rd_prelude_layers=1", "rd_recurrent_layers=1", "rd_coda_layers=1", "rd_mean_iters=2",
        "rd_backprop_iters=1", "moe_experts=4"]


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
    root = tmp_path_factory.mktemp("kuhn")
    _run(gmdata, "kuhn-poker", "generate", "--out", root / "raw" / "h.log", "--hands", "400",
         "--players", "2", "--seed", "3")
    st = _run(gmdata, "kuhn-poker", "convert", "--input", root / "raw", "--out", root / "train",
              "--shard-size", "128", "--store-events")
    assert st["converted"] == 400 and st["illegal"] == st["returns_mismatch"] == 0
    return root, st


def _convert(gmdata, tmp: Path, lines: List[str], *extra) -> List[dict]:
    """The toy converter as a function of hand-log lines (for the invariance checks)."""
    tag = str(abs(hash(tuple(lines))))
    d = tmp / tag
    if d.exists():
        shutil.rmtree(d)
    (d / "raw").mkdir(parents=True)
    (d / "raw" / "h.log").write_text("\n".join(lines) + "\n")
    _run(gmdata, "kuhn-poker", "convert", "--input", d / "raw", "--out", d / "out",
         "--players", str(len(lines[0].split(":")[3].split("|"))), "--allow-truncated", *extra)
    return load_gm_samples(str(d / "out"))


# ---------------------------------------------------------------------------
# shards
# ---------------------------------------------------------------------------

def test_shards_schema_and_meta(data):
    root, st = data
    ds = GameShardDataset(str(root / "train"), game="kuhn_poker")
    assert len(ds) == st["samples"] == 800 and ds.total_frames() == st["frames"]
    spec = gmsim.game_info("kuhn_poker", {"players": 2})["spec"]
    m = ds.shard_meta
    assert m["schema"] == "candidate.v1" and m["layout"] == "per_seat"
    assert m["config"] == {"players": 2}
    for k in ("max_entities", "entity_features", "entity_types", "global_features",
              "max_actions", "action_features", "n_seats"):
        assert m[k] == spec[k], k
    s = ds[0]
    T = s["obs"]["entities"].shape[0]
    assert set(s["obs"]) == {"entities", "entity_mask", "entity_type", "globals", "actions",
                             "action_mask"}
    assert set(s["tgt"]) == {"action_idx", "value"}
    assert s["obs"]["entity_type"].dtype == torch.int32
    assert s["obs"]["actions"].dtype == torch.float32              # bit-packed on disk
    assert s["obs"]["entity_mask"].dtype == torch.bool and s["tgt"]["action_idx"].shape == (T,)
    assert bool(s["obs"]["action_mask"].gather(1, s["tgt"]["action_idx"].unsqueeze(1)).all())


def test_teacher_soft_targets_through_the_trainer(gmdata, tmp_path):
    """Teacher distributions (JSONL "soft", `gmdata rules convert --soft-k`) land as
    tgt.action_probs_idx/val over candidate indices and train through batch_loss."""
    from data_processing.games.dataset import collate_game_sequences
    from model.games import build_game_network
    from training.train_il_game import batch_loss

    recs = []
    for i in range(6):
        ev = gmsim.random_trajectory("kuhn_poker", None, i).tolist()
        soft = [[k, [[0, 0.3], [1, 0.7], [5, 0.2]]] for k, (s, _) in enumerate(ev) if s >= 0]
        recs.append(json.dumps({"id": f"r{i}", "events": ev, "soft": soft}))
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "t.jsonl").write_text("\n".join(recs) + "\n")
    st = _run(gmdata, "rules", "convert", "--game", "kuhn_poker", "--input", tmp_path / "in",
              "--out", tmp_path / "out", "--soft-k", "2")
    assert st["converted"] == 6 and st["soft_frames"] == st["frames"]
    assert st["soft_illegal_mass"] == pytest.approx(0.2 * st["frames"], rel=1e-5)
    samples = load_gm_samples(str(tmp_path / "out"))
    s = samples[0]
    assert s["tgt"]["action_probs_idx"].tolist()[0] == [1, 0]
    assert s["tgt"]["action_probs_val"][0].tolist() == pytest.approx([0.7, 0.3])
    torch.manual_seed(0)
    net = build_game_network("kuhn_poker", d_model=32, n_heads=4, d_ff=64, n_st_layers=1,
                             core_type="transformer", core_layers=1)
    b = collate_game_sequences(samples[:4])
    mb = {"obs": b["obs"], "tgt": b["tgt"], "frame_mask": b["meta"]["frame_mask"],
          "labels": None, "legal": None}
    loss, stats, _ = batch_loss(net, mb, torch.device("cpu"), 0.2, 0.0, False)
    assert torch.isfinite(loss) and stats["action_teacher_frac"] == pytest.approx(1.0)
    assert stats["action_teacher_kl"] >= 0


def test_hq_keeps_only_expert_seats(gmdata, data, tmp_path):
    root, _ = data
    st = _run(gmdata, "kuhn-poker", "convert", "--input", root / "raw", "--out", tmp_path / "hq",
              "--hq")
    samples = load_gm_samples(str(tmp_path / "hq"))
    assert len(samples) == st["samples"] < 800
    for s in samples:
        assert s["meta"]["players"][s["meta"]["seat"]] == "expert"


def test_converter_frames_are_the_rules_frames(data):
    """Golden: every sample == the replay stream of its record (gmrules encode_frame)."""
    root, _ = data
    for s in load_gm_samples(str(root / "train"))[:200]:
        r = gmsim.replay("kuhn_poker", {"players": 2}, s["meta"]["events"])
        idx = dict(r["streams"])[s["meta"]["seat"]]
        for k in gmsim.FRAME_KEYS:
            assert np.array_equal(s["obs"][k].numpy(), r[k][idx]), k
        assert s["tgt"]["action_idx"].tolist() == r["action_idx"][idx].tolist()
        assert s["tgt"]["value"].tolist() == [r["outcome"][s["meta"]["seat"]]] * len(idx)


def test_simulator_frames_are_the_converter_frames(gmdata, tmp_path):
    """Golden IL = RL: play the gmsim env, write its episodes as hand logs, convert
    them with gmdata — the shards' frames are exactly the frames the env emitted."""
    env = CandidateVecEnv("kuhn_poker", 16, seed=0, players=3)
    env.reset(seed=21)
    gen = torch.Generator().manual_seed(0)
    obs = env.observe()
    seen = {r: [] for r in range(obs.n_rows)}
    while bool(obs.active.any()):
        for r in obs.active.nonzero().squeeze(1).tolist():
            seen[r].append({k: v[r].clone() for k, v in obs.obs.items()})
        obs, _, _, _ = env.step(env.random_actions(obs, generator=gen))
    lines = []
    for e in range(16):
        ev = env.trajectory(e)
        cards = [str(i) for s, i in ev if s < 0]
        bets = "".join("pb"[i] for s, i in ev if s >= 0)
        r = gmsim.replay("kuhn_poker", {"players": 3}, ev)
        rets = "|".join(f"{x:g}" for x in r["returns"])
        lines.append(f"STATE:{e}:{bets}:{'|'.join(cards)}:{rets}:a|b|c")
    samples = _convert(gmdata, tmp_path, lines)
    assert len(samples) == 16 * 3
    for s in samples:
        e, seat = int(s["meta"]["hand"]), s["meta"]["seat"]
        frames = seen[e * 3 + seat]
        for k in s["obs"]:
            ours = torch.stack([f[k] for f in frames])
            assert ours.dtype == s["obs"][k].dtype and torch.equal(ours, s["obs"][k]), (e, seat, k)


# ---------------------------------------------------------------------------
# invariance
# ---------------------------------------------------------------------------

def test_simulator_invariance_battery():
    for players in (2, 3, 4):
        st = invariance.check_sim_battery("kuhn_poker", {"players": players}, n_games=30,
                                          seed=players)
        assert st["privacy"] > 30
    invariance.check_sim_battery("goofspiel", {"num_cards": 5, "players": 3}, n_games=10)
    for players in (2, 4):
        assert "privacy_checks" in gmsim.check_game("kuhn_poker", {"players": players}, 200, 1)
    gmsim.check_game("goofspiel", {"num_cards": 6, "players": 4, "points_order": "random"}, 50, 2)


def test_privacy_check_catches_a_leak():
    """Negative control: frames DO depend on the seat's own card, so a
    'perturbation' that changes it must fail the privacy comparison."""
    ev = [[-1, 0], [-1, 1], [-1, 2], [0, 0], [1, 1], [2, 1], [0, 1]]
    leak = [[-1, 3], [-1, 1], [-1, 2], [0, 0], [1, 1], [2, 1], [0, 1]]      # seat 0's card
    a = gmsim.replay("kuhn_poker", {"players": 3}, ev)
    b = gmsim.replay("kuhn_poker", {"players": 3}, leak)
    with pytest.raises(AssertionError, match="entities"):
        invariance.assert_frames_prefix({"entities": a["entities"][a["seat"] == 0]},
                                        {"entities": b["entities"][b["seat"] == 0]}, 2, "leak")


def _truncations(line: str) -> List[str]:
    """The hand cut after every action (returns unknown)."""
    p = line.split(":")
    return [":".join([p[0], p[1], p[2][:t], p[3], "", p[5]]) for t in range(1, len(p[2]))]


def _redeals(line: str, seat: int, n: int) -> List[str]:
    """The hand with the other seats' cards re-dealt (seat's own card kept)."""
    p = line.split(":")
    cards = [int(c) for c in p[3].split("|")]
    pool = [c for c in range(n + 1) if c != cards[seat]]
    out = []
    for others in itertools.permutations(pool, n - 1):
        it = iter(others)
        new = [cards[seat] if s == seat else next(it) for s in range(n)]
        if new != cards:
            out.append(":".join([p[0], p[1], p[2], "|".join(map(str, new)), "", p[5]]))
    return out


def test_converter_causality_and_privacy(gmdata, tmp_path):
    def convert(lines):
        return _convert(gmdata, tmp_path, lines)

    hands = ["STATE:1:pbpb:3|1|2:3|-2|-1:a|b|c",       # a bet, a fold, a call
             "STATE:2:ppp:0|3|1:-1|2|-1:a|b|c",        # everybody passes
             "STATE:3:bpb:2|3|0:3|-1|-2:a|b|c",       # a bet, a fold, a call
             "STATE:4:pbb:1|2:-2|2:x|y"]
    for h in hands:
        n = len(h.split(":")[3].split("|"))
        assert invariance.check_converter_causality(convert, h, _truncations(h)) > 0
        for seat in range(n):
            assert invariance.check_converter_privacy(convert, h, _redeals(h, seat, n), seat) > 0
    # the returns of a record are verified: a wrong one is rejected
    bad = _convert(gmdata, tmp_path, ["STATE:9:pp:0|1:5|-5:a|b"])
    assert bad == []


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stage1(gmdata, data, tmp_path_factory):
    from training import train_multi_game

    root, _ = data
    out = tmp_path_factory.mktemp("stage1")
    _run(gmdata, "rules", "random", "--game", "goofspiel", "--config",
         '{"num_cards":4,"players":2}', "--episodes", "60", "--out", out / "goof")
    _run(gmdata, "rules", "random", "--game", "pig", "--config",
         json.dumps(PIG), "--episodes", "40", "--out", out / "pig")
    torch.manual_seed(0)
    res = train_multi_game.main([
        "--data", f"kuhn_poker={root / 'train'}", "--data", f"goofspiel={out / 'goof'}",
        "--data", f"pig={out / 'pig'}",
        "--val", f"kuhn_poker={root / 'train'}", "--eval_every", "2", "--eval_batches", "2",
        "--steps", "2", "--batch_size", "4", "--warmup_steps", "1", "--device", "cpu",
        "--mixture_frame_cap", "300", "--game_weight", "goofspiel=0.5",
        "--save_dir", str(out / "ck"), "--override", *CORE])
    return out, root, res


PIG = {"players": 2, "winscore": 10, "diceoutcomes": 6, "horizon": 30}


def test_train_multi_game_on_candidate_shards(stage1):
    out, root, res = stage1
    ck = torch.load(out / "ck" / "multi_final.pt", map_location="cpu", weights_only=False)
    assert ck["games"] == ["kuhn_poker", "goofspiel", "pig"]
    assert ck["configs"]["goofspiel"]["cand_max_actions"] == 13
    assert ck["configs"]["pig"]["cand_max_actions"] == 2
    # every corpus counts at most 300 frames; goofspiel's weight is halved
    dirs = {"kuhn_poker": root / "train", "goofspiel": out / "goof", "pig": out / "pig"}
    raw = {g: min(GameShardDataset(str(d), game=g).total_frames(), 300) ** 0.5
           * (0.5 if g == "goofspiel" else 1.0) for g, d in dirs.items()}
    for g in dirs:
        assert ck["mixture"][g] == pytest.approx(raw[g] / sum(raw.values()))
    for p in res["model"].parameters():
        assert torch.isfinite(p).all()


def test_mixture_weights():
    from training.train_multi_game import mixture_weights

    f = {"big": 10**10, "mid": 10**6, "small": 10**4}
    assert mixture_weights(f, 0.5) == pytest.approx(
        {g: n**0.5 / sum(m**0.5 for m in f.values()) for g, n in f.items()})
    w = mixture_weights(f, 0.5, frame_cap=10**8)
    assert w["big"] / w["mid"] == pytest.approx(10.0)
    w = mixture_weights(f, 0.5, frame_cap=10**8, game_frame_cap={"big": 10**6},
                        game_weight={"small": 0.0})
    assert w == pytest.approx({"big": 0.5, "mid": 0.5, "small": 0.0})
    with pytest.raises(SystemExit):
        mixture_weights(f, 0.5, game_weight={g: 0.0 for g in f})


def test_train_rl_joint_on_candidate_games(stage1):
    from training import train_rl_joint
    from training.train_rl_joint import Diag, JointRLTrainer, parse_args

    out, root, _ = stage1
    args = ["--init", str(out / "ck" / "multi_final.pt"),
            "--rl", "kuhn_poker", "--rl", "goofspiel",
            "--rl", "pig", "--data", f"kuhn_poker={root / 'train'}", "--replay_batch_size", "4",
            "--tokens_per_iter", "120", "--envs", "kuhn_poker=6", "--envs", "goofspiel=4",
            "--envs", "pig=3", "--env_kw", "kuhn_poker:players=3",
            "--env_kw", "goofspiel:num_cards=4",
            *[x for k, v in PIG.items() for x in ("--env_kw", f"pig:{k}={v}")],
            "--mirror_prob", "kuhn_poker=0.5", "--mirror_prob", "pig=0.5", "--snapshot_every", "1",
            "--epochs", "1", "--minibatches", "2", "--snapshot_dir", "memory", "--device", "cpu",
            "--seed", "3"]
    res = train_rl_joint.main(args + ["--iterations", "1", "--save_dir", str(out / "rl")])
    rec = res["history"][-1]
    assert rec["dropped"] == 0 and rec["nonfinite"] == 0
    for g in ("kuhn_poker", "goofspiel", "pig"):
        row = rec["games"][g]
        assert row["tokens"] >= 1 and row["episodes"] > 0
        assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
        for k in ("entropy", "kl", "value_loss", "adv_std"):
            assert np.isfinite(row[k]), (g, k)
    assert (out / "rl" / "rl_final.pt").exists()
    tr = JointRLTrainer(parse_args(args + ["--save_dir", str(out / "rl2")]))
    diag = Diag(tr.games, tr.n_moe)
    tr._rollouts(0, {"kuhn_poker": 60, "goofspiel": 40, "pig": 40}, diag)
    for s in tr.rl["kuhn_poker"].streams:
        # per-seat stream of a 3-player hand: its terminal reward and done arrive on the
        # seat's LAST frame, also when another seat's action ended the hand
        assert bool(s.done[-1]) and not bool(s.done[:-1].any())
        seat = int(s.seat[0])
        assert bool((s.seat == seat).all())
        assert float(s.rewards[:, seat].sum()) != 0.0
    for s in tr.rl["goofspiel"].streams:
        assert s.length == 3 and bool(s.done[-1])
    for s in tr.rl["pig"].streams:
        # interleaved: one stream per game, both movers' frames, the outcome at its end
        assert bool(s.done[-1]) and not bool(s.done[:-1].any())
        assert s.rewards[:-1].abs().sum() == 0
    # a PPO step on these streams is exact: μ = π_old
    for g in tr.rl_games:
        tr._recompute(g)
        for s in tr.rl[g].streams:
            assert torch.allclose(s.lp_old, s.mu, atol=1e-4)
    with pytest.raises(SystemExit, match="actor stream"):
        JointRLTrainer(parse_args(args + ["--save_dir", str(out / "rl3"), "--actor", "fixed"]))


def test_fetch_script_is_valid_bash():
    script = ROOT / "scripts" / "fetch_kuhn_poker.sh"
    assert script.exists()
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
