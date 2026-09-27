"""Backgammon end to end, pinned to gnubg (GNU Backgammon, GPL-3.0-or-later: an
external process in a throw-away uv environment, never a project dependency):

- rules: the legal move sets of 1 500 positions (gnubg-guided playouts and random
  boards: doubles, bar entries, bear-offs, passes, rolls with more than 64 distinct
  moves) are gnubg's (``gmdata backgammon diff --fixture``); gnubg's OWN match loop
  (its dice, opening rolls, cube, drops, Crawford / post-Crawford, gammons /
  backgammons, the Jacoby rule, resignations) replays game by game with every
  result and score (``gmdata backgammon verify``);
- data: gnubg self-play records (``gmdata backgammon generate``: our engine runs the
  matches, gnubg decides both seats and its legal move list is checked at every
  checker play) → ``gmdata backgammon convert`` (every record replayed and verified)
  → gm-shards whose labels are checked against the transcripts' own numbers: the
  candidate targets, the soft targets (gnubg's cube equities; its best move on top)
  and the engine values (match play 2·MWC − 1, money play clamp(equity × cube /
  money_scale), the actor's view: the doubler's best cube action, the taker's best
  answer, the mover's best move on every step of a split turn);
- golden IL = RL: converter frames == the rules' replay; the gmsim env started from
  every real decision's prefix shows the converter's frame; a generated record
  replays exactly in the env from its dice seed;
- causality of the converter (records cut after actions) and the gmrules
  conformance battery through gmsim;
- the candidate adapter's IL contract on real batches, train_multi_game →
  train_rl_joint (one iteration);
- scripts/fetch_backgammon.sh offline (CONVERT_ONLY=1).

Fixtures (tests/fixtures/backgammon/; ``gmdata`` = data_processing/target/release/gmdata)::

    gmdata backgammon diff --live --positions 1500 --seed 5 --random-frac 0.4 \\
        --write tests/fixtures/backgammon/gnubg_legal.jsonl.gz
    uv run --no-project --python 3.12 --with gnubg==1.1.0a16 python \\
        scripts/reference/gnubg_selfplay_dump.py \\
        --out tests/fixtures/backgammon/gnubg_selfplay.jsonl.gz --matches 12 \\
        --match-lengths 1,3,5,7,9 --money-games 16 --vs-random-matches 8 \\
        --vs-random-money 16 --plies 0 --seed 4242
    gmdata backgammon generate --out D3 --matches 3 --workers 1 --seed 11 --plies 0 \\
        --match-length 3        # D3/bgmatch-s11-m0-w000.jsonl.gz = gnubg_generated_m3.jsonl.gz
    gmdata backgammon generate --out D0 --matches 4 --workers 1 --seed 12 --plies 0 \\
        --match-length 0        # D0/bgmatch-s12-m0-w000.jsonl.gz = gnubg_generated_money.jsonl.gz

Live against gnubg (``GM_REFERENCE=1``; uv + network the first time): fresh positions
(``GM_REFERENCE_POSITIONS``, default 3000), a fresh gnubg match loop and freshly
generated matches, seeded by ``GM_REFERENCE_SEED``.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import gmsim
import numpy as np
import pytest
import torch

from data_processing.games.dataset import GameShardDataset, collate_game_sequences, load_gm_samples
from env.vec_gmsim import GmSimVecEnv
from tests import invariance

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "data_processing"
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
FIX = ROOT / "tests" / "fixtures" / "backgammon"
LEGAL = FIX / "gnubg_legal.jsonl.gz"
SELFPLAY = FIX / "gnubg_selfplay.jsonl.gz"
GENERATED = {"m3": (FIX / "gnubg_generated_m3.jsonl.gz", 3),
             "money": (FIX / "gnubg_generated_money.jsonl.gz", 0)}
GAME = "backgammon"
CPU = torch.device("cpu")
TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0, core_type="transformer",
            core_layers=1)
CORE = ["d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
        "rd_prelude_layers=1", "rd_recurrent_layers=1", "rd_coda_layers=1", "rd_mean_iters=2",
        "rd_backprop_iters=1", "moe_experts=4"]
# gmrules backgammon ids
NO_DOUBLE, DOUBLE, TAKE, DROP, PASS = 0, 1, 2, 3, 4
SUBMOVE_BASE, FULL_MOVE_MIN = 16, 9 << 24
SOFT_TAU, MONEY_SCALE = 0.02, 4.0


@pytest.fixture(scope="module")
def gmdata():
    if not os.path.exists(CARGO):
        pytest.skip("cargo not installed")
    b = subprocess.run([CARGO, "build", "--release", "-q", "-p", "gmdata"], cwd=WS,
                       capture_output=True, text=True)
    if b.returncode != 0:
        pytest.skip(f"cargo build failed: {b.stderr[-500:]}")
    return str(WS / "target" / "release" / "gmdata")


def _run(*args, check: bool = True) -> dict:
    res = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if check and res.returncode != 0:
        raise AssertionError(f"{args[1:3]} failed ({res.returncode}): {res.stdout[-800:]} "
                             f"{res.stderr[-800:]}")
    lines = res.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines else {}


def transcripts(path: Path) -> List[dict]:
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def config_of(match_length: int) -> Dict:
    return {"match_length": match_length}


# ---------------------------------------------------------------------------
# rules vs gnubg (recorded)
# ---------------------------------------------------------------------------

def test_legal_moves_match_gnubg(gmdata):
    st = _run(gmdata, GAME, "diff", "--fixture", LEGAL)
    assert st["mismatches"] == 0 and st["agree"] == st["positions"] >= 1500, st["errors"]
    assert min(st["doubles"], st["bar"], st["bearoff"]) > 100 and st["passes"] > 20, st
    assert st["max_moves"] > 64, "no roll above the 64 candidates (split turns)"


def test_gnubg_match_loop_replays(gmdata):
    """gnubg's own matches and money games (its dice, cube, scoring): every game's
    play legal, every played-out game's winner and points (cube × single / gammon /
    backgammon, drops, Jacoby) and the score / Crawford progression between games."""
    st = _run(gmdata, GAME, "verify", "--input", SELFPLAY)
    assert st["mismatches"] == 0, st["errors"]
    assert st["transcripts"] >= 50 and st["matches_checked"] >= 20, st
    for k in ("gammons", "backgammons", "drops", "crawford_games", "resigned", "split_turns"):
        assert st[k] > 0, (k, st)


# ---------------------------------------------------------------------------
# data: generated records → shards
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data(gmdata, tmp_path_factory):
    root = tmp_path_factory.mktemp("backgammon")
    out = {}
    for key, (path, ml) in GENERATED.items():
        st = _run(gmdata, GAME, "convert", "--input", path, "--out", root / key,
                  "--match-length", ml, "--soft-k", "8", "--engine-values", "--store-events",
                  "--shard-size", "2")
        out[key] = {"dir": root / key, "stats": st, "config": config_of(ml),
                    "transcripts": transcripts(path),
                    "samples": load_gm_samples(str(root / key))}
    return out


def test_convert_verifies_every_record(data):
    for key, d in data.items():
        st, tr = d["stats"], d["transcripts"]
        assert st["records"] == st["converted"] == len(tr) == len(d["samples"]), (key, st)
        for k in ("illegal", "returns_mismatch", "unparseable", "truncated", "empty"):
            assert st[k] == 0, (key, k, st)
        t = st["transcripts"]
        assert t["rejected"] == t["other_variant"] == t["bad_analysis"] == 0, (key, t)
        assert st["soft_illegal_mass"] == 0 and st["soft_frames"] > 0.5 * st["frames"], (key, st)
        assert t["engine_values"] > 0.6 * st["frames"], (key, st)
    assert data["m3"]["stats"]["transcripts"]["split_turns"] > 0


def test_shards_schema_meta_and_targets(data):
    from model.games.gmrules import CANDIDATE_META_ARCH

    for key, d in data.items():
        ds = GameShardDataset(str(d["dir"]), game=GAME)
        assert ds.total_frames() == d["stats"]["frames"]
        spec = gmsim.game_info(GAME, d["config"])["spec"]
        m = ds.shard_meta
        assert m["schema"] == "candidate.v1" and m["layout"] == "interleaved" and m["n_seats"] == 2
        assert m["config"]["match_length"] == d["config"]["match_length"]
        for k in CANDIDATE_META_ARCH:
            assert m[k] == spec["meta"][k], k
        for s in d["samples"]:
            assert set(s["obs"]) == {"entities", "entity_mask", "entity_type", "globals",
                                     "actions", "action_mask"}
            assert set(s["tgt"]) == {"action_idx", "action_probs_idx", "action_probs_val",
                                     "engine_value", "engine_value_mask", "value"}
            mask = s["obs"]["action_mask"].bool()
            idx = s["tgt"]["action_idx"]
            assert bool(mask.gather(1, idx.unsqueeze(1)).all())
            # soft targets: legal candidates only, a distribution where labelled
            pi, pv = s["tgt"]["action_probs_idx"], s["tgt"]["action_probs_val"]
            has = pi >= 0
            assert bool(mask.gather(1, pi.clamp(min=0))[has].all())
            tot = (pv * has).sum(1)
            lab = has.any(1)
            assert torch.allclose(tot[lab], torch.ones_like(tot[lab]), atol=1e-4)
            assert bool((pv[~has] == 0).all())
            # engine values: in [-1, 1], zero where unmasked
            ev, em = s["tgt"]["engine_value"], s["tgt"]["engine_value_mask"].bool()
            assert bool((ev.abs() <= 1).all()) and bool((ev[~em] == 0).all())
            assert s["meta"]["seat"] == -1               # one interleaved stream per match


def _decisions(d: dict, s: dict) -> dict:
    return gmsim.replay(GAME, d["config"], s["meta"]["events"])


def test_converter_frames_are_the_rules_frames(data):
    """Golden: every sample == the rules' replay of its record; the value target of
    each frame is the match outcome of the frame's own seat (the mover, the taker)."""
    for d in data.values():
        for s in d["samples"]:
            r = _decisions(d, s)
            assert r["terminal"]
            for k in r["obs"]:
                assert np.array_equal(s["obs"][k].numpy(), r["obs"][k]), k
            assert np.array_equal(s["tgt"]["action_idx"].numpy(), r["target"]["action_idx"])
            want = np.asarray(r["outcome"], dtype=np.float32)[r["seat"]]
            assert np.array_equal(s["tgt"]["value"].numpy(), want)
            assert set(np.unique(r["seat"]).tolist()) == {0, 1}


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _expected_labels(d: dict, t: dict, r: dict) -> List[dict]:
    """Per decision of the replay ``r`` of transcript ``t``: the engine value (None =
    no teacher) and the soft target ({action id: p}; None) the converter must have
    written — computed from the transcript's gnubg numbers alone."""
    money = d["config"]["match_length"] == 0
    ids = r["legal_ids"]
    out: List[dict] = []
    k = 0                                          # next decision

    def soft(q: Dict[int, float]) -> Dict[int, float]:
        m = max(q.values())
        w = {a: math.exp((v - m) / SOFT_TAU) for a, v in q.items()}
        z = sum(w.values())
        return {a: x / z for a, x in w.items()}

    for g in t["games"]:
        cube = 1
        last = None
        for a in g["actions"]:
            kind = a["a"]
            if kind in ("no_double", "double"):
                c = a.get("cube") or {}
                v = q = None
                if c.get("nd") is not None:
                    best = max(c["nd"], min(c["dt"], c["dp"]))
                    v = (_clamp(best * cube / MONEY_SCALE) if money
                         else 2 * max(c["mwc"][0], min(c["mwc"][1], c["mwc"][2])) - 1)
                    q = soft({NO_DOUBLE: c["nd"], DOUBLE: min(c["dt"], c["dp"])})
                out.append({"value": v, "soft": q})
                last = c
                k += 1
            elif kind in ("take", "drop"):
                c = last or {}
                v = q = None
                if c.get("dt") is not None:
                    v = (_clamp(-min(c["dt"], c["dp"]) * cube / MONEY_SCALE) if money
                         else 2 * (1 - min(c["mwc"][1], c["mwc"][2])) - 1)
                    q = soft({TAKE: -c["dt"], DROP: -c["dp"]})
                out.append({"value": v, "soft": q})
                if kind == "take":
                    cube *= 2
                k += 1
            elif kind == "move":
                eq = a.get("eq") or []
                v = None
                if eq:
                    v = _clamp(eq[0][1] * cube / MONEY_SCALE) if money else 2 * a["mwc"] - 1
                # the turn's decisions: split-turn single checker moves, then the move
                steps = 0
                while True:
                    lid = ids[k + steps]
                    steps += 1
                    if not (SUBMOVE_BASE <= min(lid) and max(lid) < FULL_MOVE_MIN):
                        break
                for j in range(steps):
                    out.append({"value": v, "soft": "move" if (steps == 1 and eq) else None})
                k += steps
            else:
                raise AssertionError(f"unexpected action {kind}")
    assert k == r["n"], (k, r["n"])
    return out


def test_labels_follow_gnubg(data):
    """Engine values and soft targets == what the transcripts' gnubg numbers say, in the
    acting seat's view (the converter's labelling, independently recomputed)."""
    n_val = n_soft = n_move = 0
    for d in data.values():
        for t, s in zip(sorted(d["transcripts"], key=lambda x: x["id"]),
                        sorted(d["samples"], key=lambda x: x["meta"]["record"])):
            assert s["meta"]["record"] == t["id"]
            r = _decisions(d, s)
            exp = _expected_labels(d, t, r)
            ev, em = s["tgt"]["engine_value"].numpy(), s["tgt"]["engine_value_mask"].numpy()
            pi, pv = s["tgt"]["action_probs_idx"].numpy(), s["tgt"]["action_probs_val"].numpy()
            for j, e in enumerate(exp):
                if e["value"] is None:
                    assert not em[j], j
                else:
                    assert em[j] and abs(ev[j] - e["value"]) < 2e-5, (t["id"], j, ev[j], e)
                    n_val += 1
                got = {int(r["legal_ids"][j][i]): float(p) for i, p in zip(pi[j], pv[j]) if i >= 0}
                if e["soft"] is None:
                    assert not got, (t["id"], j, got)
                elif e["soft"] == "move":
                    # gnubg's played move is its best: the most probable soft candidate
                    played = int(r["legal_ids"][j][int(s["tgt"]["action_idx"][j])])
                    assert got and abs(got[played] - max(got.values())) < 1e-6, (t["id"], j)
                    n_move += 1
                else:
                    assert set(got) == set(e["soft"]), (got, e["soft"])
                    for a, p in e["soft"].items():
                        assert abs(got[a] - p) < 1e-5, (t["id"], j, got, e["soft"])
                    n_soft += 1
    assert n_val > 400 and n_soft > 150 and n_move > 250, (n_val, n_soft, n_move)


def test_env_frames_equal_converter_frames_on_real_records(data):
    """Golden IL = RL: the gmsim env started from each record's prefix up to every
    decision shows exactly the converter's frame (obs, legal mask, stream position)."""
    d = data["m3"]
    for s in d["samples"][:2]:
        ev = s["meta"]["events"]
        r = _decisions(d, s)
        prefixes = [ev[:int(e)] for e in r["event"]]
        env = GmSimVecEnv(GAME, len(prefixes), config=d["config"])
        obs = env.reset_from(prefixes, seed=0)
        for t in range(len(prefixes)):
            assert bool(obs.active[t]) and int(obs.step_idx[t]) == t
            assert int(obs.to_play[t]) == int(r["seat"][t])
            for k, v in obs.obs.items():
                assert torch.equal(v[t], s["obs"][k][t]), (t, k)
            assert torch.equal(obs.legal["action"][t], s["obs"]["action_mask"][t])


def test_env_replays_generated_records_from_their_seeds(data):
    """A generated match is a pure function of its dice seed and gnubg's choices: the
    env reset with the record's seed and fed the recorded candidates reproduces the
    record event for event and pays its outcome."""
    for d in data.values():
        for s in d["samples"]:
            ev = [tuple(e) for e in s["meta"]["events"]]
            env = GmSimVecEnv(GAME, 1, config=d["config"])
            obs = env.reset(seed=int(s["meta"]["source"]["seed"]))
            acts = iter(a for seat, a in ev if seat >= 0)
            reward = None
            while not bool(env.finished[0]):
                k = list(env.legal_ids(0)).index(next(acts))
                obs, rew, done, _ = env.step({"action_idx": torch.tensor([k])})
                if bool(done[0]):
                    reward = rew[0].tolist()
            assert [tuple(e) for e in env.trajectory(0).tolist()] == ev
            assert reward == pytest.approx(s["meta"]["outcome"])


# ---------------------------------------------------------------------------
# causality / conformance
# ---------------------------------------------------------------------------

def _cut(t: dict, game: int, n_actions: int) -> dict:
    """The transcript stopped after ``n_actions`` actions of game ``game``."""
    games = [dict(g) for g in t["games"][:game + 1]]
    last = dict(games[-1])
    last["actions"] = last["actions"][:n_actions]
    for k in ("winner", "points"):
        last.pop(k, None)
    games[-1] = last
    out = {k: v for k, v in t.items() if k not in ("winner", "score")}
    out["games"] = games
    return out


def test_converter_causality(gmdata, tmp_path):
    """A match cut after any action converts to a frame-prefix of the whole match's
    stream (frames and candidate targets; the value, soft targets and engine values
    are the outcome / teacher, not the frame)."""
    t = transcripts(GENERATED["m3"][0])[0]
    n = [0]

    def convert(recs):
        n[0] += 1
        d = tmp_path / f"c{n[0]}"
        d.mkdir()
        with gzip.open(d / "t.jsonl.gz", "wt") as fh:
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
        _run(gmdata, GAME, "convert", "--input", d / "t.jsonl.gz", "--out", d / "out",
             "--match-length", 3, "--soft-k", "8", "--engine-values", "--allow-truncated")
        return load_gm_samples(str(d / "out"))

    cuts = []
    for gi, g in enumerate(t["games"]):
        m = len(g["actions"])
        cuts += [_cut(t, gi, k) for k in sorted({1, 2, 3, m // 3, m // 2, m - 1})]
    assert invariance.check_converter_causality(convert, t, cuts) == len(cuts)


def test_simulator_battery():
    """The gmrules conformance battery through gmsim (frames, causality, the action-key
    round trip, the simulator's own trajectories replaying to its frames)."""
    for cfg, n in ((config_of(3), 4), ({"match_length": 0, "jacoby": False}, 6)):
        rep = gmsim.check_game(GAME, cfg, n, 3)
        assert "terminal: " + str(n) in rep and "privacy_checks: 0" in rep, rep


# ---------------------------------------------------------------------------
# model / training
# ---------------------------------------------------------------------------

def test_il_contract_on_real_batches(data):
    from model.games import build_game_network
    from training.train_il_game import apply_meta_arch_defaults, batch_loss, check_meta_arch

    d = data["m3"]
    meta = GameShardDataset(str(d["dir"]), game=GAME).shard_meta
    kw = apply_meta_arch_defaults(GAME, dict(TINY), meta)
    assert kw["cand_max_actions"] == 64 and kw["cand_entity_types"] == 3
    torch.manual_seed(0)
    net = build_game_network(GAME, **kw).train()
    check_meta_arch(GAME, net.cfg, meta)
    b = collate_game_sequences(d["samples"][:2])
    mask = b["meta"]["frame_mask"]
    mb = {"obs": b["obs"], "tgt": b["tgt"], "frame_mask": mask, "labels": None, "legal": None}
    loss, stats, _ = batch_loss(net, mb, CPU, 0.2, 0.0, False)
    assert torch.isfinite(loss)
    assert "engine_value_mse" in stats and stats["engine_value_frac"] > 0.5
    plain = {k: v for k, v in b["tgt"].items() if k in ("action_idx", "value")}
    base, _, _ = batch_loss(net, {**mb, "tgt": plain}, CPU, 0.2, 0.0, False)
    assert not torch.allclose(loss, base), "soft targets / engine values must enter the loss"
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


@pytest.fixture(scope="module")
def stage1(data, tmp_path_factory):
    from training import train_multi_game

    out = tmp_path_factory.mktemp("backgammon_stage1")
    d = data["m3"]["dir"]
    torch.manual_seed(0)
    train_multi_game.main([
        "--data", f"{GAME}={d}", "--val", f"{GAME}={d}", "--eval_every", "2",
        "--eval_batches", "1", "--steps", "3", "--batch_size", "2", "--warmup_steps", "1",
        "--device", "cpu", "--save_dir", str(out / "ck"), "--override", *CORE])
    return out, d


def test_train_multi_game_then_rl(stage1):
    from training import train_rl_joint

    out, d = stage1
    ck = torch.load(out / "ck" / "multi_final.pt", map_location="cpu", weights_only=False)
    assert ck["games"] == [GAME] and ck["configs"][GAME]["cand_max_actions"] == 64
    res = train_rl_joint.main([
        "--init", str(out / "ck" / "multi_final.pt"), "--rl", GAME, "--data", f"{GAME}={d}",
        "--replay_batch_size", "2", "--tokens_per_iter", "60", "--envs", f"{GAME}=2",
        "--env_kw", f"{GAME}:match_length=1", "--snapshot_every", "1", "--epochs", "1",
        "--minibatches", "2", "--snapshot_dir", "memory", "--device", "cpu", "--seed", "3",
        "--iterations", "1", "--save_dir", str(out / "rl")])
    rec = res["history"][-1]
    assert rec["dropped"] == 0 and rec["nonfinite"] == 0
    row = rec["games"][GAME]
    assert row["tokens"] >= 1 and row["episodes"] > 0
    assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
    for k in ("entropy", "kl", "value_loss"):
        assert np.isfinite(row[k]), k
    assert (out / "rl" / "rl_final.pt").exists()


def test_fetch_script_convert_only(gmdata, tmp_path):
    """scripts/fetch_backgammon.sh's conversion half offline (CONVERT_ONLY=1 on raw /
    raw_val transcripts laid out like its generate step's)."""
    script = ROOT / "scripts" / "fetch_backgammon.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    out = tmp_path / "bg"
    for split in ("raw", "raw_val"):
        (out / split).mkdir(parents=True)
        shutil.copy(GENERATED["m3"][0], out / split / "bgmatch-s1-m0-w000.jsonl.gz")
    res = subprocess.run(["bash", str(script), str(out)], capture_output=True, text=True,
                         env={**os.environ, "CONVERT_ONLY": "1", "MATCH_LENGTH": "3"})
    assert res.returncode == 0, res.stderr[-800:]
    n = len(transcripts(GENERATED["m3"][0]))
    for split in ("train", "val"):
        st = json.loads((out / f"convert-{split}.json").read_text().strip().splitlines()[-1])
        assert st["converted"] == n and st["illegal"] == 0 and st["soft_frames"] > 0
        assert GameShardDataset(str(out / split), game=GAME).total_frames() == st["frames"]
    assert "WARNING" not in res.stderr


# ---------------------------------------------------------------------------
# live against gnubg (GM_REFERENCE=1)
# ---------------------------------------------------------------------------

LIVE = pytest.mark.skipif(os.environ.get("GM_REFERENCE") != "1",
                          reason="live reference: GM_REFERENCE=1")


def _live(gmdata, *args) -> Optional[dict]:
    """A gnubg-backed gmdata command: its JSON stats, or a skip when the gnubg bridge
    cannot run here (no JSON printed)."""
    if shutil.which("uv") is None:
        pytest.skip("uv not installed")
    res = subprocess.run([gmdata, GAME, *[str(a) for a in args]], capture_output=True, text=True)
    lines = res.stdout.strip().splitlines()
    if not lines:
        pytest.skip(f"gnubg bridge unavailable: {res.stderr[-400:]}")
    return json.loads(lines[-1])


@LIVE
def test_live_legal_moves_vs_gnubg(gmdata):
    n = int(os.environ.get("GM_REFERENCE_POSITIONS", "3000"))
    seed = int(os.environ.get("GM_REFERENCE_SEED", "1"))
    st = _live(gmdata, "diff", "--live", "--positions", n, "--seed", seed, "--random-frac", "0.4")
    assert st["mismatches"] == 0 and st["agree"] == n, st["errors"]


@LIVE
def test_live_gnubg_match_loop(gmdata, tmp_path):
    seed = int(os.environ.get("GM_REFERENCE_SEED", "1"))
    out = tmp_path / "selfplay.jsonl.gz"
    cmd = ["uv", "run", "--no-project", "--python", "3.12", "--with", "gnubg==1.1.0a16",
           "python", str(ROOT / "scripts" / "reference" / "gnubg_selfplay_dump.py"),
           "--out", str(out), "--matches", "6", "--match-lengths", "1,3,5", "--money-games", "6",
           "--vs-random-matches", "3", "--vs-random-money", "6", "--plies", "0",
           "--seed", str(seed)]
    if shutil.which("uv") is None or subprocess.run(cmd, capture_output=True).returncode != 0:
        pytest.skip("gnubg unavailable")
    st = _run(gmdata, GAME, "verify", "--input", out, check=False)
    assert st["mismatches"] == 0 and st["transcripts"] == 21, st["errors"]


@LIVE
def test_live_generate_and_convert(gmdata, tmp_path):
    seed = int(os.environ.get("GM_REFERENCE_SEED", "1"))
    st = _live(gmdata, "generate", "--out", tmp_path / "raw", "--matches", "2", "--workers",
               "2", "--seed", seed, "--plies", "0", "--match-length", "3")
    assert st["matches"] == 2 and st["discrepancies"] == 0 and st["failed"] == 0, st
    cv = _run(gmdata, GAME, "convert", "--input", tmp_path / "raw", "--out", tmp_path / "out",
              "--match-length", 3, "--soft-k", "8", "--engine-values")
    assert cv["converted"] == 2 and cv["illegal"] == cv["returns_mismatch"] == 0, cv
