"""Elo-ladder tests: the fit recovers orderings/gaps, and the script runs
end-to-end on tiny checkpoints (paired greedy matches, h2h cache, table)."""

import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from elo_ladder import fit_elo  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


class TestEloFit:
    def test_transitive_ordering_and_gap(self):
        records = [
            {"a": "strong", "b": "mid", "wins_a": 27, "wins_b": 3, "draws": 0},
            {"a": "mid", "b": "weak", "wins_a": 27, "wins_b": 3, "draws": 0},
            {"a": "strong", "b": "weak", "wins_a": 30, "wins_b": 0, "draws": 0},
        ]
        elo = fit_elo(records)
        assert elo["strong"] > elo["mid"] > elo["weak"]
        # 90% winrate ~ 380 Elo; allow the low-K fit wide tolerance
        assert 150 < elo["strong"] - elo["mid"] < 600

    def test_even_matchup_stays_close(self):
        records = [
            {"a": "x", "b": "y", "wins_a": 15, "wins_b": 15, "draws": 2},
        ]
        elo = fit_elo(records)
        assert abs(elo["x"] - elo["y"]) < 30

    def test_deterministic(self):
        records = [
            {"a": "x", "b": "y", "wins_a": 20, "wins_b": 10, "draws": 0},
        ]
        assert fit_elo(records) == fit_elo(records)


class TestEloLadderEndToEnd:
    def test_script_runs_on_tiny_checkpoints(self, tmp_path):
        from model.config import ModelConfig
        from model.network import GeneralsNetwork

        cfg = ModelConfig(d_model=64, n_heads=4, core_type="trm", trm_H_cycles=1,
                          trm_L_cycles=2, trm_block_layers=1)
        refs = tmp_path / "refs"
        refs.mkdir()
        for name, seed in (("ref_a", 0), ("ref_b", 1)):
            torch.manual_seed(seed)
            net = GeneralsNetwork(cfg)
            torch.save({"model_state": net.state_dict(),
                        "config": cfg.__dict__},
                       refs / f"{name}.pt")
        # Candidate is a PHASE-5-STYLE checkpoint (turn_globals, 7 globals,
        # derived army channels): the ladder must mix it with the 4-global
        # refs in the same games.
        cfg5 = ModelConfig(d_model=64, n_heads=4, core_type="trm", trm_H_cycles=1,
                           trm_L_cycles=2, trm_block_layers=1,
                           turn_globals=True, global_features=7,
                           derive_army_channels=True)
        torch.manual_seed(2)
        cand = tmp_path / "cand.pt"
        torch.save({"model_state": GeneralsNetwork(cfg5).state_dict(),
                    "config": cfg5.__dict__}, cand)

        out = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "elo_ladder.py"),
             "--refs_dir", str(refs), "--candidates", str(cand),
             "--games", "2", "--rollout_steps", "25", "--play_refs",
             "--device", "cpu"],
            capture_output=True, text=True, cwd=ROOT, timeout=600,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        assert "vs_best_ref" in out.stdout
        h2h = json.load(open(refs / "h2h.json"))
        # 1 ref-vs-ref pairing + candidate vs 2 refs
        assert len(h2h) == 3
        for r in h2h:
            assert r["wins_a"] + r["wins_b"] + r["draws"] == 2

        # Re-run: everything cached, no new games, still fits
        out2 = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "elo_ladder.py"),
             "--refs_dir", str(refs), "--candidates", str(cand),
             "--games", "2", "--rollout_steps", "25", "--play_refs",
             "--device", "cpu"],
            capture_output=True, text=True, cwd=ROOT, timeout=300,
        )
        assert out2.returncode == 0, out2.stderr[-2000:]
        assert len(json.load(open(refs / "h2h.json"))) == 3
