"""train_multi_game mixture weights: w_g ∝ m_g · min(N_g, C_g)^α."""
import math

import pytest

from training.train_multi_game import mixture_weights, parse_args


def test_plain_temperature_mixture_is_unchanged():
    w = mixture_weights({"a": 100, "b": 400}, 0.5)
    assert math.isclose(w["a"], 10 / 30) and math.isclose(w["b"], 20 / 30)


def test_frame_caps_and_game_weights():
    frames = {"big": 10**10, "mid": 10**8, "small": 10**6}
    w = mixture_weights(frames, 0.5, frame_cap=10**8)
    assert math.isclose(w["big"], w["mid"]) and math.isclose(w["mid"] / w["small"], 10.0)
    w2 = mixture_weights(frames, 0.5, game_frame_cap={"big": 10**6})
    assert math.isclose(w2["big"], w2["small"])
    w3 = mixture_weights(frames, 0.5, frame_cap=10**8, game_weight={"small": 0.5})
    assert math.isclose(w3["mid"] / w3["small"], 20.0)
    with pytest.raises(SystemExit):
        mixture_weights({"a": 1}, 0.5, game_weight={"a": 0.0})


def test_cli_parses_scientific_caps():
    a = parse_args(["--data", "chess=x", "--mixture_frame_cap", "1e8",
                    "--game_frame_cap", "mahjong=2e7", "--game_weight", "pokemon=0.5"])
    assert a.mixture_frame_cap == 10**8
    assert a.game_frame_cap == ["mahjong=2e7"] and a.game_weight == ["pokemon=0.5"]


def test_expected_epochs_counts_passes_over_the_uncapped_frames():
    from training.train_multi_game import expected_epochs

    # 100 samples of 200 frames cropped to 50: a draw of batch 4 sees 200 frames
    stats = {"a": {"samples": 100, "frames": 5000, "raw_frames": 20000}}
    e = expected_epochs({"a": 0.5}, stats, {"a": 4}, draws=1000)
    assert math.isclose(e["a"], 1000 * 0.5 * 4 * 50 / 20000)


def test_cap_mixture_is_the_kl_projection_onto_the_caps():
    from training.train_multi_game import cap_mixture

    w = {"a": 0.5, "b": 0.3, "c": 0.2}
    same, capped = cap_mixture(w, {"a": 1.0, "b": 1.0, "c": 1.0})
    assert same == pytest.approx(w) and capped == []
    out, capped = cap_mixture(w, {"a": 0.1})
    assert capped == ["a"] and out["a"] == pytest.approx(0.1)
    assert sum(out.values()) == pytest.approx(1.0)
    assert out["b"] / out["c"] == pytest.approx(1.5)           # uncapped ratios kept
    # a cap that binds only after the first redistribution (water-filling)
    out, capped = cap_mixture(w, {"a": 0.1, "b": 0.5})
    assert capped == ["a", "b"] and out == pytest.approx({"a": 0.1, "b": 0.5, "c": 0.4})
    # KKT: w' = min(u, c·w) for one c — no other feasible point is KL-closer
    c = out["c"] / w["c"]
    for g in w:
        assert out[g] == pytest.approx(min({"a": 0.1, "b": 0.5}.get(g, math.inf), c * w[g]))


def test_cap_mixture_all_capped_scales_the_caps():
    from training.train_multi_game import cap_mixture

    out, capped = cap_mixture({"a": 0.5, "b": 0.5}, {"a": 0.1, "b": 0.3})
    assert capped == ["a", "b"] and out == pytest.approx({"a": 0.25, "b": 0.75})


def test_max_epochs_caps_the_small_corpus_in_a_run(tmp_path, capsys, monkeypatch):
    """End to end: a 6-sample game and a 40-sample game; --max_epochs holds the
    small one at its cap, the checkpoint's mixture is the capped one, and the
    [mixture] log reports every game's expected epochs."""
    import torch

    from data_processing.games.dataset import save_shard
    from tests.test_core_moe import _board_samples
    from tests.test_zero1 import CORE_TF
    from training import train_multi_game as tmg

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    for g, n in (("othello", 6), ("chess", 40)):
        (tmp_path / g).mkdir()
        # random frames, served as they are (layout per_seat: no seat split)
        save_shard(str(tmp_path / g / "a.pt"), g, _board_samples(g, n, 6), meta={"layout": "per_seat"})
    steps, per_step, bs, cap = 20, 2, 2, 3.0
    tmg.main(["--data", f"othello={tmp_path}/othello", "--data", f"chess={tmp_path}/chess",
              "--steps", str(steps), "--games_per_step", str(per_step), "--batch_size", str(bs),
              "--max_epochs", str(cap), "--eval_every", "0", "--log_every", "10",
              "--device", "cpu", "--save_dir", str(tmp_path / "out"),
              "--override", *[f"{k}={v}" for k, v in CORE_TF.items()]])
    log = capsys.readouterr().out
    mix = next(line for line in log.splitlines() if line.startswith("[mixture]"))
    assert "epochs=" in mix and "(max)" in mix
    step = next(line for line in log.splitlines() if line.startswith("[step 10/"))
    assert "moe_load_cv=" in step and "moe_win_cv=" in step and "moe_win_min=" in step
    ck = torch.load(tmp_path / "out" / "multi_final.pt", map_location="cpu", weights_only=False)
    # othello: 6 samples of 6 frames (uncropped), batch 2: E = 20·2·w·2·6/36 = 3 at the cap
    w = ck["mixture"]
    assert w["othello"] == pytest.approx(cap * 6 * 6 / (steps * per_step * bs * 6))
    assert w["chess"] == pytest.approx(1 - w["othello"])
