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
