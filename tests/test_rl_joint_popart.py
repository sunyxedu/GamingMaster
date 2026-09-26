"""training/rl_joint/popart.py: output preservation, statistics, passthrough,
mirrors (EMA shadow), persistence — on this repo's ValueHead / HLGaussValueHead
and on a real MultiGameNetwork."""

import pytest
import torch
import torch.nn as nn

from model.components.heads import HLGaussValueHead, ValueHead
from model.multi_game import build_multi_game
from model.optim import ModelEMA
from tests.test_core_moe import CORE
from training.rl_joint import PASSTHROUGH, POPART, PopArt, final_linear, value_loss


def _heads(d=16, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    return nn.ModuleDict({"a": ValueHead(d), "b": ValueHead(d),
                          "hl": HLGaussValueHead(d, n_bins=32, vmax=1.5)}).to(dtype)


def _targets(seed, scale, shift, n=64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g, dtype=torch.float64) * scale + shift


class TestPreservesOutputs:
    def test_value_head_outputs_preserved_across_updates(self):
        heads = _heads()
        pa = PopArt(heads, decay=0.7)
        h = torch.randn(32, 16, dtype=torch.float64)
        with torch.no_grad():
            before = {g: pa.denormalize(g, heads[g](h)) for g in ("a", "b")}
        # very different scales per game, several updates, a masked batch
        for i, (sa, sb) in enumerate([(100.0, 0.01), (3.0, 50.0), (1e3, 1e-3)]):
            mask = torch.rand(64) > 0.3
            tfs = pa.update({"a": _targets(i, sa, -2 * sa),
                             "b": (_targets(10 + i, sb, sb), mask)}, sync=False)
            assert set(tfs) == {"a", "b"} and not tfs["a"].is_identity
            with torch.no_grad():
                for g in ("a", "b"):
                    torch.testing.assert_close(pa.denormalize(g, heads[g](h)), before[g],
                                               rtol=1e-9, atol=1e-9)

    def test_rescale_formula(self):
        heads = _heads()
        pa = PopArt({"a": heads["a"]}, decay=0.5)
        layer = final_linear(heads["a"])
        w0, b0 = layer.weight.detach().clone(), layer.bias.detach().clone()
        tf = pa.update({"a": _targets(0, 4.0, 3.0)}, sync=False)["a"]
        assert (tf.mu_old, tf.sigma_old) == (0.0, 1.0)
        torch.testing.assert_close(layer.weight, w0 * tf.sigma_old / tf.sigma_new)
        torch.testing.assert_close(
            layer.bias, (tf.sigma_old * b0 + tf.mu_old - tf.mu_new) / tf.sigma_new)

    def test_real_multi_game_network(self):
        torch.manual_seed(0)
        mg = build_multi_game(["othello", "chess"], CORE, {"chess": {"hl_gauss_value": True}})
        mg.eval()
        pa = PopArt.from_model(mg, decay=0.9)
        assert pa.modes == {"othello": POPART, "chess": PASSTHROUGH}
        obs = {"board_seq": torch.rand(2, 3, 3, 8, 8)}
        chess_obs = {"board_seq": torch.rand(2, 3, 14, 8, 8), "globals_seq": torch.rand(2, 3, 8)}
        with torch.no_grad():
            v0 = pa.denormalize("othello", mg["othello"](obs)["value"])
            c0 = mg["chess"](chess_obs)["value"]
        pa.update({"othello": _targets(1, 20.0, 5.0).float(),
                   "chess": _targets(2, 0.5, 0.0).float()}, sync=False)
        with torch.no_grad():
            v1 = pa.denormalize("othello", mg["othello"](obs)["value"])
            c1 = mg["chess"](chess_obs)["value"]
        torch.testing.assert_close(v1, v0, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(c1, c0, rtol=0, atol=0)        # untouched
        assert abs(pa.mean("othello") - 5.0) < 5.0 and pa.std("othello") > 5.0


class TestStatistics:
    def test_first_update_gives_exact_batch_moments(self):
        pa = PopArt(_heads(), decay=0.99)
        x = _targets(0, 7.0, 2.0)
        pa.update({"a": x}, sync=False)
        assert pa.mean("a") == pytest.approx(float(x.mean()), rel=1e-12)
        assert pa.std("a") == pytest.approx(float(x.std(unbiased=False)), rel=1e-9)
        assert (pa.mean("b"), pa.std("b")) == (0.0, 1.0)       # no data: identity

    def test_ema_tracks_shifted_targets(self):
        pa = PopArt(_heads(), decay=0.8)
        for i in range(60):
            pa.update({"a": _targets(i, 2.0, 10.0, n=256)}, sync=False)
        assert pa.mean("a") == pytest.approx(10.0, abs=0.2)
        assert pa.std("a") == pytest.approx(2.0, rel=0.1)

    def test_normalize_inverts_denormalize_and_std_floor(self):
        pa = PopArt(_heads(), decay=0.5, std_min=0.1)
        pa.update({"a": _targets(3, 5.0, -1.0), "b": torch.full((8,), 4.0)}, sync=False)
        x = torch.randn(10, dtype=torch.float64)
        torch.testing.assert_close(pa.normalize("a", pa.denormalize("a", x)), x)
        assert pa.std("b") == 0.1                                 # constant targets
        assert pa.mean("b") == pytest.approx(4.0)

    def test_art_value_loss_has_unit_scale(self):
        """The normalised critic loss of a game is invariant to its reward
        scale — the point of ART in a multi-game critic."""
        losses = []
        for scale in (1.0, 1000.0):
            heads = _heads(seed=1)
            pa = PopArt({"a": heads["a"]}, decay=0.5)
            G = _targets(4, 3.0 * scale, scale)
            pa.update({"a": G}, sync=False)
            h = torch.randn(64, 16, dtype=torch.float64)
            with torch.no_grad():
                pred = heads["a"](h)
            losses.append(float(value_loss(pred, pa.normalize("a", G))["loss"]))
        assert losses[1] == pytest.approx(losses[0], rel=0.05)


class TestPassthrough:
    def test_hl_gauss_head_is_identity_and_untouched(self):
        heads = _heads()
        pa = PopArt(heads)
        assert pa.mode("hl") == PASSTHROUGH and pa.mode("a") == POPART
        state = {k: v.clone() for k, v in heads["hl"].state_dict().items()}
        tfs = pa.update({"hl": _targets(0, 0.5, 0.2)}, sync=False)
        assert "hl" not in tfs
        for k, v in heads["hl"].state_dict().items():
            assert torch.equal(v, state[k])
        x = torch.randn(5, dtype=torch.float64)
        assert pa.normalize("hl", x) is x and pa.denormalize("hl", x) is x
        assert pa.target_mean("hl") == pytest.approx(float(_targets(0, 0.5, 0.2).mean()))

    def test_forced_passthrough_and_invalid_modes(self):
        heads = _heads()
        pa = PopArt(heads, modes={"a": PASSTHROUGH})
        assert pa.mode("a") == PASSTHROUGH
        with pytest.raises(ValueError, match="HL-Gauss"):
            PopArt(heads, modes={"hl": POPART})
        with pytest.raises(ValueError, match="unknown PopArt mode"):
            PopArt(heads, modes={"a": "other"})
        with pytest.raises(KeyError):
            PopArt(heads, modes={"zzz": POPART})
        with pytest.raises(ValueError, match="scalar output"):
            PopArt({"x": nn.Linear(4, 3)})


class TestMirrors:
    def test_ema_shadow_rescaled_consistently(self):
        torch.manual_seed(0)
        model = nn.ModuleDict({"a": ValueHead(8), "b": ValueHead(8)}).double()
        ema = ModelEMA(model, mu=0.5)
        with torch.no_grad():                      # make the shadow differ from the live weights
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ema.update(model)
        ema_model = ema.ema_copy(model)
        h = torch.randn(16, 8, dtype=torch.float64)
        pa = PopArt(model, decay=0.5)
        with torch.no_grad():
            live0 = pa.denormalize("a", model["a"](h))
            shadow0 = pa.denormalize("a", ema_model["a"](h))
        mirror = pa.mirror_tensors(model, ema.shadow)
        assert set(mirror) == {"a", "b"}
        pa.update({"a": _targets(0, 30.0, -4.0)}, sync=False, mirrors=[mirror])
        ema.copy_to(ema_model)
        with torch.no_grad():
            torch.testing.assert_close(pa.denormalize("a", model["a"](h)), live0)
            torch.testing.assert_close(pa.denormalize("a", ema_model["a"](h)), shadow0)

    def test_module_mirror_and_alias_guard(self):
        heads = _heads()
        copy = _heads()
        copy.load_state_dict(heads.state_dict())
        pa = PopArt({"a": heads["a"]}, decay=0.5)
        pa.update({"a": _targets(0, 9.0, 1.0)}, sync=False, mirrors=[{"a": copy["a"]}])
        for p, q in zip(heads["a"].parameters(), copy["a"].parameters()):
            torch.testing.assert_close(p, q)
        with pytest.raises(ValueError, match="aliases"):
            pa.update({"a": _targets(1, 9.0, 1.0)}, sync=False, mirrors=[{"a": heads["a"]}])


class TestPersistence:
    def test_state_dict_round_trip(self):
        heads = _heads()
        pa = PopArt(heads, decay=0.9)
        for i in range(3):
            pa.update({"a": _targets(i, 3.0, 1.0), "hl": _targets(9, 0.3, 0.0)}, sync=False)
        state = pa.state_dict()
        fresh = PopArt(_heads(), decay=0.9)
        fresh.load_state_dict(state)
        for g in ("a", "b", "hl"):
            assert fresh.mean(g) == pa.mean(g) and fresh.std(g) == pa.std(g)
            assert fresh.target_mean(g) == pa.target_mean(g)
        # identical future updates
        x = _targets(5, 2.0, 0.0)
        assert pa.update({"a": x}, sync=False)["a"] == fresh.update({"a": x}, sync=False)["a"]
        state = torch.load(_roundtrip(state), weights_only=True)   # plain types only
        fresh.load_state_dict(state)

    def test_mismatches(self):
        pa = PopArt(_heads())
        state = pa.state_dict()
        with pytest.raises(ValueError, match="mode"):
            PopArt(_heads(), modes={"a": PASSTHROUGH}).load_state_dict(state)
        sub = PopArt({"a": _heads()["a"]})
        with pytest.raises(KeyError):
            sub.load_state_dict(state)
        sub.load_state_dict(state, strict=False)


def _roundtrip(obj):
    import io
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    return buf
