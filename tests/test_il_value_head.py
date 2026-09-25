"""IL value head: HL-Gauss cross-entropy for the categorical head, MSE for the
scalar head, value_mse comparable across both, the target-support guard, and
the multi-game trainer / evaluator end to end with hl_gauss_value=True."""

import json
import math

import pytest
import torch
import torch.nn.functional as F

from data_processing.games.dataset import (
    VALUE_TARGET_BOUND,
    collate_game_sequences,
    save_shard,
)
from model.config import ModelConfig
from model.games import build_game_network
from model.losses import RLLoss, hl_gauss_cross_entropy, hl_gauss_target_probs
from training.train_il_game import batch_loss, check_value_support, prepare_batch

TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, encoder_channels=16,
            dropout=0.0, rd_prelude_layers=1, rd_recurrent_layers=1, rd_coda_layers=1,
            rd_mean_iters=2, rd_backprop_iters=1)
VMAX, SIGMA = 1.5, 0.06          # ModelConfig defaults
TARGETS = [-1.0, -1.0 / 3.0, 0.0, 0.5, 1.0]


# ---------------------------------------------------------------------------
# the loss itself
# ---------------------------------------------------------------------------

def test_defaults_are_the_documented_ones():
    cfg = ModelConfig()
    assert (cfg.hl_gauss_bins, cfg.hl_gauss_vmax, cfg.hl_gauss_sigma) == (128, VMAX, SIGMA)
    assert cfg.hl_gauss_vmax > VALUE_TARGET_BOUND        # IL targets inside the support


def test_ce_gradcheck():
    """Autograd gradient of the CE matches finite differences (fp64)."""
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 5, 16, generator=g, dtype=torch.float64, requires_grad=True)
    y = torch.rand(3, 5, generator=g) * 2 - 1
    assert torch.autograd.gradcheck(
        lambda lg: hl_gauss_cross_entropy(lg, y, VMAX, SIGMA), (logits,))


def test_ce_gradient_is_softmax_minus_target():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(4, 64, generator=g, requires_grad=True)
    y = torch.tensor(TARGETS[:4])
    hl_gauss_cross_entropy(logits, y, VMAX, SIGMA).sum().backward()
    expect = torch.softmax(logits.detach(), -1) - hl_gauss_target_probs(y, 64, VMAX, SIGMA)
    assert torch.allclose(logits.grad, expect, atol=1e-6)


@pytest.mark.parametrize("y", TARGETS)
def test_ce_minimised_at_the_target_distribution(y):
    """Stationary point (zero gradient) at logits = log p*(y), the value there is
    the target entropy, and every perturbation increases the loss."""
    n = 128
    yt = torch.tensor([y])
    p_star = hl_gauss_target_probs(yt, n, VMAX, SIGMA)
    logits = p_star.clamp_min(1e-30).log().requires_grad_(True)
    ce = hl_gauss_cross_entropy(logits, yt, VMAX, SIGMA)
    ce.sum().backward()
    assert logits.grad.abs().max() < 1e-6
    entropy = -(p_star * p_star.clamp_min(1e-30).log()).sum(-1)
    assert torch.allclose(ce.detach(), entropy, atol=1e-5)
    g = torch.Generator().manual_seed(2)
    for _ in range(5):
        bumped = logits.detach() + 0.3 * torch.randn(1, n, generator=g)
        assert hl_gauss_cross_entropy(bumped, yt, VMAX, SIGMA) > ce.detach()


@pytest.mark.parametrize("y", TARGETS)
def test_ce_minimised_at_the_target_location(y):
    """Among predicted distributions of the target family p*(mu), the loss is
    stationary and minimal at mu = y."""
    n = 128

    def loss_at(mu):
        logits = hl_gauss_target_probs(mu, n, VMAX, SIGMA).clamp_min(1e-30).log()
        return hl_gauss_cross_entropy(logits, torch.tensor([y]), VMAX, SIGMA).sum()

    mu = torch.tensor([y], requires_grad=True)
    loss_at(mu).backward()
    assert abs(float(mu.grad)) < 1e-2          # vs O(1/σ) ≈ 17 per unit off-target
    base = float(loss_at(torch.tensor([y])))
    for d in (-0.05, -0.01, 0.01, 0.05):
        assert float(loss_at(torch.tensor([y + d]))) > base


@pytest.mark.parametrize("y", TARGETS)
def test_gradient_descent_recovers_the_target(y):
    """Free logits trained on the CE converge to an expectation at the target
    (bin midpoints; the head's linspace centres read it as y·n/(n−1))."""
    n = 128
    logits = torch.zeros(1, n, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.1)
    yt = torch.tensor([y])
    for _ in range(400):
        opt.zero_grad()
        hl_gauss_cross_entropy(logits, yt, VMAX, SIGMA).sum().backward()
        opt.step()
    p = torch.softmax(logits.detach(), -1)[0]
    edges = torch.linspace(-VMAX, VMAX, n + 1)
    mids = (edges[1:] + edges[:-1]) / 2
    centers = torch.linspace(-VMAX, VMAX, n)
    assert abs(float((p * mids).sum()) - y) < 5e-3
    assert abs(float((p * centers).sum()) - y * n / (n - 1)) < 5e-3


def test_default_support_leaves_the_extreme_targets_untruncated():
    """Why vmax = 1.5 for IL targets in [-1, 1]: the ±1 target Gaussian lies
    inside the support, so the smeared target's mean is exactly ±1; a support
    ending at 1 would renormalise half of it away and pull the mean inward."""
    n = 128

    def smeared_mean(y, vmax):
        edges = torch.linspace(-vmax, vmax, n + 1)
        mids = (edges[1:] + edges[:-1]) / 2
        return float((hl_gauss_target_probs(torch.tensor([y]), n, vmax, SIGMA)[0] * mids).sum())

    for y in (-1.0, 1.0):
        assert abs(smeared_mean(y, VMAX) - y) < 1e-4
        assert abs(smeared_mean(y, 1.0) - y) > 0.8 * SIGMA * 0.9


def test_rl_loss_value_term_is_the_shared_ce():
    """The RL critic loss now calls hl_gauss_cross_entropy: bit-identical to
    the formula it had inline (RL behaviour unchanged)."""
    g = torch.Generator().manual_seed(3)
    B, T, n = 2, 7, 32
    returns = (torch.randn(B, T, generator=g) * 0.7).clamp(-1.4, 1.4)
    value_logits = torch.randn(B, T, n, generator=g)
    mask = torch.rand(B, T, generator=g) > 0.2
    crit = RLLoss(hl_gauss=True, hl_gauss_vmax=VMAX, hl_gauss_sigma=SIGMA)
    out = crit(log_probs=torch.zeros(B, T), entropy=torch.zeros(B, T),
               values=torch.zeros(B, T), returns=returns,
               advantages=torch.zeros(B, T), old_log_probs=torch.zeros(B, T),
               mask=mask, value_logits=value_logits)
    target_p = hl_gauss_target_probs(returns, n, VMAX, SIGMA)
    old_ce = -(target_p * F.log_softmax(value_logits.float(), dim=-1)).sum(-1)
    m = mask.float()
    assert torch.equal(out["value_loss"], ((old_ce * m).sum() / m.sum().clamp(min=1)))
    assert torch.equal(hl_gauss_cross_entropy(value_logits, returns, VMAX, SIGMA), old_ce)


# ---------------------------------------------------------------------------
# batch_loss wiring
# ---------------------------------------------------------------------------

def _othello_samples(n=3, t0=5, value=None):
    out = []
    for i in range(n):
        T = t0 + i
        v = torch.rand(T) * 2 - 1 if value is None else torch.full((T,), float(value))
        out.append({"obs": {"board_seq": torch.rand(T, 3, 8, 8)},
                    "tgt": {"place_idx": torch.randint(0, 64, (T,)),
                            "passed": torch.zeros(T, dtype=torch.bool),
                            "value": v},
                    "meta": {"src": f"g{i}"}})
    return out


def _net(hl: bool, **kw):
    torch.manual_seed(0)
    extra = dict(hl_gauss_value=True, hl_gauss_bins=32) if hl else {}
    return build_game_network("othello", **TINY, **extra, **kw).eval()


def _mb(samples):
    return prepare_batch("othello", collate_game_sequences(samples),
                         torch.device("cpu"), tf_prob=1.0)


def _run(net, mb, value_weight=0.2):
    return batch_loss(net, mb, torch.device("cpu"), value_weight=value_weight,
                      q_halt_weight=0.5, bf16=False, act_steps=2)


def _masked_mean(x, mask):
    m = mask.float()
    return (x * m).sum() / m.sum().clamp(min=1)


def test_scalar_head_value_mse_is_the_mse_value_loss():
    torch.manual_seed(1)
    net = _net(hl=False)
    mb = _mb(_othello_samples())
    mb["tgt"]["value"][~mb["frame_mask"]] = 7.0          # padding: excluded, no guard
    loss, stats, pred = _run(net, mb)
    assert "value_logits" not in pred
    mse = _masked_mean((pred["value"].detach() - mb["tgt"]["value"]) ** 2, mb["frame_mask"])
    assert stats["value_mse"] == stats["value_loss"]
    assert stats["value_loss"] == pytest.approx(float(mse), rel=1e-6)
    # the value term enters the loss as value_weight * MSE, as before
    loss0, _, _ = _run(net, mb, value_weight=0.0)
    assert float((loss - loss0).detach()) == pytest.approx(0.2 * float(mse), rel=1e-5)


def test_scalar_head_accepts_out_of_range_targets():
    """The scalar head has no support: behaviour unchanged, no guard."""
    net = _net(hl=False)
    loss, stats, _ = _run(net, _mb(_othello_samples(value=3.0)))
    assert math.isfinite(stats["value_loss"])


def test_hl_gauss_head_trains_on_ce_and_reports_mse():
    torch.manual_seed(2)
    net = _net(hl=True)
    mb = _mb(_othello_samples())
    mb["tgt"]["value"][~mb["frame_mask"]] = 7.0          # padding: excluded from loss + guard
    loss, stats, pred = _run(net, mb)
    tgt, fm = mb["tgt"]["value"], mb["frame_mask"]
    assert pred["value_logits"].shape == (*fm.shape, 32)
    target_p = hl_gauss_target_probs(tgt, 32, VMAX, SIGMA)
    ce = -(target_p * F.log_softmax(pred["value_logits"].detach().float(), -1)).sum(-1)
    assert stats["value_loss"] == pytest.approx(float(_masked_mean(ce, fm)), rel=1e-5)
    mse = _masked_mean((pred["value"].detach() - tgt) ** 2, fm)
    assert stats["value_mse"] == pytest.approx(float(mse), rel=1e-5)
    assert stats["value_loss"] != pytest.approx(stats["value_mse"])
    loss0, _, _ = _run(net, mb, value_weight=0.0)
    assert float((loss - loss0).detach()) == pytest.approx(0.2 * stats["value_loss"], rel=1e-4)
    loss.backward()
    head_out = net.value_head.net[-1].weight.grad
    assert head_out is not None and head_out.abs().sum() > 0


def test_hl_gauss_value_training_moves_the_expectation_to_the_target():
    """A few optimizer steps on the CE alone pull the head's expectation to
    a constant target (value_mse falls)."""
    torch.manual_seed(3)
    net = build_game_network("othello", **TINY, hl_gauss_value=True, hl_gauss_bins=32)
    net.train()
    mb = _mb(_othello_samples(value=0.6))
    opt = torch.optim.Adam(net.value_head.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        opt.zero_grad()
        _, stats, pred = batch_loss(net, mb, torch.device("cpu"), value_weight=1.0,
                                    q_halt_weight=0.0, bf16=False, act_steps=1)
        # value term only: rebuild it from pred so the policy loss is ignored
        ce = hl_gauss_cross_entropy(pred["value_logits"], mb["tgt"]["value"], VMAX, SIGMA)
        _masked_mean(ce, mb["frame_mask"]).backward()
        opt.step()
        first = stats["value_mse"] if first is None else first
    assert stats["value_mse"] < 0.1 * first


def test_guard_rejects_targets_outside_the_support():
    net = _net(hl=True)
    for bad in (2.0, -1.6, float("nan")):
        mb = _mb(_othello_samples())
        mb["tgt"]["value"][0, 1] = bad
        with pytest.raises(ValueError, match="outside the HL-Gauss support"):
            _run(net, mb)
    # a bespoke support is honoured
    net_wide = _net(hl=True, hl_gauss_vmax=2.5)
    mb = _mb(_othello_samples())
    mb["tgt"]["value"][0, 1] = 2.0
    _run(net_wide, mb)


def test_check_value_support_masks_and_edges():
    v = torch.tensor([[1.5, -1.5, 9.0]])
    check_value_support(v, torch.tensor([[True, True, False]]), 1.5)   # edges ok, pad ignored
    with pytest.raises(ValueError):
        check_value_support(v, torch.tensor([[True, True, True]]), 1.5)


# ---------------------------------------------------------------------------
# multi-game trainer + evaluator end to end
# ---------------------------------------------------------------------------

def _board_samples(game, n, T):
    out = []
    for i in range(n):
        # W/D/L outcomes, as the converters emit
        tgt = {"value": torch.randint(-1, 2, (1,)).float().expand(T).clone()}
        if game == "chess":
            tgt.update(from_idx=torch.randint(0, 64, (T,)), to_idx=torch.randint(0, 64, (T,)),
                       promotion=torch.zeros(T, dtype=torch.long))
            obs = {"board_seq": torch.rand(T, 14, 8, 8), "globals_seq": torch.rand(T, 8)}
        else:
            tgt.update(place_idx=torch.randint(0, 64, (T,)),
                       passed=torch.zeros(T, dtype=torch.bool))
            obs = {"board_seq": torch.rand(T, 3, 8, 8)}
        out.append({"obs": obs, "tgt": tgt, "meta": {"src": f"{game}{i}"}})
    return out


def test_train_and_eval_multi_game_with_hl_gauss_value(tmp_path, capsys):
    from training import eval_multi_game, train_multi_game

    torch.manual_seed(0)
    for g in ("othello", "chess"):
        (tmp_path / g).mkdir()
        save_shard(str(tmp_path / g / "a.pt"), g, _board_samples(g, 4, 6))
    core = [f"{k}={v}" for k, v in TINY.items()]
    train_multi_game.main([
        "--data", f"othello={tmp_path / 'othello'}", "--data", f"chess={tmp_path / 'chess'}",
        "--val", f"othello={tmp_path / 'othello'}", "--val", f"chess={tmp_path / 'chess'}",
        "--steps", "4", "--batch_size", "2", "--warmup_steps", "1", "--eval_every", "2",
        "--log_every", "2", "--save_dir", str(tmp_path / "ck"), "--device", "cpu",
        "--override", *core, "hl_gauss_value=True", "hl_gauss_bins=48"])
    out = capsys.readouterr().out
    assert "value_head=hl_gauss(bins=48,vmax=1.5,sigma=0.06)" in out
    val_lines = [ln for ln in out.splitlines() if ln.startswith("[val 4]")]
    assert len(val_lines) == 2
    assert all("value_mse=" in ln and "value_loss=" in ln for ln in val_lines)

    ck = torch.load(tmp_path / "ck" / "multi_final.pt", weights_only=False)
    for g in ("othello", "chess"):
        assert ck["configs"][g]["hl_gauss_value"] is True
        # per-game categorical value heads (not part of the shared core)
        assert ck["model_state"][f"nets.{g}.value_head.net.2.weight"].shape == (48, 32)
        assert not any(k.startswith("value_head") for k in ck["core_state"])

    model = eval_multi_game.load_multi(str(tmp_path / "ck" / "multi_final.pt"),
                                       torch.device("cpu"))
    assert model["othello"].value_head is not model["chess"].value_head
    js = tmp_path / "eval.json"
    eval_multi_game.main(["--ckpt", str(tmp_path / "ck" / "multi_final.pt"),
                          "--val", f"othello={tmp_path / 'othello'}",
                          "--val", f"chess={tmp_path / 'chess'}",
                          "--device", "cpu", "--json", str(js)])
    printed = capsys.readouterr().out
    assert "value_mse=" in printed
    res = json.loads(js.read_text())
    for g in ("othello", "chess"):
        (m,) = res[g].values()
        assert math.isfinite(m["value_mse"]) and math.isfinite(m["value_loss"])
        # CE (≥ target entropy ≈ log(σ/bin·√(2πe)) > 1) vs an MSE in [0, 4]
        assert m["value_loss"] > 1.0 and 0.0 <= m["value_mse"] <= 4.0


def test_multi_game_hl_gauss_guard_fires_on_bad_targets(tmp_path):
    from training import train_multi_game

    (tmp_path / "othello").mkdir()
    samples = _board_samples("othello", 2, 5)
    for s in samples:
        s["tgt"]["value"] = torch.full((5,), 2.0)        # raw score, not an outcome
    save_shard(str(tmp_path / "othello" / "a.pt"), "othello", samples)
    with pytest.raises(ValueError, match="outside the HL-Gauss support"):
        train_multi_game.main([
            "--data", f"othello={tmp_path / 'othello'}", "--steps", "1", "--batch_size", "2",
            "--warmup_steps", "1", "--save_dir", str(tmp_path / "ck"), "--device", "cpu",
            "--override", *[f"{k}={v}" for k, v in TINY.items()], "hl_gauss_value=True"])
