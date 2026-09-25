"""QK-normalisation (cfg.qk_norm): per-head q/k RMSNorm before every
attention dot product of the Stage-2 core blocks and the Stage-1
spatial / temporal attention.

* new configs default ON; checkpoint configs without the field load OFF and
  rebuild exactly the historical parameter set (strict load, identical
  outputs);
* the new submodules exist only when enabled; forward/step parity holds for
  every core × temporal mixer, and the eval step stays sync-free /
  CUDA-graph-capturable;
* the point of it: every attention logit is bounded by
  √head_dim · max|g_q| · max|g_k| whatever the activation or q/k weight
  scale, so the attention cannot collapse the way it did in the diverged
  lr-3e-4 recurrent-depth go run (attention logit growth, Wortsman et al.
  2023).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from model.components.core import (
    QK_NORM_INIT_CAP,
    CoreBlock,
    MoEFFN,
    QKNorm,
    qk_norm_init_scale,
)
from model.components.transformer import SpatiotemporalBlock
from model.config import ModelConfig, config_from_dict, upgrade_config_dict
from model.game_network import GameNetwork
from model.multi_game import (
    build_multi_game,
    core_config_overrides,
    core_state_dict,
    freeze_core,
)
from model.network import GeneralsNetwork
from tests.test_inference_graph import _forbid_host_syncs, _legal, _obs

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=2, encoder_channels=16,
            dropout=0.0, rd_prelude_layers=1, rd_recurrent_layers=1, rd_coda_layers=1,
            rd_mean_iters=2, rd_backprop_iters=1, core_layers=2)
D_HEAD = TINY["d_model"] // TINY["n_heads"]
G0 = qk_norm_init_scale(D_HEAD)          # initial QK scale (8^¼ for 8-dim heads)
INIT_CAP = D_HEAD ** 0.5 * G0 * G0       # = QK_NORM_INIT_CAP (8) for small heads
CORES = ("recurrent_depth", "transformer")
MIXERS = ("attention", "recurrent")


def _net(core="recurrent_depth", mixer="recurrent", qk_norm=True, moe=0, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = ModelConfig(**{**TINY, **kw}, core_type=core, temporal_mixer=mixer,
                      qk_norm=qk_norm, moe_experts=moe)
    return GeneralsNetwork(cfg).eval()


def _frames(B=2, T=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, T, 12, 25, 25, generator=g), torch.randn(B, T, 4, generator=g)


def _qk_names(net):
    return [n for n, _ in net.named_parameters() if "qk_norm" in n]


# ---------------------------------------------------------------------------
# config: new default on, old checkpoints off
# ---------------------------------------------------------------------------

def test_new_configs_on_old_checkpoint_configs_off():
    assert ModelConfig().qk_norm is True
    assert config_from_dict({"d_model": 32, "n_heads": 4}).qk_norm is False
    assert upgrade_config_dict({})["qk_norm"] is False
    assert upgrade_config_dict({"qk_norm": True})["qk_norm"] is True
    # the shared-core transfer path (--init_core, load_core_checkpoint) too
    assert core_config_overrides({"d_model": 32})["qk_norm"] is False
    assert core_config_overrides(ModelConfig())["qk_norm"] is True


def test_games_must_agree_on_qk_norm():
    core = dict(TINY, core_type="recurrent_depth", n_st_layers=1)
    with pytest.raises(ValueError, match="qk_norm"):
        build_multi_game(["othello", "chess"], core, {"chess": {"qk_norm": False}})


# ---------------------------------------------------------------------------
# builds: new submodules only when enabled
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("mixer", MIXERS)
def test_qk_norm_submodules_exist_only_when_enabled(core, mixer):
    off, on = _net(core, mixer, qk_norm=False), _net(core, mixer, qk_norm=True)
    assert _qk_names(off) == []
    assert not any("qk_norm" in k for k in off.state_dict())
    n_core_blocks = sum(isinstance(m, CoreBlock) for m in on.modules())
    per_st_layer = 2 if mixer == "attention" else 1          # spatial (+ temporal)
    expected = 2 * (n_core_blocks + per_st_layer * TINY["n_st_layers"])  # q + k scales
    names = _qk_names(on)
    assert len(names) == expected
    for n, p in on.named_parameters():
        if "qk_norm" in n:
            assert p.shape == (D_HEAD,) and torch.equal(p, torch.full((D_HEAD,), G0)), n
    for blk in on.st_layers:
        assert (blk.temporal_qk_norm is not None) == (mixer == "attention")
    # qk_norm only ADDS parameters: the others keep their names, order and
    # (same seed) initial values
    on_sd, off_sd = on.state_dict(), off.state_dict()
    assert [n for n, _ in on.named_parameters() if "qk_norm" not in n] == \
        [n for n, _ in off.named_parameters()]
    for k, v in off_sd.items():
        assert torch.equal(v, on_sd[k]), k


# ---------------------------------------------------------------------------
# forward / step parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mixer", MIXERS)
@pytest.mark.parametrize("qk_norm", [False, True])
def test_spatiotemporal_block_step_matches_forward(mixer, qk_norm):
    torch.manual_seed(3)
    cfg = ModelConfig(d_model=32, n_heads=4, d_ff=64, dropout=0.0,
                      temporal_mixer=mixer, qk_norm=qk_norm)
    blk = SpatiotemporalBlock(cfg).eval()
    if qk_norm:      # non-trivial scales, so a missing norm would show
        with torch.no_grad():
            for n, p in blk.named_parameters():
                if "qk_norm" in n:
                    p.uniform_(0.5, 2.0)
    x = torch.randn(2, 6, 9, 32)
    with torch.no_grad():
        full = blk(x)
        cache, outs = None, []
        for t in range(x.shape[1]):
            o, cache = blk.forward_step(x[:, t:t + 1], cache)
            outs.append(o)
    torch.testing.assert_close(torch.cat(outs, 1), full, rtol=0, atol=1e-5)


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("mixer", MIXERS)
@pytest.mark.parametrize("qk_norm", [False, True])
@pytest.mark.parametrize("moe", [0, 4])
def test_network_step_matches_forward(core, mixer, qk_norm, moe):
    net = _net(core, mixer, qk_norm=qk_norm, moe=moe)
    grid, glb = _frames()
    B, T = grid.shape[:2]
    with torch.no_grad():
        full = net(grid, glb, act_steps=2)
        cache = None
        for t in range(T):
            s = net.step(grid[:, t], glb[:, t], st_cache=cache, step_idx=t, act_steps=2)
            cache = s["st_cache"]
            torch.testing.assert_close(s["source_logits"], full["source_logits"][:, t],
                                       rtol=0, atol=1e-5)
            torch.testing.assert_close(s["value"], full["value"][:, t], rtol=0, atol=1e-5)


@pytest.mark.parametrize("qk_norm", [False, True])
def test_eval_step_stays_sync_free(qk_norm, monkeypatch):
    """The graph-capturable calling convention (tensor step_idx, static
    recurrent cache, dense MoE) never reads a tensor on the host."""
    net = _net("recurrent_depth", "recurrent", qk_norm=qk_norm, moe=4)
    B = 2
    net.prepare_inference()
    cache = net.initial_st_cache(B)
    with torch.no_grad(), monkeypatch.context() as mp, MoEFFN.inference_path("dense"):
        _forbid_host_syncs(mp)
        out = GameNetwork.step(net, _obs(B), _legal(B), st_cache=cache,
                               step_idx=torch.tensor(3), act_steps=2)
        out = GameNetwork.step(net, _obs(B, seed=1), _legal(B, seed=1),
                               st_cache=out["st_cache"], step_idx=torch.tensor(4),
                               act_steps=2)
    assert torch.isfinite(out["action_log_prob"]).all()


# ---------------------------------------------------------------------------
# old checkpoints: strict load, identical outputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("core", CORES + ("trm",))
@pytest.mark.parametrize("mixer", MIXERS)
def test_old_config_round_trip_loads_strictly_with_identical_outputs(core, mixer, tmp_path):
    """A checkpoint written before qk_norm existed (config dict without the
    key) rebuilds with qk_norm=False, loads strictly and reproduces the
    saved network exactly; the new default would not even load it."""
    net = _net(core, mixer, qk_norm=False, trm_H_cycles=1, trm_L_cycles=1,
               trm_block_layers=1)
    old_cfg = {k: v for k, v in net.cfg.__dict__.items() if k != "qk_norm"}
    torch.save({"config": old_cfg, "model_state": net.state_dict()}, tmp_path / "old.pt")

    ck = torch.load(tmp_path / "old.pt", weights_only=False)
    cfg = config_from_dict(ck["config"])
    assert cfg.qk_norm is False
    torch.manual_seed(1234)                           # different init
    net2 = GeneralsNetwork(cfg).eval()
    net2.load_state_dict(ck["model_state"], strict=True)

    grid, glb = _frames(T=3)
    with torch.no_grad():
        a, b = net(grid, glb, act_steps=2), net2(grid, glb, act_steps=2)
        assert torch.equal(a["source_logits"], b["source_logits"])
        assert torch.equal(a["value"], b["value"])
        ca = cb = None
        for t in range(3):
            torch.manual_seed(t)
            sa = net.step(grid[:, t], glb[:, t], st_cache=ca, step_idx=t, act_steps=2)
            torch.manual_seed(t)
            sb = net2.step(grid[:, t], glb[:, t], st_cache=cb, step_idx=t, act_steps=2)
            ca, cb = sa["st_cache"], sb["st_cache"]
            for k in ("source_logits", "value", "action_log_prob"):
                assert torch.equal(sa[k], sb[k]), (t, k)

    with pytest.raises(RuntimeError, match="qk_norm"):
        GeneralsNetwork(ModelConfig(**old_cfg)).load_state_dict(ck["model_state"], strict=True)


def test_old_single_game_checkpoint_resumes_through_train_il_game():
    """train_il_game / train_rl_game rebuild from the saved config via
    resume_arch_overrides → build_net_tolerant."""
    from model.games import build_game_network
    from training.train_il_game import build_net_tolerant, resume_arch_overrides

    kw = dict(TINY, core_type="recurrent_depth", n_st_layers=1, qk_norm=False)
    torch.manual_seed(0)
    net = build_game_network("othello", **kw)
    old_cfg = {k: v for k, v in net.cfg.__dict__.items() if k != "qk_norm"}
    ckpt = {"game": "othello", "config": old_cfg, "model_state": net.state_dict()}
    net2, _ = build_net_tolerant("othello", resume_arch_overrides(ckpt, {}))
    assert net2.cfg.qk_norm is False
    net2.load_state_dict(ckpt["model_state"], strict=True)


def test_old_multi_game_checkpoint_loads(tmp_path):
    """eval_multi_game.load_multi and train_multi_game.load_core_checkpoint
    on a multi-game checkpoint saved before qk_norm existed."""
    from training.eval_multi_game import load_multi
    from training.train_multi_game import load_core_checkpoint

    core = dict(TINY, core_type="recurrent_depth", n_st_layers=1, qk_norm=False)
    torch.manual_seed(0)
    mg = build_multi_game(["othello", "chess"], core)
    strip = lambda d: {k: v for k, v in d.items() if k != "qk_norm"}   # noqa: E731
    first = mg["othello"]
    torch.save({
        "games": ["othello", "chess"],
        "configs": {g: strip(mg[g].cfg.__dict__) for g in ("othello", "chess")},
        "core_config": strip(core_config_overrides(first.cfg)),
        "core_state": core_state_dict(first),
        "model_state": mg.state_dict(),
        "train_model_state": mg.state_dict(),
    }, tmp_path / "multi_old.pt")

    loaded = load_multi(str(tmp_path / "multi_old.pt"), torch.device("cpu"))
    assert loaded["othello"].cfg.qk_norm is False
    for k, v in mg.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v), k
    state, overrides = load_core_checkpoint(str(tmp_path / "multi_old.pt"))
    assert overrides["qk_norm"] is False


# ---------------------------------------------------------------------------
# the stabiliser: logits bounded whatever the scale
# ---------------------------------------------------------------------------

def _spy_sdpa(monkeypatch):
    """Record (q, k, is_causal) of every scaled_dot_product_attention call."""
    rec = []
    orig = F.scaled_dot_product_attention

    def spy(q, k, v, *a, **kw):
        rec.append((q.detach().float(), k.detach().float(), bool(kw.get("is_causal", False))))
        return orig(q, k, v, *a, **kw)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    return rec


def _logits_entropy(q, k, causal):
    """(max |logit| over attended pairs, mean attention entropy)."""
    logits = q @ k.transpose(-2, -1) * q.shape[-1] ** -0.5
    if causal:
        keep = torch.ones(logits.shape[-2:], dtype=torch.bool).tril()
        top = logits.masked_fill(~keep, 0.0).abs().max()
        logits = logits.masked_fill(~keep, float("-inf"))
    else:
        top = logits.abs().max()
    p = logits.softmax(-1)
    return top.item(), -torch.special.xlogy(p, p).sum(-1).mean().item()


def test_qk_norm_bound_holds_for_any_input_scale_and_is_tight():
    torch.manual_seed(0)
    D = 16
    qkn = QKNorm(D)
    with torch.no_grad():
        qkn.q_norm.weight.uniform_(-2.0, 2.0)
        qkn.k_norm.weight.uniform_(-1.5, 1.5)
    bound = D ** 0.5 * qkn.q_norm.weight.abs().max() * qkn.k_norm.weight.abs().max()
    with torch.no_grad():
        for scale in (1e-3, 1.0, 1e3, 1e6):
            q, k = torch.randn(2, 3, 40, D) * scale, torch.randn(2, 3, 40, D) * scale
            top, _ = _logits_entropy(*qkn(q, k), causal=False)
            assert top <= bound.item() * (1 + 1e-5), (scale, top, bound)
        # at init, aligned one-hot rows reach the initial cap √D·g0²
        fresh = QKNorm(D)
        e = torch.zeros(1, 1, 1, D)
        e[..., 0] = 100.0
        top, _ = _logits_entropy(*fresh(e, e), causal=False)
        assert abs(top - QK_NORM_INIT_CAP) < 1e-4


def test_qk_norm_init_scale_starts_the_cap_at_8():
    """Small heads start with scales > 1 so the logit cap √D·g_q·g_k is 8
    (unit scales would give √32 ≈ 5.7 for the study's 32-dim heads); heads
    of ≥ 64 dims keep the standard unit init."""
    assert QK_NORM_INIT_CAP == 8.0 and INIT_CAP == pytest.approx(8.0)
    for d in (8, 16, 32, 64, 128, 256):
        assert d ** 0.5 * qk_norm_init_scale(d) ** 2 == pytest.approx(max(d ** 0.5, 8.0))
    assert qk_norm_init_scale(32) == pytest.approx(2 ** 0.25)
    assert qk_norm_init_scale(64) == 1.0 and qk_norm_init_scale(128) == 1.0
    qkn = QKNorm(32)
    for w in (qkn.q_norm.weight, qkn.k_norm.weight):
        assert torch.equal(w, torch.full((32,), qk_norm_init_scale(32)))
    # GameNetwork's blanket init pass leaves the QK scales alone
    net = _net("recurrent_depth", "attention", qk_norm=True)
    assert all(torch.equal(p, torch.full((D_HEAD,), G0))
               for n, p in net.named_parameters() if "qk_norm" in n)


@pytest.mark.parametrize("qk_norm", [False, True])
def test_core_block_logits_bounded_under_qk_weight_growth(qk_norm, monkeypatch):
    """CoreBlock pre-normalises its input (sandwich n1), so input scale is
    already harmless there; the logit growth of a diverging run comes from
    the q/k projection weights.  ×100 on them multiplies raw logits by 10⁴
    (attention collapses to one-hot rows); with QK-norm nothing changes."""
    torch.manual_seed(0)
    blk = CoreBlock(ModelConfig(d_model=32, n_heads=4, qk_norm=qk_norm),
                    out_std=0.02, moe=False).eval()
    x = torch.randn(2, 24, 32)
    rec = _spy_sdpa(monkeypatch)
    with torch.no_grad():
        blk(x)
        blk(100.0 * x)                     # input scale: removed by n1 either way
        blk.qkv.weight.mul_(100.0)
        blk(x)
    (top0, ent0), (top_in, ent_in), (top_w, ent_w) = [_logits_entropy(*r) for r in rec]
    assert ent_in == pytest.approx(ent0, abs=1e-4)
    if qk_norm:
        assert max(top0, top_in, top_w) <= INIT_CAP * (1 + 1e-5)
        assert ent_w == pytest.approx(ent0, abs=1e-4)
    else:
        assert top_w > 100 * INIT_CAP
        assert ent_w < 0.25 * ent0


@pytest.mark.parametrize("qk_norm", [False, True])
def test_stage1_logits_bounded_under_input_and_weight_scale(qk_norm, monkeypatch):
    """Stage-1 is post-norm: the spatial attention sees the raw block input,
    so ×100 input scale goes straight into its logits; the temporal attention
    sees a LayerNorm output, so its growth comes from its q/k weights."""
    torch.manual_seed(0)
    blk = SpatiotemporalBlock(ModelConfig(d_model=32, n_heads=4, d_ff=64, dropout=0.0,
                                          temporal_mixer="attention",
                                          qk_norm=qk_norm)).eval()
    x = torch.randn(2, 5, 12, 32)
    rec = _spy_sdpa(monkeypatch)
    with torch.no_grad():
        blk(x)
        blk.temporal_attn.in_proj_weight.mul_(100.0)
        blk(100.0 * x)
        if qk_norm:                         # the step path is normalised too
            cache = None
            for t in range(x.shape[1]):
                _, cache = blk.forward_step(100.0 * x[:, t:t + 1], cache)
    (sp0, tp0), (sp1, tp1) = rec[0:2], rec[2:4]
    assert not sp0[2] and tp0[2]           # spatial non-causal, temporal causal
    (s_top0, s_ent0), (s_top1, s_ent1) = _logits_entropy(*sp0), _logits_entropy(*sp1)
    (t_top0, t_ent0), (t_top1, t_ent1) = _logits_entropy(*tp0), _logits_entropy(*tp1)
    if qk_norm:
        bound = INIT_CAP * (1 + 1e-5)
        assert max(s_top0, s_top1, t_top0, t_top1) <= bound
        assert s_ent1 == pytest.approx(s_ent0, abs=1e-4)
        assert t_ent1 == pytest.approx(t_ent0, abs=1e-3)
        assert len(rec) == 4 + 2 * x.shape[1]
        for q, k, causal in rec[4:]:
            assert _logits_entropy(q, k, causal)[0] <= bound
    else:
        assert s_top1 > 100 * INIT_CAP and s_ent1 < 0.25 * s_ent0
        assert t_top1 > 100 * INIT_CAP and t_ent1 < 0.5 * t_ent0


# ---------------------------------------------------------------------------
# training: scales learn (through activation checkpointing), bf16 works
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("mixer", MIXERS)
def test_qk_norm_scales_receive_gradients(core, mixer):
    net = _net(core, mixer, qk_norm=True, moe=4).train()
    grid, glb = _frames(T=3)
    torch.manual_seed(0)
    out = net(grid, glb, act_steps=2)
    (out["source_logits"].float().logsumexp(-1).sum() + out["value"].sum()).backward()
    for n, p in net.named_parameters():
        if "qk_norm" in n:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
            assert p.grad.abs().sum() > 0, n


def test_frozen_core_keeps_qk_norm_scales_trainable_with_norms():
    net = _net("recurrent_depth", "attention", qk_norm=True)
    freeze_core(net, train_norms=True)
    assert all(p.requires_grad for n, p in net.named_parameters() if "qk_norm" in n)
    freeze_core(net, train_norms=False)
    assert not any(p.requires_grad for n, p in net.named_parameters() if "qk_norm" in n)


@pytest.mark.parametrize("mixer", MIXERS)
def test_qk_norm_scales_are_not_weight_decayed(mixer):
    """Weight decay on the QK scales would pull the attention temperature
    toward uniform; both optimiser groupings (optim_param_groups: single-game
    IL / RL trainers; multi_game_param_groups: train_multi_game) keep them at
    wd 0 and leave qk_norm=False models' groups exactly as before."""
    from model.optim import AdamATan2
    from model.rl_utils import optim_param_groups, qk_norm_param_ids
    from training.train_multi_game import multi_game_param_groups

    def is_memory(n):
        return n.endswith("retention_bias") or n.endswith("temporal_rnn.h_init")

    off = _net("recurrent_depth", mixer, qk_norm=False)
    assert qk_norm_param_ids(off) == set()
    g_off = optim_param_groups(off, 0.1)
    assert [id(p) for p in g_off[0]["params"]] == \
        [id(p) for n, p in off.named_parameters() if not is_memory(n)]
    mg_off = multi_game_param_groups(off, 0.1)
    assert len(mg_off) == 1 and [id(p) for p in mg_off[0]["params"]] == \
        [id(p) for p in off.parameters()]

    on = _net("recurrent_depth", mixer, qk_norm=True)
    qk = {id(p) for n, p in on.named_parameters() if "qk_norm" in n}
    assert qk_norm_param_ids(on) == qk
    g_on = optim_param_groups(on, 0.1)
    assert {id(p) for p in g_on[-1]["params"]} == qk | {
        id(p) for n, p in on.named_parameters() if is_memory(n)}
    assert g_on[-1]["weight_decay"] == 0.0
    mg_on = multi_game_param_groups(on, 0.1)
    assert {id(p) for p in mg_on[1]["params"]} == qk and mg_on[1]["weight_decay"] == 0.0
    assert sum(len(g["params"]) for g in mg_on) == len(list(on.parameters()))

    # pure decay (zero gradients): QK scales keep their init, other norm scales shrink
    opt = AdamATan2(mg_on, lr=1e-2, weight_decay=0.1)
    for p in on.parameters():
        p.grad = torch.zeros_like(p)
    for _ in range(50):
        opt.step()
    assert torch.equal(on.core.prelude[0].qk_norm.q_norm.weight, torch.full((D_HEAD,), G0))
    assert torch.equal(on.st_layers[0].spatial_qk_norm.k_norm.weight,
                       torch.full((D_HEAD,), G0))
    assert (on.core.prelude[0].norms[0].weight < 1).all()


def test_core_block_bf16_autocast():
    torch.manual_seed(0)
    blk = CoreBlock(ModelConfig(d_model=32, n_heads=4, qk_norm=True),
                    out_std=0.02, moe=False).eval()
    x = torch.randn(2, 10, 32)
    with torch.no_grad():
        ref = blk(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = blk(x)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.float(), ref, rtol=5e-2, atol=5e-2)


# ---------------------------------------------------------------------------
# CUDA only: sync-free and graph == eager, qk_norm on and off
# ---------------------------------------------------------------------------

@needs_cuda
@pytest.mark.parametrize("qk_norm", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_eval_step_has_no_host_sync_cuda(qk_norm, bf16):
    dev = torch.device("cuda")
    net = _net("recurrent_depth", "recurrent", qk_norm=qk_norm, moe=4).to(dev)
    B = 2
    net.prepare_inference(dev)
    cache = net.initial_st_cache(B)
    t = torch.tensor(3, device=dev)
    # inputs are built BEFORE the sync-check window: host->device copies of
    # the test's own inputs are syncs by definition and not the model's
    obs, legal = _obs(B, dev), _legal(B, dev)
    obs2, legal2 = _obs(B, dev), _legal(B, dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        GameNetwork.step(net, obs, legal, st_cache=cache, step_idx=t, act_steps=2)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            out = GameNetwork.step(net, obs2, legal2, st_cache=cache, step_idx=t,
                                   act_steps=2)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    assert torch.isfinite(out["action_log_prob"]).all()


@needs_cuda
@pytest.mark.parametrize("qk_norm", [False, True])
@pytest.mark.parametrize("core", CORES)
def test_graph_replay_matches_eager_cuda(qk_norm, core):
    from model.cuda_graph import GraphedStep

    dev = torch.device("cuda")
    net = _net(core, "recurrent", qk_norm=qk_norm, moe=4).to(dev)
    net.action_decoder.greedy = True
    B = 2
    g = GraphedStep(net, _obs(B, dev), _legal(B, dev), act_steps=2)
    cache = None
    for t in range(4):
        obs, legal = _obs(B, dev, seed=t), _legal(B, dev, seed=t)
        with torch.no_grad():
            e = GameNetwork.step(net, obs, legal, st_cache=cache, step_idx=t, act_steps=2)
        cache = e["st_cache"]
        r = g(obs, t, legal)
        for key in ("source_logits", "value", "action_log_prob"):
            torch.testing.assert_close(r[key], e[key], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# review fixes: fatal qk_norm mismatch on lenient loads, boolean overrides,
# diagnostics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("core", CORES + ("trm",))
def test_lenient_load_rejects_qk_norm_mismatch(core):
    """train_rl builds its KL reference / opening anchor as a deepcopy of the
    learner filled by load_model_state, which only warns on unused keys.
    A qk_norm=True checkpoint into a qk_norm=False learner must not load as
    a QK-norm-less (different) network; the reverse fails on missing keys."""
    import copy

    from model.rl_utils import load_model_state

    kw = dict(trm_H_cycles=1, trm_L_cycles=1, trm_block_layers=1)
    learner = _net(core, "attention", qk_norm=False, **kw)
    anchor = _net(core, "attention", qk_norm=True, seed=1, **kw)
    ref = copy.deepcopy(learner)
    with pytest.raises(RuntimeError, match="qk_norm=False"):
        load_model_state(ref, anchor.state_dict(), ctx="KL reference checkpoint")
    with pytest.raises(RuntimeError, match="missing"):
        load_model_state(copy.deepcopy(anchor), learner.state_dict(), ctx="KL reference")
    # matching configs still load (and other unused keys still only warn)
    extra = {**anchor.state_dict(), "world_model.w": torch.zeros(1)}
    load_model_state(_net(core, "attention", qk_norm=True, seed=2, **kw), extra, ctx="ok")


def test_boolean_overrides_parse_in_any_case():
    from training.train_il_game import parse_overrides

    got = parse_overrides(["qk_norm=false", "a=FALSE", "b=True", "c=true",
                           "d_model=256", "temporal_mixer=recurrent", "lr=3e-4"])
    assert got == {"qk_norm": False, "a": False, "b": True, "c": True,
                   "d_model": 256, "temporal_mixer": "recurrent", "lr": 3e-4}
    assert got["qk_norm"] is False
    from model.games import build_game_network
    net = build_game_network("othello", **dict(TINY, n_st_layers=1),
                             **parse_overrides(["qk_norm=false"]))
    assert net.cfg.qk_norm is False and _qk_names(net) == []


def test_qk_caps_reports_the_logit_bound():
    from training.diagnostics import qk_caps

    assert qk_caps(_net(qk_norm=False)) == {}
    net = _net("recurrent_depth", "attention", qk_norm=True)
    caps = qk_caps(net)
    assert len(caps) == sum(isinstance(m, QKNorm) for m in net.modules())
    assert all(c == pytest.approx(INIT_CAP) for c in caps.values())
    with torch.no_grad():
        net.core.prelude[0].qk_norm.q_norm.weight[3] = -3.0
    assert qk_caps(net)["core.prelude.0.qk_norm"] == pytest.approx(D_HEAD ** 0.5 * 3.0 * G0)


@pytest.mark.parametrize("qk_norm", [False, True])
def test_attention_probe_is_read_only_and_names_every_attention(qk_norm):
    from training.diagnostics import attention_probe, qk_caps

    net = _net("recurrent_depth", "attention", qk_norm=qk_norm)
    grid, glb = _frames(T=3)
    with torch.no_grad():
        ref = net(grid, glb, act_steps=2)
        with attention_probe(net) as attn:
            out = net(grid, glb, act_steps=2)
    for k in ("source_logits", "value"):
        assert torch.equal(out[k], ref[k]), k
    blocks = [n for n, m in net.named_modules() if isinstance(m, CoreBlock)]
    st = [f"st_layers.{i}" for i in range(TINY["n_st_layers"])]
    assert set(attn) == set(blocks) | set(st) | {n + "[causal]" for n in st}
    for s in attn.values():
        assert 0.0 <= s["entropy"] <= 1.0 + 1e-6 and s["max_logit"] > 0
    assert attn["core.recurrent.0"]["calls"] == 2                 # once per iteration
    if qk_norm:
        assert max(s["max_logit"] for s in attn.values()) <= max(qk_caps(net).values()) * (1 + 1e-5)
    # hooks and the SDPA patch are gone afterwards
    assert F.scaled_dot_product_attention.__name__ == "scaled_dot_product_attention"
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in net.modules())


def test_attention_probe_matches_direct_computation_and_sees_collapse(monkeypatch):
    """Probe numbers = the test helper's on the same call; ×100 q/k weights
    without QK-norm show up as a large logit and a near-zero entropy."""
    from training.diagnostics import attention_probe

    torch.manual_seed(0)
    blk = CoreBlock(ModelConfig(d_model=32, n_heads=4, qk_norm=False), out_std=0.02,
                    moe=False).eval()
    x = torch.randn(2, 24, 32)
    rec = _spy_sdpa(monkeypatch)
    with torch.no_grad():
        with attention_probe(blk) as a0:
            blk(x)
        blk.qkv.weight.mul_(100.0)
        with attention_probe(blk) as a1:
            blk(x)
    for (q, k, causal), a in zip(rec, (a0, a1)):
        top, _ = _logits_entropy(q, k, causal)
        assert a["<root>"]["max_logit"] == pytest.approx(top, rel=1e-5)
    assert a0["<root>"]["entropy"] > 0.9 and a1["<root>"]["entropy"] < 0.1
    assert a1["<root>"]["max_logit"] > 100 * a0["<root>"]["max_logit"]


def test_attention_probe_causal_rows_and_chunking():
    from training.diagnostics import _attn_stats

    torch.manual_seed(0)
    q, k = torch.randn(3, 2, 7, 8), torch.randn(3, 2, 7, 8)
    full = _attn_stats(q, k, None, True, None, max_elems=1 << 20)
    chunked = _attn_stats(q, k, None, True, None, max_elems=49)   # one row-block per chunk
    assert full[2] == chunked[2] == 3 * 2 * 6                     # row 0 has one key
    assert full[0] == pytest.approx(chunked[0]) and full[1] == pytest.approx(chunked[1])
    # an explicit boolean causal mask is the same thing
    tri = torch.ones(7, 7, dtype=torch.bool).tril()
    assert _attn_stats(q, k, tri, False, None, 1 << 20) == pytest.approx(full)
    # uniform attention has normalised entropy 1
    zero = torch.zeros(1, 1, 5, 8)
    top, ent, rows = _attn_stats(zero, zero, None, False, None, 1 << 20)
    assert top == 0.0 and ent / rows == pytest.approx(1.0)


def test_logit_lse_absmax():
    from training.diagnostics import logit_lse_absmax

    fm = torch.tensor([[True, True, False]])
    cat = torch.zeros(1, 3, 4)
    cat[0, 1] = torch.tensor([50.0, 0.0, 0.0, 0.0])
    cat[0, 2] = 1e3                                   # padded frame: ignored
    bern = torch.tensor([[-7.0, 2.0, 99.0]])          # [B, T] Bernoulli logit
    masked = torch.full((1, 3, 2, 2), float("-inf"))  # fully masked rows: skipped
    pred = {"a_logits": cat, "wait_logits": bern, "b_logits": masked,
            "value": torch.full((1, 3), 1e6)}
    assert logit_lse_absmax(pred, fm) == pytest.approx(50.0, abs=1e-3)
    assert logit_lse_absmax({"wait_logits": bern}, fm) == pytest.approx(7.0)
    assert logit_lse_absmax({"value": bern}, fm) is None


def test_train_multi_game_logs_qk_norm_and_attention(tmp_path, capsys):
    from data_processing.games.dataset import save_shard
    from tests.test_core_moe import _board_samples
    from training import train_multi_game

    d = tmp_path / "othello"
    d.mkdir()
    save_shard(str(d / "a.pt"), "othello", _board_samples("othello", 4, 6))
    core = dict(TINY, n_st_layers=1, core_type="recurrent_depth", temporal_mixer="attention")
    train_multi_game.main([
        "--data", f"othello={d}", "--val", f"othello={d}", "--steps", "2",
        "--batch_size", "2", "--warmup_steps", "1", "--eval_every", "2", "--log_every", "1",
        "--save_dir", str(tmp_path / "ck"), "--device", "cpu",
        "--override", *[f"{k}={v}" for k, v in core.items()], "qk_norm=true"])
    out = capsys.readouterr().out
    assert "qk_norm=True" in out and "qk_cap_max=" in out
    attn = [ln for ln in out.splitlines() if ln.startswith("[attn 2] othello:")]
    assert len(attn) == 1 and "max_logit=" in attn[0] and "min_entropy=" in attn[0]
    assert "st_layers.0[causal]=" in attn[0] and "lse_absmax=" in attn[0]
