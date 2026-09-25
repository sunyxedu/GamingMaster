"""TRM training recipe of the reference implementation
(SamsungSAILMontreal/TinyRecursiveModels): deep supervision with carried
latents, ACT halting with exploration, AdamATan2, EMA, reference block."""

import math

import torch

from model.components.trm import ReferenceTRMBlock, TRMBlock
from model.config import ModelConfig, config_from_dict, upgrade_config_dict
from model.network import GeneralsNetwork
from model.optim import AdamATan2, ModelEMA, cosine_schedule_with_warmup_lr_lambda
from training.deep_supervision import halt_update, run_deep_supervision

TINY = dict(d_model=64, n_heads=4, d_ff=128, n_st_layers=1, trm_H_cycles=2,
            trm_L_cycles=2, trm_block_layers=1, encoder_channels=16, dropout=0.0,
            temporal_mixer="recurrent")


def _net(**kw):
    torch.manual_seed(0)
    return GeneralsNetwork(ModelConfig(**{**TINY, **kw}))


def _batch(B=2, T=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    grid = torch.rand(B, T, 12, 25, 25, generator=g)
    glb = torch.randn(B, T, 4, generator=g)
    return grid, glb


# ---------------------------------------------------------------------------
# defaults and legacy compatibility
# ---------------------------------------------------------------------------

def test_reference_defaults():
    cfg = ModelConfig()
    assert (cfg.trm_H_cycles, cfg.trm_L_cycles, cfg.trm_block_layers) == (3, 6, 2)
    assert cfg.halt_max_steps == 16 and cfg.halt_exploration_prob == 0.1
    assert cfg.trm_arch == "trm"
    net = GeneralsNetwork(ModelConfig(**TINY))
    assert isinstance(net.trm.reasoning.layers[0], ReferenceTRMBlock)


def test_old_checkpoint_config_rebuilds_legacy_block():
    old = {**TINY, "halt_max_steps": 5}          # saved before trm_arch existed
    cfg = config_from_dict(old)
    assert cfg.trm_arch == "legacy" and cfg.halt_max_steps == 5
    assert upgrade_config_dict({"trm_arch": "trm"})["trm_arch"] == "trm"
    net = GeneralsNetwork(cfg)
    assert isinstance(net.trm.reasoning.layers[0], TRMBlock)


def test_reference_block_has_no_bias_no_dropout_and_lecun_init():
    torch.manual_seed(0)
    blk = ReferenceTRMBlock(ModelConfig(d_model=256, n_heads=4))
    names = [n for n, _ in blk.named_parameters()]
    assert not any("bias" in n for n in names)
    assert not any(isinstance(m, torch.nn.Dropout) for m in blk.modules())
    w = blk.qkv_proj.weight
    assert abs(w.std().item() - 256 ** -0.5) < 0.002
    # inter = find_multiple(round(4 * 256 * 2/3), 256) = 768
    assert blk.mlp.down_proj.weight.shape == (256, 768)
    # the network-wide std-0.02 init must not overwrite it
    net = _net()
    w = net.trm.reasoning.layers[0].qkv_proj.weight
    assert abs(w.std().item() - 64 ** -0.5) < 0.02


# ---------------------------------------------------------------------------
# halting
# ---------------------------------------------------------------------------

def test_halt_update_max_steps_and_q_rule():
    steps = torch.zeros(1, 4, dtype=torch.long)
    active = torch.ones(1, 4, dtype=torch.bool)
    q = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    s1, a1 = halt_update(q, steps, active, n_sup=3, exploration_prob=0.0)
    assert s1.tolist() == [[1, 1, 1, 1]]
    assert a1.tolist() == [[False, True, False, True]]
    # eval semantics: only the step limit halts
    _, a_eval = halt_update(q, steps, active, n_sup=3, exploration_prob=0.0,
                            training=False)
    assert a_eval.all()
    # the last step always halts, exploration or not
    s3, a3 = halt_update(-torch.ones(1, 4), torch.full((1, 4), 2), active,
                         n_sup=3, exploration_prob=1.0)
    assert not a3.any() and s3.tolist() == [[3, 3, 3, 3]]


def test_halt_update_exploration_keeps_frames_at_least_two_steps():
    torch.manual_seed(0)
    q = torch.ones(1, 1000)                       # every frame wants to halt
    steps = torch.zeros(1, 1000, dtype=torch.long)
    active = torch.ones(1, 1000, dtype=torch.bool)
    _, a1 = halt_update(q, steps, active, n_sup=16, exploration_prob=0.1)
    frac = a1.float().mean().item()
    # exploring frames draw min_halt ∈ {2..16} ≥ 2, so they survive step 1
    assert 0.05 < frac < 0.15


def test_inactive_frames_do_not_advance():
    steps = torch.tensor([[2, 5]])
    active = torch.tensor([[True, False]])
    s, a = halt_update(torch.full((1, 2), -1.0), steps, active, 16, 0.0)
    assert s.tolist() == [[3, 5]] and a.tolist() == [[True, False]]


# ---------------------------------------------------------------------------
# carried forward
# ---------------------------------------------------------------------------

def test_carried_steps_equal_multi_step_forward():
    """k supervision steps with an explicit carry == forward(act_steps=k)."""
    net = _net().eval()
    grid, glb = _batch()
    with torch.no_grad():
        ref = net(grid, glb, act_steps=3)
        carry = None
        for _ in range(3):
            out = net(grid, glb, carry=carry, return_carry=True)
            carry = out["carry"]
    assert torch.allclose(out["source_logits"], ref["source_logits"], atol=1e-5)
    assert torch.allclose(out["q_halt"], ref["q_halt"], atol=1e-5)


def test_active_rows_match_full_batch_and_frozen_rows_keep_carry():
    net = _net().eval()
    grid, glb = _batch()
    with torch.no_grad():
        first = net(grid, glb, return_carry=True)
        full = net(grid, glb, carry=first["carry"], return_carry=True)
        active = torch.tensor([[True, False, True], [False, True, True]])
        part = net(grid, glb, carry=first["carry"], active=active, return_carry=True)
    rows = active.reshape(-1)
    for i in range(2):
        a, b = full["carry"][i], part["carry"][i]
        assert torch.allclose(a[rows], b[rows], atol=1e-5)
        assert torch.equal(b[~rows], first["carry"][i][~rows])
    assert torch.allclose(full["q_halt"][active], part["q_halt"][active], atol=1e-5)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def test_run_deep_supervision_one_optimizer_step_per_supervision_step():
    net = _net(halt_max_steps=4, halt_exploration_prob=0.0)
    grid, glb = _batch()
    mask = torch.ones(2, 3, dtype=torch.bool)
    calls = {"fb": 0, "opt": 0}
    torch.nn.init.constant_(net.trm.q_head.bias, -5.0)   # never halts

    def fb(i, carry, active, weight):
        calls["fb"] += 1
        assert weight == 1.0                            # nothing halts
        if carry is not None:
            assert not carry[0].requires_grad          # carry is detached
        out = net(grid, glb, carry=carry, active=active, return_carry=True)
        out["source_logits"].square().mean().backward()
        return out, {"loss": 1.0}

    def opt():
        calls["opt"] += 1

    stats = run_deep_supervision(1, [mask], fb, opt, n_sup=4, exploration_prob=0.0)
    assert calls == {"fb": 4, "opt": 4}
    assert stats["sup_steps"] == 4 and stats["halt_depth"] == 4


def test_run_deep_supervision_stops_when_all_frames_halt():
    net = _net(halt_max_steps=16)
    torch.nn.init.constant_(net.trm.q_head.bias, 5.0)    # halts at once
    grid, glb = _batch()
    mask = torch.tensor([[True, True, False], [True, False, False]])
    n = {"opt": 0}

    def fb(i, carry, active, weight):
        out = net(grid, glb, carry=carry, active=active, return_carry=True)
        return out, {}

    def opt():
        n["opt"] += 1

    stats = run_deep_supervision(1, [mask], fb, opt, n_sup=16, exploration_prob=0.0)
    assert n["opt"] == 1 and stats["halt_depth"] == 1.0


# ---------------------------------------------------------------------------
# optimiser / schedule / EMA
# ---------------------------------------------------------------------------

def _kernel_reference(param, grad, exp_avg, exp_avg_sq, lr, wd, beta1, beta2, t):
    """adam-atan2 0.0.3 adam_atan2_test._adam_atan2_reference_impl (the
    package's own spec of its CUDA kernel)."""
    param = param.clone()
    exp_avg = exp_avg.clone()
    exp_avg_sq = exp_avg_sq.clone()
    param.mul_(1 - lr * wd)
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    denom = exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** t)
    param.add_(torch.atan2(exp_avg, denom), alpha=-lr / (1 - beta1 ** t))
    return param, exp_avg, exp_avg_sq


def test_adam_atan2_matches_reference_kernel_over_steps():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(64))
    ref_p = p.detach().clone()
    m = torch.zeros(64)
    v = torch.zeros(64)
    opt = AdamATan2([p], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    for t in range(1, 8):
        g = torch.randn(64)
        p.grad = g.clone()
        opt.step()
        ref_p, m, v = _kernel_reference(ref_p, g, m, v, 1e-2, 0.1, 0.9, 0.95, t)
        assert torch.allclose(p.detach(), ref_p, atol=1e-6, rtol=0)  # fp32 ulp: lerp vs addcmul
    # first step: |update| = lr/(1-β1) · atan2(0.1 g, |g|) — no 1.27 factor
    q = torch.nn.Parameter(torch.zeros(1))
    q.grad = torch.ones(1)
    AdamATan2([q], lr=1e-3, weight_decay=0.0).step()
    assert math.isclose(-q.item(), 1e-3 / 0.1 * math.atan2(0.1, 1.0), rel_tol=1e-5)


def test_loss_weight_is_active_fraction():
    net = _net(halt_max_steps=3, halt_exploration_prob=0.0)
    grid, glb = _batch()
    mask = torch.tensor([[True, True, True], [True, False, False]])
    seen = []

    def fb(i, carry, active, weight):
        seen.append((int(active.sum()), weight))
        out = net(grid, glb, carry=carry, active=active, return_carry=True)
        # halt exactly two of the frames after the first step
        q = torch.full_like(out["q_halt"], -1.0)
        if len(seen) == 1:
            q[0, 0] = q[0, 1] = 1.0
        out["q_halt"] = q
        return out, {}

    run_deep_supervision(1, [mask], fb, lambda: None, n_sup=3, exploration_prob=0.0)
    assert seen == [(4, 1.0), (2, 0.5), (2, 0.5)]


def test_schedule_is_constant_after_warmup_by_default():
    f = lambda s: cosine_schedule_with_warmup_lr_lambda(  # noqa: E731
        s, base_lr=1e-4, num_warmup_steps=10, num_training_steps=100, min_ratio=1.0)
    assert f(0) == 0.0 and math.isclose(f(5), 5e-5)
    assert f(10) == f(50) == f(100) == 1e-4


def test_ema_update_and_state():
    net = torch.nn.Linear(3, 1)
    ema = ModelEMA(net, mu=0.5)
    w0 = net.weight.detach().clone()
    with torch.no_grad():
        net.weight.add_(2.0)
    ema.update(net)
    assert torch.allclose(ema.shadow["weight"], w0 + 1.0)
    sd = ema.ema_model_state(net)
    assert torch.allclose(sd["weight"], w0 + 1.0)
    assert torch.allclose(net.weight, w0 + 2.0)        # live weights untouched
