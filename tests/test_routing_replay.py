"""MoE routing replay (model.components.core): the RL actor records every
token's selected experts, the learner's forward replays them.

Properties pinned here:
  * replaying a forward's own routing reproduces it exactly (outputs and
    gradients, also through the activation-checkpoint recompute);
  * a replayed routing overrides the top-k (and the balance bias), while the
    gates still come from the current affinities — the router keeps its
    gradient;
  * GameNetwork.step(record_routes=True) → forward(routes=...) reproduces the
    step's outputs frame by frame even when the learner's own routing would
    differ (a perturbed balance bias stands in for a near-tie flip), for the
    deep and the recurrent-depth core;
  * shape / depth mismatches are refused.
"""

import pytest
import torch

from model.components import core as core_mod
from model.components.core import (
    CoreBlock,
    MoEFFN,
    moe_layers,
    record_moe_routes,
    route_dtype,
    stack_routes,
)
from model.config import ModelConfig
from model.game_network import GameNetwork
from model.games.generals import GeneralsDecoder, GeneralsEncoder

H = W = 4
CFG = dict(d_model=32, n_heads=4, d_ff=64, encoder_channels=8, n_st_layers=1, dropout=0.0,
           grid_h=H, grid_w=W, moe_experts=6, moe_top_k=2, moe_shared_experts=1,
           rd_prelude_layers=1, rd_recurrent_layers=1, rd_coda_layers=1, rd_mean_iters=2,
           rd_backprop_iters=1)


def _moe(k=2, E=6):
    torch.manual_seed(0)
    return MoEFFN(16, n_experts=E, top_k=k, n_shared=1, expansion=2.0, std=0.3, out_std=0.3)


def _manual(m: MoEFFN, x: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
    """Σ_shared FFN + Σ_j g_j FFN_{route_j}, g from the current affinities."""
    xf = x.reshape(-1, x.shape[-1])
    aff = torch.sigmoid(m.centroids(xf).float())
    r = route.reshape(xf.shape[0], -1).long()
    out = []
    for t in range(xf.shape[0]):
        s = aff[t, r[t]]
        y = sum(sh(xf[t:t + 1])[0] for sh in m.shared)
        for g, e in zip(s / s.sum(), r[t]):
            gg, uu = (xf[t] @ m.w_gate_up[e]).chunk(2)
            y = y + g * ((torch.nn.functional.silu(gg) * uu) @ m.w_down[e])
        out.append(y)
    return torch.stack(out).reshape(x.shape)


# ---------------------------------------------------------------------------
# MoEFFN
# ---------------------------------------------------------------------------

def test_replaying_the_own_routing_is_exact_and_overrides_the_bias():
    m = _moe().eval()
    x = torch.randn(3, 5, 16)
    with torch.no_grad(), record_moe_routes(m) as calls:
        y = m(x)
    assert len(calls) == 1 and calls[0].shape == (15, 2)
    with torch.no_grad():
        assert torch.equal(m(x, route=calls[0]), y)
        # a bias that moves the selection: the own routing changes, the replay does not
        m.balance_bias.copy_(torch.tensor([3.0, -3.0, 3.0, -3.0, 0.0, 0.0]))
        assert not torch.allclose(m(x), y)
        assert torch.equal(m(x, route=calls[0]), y)


def test_a_replayed_route_uses_the_current_gates():
    m = _moe().eval()
    x = torch.randn(2, 4, 16)
    g = torch.Generator().manual_seed(1)
    route = torch.stack([torch.randperm(6, generator=g)[:2] for _ in range(8)]).to(torch.uint8)
    with torch.no_grad():
        assert torch.allclose(m(x, route=route), _manual(m, x, route), atol=1e-5)
    # the router's centroids get a gradient through the replayed gates
    m.train()
    m(x, route=route).square().sum().backward()
    assert m.centroids.weight.grad.abs().sum() > 0


def test_route_shape_is_checked():
    m = _moe().eval()
    with pytest.raises(ValueError, match="replayed MoE route"):
        m(torch.randn(2, 4, 16), route=torch.zeros(8, 3, dtype=torch.long))


def test_route_dtype_is_the_narrowest_that_fits():
    assert route_dtype(64) == torch.uint8 and route_dtype(256) == torch.uint8
    assert route_dtype(257) == torch.int16 and route_dtype(40000) == torch.int32


def test_checkpointed_block_replays_through_the_recompute():
    """Train mode, grad on: the block runs under the activation checkpoint and
    its backward recomputes the forward — with the replayed route an explicit
    argument, the recompute takes the same experts; recording skips it."""
    torch.manual_seed(0)
    cfg = ModelConfig(**CFG)
    blk = CoreBlock(cfg, out_std=0.05, moe=True).train()
    x = torch.randn(2, 6, 32, requires_grad=True)
    with record_moe_routes(blk) as calls:
        core_mod.ckpt(blk, x, None).square().sum().backward()
    assert len(calls) == 1                                   # the recompute is not recorded
    ref = {n: p.grad.clone() for n, p in blk.named_parameters()}
    gx = x.grad.clone()
    blk.zero_grad()
    x.grad = None
    route = calls[0].view(2, 6, -1)
    core_mod.ckpt(blk, x, None, route).square().sum().backward()
    for n, p in blk.named_parameters():
        assert torch.allclose(p.grad, ref[n], atol=1e-6), n
    assert torch.allclose(x.grad, gx, atol=1e-6)


def test_dense_block_refuses_a_route():
    blk = CoreBlock(ModelConfig(**CFG), out_std=0.05, moe=False).eval()
    with pytest.raises(ValueError, match="non-MoE"):
        blk(torch.randn(1, 3, 32), None, torch.zeros(1, 3, 2, dtype=torch.long))


# ---------------------------------------------------------------------------
# GameNetwork: step records, forward replays
# ---------------------------------------------------------------------------

def _net(core_type: str, temporal_mixer: str) -> GameNetwork:
    torch.manual_seed(0)
    cfg = ModelConfig(**CFG, core_type=core_type, core_layers=2, temporal_mixer=temporal_mixer)
    return GameNetwork(cfg, GeneralsEncoder(cfg), GeneralsDecoder(cfg)).eval()


def _frames(B=2, T=3):
    g = torch.Generator().manual_seed(3)
    return (torch.rand(B, T, 12, H, W, generator=g), torch.randn(B, T, 4, generator=g))


def _perturb_routing(net: GameNetwork) -> None:
    """Move every MoE layer's balance bias far enough to change the routing:
    what a numerics-induced near-tie flip does to a few tokens."""
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for m in moe_layers(net):
            m.balance_bias.copy_(torch.randn(m.n_experts, generator=g) * 2.0)


@pytest.mark.parametrize("core_type,depth", [("transformer", None), ("recurrent_depth", 2),
                                             ("recurrent_depth", 3)])
@pytest.mark.parametrize("temporal_mixer", ["recurrent", "attention"])
def test_step_routes_replayed_by_forward(core_type, depth, temporal_mixer):
    net = _net(core_type, temporal_mixer)
    grid, glob = _frames()
    B, T = grid.shape[:2]
    S = H * W + 2
    C = net.moe_route_calls(depth)
    assert C == (2 if core_type == "transformer" else 1 + depth + 1)
    steps, routes, cache = [], [], None
    with torch.no_grad():
        for t in range(T):
            out = net.step({"grid_seq": grid[:, t], "globals_seq": glob[:, t]}, st_cache=cache,
                           step_idx=t, act_steps=depth, record_routes=True)
            cache = out["st_cache"]
            steps.append(out["source_logits"])
            routes.append(out["moe_routes"])
            assert out["moe_routes"].shape == (B, C, S, 2)
            assert out["moe_routes"].dtype == torch.uint8
        routes = torch.stack(routes, 1)                      # [B, T, C, S, k]
        obs = {"grid_seq": grid, "globals_seq": glob}
        same = net(obs, act_steps=depth)
        _perturb_routing(net)
        own = net(obs, act_steps=depth)
        replay = net(obs, act_steps=depth, routes=routes)
    ref = torch.stack(steps, 1)
    assert torch.allclose(same["source_logits"], ref, atol=1e-5)
    assert not torch.allclose(own["source_logits"], ref, atol=1e-3)    # the routing moved
    assert torch.allclose(replay["source_logits"], ref, atol=1e-5)
    assert torch.allclose(replay["value"], same["value"], atol=1e-5)


def test_forward_refuses_routes_of_another_depth():
    net = _net("recurrent_depth", "recurrent")
    grid, glob = _frames(T=1)
    with torch.no_grad():
        out = net.step({"grid_seq": grid[:, 0], "globals_seq": glob[:, 0]}, act_steps=2,
                       record_routes=True)
        routes = out["moe_routes"].unsqueeze(1)
        with pytest.raises(ValueError, match="same core depth"):
            net({"grid_seq": grid, "globals_seq": glob}, act_steps=3, routes=routes)


def test_step_without_moe_records_nothing():
    torch.manual_seed(0)
    cfg = ModelConfig(**{**CFG, "moe_experts": 0}, core_type="transformer", core_layers=1)
    net = GameNetwork(cfg, GeneralsEncoder(cfg), GeneralsDecoder(cfg)).eval()
    assert net.moe_route_calls() == 0
    grid, glob = _frames(T=1)
    with torch.no_grad():
        out = net.step({"grid_seq": grid[:, 0], "globals_seq": glob[:, 0]}, record_routes=True)
    assert "moe_routes" not in out


def test_stack_routes_layout():
    calls = [torch.arange(12).view(6, 2) + 100 * c for c in range(3)]
    r = stack_routes(calls, rows=2, tokens=3, n_experts=1000)
    assert r.shape == (2, 3, 3, 2) and r.dtype == torch.int16
    assert r[1, 2, 0].tolist() == [206, 207]      # row 1, call 2, token 0
