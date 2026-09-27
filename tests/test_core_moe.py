"""Recurrent-depth / deep cores, DeepSeek-V3 MoE, and the shared multi-game
core: mathematical properties, not just smoke."""

import math

import pytest
import torch

from data_processing.games.dataset import save_shard
from model.components import core as core_mod
from model.components.core import (
    CoreBlock,
    MoEFFN,
    RecurrentDepthCore,
    sample_recurrence,
)
from model.components.trm import trunc_normal_init_
from model.config import ModelConfig, config_from_dict
from model.multi_game import (
    build_multi_game,
    core_state_dict,
    freeze_core,
    is_core_key,
    load_core,
)
from model.network import GeneralsNetwork
from model.optim import ModelEMA

TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, encoder_channels=16,
            dropout=0.0, rd_prelude_layers=1, rd_recurrent_layers=1, rd_coda_layers=1,
            rd_mean_iters=3, rd_backprop_iters=2)


# ---------------------------------------------------------------------------
# recurrent depth
# ---------------------------------------------------------------------------

def test_sampled_recurrence_mean_is_rbar_plus_one():
    """τ ~ N(log r̄ − σ²/2, σ) ⇒ E[e^τ] = r̄ ⇒ E[Poisson(e^τ) + 1] = r̄ + 1."""
    torch.manual_seed(0)
    rs = [sample_recurrence(8.0) for _ in range(20000)]
    assert abs(sum(rs) / len(rs) - 9.0) < 0.15
    assert min(rs) >= 1
    assert max(sample_recurrence(8.0, max_iters=5) for _ in range(200)) <= 5


def test_fixed_inference_state_is_a_draw_of_the_training_distribution():
    cfg = ModelConfig(**TINY)
    rd = RecurrentDepthCore(cfg)
    like = torch.zeros(1, 4096, 256)
    z = rd.fixed_state(4096, like)
    ref = trunc_normal_init_(torch.empty(4096, 256), std=rd.state_std)
    assert abs(z.std().item() - math.sqrt(0.4)) < 0.01           # σ_s² = 2/5
    assert abs(z.abs().max().item() - ref.abs().max().item()) < 0.02   # same truncation
    assert torch.equal(z, rd.fixed_state(4096, like))            # deterministic


def test_eval_deterministic_train_stochastic():
    torch.manual_seed(0)
    rd = RecurrentDepthCore(ModelConfig(**TINY))
    x = torch.randn(2, 5, 32)
    rd.eval()
    with torch.no_grad():
        a, r1 = rd(x)
        b, _ = rd(x)
    assert torch.equal(a, b) and r1 == 3
    rd.train()
    with torch.no_grad():
        c, _ = rd(x, steps=3)
        d, _ = rd(x, steps=3)
    assert not torch.equal(c, d)          # s_0 is sampled in training


def test_truncated_bptt_differentiates_exactly_k_iterations():
    torch.manual_seed(0)
    rd = RecurrentDepthCore(ModelConfig(**TINY)).train()
    calls = {"grad": 0, "nograd": 0}

    def hook(mod, inp, out):
        if core_mod._RECOMPUTING:
            return
        calls["grad" if torch.is_grad_enabled() else "nograd"] += 1

    rd.recurrent[0].register_forward_hook(hook)
    x = torch.randn(2, 5, 32, requires_grad=True)
    y, r = rd(x, steps=6)
    assert r == 6 and calls == {"grad": 2, "nograd": 4}      # k = rd_backprop_iters
    y.square().mean().backward()
    assert rd.prelude[0].qkv.weight.grad.abs().sum() > 0
    assert rd.adapter.weight.grad.abs().sum() > 0
    assert x.grad.abs().sum() > 0


def test_step_matches_forward_recurrent_depth_with_moe():
    torch.manual_seed(0)
    net = GeneralsNetwork(ModelConfig(**TINY, moe_experts=4, temporal_mixer="recurrent")).eval()
    g, gl = torch.rand(2, 1, 12, 25, 25), torch.randn(2, 1, 4)
    with torch.no_grad():
        s = net.step(g[:, 0], gl[:, 0])
        f = net(g, gl)
    assert torch.allclose(s["source_logits"], f["source_logits"][:, 0], atol=1e-5)
    assert s["act_depth"].tolist() == [3, 3]


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

def _naive_moe(m: MoEFFN, x):
    xf = x.reshape(-1, x.shape[-1])
    aff = torch.sigmoid(m.centroids(xf).float())
    out = []
    for t in range(xf.shape[0]):
        sel = (aff[t] + m.balance_bias).topk(m.top_k).indices
        gates = aff[t, sel] / aff[t, sel].sum()
        y = sum(sh(xf[t:t + 1])[0] for sh in m.shared)
        for gi, e in zip(gates, sel):
            gg, uu = (xf[t] @ m.w_gate_up[e]).chunk(2)
            y = y + gi * ((torch.nn.functional.silu(gg) * uu) @ m.w_down[e])
        out.append(y)
    return torch.stack(out).reshape(x.shape)


def test_moe_matches_naive_per_token_computation():
    torch.manual_seed(0)
    m = MoEFFN(16, n_experts=5, top_k=2, n_shared=1, expansion=2.0, std=0.2, out_std=0.2)
    with torch.no_grad():
        m.balance_bias.copy_(torch.tensor([0.3, -0.2, 0.0, 0.1, -0.5]))
    x = torch.randn(3, 7, 16)
    assert torch.allclose(m(x), _naive_moe(m, x), atol=1e-5)


def test_balance_bias_steers_selection_not_gates():
    torch.manual_seed(0)
    m = MoEFFN(8, n_experts=4, top_k=1, n_shared=0, expansion=2.0, std=0.2, out_std=0.2)
    with torch.no_grad():
        m.balance_bias.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))
    m.train()
    x = torch.randn(2, 6, 8)
    m(x)
    assert m.load.tolist() == [12.0, 0.0, 0.0, 0.0]        # every token picks expert 0
    # top-1 gates are s/s = 1 whatever the bias: output = expert 0 exactly
    with torch.no_grad():
        m.balance_bias.copy_(torch.tensor([20.0, 0.0, 0.0, 0.0]))
    y2 = m(x)
    m.balance_bias.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(m(x), y2)


def test_balance_update_direction_and_valid_masking():
    m = MoEFFN(8, n_experts=4, top_k=1, n_shared=0, expansion=2.0, std=0.2, out_std=0.2)
    m.train()
    with torch.no_grad():
        m.balance_bias.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))
    x = torch.randn(3, 5, 8)
    m(x, valid=torch.tensor([True, False, True]))            # row 1 is padding
    assert m.load.sum().item() == 10.0                       # 2 rows × 5 tokens × k=1
    load = m.update_balance(0.1)
    assert load.tolist() == [10.0, 0.0, 0.0, 0.0]
    # [9.9, 0.1, 0.1, 0.1] re-centred (mean 2.55): routing-neutral
    assert torch.allclose(m.balance_bias, torch.tensor([7.35, -2.45, -2.45, -2.45]))
    assert m.load.sum() == 0                                 # tally reset


def test_checkpoint_recompute_is_not_double_counted():
    torch.manual_seed(0)
    cfg = ModelConfig(**TINY, moe_experts=4)
    blk = CoreBlock(cfg, out_std=0.05, moe=True).train()
    x = torch.randn(2, 6, 32, requires_grad=True)
    y = core_mod.ckpt(blk, x, None)
    y.sum().backward()                                        # triggers the recompute
    assert blk.ffn.load.sum().item() == 2 * 6 * cfg.moe_top_k


def test_centroids_receive_gradient_through_gates():
    torch.manual_seed(0)
    m = MoEFFN(8, n_experts=4, top_k=2, n_shared=1, expansion=2.0, std=0.2, out_std=0.2)
    m(torch.randn(2, 5, 8)).square().sum().backward()
    assert m.centroids.weight.grad.abs().sum() > 0
    assert "balance_bias" not in dict(m.named_parameters())


# ---------------------------------------------------------------------------
# legacy + shared multi-game core
# ---------------------------------------------------------------------------

def test_old_checkpoint_configs_keep_the_trm_core():
    cfg = config_from_dict({"d_model": 32, "n_heads": 4})
    assert cfg.core_type == "trm"
    assert ModelConfig().core_type == "recurrent_depth"


CORE = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0, encoder_channels=16,
            rd_prelude_layers=1, rd_recurrent_layers=1, rd_coda_layers=1, rd_mean_iters=2,
            rd_backprop_iters=1, moe_experts=4)


def test_multi_game_shares_exactly_the_core():
    mg = build_multi_game(["othello", "chess"], CORE)
    a, b = mg["othello"], mg["chess"]
    assert a.core is b.core and a.st_layers is b.st_layers and a.hl_token is b.hl_token
    assert a.obs_encoder is not b.obs_encoder and a.value_head is not b.value_head
    n_unique = sum(p.numel() for p in mg.parameters())
    n_core = sum(p.numel() for n, p in a.named_parameters() if is_core_key(n))
    n_adapt = sum(p.numel() for g in ("othello", "chess")
                  for n, p in mg[g].named_parameters() if not is_core_key(n))
    assert n_unique == n_core + n_adapt


def test_core_config_mismatch_is_rejected():
    with pytest.raises(ValueError, match="disagree on core field"):
        build_multi_game(["othello", "chess"], CORE, {"chess": {"moe_experts": 2}})


@pytest.mark.parametrize("field,value", [("moe_bias_clip", 0.5), ("trm_rms_eps", 1e-6)])
def test_core_module_constructor_fields_are_core_fields(field, value):
    """The shared core modules keep these at construction (MoE balance-bias
    clip, the TRM block's RMSNorm eps): the core the games share is built from
    the FIRST game's config, so another value in a later game would silently
    not apply — they must agree like every core field, and --init_core /
    games added on resume take them from the checkpoint (core_config_overrides)."""
    from model.multi_game import CORE_CONFIG_FIELDS, core_config_overrides

    assert field in CORE_CONFIG_FIELDS
    assert core_config_overrides(ModelConfig(**{field: value}))[field] == value
    with pytest.raises(ValueError, match=f"disagree on core field {field}"):
        build_multi_game(["othello", "chess"], CORE, {"chess": {field: value}})
    mg = build_multi_game(["othello", "chess"], {**CORE, "moe_bias_clip": 0.5})
    assert {m.bias_clip for m in core_mod.moe_layers(mg)} == {0.5}


def test_core_transfer_and_freeze():
    mg = build_multi_game(["othello", "chess"], CORE)
    src = core_state_dict(mg["othello"])
    new = build_multi_game(["go"], CORE)["go"]
    load_core(new, src)
    for k, v in core_state_dict(new).items():
        assert torch.equal(v, src[k])
    n_train = freeze_core(new, train_norms=True)
    trainable = {n for n, p in new.named_parameters() if p.requires_grad}
    assert all((not is_core_key(n)) or "norm" in n or n in ("hl_pos", "ll_pos")
               for n in trainable)
    assert any("obs_encoder" in n for n in trainable)
    assert n_train == sum(p.numel() for p in new.parameters() if p.requires_grad)


def test_ema_state_covers_every_alias_of_a_shared_parameter():
    mg = build_multi_game(["othello", "chess"], CORE)
    ema = ModelEMA(mg, mu=0.0)                       # shadow := live weights
    with torch.no_grad():
        mg["othello"].hl_token.add_(1.0)
    ema.update(mg)
    sd = ema.ema_model_state(mg)
    assert torch.equal(sd["nets.othello.hl_token"], sd["nets.chess.hl_token"])
    assert torch.equal(sd["nets.chess.hl_token"], mg["othello"].hl_token.detach())


def _board_samples(game, n, T, board=8, chans=3):
    out = []
    for i in range(n):
        tgt = {"value": torch.rand(T) * 2 - 1}
        if game == "chess":
            tgt.update(from_idx=torch.randint(0, board * board, (T,)),
                       to_idx=torch.randint(0, board * board, (T,)),
                       promotion=torch.zeros(T, dtype=torch.long))
            obs = {"board_seq": torch.rand(T, 14, board, board), "globals_seq": torch.rand(T, 8)}
        else:
            tgt.update(place_idx=torch.randint(0, board * board, (T,)),
                       passed=torch.zeros(T, dtype=torch.bool))
            obs = {"board_seq": torch.rand(T, chans, board, board)}
        out.append({"obs": obs, "tgt": tgt, "meta": {"src": f"{game}{i}"}})
    return out


def test_train_multi_game_joint_then_frozen_adaptation(tmp_path):
    from training import train_multi_game

    for g in ("othello", "chess"):
        d = tmp_path / g
        d.mkdir()
        save_shard(str(d / "a.pt"), g, _board_samples(g, 4, 6))
    core_args = [f"{k}={v}" for k, v in CORE.items()]
    train_multi_game.main([
        "--data", f"othello={tmp_path / 'othello'}", "--data", f"chess={tmp_path / 'chess'}",
        "--val", f"othello={tmp_path / 'othello'}",
        "--steps", "3", "--batch_size", "2", "--warmup_steps", "1", "--eval_every", "3",
        "--save_dir", str(tmp_path / "ck"), "--device", "cpu", "--override", *core_args])
    ck = torch.load(tmp_path / "ck" / "multi_final.pt", weights_only=False)
    assert ck["games"] == ["othello", "chess"] and ck["core_state"]
    assert abs(sum(ck["mixture"].values()) - 1) < 1e-9

    # adapt the brain to a new game with the core frozen
    (tmp_path / "othello2").mkdir()
    save_shard(str(tmp_path / "othello2" / "a.pt"), "othello", _board_samples("othello", 4, 6))
    train_multi_game.main([
        "--data", f"othello={tmp_path / 'othello2'}", "--steps", "3", "--batch_size", "2",
        "--warmup_steps", "1", "--ema_rate", "0", "--save_dir", str(tmp_path / "ck2"),
        "--device", "cpu", "--init_core", str(tmp_path / "ck" / "multi_final.pt"),
        "--freeze_core"])
    ck2 = torch.load(tmp_path / "ck2" / "multi_final.pt", weights_only=False)
    for k, v in ck["core_state"].items():
        if k.endswith("load"):
            continue
        assert torch.equal(ck2["core_state"][k], v), k     # frozen core unchanged


def test_bias_centring_is_routing_neutral_and_clip_bounds():
    torch.manual_seed(0)
    m = MoEFFN(8, n_experts=4, top_k=2, n_shared=0, expansion=2.0, std=0.2, out_std=0.2)
    m.train()
    x = torch.randn(3, 5, 8)
    with torch.no_grad():
        m.balance_bias.copy_(torch.tensor([0.3, 1.1, 0.7, 0.9]))
        y0 = m(x)
        m.balance_bias -= m.balance_bias.mean()           # constant shift
        y1 = m(x)
    assert torch.allclose(y0, y1)                          # selection unchanged
    m.bias_clip = 0.05
    for _ in range(50):
        m(torch.randn(3, 5, 8))
        m.update_balance(0.01)
    assert m.balance_bias.abs().max() <= 0.05 + 1e-7
    assert abs(m.balance_bias.mean().item()) < 0.05


def _sparse_experts_indexed(moe, xf, top_i, gates):
    """The pre-unbind reference: per-expert ``w[e]`` indexing in the loop."""
    import torch.nn.functional as F
    out = xf.new_zeros(xf.shape)
    for e in range(moe.n_experts):
        rows, slot = (top_i == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        h = xf.index_select(0, rows)
        g, u = (h @ moe.w_gate_up[e].to(h.dtype)).chunk(2, dim=-1)
        h = (F.silu(g) * u) @ moe.w_down[e].to(h.dtype)
        out.index_add_(0, rows, h * gates[rows, slot].unsqueeze(-1))
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sparse_experts_unbind_matches_indexed_reference(dtype):
    """unbind-ed expert weights give bit-identical outputs and weight
    gradients (incl. experts that received no token) to w[e] indexing."""
    from model.components.core import MoEFFN

    torch.manual_seed(0)
    moe = MoEFFN(32, 8, 2, 1, 2.0, 0.02, 0.02)
    xf = torch.randn(40, 32).to(dtype)
    top_i = torch.randint(0, 6, (40, 2))           # experts 6, 7 get no token
    gates = torch.rand(40, 2).to(dtype)
    grads = []
    for fn in (moe._sparse_experts, lambda x, t, g: _sparse_experts_indexed(moe, x, t, g)):
        moe.zero_grad(set_to_none=True)
        out = fn(xf, top_i, gates)
        (out.float() * torch.linspace(-1, 1, out.numel()).reshape(out.shape)).sum().backward()
        grads.append((out.detach().clone(), moe.w_gate_up.grad.clone(), moe.w_down.grad.clone()))
    for a, b in zip(*grads):
        assert torch.equal(a, b)
    assert not grads[0][1][6:].any()
