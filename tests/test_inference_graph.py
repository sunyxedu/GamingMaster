"""Sync-free, CUDA-graph-capturable per-decision inference.

* dense (all-experts) MoE inference path == sparse routed path; every
  autograd path stays sparse;
* the sampling / masking helpers are branch-free rewrites with identical
  results (``categorical_sample`` is bit-identical to ``torch.multinomial``
  and still rejects invalid distributions, without a host sync);
* the eval ``GameNetwork.step`` path never reads a tensor on the host
  (CPU: forbidden-call monkeypatching; CUDA: ``set_sync_debug_mode("error")``);
* a tensor ``step_idx`` + materialised initial state reproduce the int /
  ``st_cache=None`` path, and the rollout log-prob still replays exactly;
* CUDA only: ``GraphedStep`` replays == eager steps.
"""

from __future__ import annotations

import contextlib

import pytest
import torch
import torch.nn.functional as F

from model.components.core import MoEFFN
from model.config import ModelConfig
from model.game_network import GameNetwork
from model.games.base import (
    categorical_sample,
    masked_log_prob,
    masked_sample,
    safe_masked_logits,
)
from model.network import GeneralsNetwork

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=2, encoder_channels=16,
            dropout=0.0, rd_prelude_layers=1, rd_recurrent_layers=1, rd_coda_layers=1,
            rd_mean_iters=2, rd_backprop_iters=1, core_layers=2,
            temporal_mixer="recurrent")
H = W = 25
ACTION_KEYS = ("action_source", "action_direction", "action_split", "action_pass")
LOGIT_KEYS = ("source_logits", "direction_logits", "split_logits", "pass_logits", "value")


def _net(core="recurrent_depth", moe=4, joint=False, device="cpu", **kw):
    torch.manual_seed(0)
    cfg = ModelConfig(**{**TINY, **kw}, core_type=core, moe_experts=moe, joint_decoder=joint)
    return GeneralsNetwork(cfg).to(device).eval()


def _obs(B, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"grid_seq": torch.rand(B, 12, H, W, generator=g).to(device),
            "globals_seq": torch.randn(B, 4, generator=g).to(device)}


def _legal(B, device="cpu", seed=0, all_masked_row=None):
    g = torch.Generator().manual_seed(seed + 1)
    src = torch.rand(B, 1, H, W, generator=g) < 0.1
    dirm = torch.rand(B, 4, H, W, generator=g) < 0.6
    if all_masked_row is not None:
        src[all_masked_row] = False
    return {"source": src.to(device), "direction": dirm.to(device)}


# ---------------------------------------------------------------------------
# MoE: dense inference path == sparse routed path
# ---------------------------------------------------------------------------

def _moe(E=5, k=2, n_shared=1, d=16, seed=0):
    torch.manual_seed(seed)
    m = MoEFFN(d, n_experts=E, top_k=k, n_shared=n_shared, expansion=2.0,
               std=0.2, out_std=0.2)
    with torch.no_grad():
        m.balance_bias.copy_(torch.linspace(-0.5, 0.4, E))
    return m.eval()


def _sparse(m, x):
    """The routed path, forced."""
    with MoEFFN.inference_path("sparse"):
        return m(x)


def _dense(m, x):
    """The all-experts path, forced (the automatic choice is sparse on CPU)."""
    with MoEFFN.inference_path("dense"):
        return m(x)


@contextlib.contextmanager
def _dense_max(n):
    old = MoEFFN.DENSE_MAX_TOKEN_EXPERTS
    MoEFFN.DENSE_MAX_TOKEN_EXPERTS = n
    try:
        yield
    finally:
        MoEFFN.DENSE_MAX_TOKEN_EXPERTS = old


@pytest.mark.parametrize("E,k,n_shared", [(5, 2, 1), (8, 1, 0), (4, 4, 2), (8, 3, 1)])
def test_dense_moe_equals_sparse(E, k, n_shared):
    m = _moe(E, k, n_shared)
    x = torch.randn(3, 7, 16)
    with torch.no_grad():
        dense, sparse = _dense(m, x), _sparse(m, x)
    torch.testing.assert_close(dense, sparse, rtol=1e-5, atol=1e-6)


def test_autograd_always_takes_the_sparse_path(monkeypatch):
    """Eval mode WITH autograd is the PPO update regime (train_rl*.py run the
    update in eval mode): it must stay on the routed path even when dense is
    forced — the dense backward would put 0·inf = NaN into w_down.grad for
    an overflowing expert the token was never routed to."""
    m = _moe(4, 2, 1)
    with torch.no_grad():
        m.balance_bias.copy_(torch.tensor([0.0, 0.0, 0.0, -100.0]))   # never pick 3
        m.w_gate_up[3].fill_(1e30)                                     # overflows
    x = torch.randn(2, 5, 16)

    def boom(*_a, **_k):
        raise AssertionError("dense MoE path under autograd")

    with monkeypatch.context() as mp, MoEFFN.inference_path("dense"):
        mp.setattr(m, "_dense_experts", boom)
        assert m.expert_path(10, torch.device("cuda")) == "sparse"
        m(x).square().sum().backward()
    for name, prm in m.named_parameters():
        assert torch.isfinite(prm.grad).all(), name


def test_dense_moe_ignores_non_finite_unselected_expert():
    """An expert no token routes to may blow up (0·inf = NaN) without
    leaking into the dense sum — the routed path never computes it."""
    m = _moe(4, 2, 1)
    with torch.no_grad():
        m.balance_bias.copy_(torch.tensor([0.0, 0.0, 0.0, -100.0]))   # never pick 3
        m.w_down[3].fill_(float("inf"))
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        dense = _dense(m, x)
    assert torch.isfinite(dense).all()
    torch.testing.assert_close(dense, _sparse(m, x), rtol=1e-5, atol=1e-6)


def test_dense_moe_under_bf16_autocast_matches_sparse():
    m = _moe(8, 2, 1)
    x = torch.randn(2, 11, 16)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        dense, sparse = _dense(m, x), _sparse(m, x)
    assert dense.dtype == sparse.dtype
    torch.testing.assert_close(dense.float(), sparse.float(), rtol=2e-2, atol=2e-2)


def test_moe_path_selection(monkeypatch):
    """Automatic: dense exactly in eval mode, autograd off, on an accelerator,
    with N·E <= DENSE_MAX_TOKEN_EXPERTS.  inference_path() forces the no-grad
    eval path; training and autograd are always sparse."""
    m = _moe(4, 2, 1)
    cuda, cpu = torch.device("cuda"), torch.device("cpu")
    with torch.no_grad():
        with _dense_max(40):
            assert m.expert_path(10, cuda) == "dense"          # 40 <= 40
            assert m.expert_path(11, cuda) == "sparse"         # 44 > 40
            assert m.expert_path(10, "mps") == "dense"
            assert m.expert_path(10, cpu) == "sparse"          # CPU: routed is cheaper
        with MoEFFN.inference_path("dense"), _dense_max(0):
            assert m.expert_path(10 ** 6, cpu) == "dense"
        with MoEFFN.inference_path("sparse"):
            assert m.expert_path(1, cuda) == "sparse"
        assert MoEFFN._INFERENCE_PATH == "auto"                # restored
        m.train()
        with MoEFFN.inference_path("dense"):
            assert m.expert_path(1, cuda) == "sparse"          # training
        m.eval()
    with MoEFFN.inference_path("dense"):
        assert m.expert_path(1, cuda) == "sparse"              # autograd on
    with pytest.raises(ValueError, match="auto"):
        with MoEFFN.inference_path("fast"):
            pass

    x = torch.randn(2, 5, 16)                                  # N = 10

    def boom(*_a, **_k):
        raise AssertionError("wrong MoE path")

    with torch.no_grad():
        with monkeypatch.context() as mp, MoEFFN.inference_path("dense"):
            mp.setattr(m, "_sparse_experts", boom)
            m(x)
        with monkeypatch.context() as mp:                      # CPU auto → sparse
            mp.setattr(m, "_dense_experts", boom)
            m(x)
    m.train()
    with monkeypatch.context() as mp:                  # training → sparse + load tally
        mp.setattr(m, "_dense_experts", boom)
        m(x)
    assert m.load.sum().item() == 2 * 5 * m.top_k
    m.eval()
    m.load.zero_()
    with torch.no_grad(), MoEFFN.inference_path("dense"):
        m(x)
    assert m.load.sum().item() == 0                    # no tally at inference


def test_recurrent_depth_eval_dense_vs_sparse_network():
    net = _net(moe=4)
    obs, legal = _obs(2), _legal(2)
    with torch.no_grad():
        torch.manual_seed(5)
        with MoEFFN.inference_path("dense"):
            a = GameNetwork.step(net, obs, legal, act_steps=2)
        torch.manual_seed(5)
        with MoEFFN.inference_path("sparse"):
            b = GameNetwork.step(net, obs, legal, act_steps=2)
    for key in LOGIT_KEYS:
        torch.testing.assert_close(a[key], b[key], rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Branch-free sampling helpers: identical results
# ---------------------------------------------------------------------------

def test_categorical_sample_is_bitwise_multinomial():
    for seed in range(60):
        B, N = seed % 4 + 1, (2, 4, 625, 17)[seed % 4]
        probs = torch.softmax(torch.randn(B, N) * 3, -1)
        torch.manual_seed(seed)
        ref = torch.multinomial(probs, 1).squeeze(-1)
        ref_next = torch.rand(2)
        torch.manual_seed(seed)
        got = categorical_sample(probs)
        assert torch.equal(got, ref)
        assert torch.equal(torch.rand(2), ref_next)      # same RNG consumption


@pytest.mark.parametrize("bad", [
    [[0.2, float("nan"), 0.8]], [[0.2, float("inf"), 0.8]], [[0.5, -0.1, 0.6]],
    [[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]])
def test_categorical_sample_rejects_invalid_probs_like_multinomial(bad):
    """Still loud on an invalid distribution (async assert, no host read)."""
    probs = torch.tensor(bad)
    with pytest.raises(RuntimeError):
        torch.multinomial(probs, 1)
    with pytest.raises(RuntimeError, match="probability tensor"):
        categorical_sample(probs)


def test_nan_direction_logits_fail_loudly_not_a_nan_log_prob():
    """The direction / split heads have no NaN-row fallback (only the source
    head does): a NaN there must crash the rollout, as multinomial did,
    rather than record a NaN action_log_prob for the PPO update."""
    net = _net(moe=0)
    B, N = 2, H * W
    pred = {"source_logits": torch.randn(B, N), "src_idx": torch.tensor([3, 7]),
            "direction_logits": torch.randn(B, 4), "split_logits": torch.randn(B, 2),
            "pass_logits": torch.randn(B)}
    pred["direction_logits"][0] = float("nan")
    with pytest.raises(RuntimeError, match="probability tensor"):
        net.action_decoder.sample(pred)
    pred["direction_logits"][0] = 0.0
    pred["split_logits"][1, 0] = float("nan")
    with pytest.raises(RuntimeError, match="probability tensor"):
        net.action_decoder.sample(pred)


def _old_safe_masked_logits(logits, mask):
    """The pre-rewrite (branchy, syncing) implementation, verbatim."""
    safe = logits.detach().clone().float()
    if mask is not None:
        safe = safe.masked_fill(~mask, float("-inf"))
        all_masked = (safe.isinf() & (safe < 0)).all(dim=-1)
        if all_masked.any():
            safe[..., 0] = safe[..., 0].masked_fill(all_masked, 0.0)
    nan_rows = safe.isnan().any(dim=-1)
    if nan_rows.any():
        safe[nan_rows] = 0.0
    return safe


def _old_masked_log_prob(logits, index, mask=None):
    logits = logits.float()
    if mask is not None:
        safe = logits.masked_fill(~mask, float("-inf"))
        all_masked = (safe.isinf() & (safe < 0)).all(dim=-1)
        if all_masked.any():
            safe = safe.clone()
            safe[..., 0] = safe[..., 0].masked_fill(all_masked, 0.0)
    else:
        safe = logits
    return F.log_softmax(safe, dim=-1).gather(-1, index.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize("shape", [(6, 9), (2, 3, 9), (9,)])
@pytest.mark.parametrize("with_mask", [True, False])
def test_safe_masked_logits_unchanged(shape, with_mask):
    torch.manual_seed(0)
    logits = torch.randn(*shape).to(torch.bfloat16)
    mask = torch.rand(*shape) < 0.4 if with_mask else None
    flat = logits.view(-1, shape[-1])
    flat[0, 3] = float("nan")                                  # a NaN row
    if len(shape) > 1:
        flat[1, 0] = float("-inf")                             # a raw -inf entry
        if mask is not None:
            mask.view(-1, shape[-1])[-1] = False               # an all-masked row
    got, ref = safe_masked_logits(logits, mask), _old_safe_masked_logits(logits, mask)
    assert got.dtype == torch.float32 and torch.equal(got, ref)
    for healthy in (torch.randn(*shape), torch.randn(*shape) * 50):
        m2 = torch.ones(*shape, dtype=torch.bool) if with_mask else None
        assert torch.equal(safe_masked_logits(healthy, m2), _old_safe_masked_logits(healthy, m2))


@pytest.mark.parametrize("all_masked_row", [True, False])
def test_masked_log_prob_unchanged_values_and_grads(all_masked_row):
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)
    mask = torch.rand(2, 3, 7) < 0.5
    mask[..., 4] = True
    if all_masked_row:
        mask[1, 2] = False
    idx = torch.randint(0, 7, (2, 3))
    idx[1, 2] = 0
    outs = []
    for fn in (masked_log_prob, _old_masked_log_prob):
        x = logits.clone().requires_grad_(True)
        lp = fn(x, idx, mask)
        finite = torch.isfinite(lp)
        lp[finite].sum().backward()
        outs.append((lp.detach(), x.grad))
    assert torch.equal(outs[0][0], outs[1][0])
    assert torch.equal(outs[0][1], outs[1][1])


def test_masked_sample_same_draws_as_before():
    """masked_sample's draw == the old softmax + multinomial draw."""
    logits = torch.randn(4, 30)
    mask = torch.rand(4, 30) < 0.3
    mask[2] = False
    for seed in range(20):
        torch.manual_seed(seed)
        got = masked_sample(logits, mask)
        torch.manual_seed(seed)
        ref = torch.multinomial(
            F.softmax(_old_safe_masked_logits(logits, mask), -1), 1).squeeze(-1)
        assert torch.equal(got, ref)
    assert (got[2] == 0).item()                       # all-masked → index 0


# ---------------------------------------------------------------------------
# The eval step path contains no host sync (CPU, by construction)
# ---------------------------------------------------------------------------

_SYNC_METHODS = ("item", "tolist", "numpy", "nonzero", "multinomial", "masked_select",
                 "unique", "bincount", "__bool__", "__int__", "__float__", "__index__")
_SYNC_FUNCS = ("nonzero", "argwhere", "multinomial", "masked_select", "unique",
               "bincount")


def _has_bool_index(idx) -> bool:
    items = idx if isinstance(idx, tuple) else (idx,)
    return any(isinstance(i, torch.Tensor) and i.dtype == torch.bool for i in items)


def _forbid_host_syncs(mp):
    """Every Python-level way to read a tensor on the host (or to create a
    data-dependent shape) raises for the duration of ``mp``."""
    def boom(name):
        def f(*_a, **_k):
            raise AssertionError(f"host sync on the eval step path: {name}")
        return f

    for n in _SYNC_METHODS:
        mp.setattr(torch.Tensor, n, boom(f"Tensor.{n}"))
    for n in _SYNC_FUNCS:
        mp.setattr(torch, n, boom(f"torch.{n}"))
    get, put = torch.Tensor.__getitem__, torch.Tensor.__setitem__

    def getitem(self, idx):
        if _has_bool_index(idx):
            raise AssertionError("boolean-mask indexing (nonzero) on the eval step path")
        return get(self, idx)

    def setitem(self, idx, val):
        if _has_bool_index(idx):
            raise AssertionError("boolean-mask assignment (nonzero) on the eval step path")
        return put(self, idx, val)

    mp.setattr(torch.Tensor, "__getitem__", getitem)
    mp.setattr(torch.Tensor, "__setitem__", setitem)


STEP_CASES = {
    "rd_moe_rl": dict(net=dict(core="recurrent_depth", moe=4), legal=False),
    "rd_moe_legal": dict(net=dict(core="recurrent_depth", moe=4), legal=True),
    "rd_moe_legal_greedy": dict(net=dict(core="recurrent_depth", moe=4), legal=True,
                                greedy=True),
    "rd_dense_legal": dict(net=dict(core="recurrent_depth", moe=0), legal=True),
    "transformer_moe_legal": dict(net=dict(core="transformer", moe=4), legal=True),
    "joint_rd_moe_legal": dict(net=dict(core="recurrent_depth", moe=4, joint=True),
                               legal=True),
    "rd_moe_hl_gauss": dict(net=dict(core="recurrent_depth", moe=4, hl_gauss_value=True),
                            legal=False),
}


@pytest.mark.parametrize("case", list(STEP_CASES))
@pytest.mark.parametrize("tensor_t", [True, False])
def test_eval_step_path_is_sync_free_cpu(case, tensor_t, monkeypatch):
    spec = STEP_CASES[case]
    net = _net(**spec["net"])
    net.action_decoder.greedy = spec.get("greedy", False)
    B = 2
    obs = _obs(B)
    legal = _legal(B, all_masked_row=1) if spec["legal"] else None
    net.prepare_inference()
    cache = net.initial_st_cache(B)
    t = torch.tensor(3) if tensor_t else 3
    # dense MoE forced: the automatic choice is the routed path on the CPU
    # (where nonzero is no sync); on an accelerator batch 2 is dense anyway.
    with torch.no_grad(), monkeypatch.context() as mp, MoEFFN.inference_path("dense"):
        _forbid_host_syncs(mp)
        out = GameNetwork.step(net, obs, legal, st_cache=cache, step_idx=t, act_steps=2)
        out2 = GameNetwork.step(net, obs, legal, st_cache=out["st_cache"],
                                step_idx=t + 1, act_steps=2)
    assert out2["action_log_prob"].shape == (B,)
    assert torch.isfinite(out2["action_log_prob"]).all()


def test_forbidden_call_guard_catches_the_old_paths(monkeypatch):
    """The guard is live: the previous implementations trip it."""
    logits, mask = torch.randn(2, 5), torch.rand(2, 5) < 0.5
    probs = torch.softmax(logits, -1)
    m = _moe(4, 2, 1)
    with monkeypatch.context() as mp:
        _forbid_host_syncs(mp)
        with pytest.raises(AssertionError, match="host sync"):
            _old_safe_masked_logits(logits, mask)
        with pytest.raises(AssertionError, match="host sync"):
            torch.multinomial(probs, 1)
        with pytest.raises(AssertionError, match="host sync"):
            _sparse(m, torch.randn(1, 3, 16))
        with pytest.raises(AssertionError, match="boolean-mask"):
            logits[mask]


# ---------------------------------------------------------------------------
# Graph-friendly calling convention == historical one; RL exactness holds
# ---------------------------------------------------------------------------

def test_initial_st_cache_is_the_none_state():
    net = _net()
    cache = net.initial_st_cache(3)
    S = net.obs_encoder.n_tokens + 2
    assert len(cache) == TINY["n_st_layers"]
    for c, layer in zip(cache, net.st_layers):
        assert c.shape == (3 * S, TINY["d_model"]) and c.dtype == torch.float32
        assert c.is_contiguous() and not c.requires_grad
        assert torch.equal(c, layer.temporal_rnn.initial_state(3 * S))
    # documented: the materialised cache is stepped with a tensor step_idx;
    # an int 0 with a cache present trips the forgotten-step_idx guard
    with torch.no_grad(), pytest.raises(ValueError, match="step_idx"):
        GameNetwork.step(net, _obs(3), st_cache=cache, step_idx=0, act_steps=1)
    with pytest.raises(ValueError, match="recurrent"):
        _net(temporal_mixer="attention").initial_st_cache(1)


@pytest.mark.parametrize("legal_on", [False, True])
def test_tensor_step_idx_and_static_cache_match_int_path(legal_on):
    net = _net(moe=4)
    B = 2
    cache_a, cache_b = None, net.initial_st_cache(B)
    for t in range(4):
        obs = _obs(B, seed=t)
        legal = _legal(B, seed=t) if legal_on else None
        with torch.no_grad():
            torch.manual_seed(100 + t)
            a = GameNetwork.step(net, obs, legal, st_cache=cache_a, step_idx=t, act_steps=2)
            torch.manual_seed(100 + t)
            b = GameNetwork.step(net, obs, legal, st_cache=cache_b,
                                 step_idx=torch.tensor(t), act_steps=2)
        for key in LOGIT_KEYS:
            torch.testing.assert_close(a[key], b[key], rtol=1e-5, atol=1e-6)
        for key in ACTION_KEYS:
            assert torch.equal(a[key], b[key]), (t, key)
        torch.testing.assert_close(a["action_log_prob"], b["action_log_prob"],
                                   rtol=1e-5, atol=1e-6)
        for ca, cb in zip(a["st_cache"], b["st_cache"]):
            torch.testing.assert_close(ca, cb, rtol=1e-5, atol=1e-6)
        cache_a, cache_b = a["st_cache"], b["st_cache"]


def test_time_pe_tensor_rows_match_int_rows():
    from model.components.transformer import _sinusoidal_pe, sinusoidal_pe_rows
    for t0, T in [(0, 1), (7, 1), (1234, 1), (3, 5)]:
        ref = _sinusoidal_pe(t0 + T, 32, torch.device("cpu"))[t0:t0 + T]
        assert torch.equal(sinusoidal_pe_rows(t0, T, 32, torch.device("cpu")), ref)
        got = sinusoidal_pe_rows(torch.tensor(t0), T, 32, torch.device("cpu"))
        torch.testing.assert_close(got, ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize("replay_sparse", [False, True])
def test_rollout_log_prob_replays_exactly(replay_sparse):
    """Rollout through the graph-friendly step (tensor time, static cache,
    dense MoE) → sequence forward + compute_log_prob: ratio == 1, also when
    the replay routes the MoE sparsely (as the with-grad PPO update always
    does)."""
    cfg_net = _net(moe=4)
    T = 5
    cache = cfg_net.initial_st_cache(1)
    rec = {k: [] for k in ("grid", "glb", "src_m", "dir_m", "src", "dir", "spl", "pass", "lp")}
    for t in range(T):
        obs, legal = _obs(1, seed=t), _legal(1, seed=t)
        with torch.no_grad(), MoEFFN.inference_path("dense"):
            out = GameNetwork.step(cfg_net, obs, legal, st_cache=cache,
                                   step_idx=torch.tensor(t), act_steps=2)
        cache = out["st_cache"]
        rec["grid"].append(obs["grid_seq"][0])
        rec["glb"].append(obs["globals_seq"][0])
        rec["src_m"].append(legal["source"][0])
        rec["dir_m"].append(legal["direction"][0])
        for k, key in zip(("src", "dir", "spl", "pass", "lp"),
                          ACTION_KEYS + ("action_log_prob",)):
            rec[k].append(out[key][0])
    actions = {"source_idx": torch.stack(rec["src"])[None],
               "direction": torch.stack(rec["dir"])[None],
               "split": torch.stack(rec["spl"])[None],
               "passed": torch.stack(rec["pass"])[None]}
    src_m = torch.stack(rec["src_m"])[None]
    with torch.no_grad(), MoEFFN.inference_path("sparse" if replay_sparse else "dense"):
        pred = cfg_net(torch.stack(rec["grid"])[None], torch.stack(rec["glb"])[None],
                       legal_mask=src_m,
                       source_y=actions["source_idx"] // W,
                       source_x=actions["source_idx"] % W,
                       act_steps=2, direction_mask=torch.stack(rec["dir_m"])[None])
        lp, _, _ = cfg_net.compute_log_prob(pred, actions, legal={"source": src_m})
    ratio = (lp - torch.stack(rec["lp"])[None]).exp()
    torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=0, atol=1e-4)


def test_recurrent_depth_fixed_state_precompute_is_a_lookup(monkeypatch):
    net = _net()
    net.prepare_inference(torch.device("cpu"))
    assert len(net.core._fixed_state) == 1
    obs = _obs(1)

    def no_rebuild(*_a, **_k):
        raise AssertionError("fixed state rebuilt inside the step")

    monkeypatch.setattr(torch, "Generator", no_rebuild)
    with torch.no_grad():
        GameNetwork.step(net, obs, act_steps=1)


# ---------------------------------------------------------------------------
# CUDA only: no sync under set_sync_debug_mode("error"); graph == eager
# ---------------------------------------------------------------------------

@needs_cuda
@pytest.mark.parametrize("case", ["rd_moe_rl", "rd_moe_legal", "transformer_moe_legal",
                                  "joint_rd_moe_legal"])
@pytest.mark.parametrize("bf16", [False, True])
def test_eval_step_has_no_host_sync_cuda(case, bf16):
    spec = STEP_CASES[case]
    dev = torch.device("cuda")
    net = _net(device=dev, **spec["net"])
    B = 2
    obs = _obs(B, dev)
    legal = _legal(B, dev, all_masked_row=1) if spec["legal"] else None
    net.prepare_inference(dev)
    cache = net.initial_st_cache(B)
    t = torch.tensor(3, device=dev)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16)
    with torch.no_grad(), autocast:
        GameNetwork.step(net, obs, legal, st_cache=cache, step_idx=t, act_steps=2)  # lazy init
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            out = GameNetwork.step(net, obs, legal, st_cache=cache, step_idx=t, act_steps=2)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    assert torch.isfinite(out["action_log_prob"]).all()


@needs_cuda
@pytest.mark.parametrize("legal_on", [False, True])
@pytest.mark.parametrize("core,moe", [("recurrent_depth", 4), ("recurrent_depth", 0),
                                      ("transformer", 4)])
def test_graph_replay_matches_eager_cuda(core, moe, legal_on):
    from model.cuda_graph import GraphedStep

    dev = torch.device("cuda")
    net = _net(core=core, moe=moe, device=dev)
    net.action_decoder.greedy = True           # deterministic actions to compare
    B = 2
    g = GraphedStep(net, _obs(B, dev), _legal(B, dev) if legal_on else None, act_steps=3)
    cache = None
    for t in range(5):
        obs = _obs(B, dev, seed=t)
        legal = _legal(B, dev, seed=t) if legal_on else None
        with torch.no_grad():
            e = GameNetwork.step(net, obs, legal, st_cache=cache, step_idx=t, act_steps=3)
        cache = e["st_cache"]
        r = g(obs, t, legal)
        for key in LOGIT_KEYS + ("action_log_prob",):
            torch.testing.assert_close(r[key], e[key], rtol=1e-5, atol=1e-5)
        for key in ACTION_KEYS:
            assert torch.equal(r[key], e[key]), (t, key)
        for cs, ce in zip(g.st_cache, cache):
            torch.testing.assert_close(cs, ce, rtol=1e-5, atol=1e-5)
    # new episode: the state resets to the t=0 state
    g.reset()
    with torch.no_grad():
        e0 = GameNetwork.step(net, _obs(B, dev, seed=9), None if not legal_on
                              else _legal(B, dev, seed=9), act_steps=3)
    r0 = g(_obs(B, dev, seed=9), 0, _legal(B, dev, seed=9) if legal_on else None)
    torch.testing.assert_close(r0["source_logits"], e0["source_logits"], rtol=1e-5, atol=1e-5)


@needs_cuda
def test_graph_replay_matches_eager_bf16_cuda():
    from model.cuda_graph import GraphedStep

    dev = torch.device("cuda")
    net = _net(moe=4, device=dev)
    g = GraphedStep(net, _obs(1, dev), act_steps=2, autocast_dtype=torch.bfloat16)
    cache = None
    for t in range(4):
        obs = _obs(1, dev, seed=t)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            e = GameNetwork.step(net, obs, st_cache=cache, step_idx=t, act_steps=2)
        cache = e["st_cache"]
        r = g(obs, t)
        for key in LOGIT_KEYS:
            torch.testing.assert_close(r[key].float(), e[key].float(), rtol=1e-2, atol=1e-2)


@needs_cuda
def test_graphed_step_rejects_training_mode():
    from model.cuda_graph import GraphedStep

    net = _net(device="cuda").train()
    with pytest.raises(ValueError, match="eval"):
        GraphedStep(net, _obs(1, "cuda"))


def test_graphed_step_requires_cuda_network():
    from model.cuda_graph import GraphedStep

    with pytest.raises(ValueError, match="CUDA"):
        GraphedStep(_net(), _obs(1))



@needs_cuda
def test_graph_replay_samples_like_eager_cuda():
    """Sampling inside the graph (exponential_ race + bernoulli on the
    default CUDA generator): the same seed gives the eager step's actions
    and log-probs, and successive replays draw fresh samples."""
    from model.cuda_graph import GraphedStep

    dev = torch.device("cuda")
    net = _net(moe=4, device=dev)
    assert not net.action_decoder.greedy
    B = 8
    g = GraphedStep(net, _obs(B, dev), _legal(B, dev), act_steps=2)
    cache = None
    for t in range(3):
        obs, legal = _obs(B, dev, seed=t), _legal(B, dev, seed=t)
        torch.manual_seed(1000 + t)
        with torch.no_grad():
            e = GameNetwork.step(net, obs, legal, st_cache=cache, step_idx=t, act_steps=2)
        cache = e["st_cache"]
        torch.manual_seed(1000 + t)
        r = g(obs, t, legal)
        for key in ACTION_KEYS:
            assert torch.equal(r[key], e[key]), (t, key)
        torch.testing.assert_close(r["action_log_prob"], e["action_log_prob"],
                                   rtol=1e-5, atol=1e-5)
    obs, legal = _obs(B, dev, seed=5), _legal(B, dev, seed=5)
    draws = []
    for _ in range(2):
        g.reset()
        r = g(obs, 0, legal)
        draws.append({k: r[k].clone() for k in ACTION_KEYS})
    assert any(not torch.equal(draws[0][k], draws[1][k]) for k in ACTION_KEYS)


@needs_cuda
def test_graph_captures_moe_above_the_auto_dense_batch_cuda():
    """Above DENSE_MAX_TOKEN_EXPERTS the automatic choice is the (uncapturable)
    routed path; GraphedStep forces the dense one, so capture still works."""
    from model.cuda_graph import GraphedStep

    dev = torch.device("cuda")
    net = _net(moe=4, device=dev)
    net.action_decoder.greedy = True
    S = net.obs_encoder.n_tokens + 2
    B = MoEFFN.DENSE_MAX_TOKEN_EXPERTS // (S * 4) + 1
    moe = next(m for m in net.modules() if isinstance(m, MoEFFN))
    with torch.no_grad():
        assert moe.expert_path(B * S, dev) == "sparse"
    g = GraphedStep(net, _obs(B, dev), _legal(B, dev), act_steps=2)
    cache = None
    for t in range(2):
        obs, legal = _obs(B, dev, seed=t), _legal(B, dev, seed=t)
        with torch.no_grad(), MoEFFN.inference_path("dense"):
            e = GameNetwork.step(net, obs, legal, st_cache=cache, step_idx=t, act_steps=2)
        cache = e["st_cache"]
        r = g(obs, t, legal)
        for key in LOGIT_KEYS:
            torch.testing.assert_close(r[key], e[key], rtol=1e-5, atol=1e-5)


@needs_cuda
def test_graph_reset_reads_live_h_init_cuda():
    """Weights loaded in place after capture: replay reads them by address,
    and reset() starts the episode from the NEW h_init."""
    from model.cuda_graph import GraphedStep

    dev = torch.device("cuda")
    net = _net(moe=0, device=dev)
    net.action_decoder.greedy = True
    g = GraphedStep(net, _obs(1, dev), act_steps=1)
    with torch.no_grad():
        for layer in net.st_layers:
            layer.temporal_rnn.h_init.add_(0.5)
    g.reset()
    obs = _obs(1, dev, seed=3)
    r = g(obs, 0)
    with torch.no_grad():
        e = GameNetwork.step(net, obs, act_steps=1)
    torch.testing.assert_close(r["source_logits"], e["source_logits"], rtol=1e-5, atol=1e-5)
    for cs, ce in zip(g.st_cache, e["st_cache"]):
        torch.testing.assert_close(cs, ce, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_graph_captures_on_the_network_device_multi_gpu():
    from model.cuda_graph import GraphedStep

    dev = torch.device("cuda:1")
    net = _net(moe=4, device=dev)
    net.action_decoder.greedy = True
    with torch.cuda.device(0):
        g = GraphedStep(net, _obs(1, dev), act_steps=2)
        obs = _obs(1, dev, seed=1)
        r = g(obs, 0)
        with torch.no_grad():
            e = GameNetwork.step(net, obs, act_steps=2)
    assert r["source_logits"].device == dev
    torch.testing.assert_close(r["source_logits"], e["source_logits"], rtol=1e-5, atol=1e-5)


def test_graphed_step_reset_uses_live_h_init_cpu_logic():
    """reset()'s logic without CUDA: it re-reads h_init, not a capture-time copy."""
    from model.cuda_graph import GraphedStep

    net = _net(moe=0)
    g = GraphedStep.__new__(GraphedStep)
    g.net, g.batch = net, 2
    g.st_cache = net.initial_st_cache(2)
    with torch.no_grad():
        for layer in net.st_layers:
            layer.temporal_rnn.h_init.add_(0.25)
        for c in g.st_cache:
            c.fill_(7.0)
    g.reset()
    for c, layer in zip(g.st_cache, net.st_layers):
        assert torch.equal(c, layer.temporal_rnn.initial_state(c.shape[0]))


def test_graphed_step_input_checks():
    """Mismatched keys / shapes are rejected instead of silently ignored or
    broadcast by copy_ into the static buffers."""
    from model.cuda_graph import GraphedStep

    static = {"source": torch.zeros(2, 1, 5, 5, dtype=torch.bool)}
    GraphedStep._check_inputs("legal", {"source": torch.ones(2, 1, 5, 5, dtype=torch.bool)},
                              static)
    with pytest.raises(ValueError, match="keys"):
        GraphedStep._check_inputs(
            "legal", {"source": torch.ones(2, 1, 5, 5, dtype=torch.bool),
                      "direction": torch.ones(2, 4, 5, 5, dtype=torch.bool)}, static)
    with pytest.raises(ValueError, match="shape"):
        GraphedStep._check_inputs("legal", {"source": torch.ones(1, 1, 5, 5, dtype=torch.bool)},
                                  static)
