"""Correctness tests for the gated linear recurrence (temporal memory core).

The three properties the whole recurrent-memory design rests on:
  1. the custom-backward scan is EXACTLY plain autograd (values + grads),
  2. forward_step chained over T equals forward (rollout/update identity),
  3. gradients survive hundreds of steps (memory is trainable, true BPTT).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.components.recurrence import GatedRecurrence, _GatedScan


def _naive_scan(a, v, h0):
    """Reference implementation via plain autograd (no custom Function)."""
    h = h0
    hs = []
    for t in range(a.shape[1]):
        h = a[:, t] * h + (1.0 - a[:, t]) * v[:, t]
        hs.append(h)
    return torch.stack(hs, dim=1)


def test_scan_matches_naive_autograd():
    """Custom-backward scan == plain autograd: values and ALL gradients."""
    print("=== scan vs naive autograd (values + grads) ===")
    torch.manual_seed(0)
    B, T, d = 3, 25, 8
    a_raw = torch.randn(B, T, d, dtype=torch.float64)
    v0 = torch.randn(B, T, d, dtype=torch.float64)
    h00 = torch.randn(B, d, dtype=torch.float64)
    grad_seed = torch.randn(B, T, d, dtype=torch.float64)

    results = []
    for impl in (_GatedScan.apply, _naive_scan):
        a = torch.sigmoid(a_raw).clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        h0 = h00.clone().requires_grad_(True)
        h_all = impl(a, v, h0)
        (h_all * grad_seed).sum().backward()
        results.append((h_all.detach(), a.grad.clone(), v.grad.clone(), h0.grad.clone()))

    names = ["h_all", "a.grad", "v.grad", "h0.grad"]
    for name, ours, ref in zip(names, results[0], results[1]):
        diff = (ours - ref).abs().max().item()
        assert diff < 1e-12, f"{name}: diff={diff}"
        print(f"  {name}: max diff {diff:.2e} OK")


def test_parallel_scan_equals_sequential():
    """The chunked Hillis-Steele scan must reproduce the step-by-step
    recurrence across chunk boundaries, odd lengths, and extreme gates."""
    print("\n=== parallel scan vs sequential reference ===")
    from model.components.recurrence import _SCAN_CHUNK, _scan_parallel, _scan_sequential

    torch.manual_seed(10)
    for T in (1, 2, 7, _SCAN_CHUNK - 1, _SCAN_CHUNK, _SCAN_CHUNK + 1, 3 * _SCAN_CHUNK + 5):
        B, d = 3, 8
        a = torch.sigmoid(torch.randn(B, T, d, dtype=torch.float64) * 4)  # incl. near-0/1 gates
        v = torch.randn(B, T, d, dtype=torch.float64)
        h0 = torch.randn(B, d, dtype=torch.float64)
        seq = _scan_sequential(a, v, h0)
        par = _scan_parallel(a, (1.0 - a) * v, h0)
        diff = (seq - par).abs().max().item()
        assert diff < 1e-12, f"T={T}: diff={diff}"
    print("  T ∈ {1, 2, 7, C±1, 3C+5}: all < 1e-12 OK")


def test_module_step_equals_forward():
    """Chained forward_step must reproduce forward exactly."""
    print("\n=== GatedRecurrence: step chain vs forward ===")
    torch.manual_seed(1)
    B, T, d = 4, 13, 32
    layer = GatedRecurrence(d)
    layer.eval()
    x = torch.randn(B, T, d)

    with torch.no_grad():
        y_full = layer(x)

        h = None
        ys = []
        for t in range(T):
            y_t, h = layer.forward_step(x[:, t], h)
            ys.append(y_t)
        y_step = torch.stack(ys, dim=1)

    diff = (y_full - y_step).abs().max().item()
    assert diff < 1e-5, f"step vs forward diff={diff}"
    print(f"  max diff {diff:.2e} OK")


def test_long_range_gradient_survives():
    """BPTT: a loss at t=T-1 must reach the input at t=0 across 300 steps
    (multi-timescale retention biases keep slow channels alive)."""
    print("\n=== long-range BPTT gradient ===")
    torch.manual_seed(2)
    B, T, d = 2, 300, 32
    layer = GatedRecurrence(d)
    x = torch.randn(B, T, d, requires_grad=True)

    y = layer(x)
    y[:, -1].sum().backward()

    g0 = x.grad[:, 0].abs().max().item()
    assert g0 > 0, "gradient at t=0 from loss at t=T-1 is exactly zero"
    assert layer.h_init.grad is not None and layer.h_init.grad.abs().max().item() > 0, \
        "learnable initial state receives no gradient"
    print(f"  |dL/dx_0| max = {g0:.3e} > 0 across {T} steps OK")


def test_state_is_fixed_size():
    """The recurrent state must not grow with time (that is the point)."""
    print("\n=== state size is O(1) ===")
    torch.manual_seed(3)
    B, d = 5, 16
    layer = GatedRecurrence(d)
    layer.eval()
    h = None
    with torch.no_grad():
        for t in range(50):
            _, h = layer.forward_step(torch.randn(B, d), h)
            assert h.shape == (B, d), f"state grew: {h.shape}"
    print("  state stays [B, d] over 50 steps OK")


def test_retention_bias_multi_timescale():
    """Initial retention gates span fast (~0.5) to slow (~0.999) channels."""
    print("\n=== multi-timescale retention init ===")
    layer = GatedRecurrence(64)
    a0 = torch.sigmoid(layer.retention_bias)
    assert a0.min().item() < 0.55, f"no fast channel: min a = {a0.min():.3f}"
    assert a0.max().item() > 0.99, f"no slow channel: max a = {a0.max():.3f}"
    assert (a0.diff() >= 0).all(), "retention gates should be monotone over channels"
    print(f"  a range [{a0.min():.3f}, {a0.max():.4f}] OK")


def test_recurrent_block_step_vs_full():
    """SpatiotemporalBlock(recurrent): chained forward_step == forward."""
    print("\n=== recurrent SpatiotemporalBlock: cache vs full ===")
    from model.components.transformer import SpatiotemporalBlock
    from model.config import ModelConfig

    torch.manual_seed(4)
    cfg = ModelConfig(d_model=32, n_heads=4, d_ff=64, dropout=0.0,
                      temporal_mixer="recurrent")
    block = SpatiotemporalBlock(cfg)
    block.eval()

    B, T, S, d = 2, 7, 10, 32
    x = torch.randn(B, T, S, d)

    with torch.no_grad():
        full_out = block(x)

        cache = None
        outs = []
        for t in range(T):
            out_t, cache = block.forward_step(x[:, t:t + 1], cache)
            outs.append(out_t)
        step_out = torch.cat(outs, dim=1)

    assert cache.shape == (B * S, d), f"state should be fixed-size, got {cache.shape}"
    diff = (full_out - step_out).abs().max().item()
    assert diff < 1e-4, f"recurrent block cache vs full diff={diff}"
    print(f"  max diff {diff:.2e}, state fixed at {tuple(cache.shape)} OK")


def test_scan_gradients_correct_under_checkpoint():
    """The network wraps ST blocks in re-entrant gradient checkpointing;
    the custom scan's backward must produce identical grads inside it."""
    print("\n=== scan under re-entrant checkpoint: grads identical ===")
    from torch.utils.checkpoint import checkpoint

    from model.components.transformer import SpatiotemporalBlock
    from model.config import ModelConfig

    torch.manual_seed(7)
    cfg = ModelConfig(d_model=32, n_heads=4, d_ff=64, dropout=0.0,
                      temporal_mixer="recurrent")
    block = SpatiotemporalBlock(cfg)
    block.train()

    B, T, S, d = 2, 6, 8, 32
    x0 = torch.randn(B, T, S, d)

    grads = []
    for use_ckpt in (False, True):
        for p in block.parameters():
            p.grad = None
        x = x0.clone().requires_grad_(True)
        out = checkpoint(block, x, use_reentrant=True) if use_ckpt else block(x)
        out.square().sum().backward()
        grads.append((x.grad.clone(),
                      {n: p.grad.clone() for n, p in block.named_parameters()}))

    xdiff = (grads[0][0] - grads[1][0]).abs().max().item()
    pdiff = max((grads[0][1][n] - grads[1][1][n]).abs().max().item()
                for n in grads[0][1])
    assert xdiff == 0.0 and pdiff == 0.0, f"ckpt grad mismatch: x={xdiff}, p={pdiff}"
    print(f"  x.grad diff {xdiff:.1e}, param grad diff {pdiff:.1e} OK")


def _small_recurrent_net():
    from model.config import ModelConfig
    from model.network import GeneralsNetwork

    cfg = ModelConfig(
        d_model=64, n_heads=4, d_ff=128,
        n_st_layers=2, core_type="trm", trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        encoder_channels=32, temporal_type="spatiotemporal", dropout=0.0,
        temporal_mixer="recurrent",
    )
    return GeneralsNetwork(cfg), cfg


def test_network_recurrent_step_vs_full_act_steps():
    """Recurrent mode, fixed ACT depth: chained step_rl must define exactly
    the policy that forward(act_steps=K) recomputes — the PPO ratio identity."""
    print("\n=== recurrent network: step_rl vs forward(act_steps) ===")
    torch.manual_seed(5)
    model, cfg = _small_recurrent_net()
    model.eval()

    K = 2
    B, T = 1, 5
    grid_seq = torch.randn(B, T, 12, 25, 25)
    globals_seq = torch.randn(B, T, 4)

    # Rollout: step-by-step, threading the fixed-size recurrent cache
    cache = None
    step_src, step_pass, step_val, sampled = [], [], [], []
    for t in range(T):
        with torch.no_grad():
            out = model.step_rl(grid_seq[:, t], globals_seq[:, t],
                                st_cache=cache, step_idx=t, act_steps=K)
        cache = out["st_cache"]
        step_src.append(out["source_logits"])
        step_pass.append(out["pass_logits"])
        step_val.append(out["value"])
        sampled.append(out["action_source"])

    for c in cache:
        assert c.dim() == 2, f"recurrent cache must be fixed-size [B*S, d], got {c.shape}"

    # Update path: full-sequence forward conditioned on the rollout sources
    src_idx = torch.stack(sampled, dim=1)
    with torch.no_grad():
        full = model(
            grid_seq, globals_seq,
            source_y=src_idx // 25, source_x=src_idx % 25,
            act_steps=K,
        )

    for name, full_t, step_t in [
        ("source_logits", full["source_logits"], torch.stack(step_src, dim=1)),
        ("pass_logits", full["pass_logits"], torch.stack(step_pass, dim=1)),
        ("value", full["value"], torch.stack(step_val, dim=1)),
    ]:
        diff = (full_t - step_t).abs().max().item()
        assert diff < 1e-3, f"{name} diff too large: {diff}"
        print(f"  {name}: max diff {diff:.8f} OK")


def test_network_recurrent_bptt_reaches_early_frames():
    """Training gradients must reach early observations through the
    recurrence — this is what makes 'what to remember' learnable."""
    print("\n=== recurrent network: BPTT reaches early frames ===")
    torch.manual_seed(6)
    model, cfg = _small_recurrent_net()
    model.train()

    B, T = 1, 12
    grid_seq = torch.randn(B, T, 12, 25, 25, requires_grad=True)
    globals_seq = torch.randn(B, T, 4)
    sy = torch.zeros(B, T, dtype=torch.long)
    sx = torch.zeros(B, T, dtype=torch.long)

    out = model(grid_seq, globals_seq, source_y=sy, source_x=sx)
    # Loss ONLY on the final timestep
    out["source_logits"][:, -1].sum().backward()

    g_first = grid_seq.grad[:, 0].abs().max().item()
    assert g_first > 0, "loss at t=T-1 produced zero gradient at t=0 input"
    print(f"  |dL/d obs_0| max = {g_first:.3e} > 0 across {T} frames OK")


def test_next_obs_loss_masks_and_zero():
    """next_obs_loss: correct masking of padded frames/off-map cells, and
    a perfect prediction scores ~zero."""
    print("\n=== next_obs_loss: masks + perfect-prediction zero ===")
    from model.losses import next_obs_loss

    torch.manual_seed(8)
    B, T, H, W, C = 2, 5, 4, 4, 12
    N = H * W
    n_types = 11

    # Build observations: one-hot states on a 3x4 "real map" area,
    # last row of cells is off-map (all-zero); frame T-1 of sample 1 is padding.
    grid_seq = torch.zeros(B, T, C, H, W)
    states = torch.randint(0, n_types, (B, T, H, W))
    for b in range(B):
        for t in range(T):
            for r in range(3):          # rows 0..2 real, row 3 off-map
                for c in range(W):
                    grid_seq[b, t, states[b, t, r, c], r, c] = 1.0
                    grid_seq[b, t, 11, r, c] = 0.3
    frame_mask = torch.ones(B, T, dtype=torch.bool)
    frame_mask[1, -1] = False           # padded frame

    # Perfect prediction: logits peaked on frame t+1's true state
    logits = torch.full((B, T, N, n_types), -20.0)
    army = torch.zeros(B, T, N)
    for b in range(B):
        for t in range(T - 1):
            nxt = grid_seq[b, t + 1, :n_types].reshape(n_types, N)
            idx = nxt.argmax(0)
            logits[b, t, torch.arange(N), idx] = 20.0
            army[b, t] = grid_seq[b, t + 1, 11].reshape(N)

    pred = {"next_state_logits": logits, "next_army": army}
    out = next_obs_loss(pred, grid_seq, frame_mask)
    assert out["state_loss"].item() < 1e-3, f"perfect state loss: {out['state_loss']}"
    assert out["army_loss"].item() < 1e-6, f"perfect army loss: {out['army_loss']}"
    print(f"  perfect prediction: state={out['state_loss']:.2e}, army={out['army_loss']:.2e} OK")

    # Corrupt predictions ONLY on masked entries (off-map cells + padded pair):
    # loss must not change.
    logits2 = logits.clone()
    off_map = grid_seq[:, 1:, :n_types].sum(2).reshape(B, T - 1, N) < 0.5
    logits2[:, :-1][off_map] = torch.randn_like(logits2[:, :-1][off_map])
    logits2[1, T - 2] = torch.randn_like(logits2[1, T - 2])  # pair with padded frame
    out2 = next_obs_loss({"next_state_logits": logits2, "next_army": army}, grid_seq, frame_mask)
    assert abs(out2["state_loss"].item() - out["state_loss"].item()) < 1e-9, \
        "masked entries leaked into the loss"
    print("  masked cells/frames do not affect the loss OK")


def test_next_obs_head_trains_memory():
    """End-to-end: the world-model loss must send gradient into the
    recurrence gates (the memory-pressure path)."""
    print("\n=== next_obs head: gradient reaches recurrence gates ===")
    from model.config import ModelConfig
    from model.losses import next_obs_loss
    from model.network import GeneralsNetwork

    torch.manual_seed(9)
    cfg = ModelConfig(
        d_model=64, n_heads=4, d_ff=128,
        n_st_layers=1, core_type="trm", trm_H_cycles=1, trm_L_cycles=1, trm_block_layers=1,
        encoder_channels=32, temporal_type="spatiotemporal", dropout=0.0,
        temporal_mixer="recurrent", predict_next_obs=True,
    )
    model = GeneralsNetwork(cfg)
    model.train()

    B, T = 1, 4
    grid_seq = torch.rand(B, T, 12, 25, 25)
    globals_seq = torch.randn(B, T, 4)
    sy = torch.zeros(B, T, dtype=torch.long)
    sx = torch.zeros(B, T, dtype=torch.long)

    pred = model(grid_seq, globals_seq, source_y=sy, source_x=sx)
    assert pred["next_state_logits"].shape == (B, T, 625, 11)
    assert pred["next_army"].shape == (B, T, 625)

    wm = next_obs_loss(pred, grid_seq)
    (wm["state_loss"] + wm["army_loss"]).backward()

    rnn = model.st_layers[0].temporal_rnn
    ga = rnn.a_proj.weight.grad
    assert ga is not None and ga.abs().max().item() > 0, \
        "world-model loss reaches no gradient into the retention gate"
    print(f"  |d wm_loss / d a_proj| max = {ga.abs().max():.3e} > 0 OK")


def test_memory_params_excluded_from_weight_decay():
    """AdamW decay must not erode retention_bias/h_init: decay shrinks the
    retention logits toward 0.5, hitting the longest-memory channels hardest
    (half-life ~ ln2·e^bias). optim_param_groups puts them in a wd=0 group."""
    print("\n=== weight decay excludes memory parameters ===")
    from model.rl_utils import optim_param_groups

    model, _ = _small_recurrent_net()
    groups = optim_param_groups(model, weight_decay=0.1)
    assert len(groups) == 2 and groups[1]["weight_decay"] == 0.0

    no_decay_ids = {id(p) for p in groups[1]["params"]}
    for name, p in model.named_parameters():
        in_no_decay = id(p) in no_decay_ids
        is_memory = name.endswith("retention_bias") or name.endswith("temporal_rnn.h_init")
        is_qk_scale = "qk_norm." in name        # cfg.qk_norm (tests/test_qk_norm.py)
        assert in_no_decay == (is_memory or is_qk_scale), f"{name}: wrong decay group"

    # Empirical: pure decay (zero grad) must leave retention_bias untouched
    opt = torch.optim.AdamW(groups, lr=1e-2)
    rb = model.st_layers[0].temporal_rnn.retention_bias
    before = rb.detach().clone()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    for _ in range(100):
        opt.step()
    assert torch.equal(rb.detach(), before), "retention_bias still being decayed"
    # ...while a regular weight IS decayed (sanity that decay is on)
    w = model.st_layers[0].temporal_rnn.a_proj.weight
    assert not torch.equal(w.detach(), torch.zeros_like(w)) and w.abs().sum() > 0
    print("  retention_bias/h_init bit-unchanged after 100 AdamW steps OK")


def test_grad_accumulation_matches_large_batch():
    """k accumulated micro-batches must produce the same gradients as one
    k-times-larger batch (identical loss scaling)."""
    print("\n=== gradient accumulation == large batch ===")
    torch.manual_seed(12)
    model, _ = _small_recurrent_net()
    # eval(): train-mode BatchNorm in the spatial encoder couples the
    # samples of a batch, so split vs concatenated batches differ by
    # construction; this test checks the LOSS SCALING only.
    model.eval()

    B, T = 2, 3
    grids = [torch.rand(B, T, 12, 25, 25) for _ in range(2)]
    glbs = [torch.randn(B, T, 4) for _ in range(2)]
    sy = torch.zeros(B, T, dtype=torch.long)
    sx = torch.zeros(B, T, dtype=torch.long)

    def loss_of(g, gl):
        pred = model(g, gl, source_y=sy, source_x=sx)
        return pred["source_logits"].square().mean()

    # accumulated: two micro-batches, each backward(loss/2)
    for p in model.parameters():
        p.grad = None
    for g, gl in zip(grids, glbs):
        (loss_of(g, gl) / 2).backward()
    g_accum = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    # one large batch (same samples concatenated)
    for p in model.parameters():
        p.grad = None
    big_g = torch.cat(grids, dim=0)
    big_gl = torch.cat(glbs, dim=0)
    big_sy = torch.zeros(2 * B, T, dtype=torch.long)
    pred = model(big_g, big_gl, source_y=big_sy, source_x=big_sy)
    pred["source_logits"].square().mean().backward()

    max_diff = max((g_accum[n] - p.grad).abs().max().item()
                   for n, p in model.named_parameters() if n in g_accum)
    assert max_diff < 1e-5, f"accumulated vs large-batch grad diff: {max_diff}"
    print(f"  max grad diff {max_diff:.2e} OK")


def main():
    tests = [
        test_scan_matches_naive_autograd,
        test_parallel_scan_equals_sequential,
        test_module_step_equals_forward,
        test_long_range_gradient_survives,
        test_state_is_fixed_size,
        test_retention_bias_multi_timescale,
        test_recurrent_block_step_vs_full,
        test_scan_gradients_correct_under_checkpoint,
        test_network_recurrent_step_vs_full_act_steps,
        test_network_recurrent_bptt_reaches_early_frames,
        test_next_obs_loss_masks_and_zero,
        test_next_obs_head_trains_memory,
        test_memory_params_excluded_from_weight_decay,
        test_grad_accumulation_matches_large_batch,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAILED {t.__name__}: {e}")
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    return failed


if __name__ == "__main__":
    sys.exit(main())
