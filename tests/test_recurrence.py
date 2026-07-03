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


def main():
    tests = [
        test_scan_matches_naive_autograd,
        test_module_step_equals_forward,
        test_long_range_gradient_survives,
        test_state_is_fixed_size,
        test_retention_bias_multi_timescale,
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
