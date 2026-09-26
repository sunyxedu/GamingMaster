"""training/rl_joint/gae.py against an independent reference implementation."""

import pytest
import torch

from model.rl_utils import compute_gae
from training.rl_joint import gae


def reference_gae(rewards, values, terminated, truncated, trunc_values, bootstrap, length,
                  gamma, lam):
    """Episode-by-episode GAE in plain Python floats for ONE row.

    Splits the row's first ``length`` steps into episodes, and for each
    computes δ_t = r_t + γ_t·V(s_{t+1}) − V(s_t) with V(s_{t+1}) = 0 after a
    termination, the final-observation value after a truncation, the
    row's bootstrap after its last step, and the next step's value
    otherwise; then A_t = Σ_k (Π_{j<k} γλ)·δ_{t+k} within the episode.
    """
    adv = [0.0] * len(rewards)
    start = 0
    while start < length:
        end = start
        while end < length - 1 and not (terminated[end] or truncated[end]):
            end += 1
        steps = list(range(start, end + 1))
        deltas = []
        for t in steps:
            if terminated[t]:
                nxt = 0.0
            elif truncated[t]:
                nxt = trunc_values[t]
            elif t == length - 1:
                nxt = bootstrap
            else:
                nxt = values[t + 1]
            deltas.append(rewards[t] + gamma[t] * nxt - values[t])
        for i, t in enumerate(steps):
            acc, disc = 0.0, 1.0
            for j in range(i, len(steps)):
                acc += disc * deltas[j]
                disc *= gamma[steps[j]] * lam[steps[j]]
            adv[t] = acc
        start = end + 1
    ret = [adv[t] + values[t] if t < length else 0.0 for t in range(len(rewards))]
    return adv, ret


def _random_rollout(seed, B=5, T=23):
    g = torch.Generator().manual_seed(seed)
    rewards = torch.randn(B, T, generator=g, dtype=torch.float64)
    values = torch.randn(B, T, generator=g, dtype=torch.float64)
    u = torch.rand(B, T, generator=g)
    terminated = u < 0.12
    truncated = (u >= 0.12) & (u < 0.2)
    trunc_values = torch.randn(B, T, generator=g, dtype=torch.float64)
    bootstrap = torch.randn(B, generator=g, dtype=torch.float64)
    lengths = torch.randint(1, T + 1, (B,), generator=g)
    lengths[0] = T
    valid = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
    return rewards, values, terminated, truncated, trunc_values, bootstrap, valid, lengths


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("gamma_kind", ["scalar", "per_row", "per_step"])
def test_matches_reference(seed, gamma_kind):
    r, v, term, trunc, tv, boot, valid, lengths = _random_rollout(seed)
    B, T = r.shape
    g = torch.Generator().manual_seed(100 + seed)
    if gamma_kind == "scalar":
        gamma, lam = 0.97, 0.9
        gm, lm = torch.full((B, T), 0.97), torch.full((B, T), 0.9)
    elif gamma_kind == "per_row":            # a mixed batch: one γ/λ per row (game)
        gamma = torch.rand(B, generator=g, dtype=torch.float64) * 0.2 + 0.8
        lam = torch.rand(B, generator=g, dtype=torch.float64) * 0.3 + 0.7
        gm, lm = gamma.unsqueeze(1).expand(B, T), lam.unsqueeze(1).expand(B, T)
    else:                                    # per-step discount (variable time steps)
        gamma = torch.rand(B, T, generator=g, dtype=torch.float64) * 0.2 + 0.8
        lam = torch.rand(B, T, generator=g, dtype=torch.float64) * 0.3 + 0.7
        gm, lm = gamma, lam
    adv, ret = gae(r, v, term, gamma=gamma, lam=lam, truncated=trunc, trunc_values=tv,
                   bootstrap=boot, valid=valid)
    assert adv.dtype == torch.float64
    for b in range(B):
        ra, rr = reference_gae(r[b].tolist(), v[b].tolist(), term[b].tolist(),
                               trunc[b].tolist(), tv[b].tolist(), float(boot[b]),
                               int(lengths[b]), gm[b].tolist(), lm[b].tolist())
        torch.testing.assert_close(adv[b], torch.tensor(ra, dtype=torch.float64))
        torch.testing.assert_close(ret[b], torch.tensor(rr, dtype=torch.float64))
    assert (adv[~valid] == 0).all() and (ret[~valid] == 0).all()


def test_matches_rl_utils_compute_gae():
    g = torch.Generator().manual_seed(7)
    r, v = torch.randn(40, generator=g), torch.randn(40, generator=g)
    dones = torch.rand(40, generator=g) < 0.15
    dones[-1] = False
    a_ref, ret_ref = compute_gae(r, v, dones, gamma=0.95, lam=0.8, last_value=0.37)
    a, ret = gae(r, v, dones, gamma=0.95, lam=0.8, bootstrap=0.37)
    torch.testing.assert_close(a, a_ref)
    torch.testing.assert_close(ret, ret_ref)


def test_lambda_extremes():
    g = torch.Generator().manual_seed(3)
    r, v = torch.randn(2, 12, generator=g, dtype=torch.float64), torch.randn(
        2, 12, generator=g, dtype=torch.float64)
    term = torch.zeros(2, 12, dtype=torch.bool)
    term[:, -1] = True
    # λ = 0: one-step TD residuals
    a0, _ = gae(r, v, term, gamma=0.9, lam=0.0)
    nxt = torch.cat([v[:, 1:], torch.zeros(2, 1, dtype=torch.float64)], 1)
    torch.testing.assert_close(a0, r + 0.9 * nxt - v)
    # λ = 1: Monte-Carlo returns − V
    _, ret1 = gae(r, v, term, gamma=0.9, lam=1.0)
    mc = torch.zeros_like(r)
    acc = torch.zeros(2, dtype=torch.float64)
    for t in reversed(range(12)):
        acc = r[:, t] + 0.9 * acc
        mc[:, t] = acc
    torch.testing.assert_close(ret1, mc)


def test_truncation_bootstraps_from_final_observation():
    # one row, two episodes: steps 0-2 cut at 2 (V(final obs) = 10), 3-4 terminated
    r = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0]])
    v = torch.zeros(1, 5)
    term = torch.tensor([[False, False, False, False, True]])
    trunc = torch.tensor([[False, False, True, False, False]])
    tv = torch.full((1, 5), float("nan"))
    tv[0, 2] = 10.0
    adv, _ = gae(r, v, term, gamma=0.5, lam=1.0, truncated=trunc, trunc_values=tv)
    # episode 1: G_2 = 1 + 0.5·10 = 6, G_1 = 3, G_0 = 1.5 — no leak from episode 2
    torch.testing.assert_close(adv[0, :3], torch.tensor([1.5, 3.0, 6.0]))
    torch.testing.assert_close(adv[0, 3:], torch.tensor([0.5, 1.0]))


def test_input_validation():
    r = torch.zeros(2, 4)
    term = torch.zeros(2, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="prefix"):
        gae(r, r, term, gamma=0.9, lam=0.9, valid=torch.tensor([[True, False, True, True]] * 2))
    both = term.clone()
    both[0, 1] = True
    with pytest.raises(ValueError, match="both"):
        gae(r, r, both, gamma=0.9, lam=0.9, truncated=both, trunc_values=r)
    with pytest.raises(ValueError, match="trunc_values"):
        gae(r, r, term, gamma=0.9, lam=0.9, truncated=both)
    with pytest.raises(ValueError, match="broadcasts"):
        gae(r, r, term, gamma=torch.ones(3), lam=0.9)
    with pytest.raises(ValueError, match="share one shape"):
        gae(r, torch.zeros(2, 5), term, gamma=0.9, lam=0.9)
