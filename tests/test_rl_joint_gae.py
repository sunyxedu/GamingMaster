"""training/rl_joint/gae.py against an independent reference implementation."""

import pytest
import torch

from model.rl_utils import compute_gae
from training.rl_joint import gae, seat_gae


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
    # a per-row trunc_values would silently be read as tv[t] by every row
    with pytest.raises(ValueError, match="trunc_values shape"):
        gae(r, r, term, gamma=0.9, lam=0.9, truncated=both & ~term, trunc_values=torch.ones(2))


# ======================================================================
# interleaved multi-seat streams
# ======================================================================

def reference_seat_gae(rewards, values, seat, term, trunc, tv, boot, length, gamma, lam, S):
    """Split ONE interleaved row into per-seat decision chains in plain
    Python and run :func:`reference_gae` on each.  Per-seat rewards
    ``rewards[t][s]`` are summed from a seat's decision up to (excluding) its
    next decision or through its episode's last ply; a seat's chain ends
    (terminated / truncated with the per-seat final value) at its last
    decision of an episode; after the row's last ply it bootstraps."""
    # episode id of every ply and each episode's end kind
    ep, ends, e = [], [], 0
    for t in range(length):
        ep.append(e)
        if term[t] or trunc[t]:
            ends.append((t, "term" if term[t] else "trunc"))
            e += 1
    adv = [0.0] * len(values)
    for s in range(S):
        idx = [t for t in range(length) if seat[t] == s]
        if not idx:
            continue
        r_s, v_s, te_s, tr_s, tv_s, g_s, l_s = [], [], [], [], [], [], []
        for i, t in enumerate(idx):
            nxt = idx[i + 1] if i + 1 < len(idx) else None
            same_ep_next = nxt is not None and ep[nxt] == ep[t]
            ep_end = ends[ep[t]] if ep[t] < len(ends) else None
            stop = nxt if same_ep_next else (ep_end[0] + 1 if ep_end else length)
            r_s.append(sum(rewards[u][s] for u in range(t, stop)))
            v_s.append(values[t])
            last_of_ep = not same_ep_next
            te_s.append(last_of_ep and ep_end is not None and ep_end[1] == "term")
            tr_s.append(last_of_ep and ep_end is not None and ep_end[1] == "trunc")
            tv_s.append(tv[ep_end[0]][s] if tr_s[-1] else 0.0)
            g_s.append(gamma[t])
            l_s.append(lam[t])
        a, _ = reference_gae(r_s, v_s, te_s, tr_s, tv_s, boot[s], len(idx), g_s, l_s)
        for i, t in enumerate(idx):
            adv[t] = a[i]
    return adv


def _random_seat_rollout(seed, B=6, T=31, S=2, turn_based=True):
    g = torch.Generator().manual_seed(seed)
    if turn_based:        # alternating movers, restarting at seat 0 after each episode
        seat = torch.zeros(B, T, dtype=torch.long)
    else:                 # arbitrary acting seat (mahjong-style claims, passes)
        seat = torch.randint(0, S, (B, T), generator=g)
    u = torch.rand(B, T, generator=g)
    term = u < 0.1
    trunc = (u >= 0.1) & (u < 0.15)
    if turn_based:
        for b in range(B):
            k = 0
            for t in range(T):
                seat[b, t] = k % S
                k = 0 if (term[b, t] or trunc[b, t]) else k + 1
    rewards = torch.randn(B, T, S, generator=g, dtype=torch.float64)
    rewards = torch.where(torch.rand(B, T, S, generator=g) < 0.6, 0.0, rewards)
    values = torch.randn(B, T, generator=g, dtype=torch.float64)
    tv = torch.randn(B, T, S, generator=g, dtype=torch.float64)
    boot = torch.randn(B, S, generator=g, dtype=torch.float64)
    lengths = torch.randint(1, T + 1, (B,), generator=g)
    lengths[0] = T
    valid = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
    gamma = torch.rand(B, T, generator=g, dtype=torch.float64) * 0.2 + 0.8
    lam = torch.rand(B, T, generator=g, dtype=torch.float64) * 0.3 + 0.7
    return rewards, values, seat, term, trunc, tv, boot, valid, lengths, gamma, lam


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("S,turn_based", [(2, True), (4, True), (3, False)])
def test_seat_gae_matches_per_seat_reference(seed, S, turn_based):
    r, v, seat, term, trunc, tv, boot, valid, lengths, gm, lm = _random_seat_rollout(
        seed, S=S, turn_based=turn_based)
    adv, ret = seat_gae(r, v, seat, term, gamma=gm, lam=lm, truncated=trunc,
                        trunc_values=tv, bootstrap=boot, valid=valid)
    for b in range(r.shape[0]):
        want = reference_seat_gae(r[b].tolist(), v[b].tolist(), seat[b].tolist(),
                                  term[b].tolist(), trunc[b].tolist(), tv[b].tolist(),
                                  boot[b].tolist(), int(lengths[b]), gm[b].tolist(),
                                  lm[b].tolist(), S)
        torch.testing.assert_close(adv[b], torch.tensor(want, dtype=torch.float64))
    torch.testing.assert_close(ret[valid], (adv + v)[valid])
    assert (adv[~valid] == 0).all() and (ret[~valid] == 0).all()


@pytest.mark.parametrize("seed", range(3))
def test_seat_gae_with_one_seat_is_gae(seed):
    r, v, term, trunc, tv, boot, valid, _ = _random_rollout(seed)
    want = gae(r, v, term, gamma=0.97, lam=0.9, truncated=trunc, trunc_values=tv,
               bootstrap=boot, valid=valid)
    got = seat_gae(r, v, torch.zeros_like(term, dtype=torch.long), term, gamma=0.97, lam=0.9,
                   n_seats=1, truncated=trunc, trunc_values=tv.unsqueeze(-1),
                   bootstrap=boot.unsqueeze(-1), valid=valid)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b)


def _board_game(values, final):
    """One 2-seat alternating board game ending after its last ply."""
    T = len(values)
    seat = torch.tensor([t % 2 for t in range(T)])
    term = torch.zeros(T, dtype=torch.bool)
    term[-1] = True
    rewards = torch.zeros(T, 2)
    rewards[-1] = torch.tensor(final)
    return torch.tensor(values), seat, term, rewards


def test_seat_gae_board_game_outcome_credit():
    """Both seats are credited their own outcome — the acting-seat and the
    per-seat reward forms agree, and match main's per-seat loop recipe
    (per seat: reward 0 except its last frame, done there)."""
    v, seat, term, rewards = _board_game([0.1, -0.2, 0.3, -0.1, 0.5, -0.4, 0.8], [1.0, -1.0])
    adv, ret = seat_gae(rewards, v, seat, term, gamma=1.0, lam=0.95)
    credited = torch.zeros(7)
    credited[-1], credited[-2] = 1.0, -1.0           # each seat's last frame
    adv2, _ = seat_gae(credited, v, seat, term, gamma=1.0, lam=0.95, n_seats=2)
    torch.testing.assert_close(adv, adv2)
    want = torch.zeros(7)
    for s, outcome in ((0, 1.0), (1, -1.0)):
        idx = (seat == s).nonzero().squeeze(1)
        rr = torch.zeros(len(idx))
        rr[-1] = outcome
        dd = torch.zeros(len(idx), dtype=torch.bool)
        dd[-1] = True
        a, _ = compute_gae(rr, v[idx], dd, gamma=1.0, lam=0.95)
        want[idx] = a
    torch.testing.assert_close(adv, want)
    assert (adv[1::2] < 0).all()                      # the losing seat's frames
    # the review's worked example (γ = 1, λ = 0.95); gae on the raw row flips seat 1's signs
    torch.testing.assert_close(adv, torch.tensor([0.832, -0.726, 0.665, -0.87, 0.49,
                                                  -0.6, 0.2]), atol=1e-3, rtol=0)


def test_seat_gae_matches_train_rl_game_when_available():
    """Same numbers as training/train_rl_game.seat_gae (main's per-game
    loop), batched: rows of different lengths, tail-padded."""
    trg = pytest.importorskip("training.train_rl_game")
    if not hasattr(trg, "seat_gae"):
        pytest.skip("training.train_rl_game.seat_gae not on this branch")
    g = torch.Generator().manual_seed(0)
    games = []
    for T, final in ((9, (1.0, -1.0)), (4, (-1.0, 1.0)), (7, (0.0, 0.0))):
        v, seat, term, rewards = _board_game(torch.randn(T, generator=g).tolist(), final)
        games.append((v, seat, term, rewards, final))
    T_max = max(len(x[0]) for x in games)
    B = len(games)
    V, SEAT, TERM = torch.zeros(B, T_max), torch.zeros(B, T_max, dtype=torch.long), \
        torch.zeros(B, T_max, dtype=torch.bool)
    R, VALID = torch.zeros(B, T_max, 2), torch.zeros(B, T_max, dtype=torch.bool)
    for i, (v, seat, term, rewards, _) in enumerate(games):
        n = len(v)
        V[i, :n], SEAT[i, :n], TERM[i, :n], R[i, :n], VALID[i, :n] = v, seat, term, rewards, True
    adv, ret = seat_gae(R, V, SEAT, TERM, gamma=0.97, lam=0.9, valid=VALID)
    for i, (v, seat, _, _, final) in enumerate(games):
        traj = {"place": torch.zeros(len(v)), "seat": seat + 1, "value": v,
                "reward": {1: final[0], 2: final[1]}}
        a, r = trg.seat_gae(traj, 0.97, 0.9)
        torch.testing.assert_close(adv[i, :len(v)], a)
        torch.testing.assert_close(ret[i, :len(v)], r)


def test_seat_gae_multi_episode_row_and_bootstrap():
    """A pooled stream: episode 1 (plies 0-2, seat 0 moves last) terminates,
    episode 2 (plies 3-5) is still running when the row stops — each seat
    bootstraps from its own V, and no reward crosses the boundary."""
    v = torch.zeros(6)
    seat = torch.tensor([0, 1, 0, 0, 1, 0])
    term = torch.tensor([False, False, True, False, False, False])
    rewards = torch.zeros(6, 2)
    rewards[2] = torch.tensor([1.0, -1.0])
    rewards[3, 1] = 5.0                      # seat 1, episode 2, before its decision at 4
    rewards[4, 0] = 2.0                      # seat 0: between its decisions at 3 and 5
    boot = torch.tensor([10.0, 20.0])
    adv, _ = seat_gae(rewards, v, seat, term, gamma=0.5, lam=1.0, bootstrap=boot)
    # episode 1: seat 0 at 0 → 2 → end(+1); seat 1 at 1 → end(−1)
    torch.testing.assert_close(adv[:3], torch.tensor([0.5, -1.0, 1.0]))
    # episode 2: seat 0: 3 → 5 → boot 10 (the +2 received before ply 5);
    # seat 1: its 5.0 came before its first decision (dropped), then → boot 20
    torch.testing.assert_close(adv[3:], torch.tensor([2.0 + 0.5 * 5.0, 10.0, 5.0]))


def test_seat_gae_validation():
    v = torch.zeros(2, 4)
    seat = torch.zeros(2, 4, dtype=torch.long)
    term = torch.zeros(2, 4, dtype=torch.bool)
    r = torch.zeros(2, 4, 2)
    with pytest.raises(ValueError, match=r"seat ids"):
        seat_gae(r, v, seat + 2, term, gamma=0.9, lam=0.9)
    with pytest.raises(ValueError, match="trunc_values shape"):
        t2 = term.clone()
        t2[0, 1] = True
        seat_gae(r, v, seat, term, gamma=0.9, lam=0.9, truncated=t2, trunc_values=v)
    with pytest.raises(ValueError, match="bootstrap shape"):
        seat_gae(r, v, seat, term, gamma=0.9, lam=0.9, bootstrap=torch.zeros(2))
    with pytest.raises(ValueError, match="rewards"):
        seat_gae(torch.zeros(2, 5), v, seat, term, gamma=0.9, lam=0.9)
    with pytest.raises(ValueError, match="n_seats"):
        seat_gae(r, v, seat, term, gamma=0.9, lam=0.9, n_seats=3)
