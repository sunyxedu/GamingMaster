"""The rl_joint pieces compose into one mixed-game PPO iteration on a real
MultiGameNetwork (othello + gomoku sharing one MoE core) through the real
decoder contract (``log_prob_entropy``): exact replay (|log π/π_old| = 0 on
the first minibatch), a stale behaviour policy caught by the |log π_old/μ|
canary, PopArt-consistent values, finite gradients into the shared core
from both games, and controller / scheduler updates from the iteration's
measurements."""

import copy

import torch

from model.multi_game import build_multi_game, is_core_key
from tests.test_core_moe import CORE
from training.rl_joint import (
    AdaptiveEntropyCoef,
    AdaptiveKLBeta,
    AdvantageNormalizer,
    PopArt,
    TokenQuotaScheduler,
    combine_game_losses,
    decoupled_ppo_loss,
    entropy_loss,
    gae,
    kl_penalty,
    top_q_mask,
    value_loss,
)

B, T, DEPTH = 3, 6, 2
GAMMA = {"othello": 1.0, "gomoku": 0.99}


def _rollout(net, n_points, side, gen):
    obs = {"board_seq": torch.rand(B, T, 3, side, side, generator=gen)}
    legal = torch.rand(B, T, n_points, generator=gen) > 0.3
    legal[..., 0] = True
    lengths = torch.tensor([T, T - 2, 3])
    valid = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
    with torch.no_grad():
        pred = net(obs, legal={"place": legal}, act_steps=DEPTH, frame_mask=valid)
    place = torch.multinomial(legal.float().reshape(-1, n_points), 1,
                              generator=gen).reshape(B, T)
    actions = {"place_idx": place, "passed": torch.zeros(B, T, dtype=torch.bool)}
    with torch.no_grad():
        lp, _, _ = net.action_decoder.log_prob_entropy(pred, actions, {"place": legal})
    # the actors ran slightly stale weights: μ ≠ π_old
    lp_mu = lp + 0.05 * torch.randn(B, T, generator=gen)
    rewards = torch.zeros(B, T)
    term = torch.zeros(B, T, dtype=torch.bool)
    for b, n in enumerate(lengths.tolist()):
        rewards[b, n - 1] = 1.0 if b % 2 else -1.0
        term[b, n - 1] = True
    return dict(obs=obs, legal={"place": legal}, actions=actions, valid=valid, lp_mu=lp_mu,
                head_out=pred["value"], rewards=rewards, term=term)


def _forward(net, ro):
    pred = net(ro["obs"], legal=ro["legal"], act_steps=DEPTH, frame_mask=ro["valid"])
    lp, ent, _ = net.action_decoder.log_prob_entropy(pred, ro["actions"], ro["legal"])
    return pred, lp, ent


def test_one_mixed_game_iteration():
    torch.manual_seed(0)
    mg = build_multi_game(["othello", "gomoku"], CORE, {"gomoku": {"board_h": 9, "board_w": 9}})
    mg.eval()        # rollouts and updates in eval mode: the replay is exact
    games = mg.games
    ref = copy.deepcopy(mg).requires_grad_(False)          # frozen Stage-1 reference
    gen = torch.Generator().manual_seed(1)
    ro = {"othello": _rollout(mg["othello"], 64, 8, gen),
          "gomoku": _rollout(mg["gomoku"], 81, 9, gen)}

    popart = PopArt.from_model(mg, decay=0.9)
    advnorm = AdvantageNormalizer(games, decay=0.9)
    entc = AdaptiveEntropyCoef(games, target={"othello": 2.0, "gomoku": 3.0})
    klb = AdaptiveKLBeta(games, budget=0.01)
    sched = TokenQuotaScheduler({g: 1.0 for g in games}, floor=0.2)

    # ---- targets --------------------------------------------------------
    for g, r in ro.items():
        V = popart.denormalize(g, r["head_out"])
        r["adv"], r["ret"] = gae(r["rewards"], V, r["term"], gamma=GAMMA[g], lam=0.95,
                                 valid=r["valid"])
    popart.update({g: (r["ret"], r["valid"]) for g, r in ro.items()})
    advnorm.update({g: (r["adv"], r["valid"]) for g, r in ro.items()})
    for g, r in ro.items():
        r["adv_n"] = advnorm.normalize(g, r["adv"], r["valid"])
        r["keep"] = top_q_mask(r["adv_n"], r["valid"], q=0.5, per_row=True, min_count=2)
        with torch.no_grad():
            pred, r["lp_old"], _ = _forward(mg[g], r)       # π_old: same weights, before epoch 1
            # PopArt preserved the critic's outputs in reward units (the
            # rollout's head outputs were in reward units: identity stats)
            torch.testing.assert_close(popart.denormalize(g, pred["value"])[r["valid"]],
                                       r["head_out"][r["valid"]], rtol=1e-5, atol=1e-5)
            assert popart.std(g) != 1.0
            _, r["lp_ref"], _ = _forward(ref[g], r)

    opt = torch.optim.Adam([p for p in mg.parameters() if p.requires_grad], lr=1e-3)
    core = [p for n, p in mg["othello"].named_parameters() if is_core_key(n) and p.requires_grad]
    canaries = []
    for epoch in range(2):
        pol, val, ent, kl = {}, {}, {}, {}
        for g, r in ro.items():
            pred, lp, H = _forward(mg[g], r)
            pol[g] = decoupled_ppo_loss(lp, r["lp_old"], r["lp_mu"], r["adv_n"], r["valid"],
                                        policy_mask=r["keep"], reduction="sum")
            val[g] = value_loss(pred["value"], popart.normalize(g, r["ret"]), r["valid"],
                                reduction="sum")
            ent[g] = entropy_loss(H, r["valid"], coef=entc[g], reduction="sum")
            kl[g] = kl_penalty(lp, r["lp_ref"], r["valid"], beta=klb[g], reduction="sum")
        total = (combine_game_losses(pol) + 0.5 * combine_game_losses(val)
                 + combine_game_losses(ent) + combine_game_losses(kl))
        canaries.append({g: (float(pol[g]["abs_log_ratio"]), float(pol[g]["abs_log_old_mu"]))
                         for g in games})
        opt.zero_grad()
        total.backward()
        assert torch.isfinite(total)
        grads = [p.grad for p in mg.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(gr).all() for gr in grads)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in core)
        for g in games:
            assert all(p.grad is not None for p in mg[g].action_decoder.parameters()
                       if p.requires_grad)
        opt.step()

    for g in games:
        first, second = canaries[0][g], canaries[1][g]
        assert first[0] == 0.0                # exact replay on the first minibatch
        assert first[1] > 0.01                # the stale actors are visible
        assert second[0] > 0.0                # the policy moved after one step

    # ---- between iterations ---------------------------------------------
    entc.update({g: float(ent[g]["entropy"]) for g in games})
    klb.update({g: float(kl[g]["kl_k3"]) for g in games})
    sched.record({g: int(ro[g]["valid"].sum()) for g in games})
    q = sched.quotas(100)
    assert sum(q.values()) == 100 and all(v >= 20 for v in q.values())
