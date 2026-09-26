"""The rl_joint pieces compose into one mixed-game PPO iteration on a real
MultiGameNetwork (othello + gomoku sharing one MoE core) through the real
decoder contract (``log_prob_entropy``): interleaved two-seat rows through
``seat_gae`` (othello trains only its learner seat), exact replay
(|log π/π_old| = 0 on the first minibatch), a stale behaviour policy caught
by the |log π_old/μ| canary, PopArt-consistent values, the full-head KL
anchor, finite gradients into the shared core from both games, and
controller / scheduler updates from the iteration's measurements."""

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
    head_terms_kl,
    seat_gae,
    top_q_mask,
    value_loss,
)

B, T, DEPTH = 3, 6, 2
GAMMA = {"othello": 1.0, "gomoku": 0.99}
LEARNER_SEATS = {"othello": (0,), "gomoku": (0, 1)}


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
    # one interleaved sequence per game, both movers alternating; each seat
    # receives its own outcome after the last ply
    seat = (torch.arange(T) % 2).expand(B, T)
    rewards = torch.zeros(B, T, 2)
    term = torch.zeros(B, T, dtype=torch.bool)
    for b, n in enumerate(lengths.tolist()):
        win = 1.0 if b % 2 else -1.0
        rewards[b, n - 1] = torch.tensor([win, -win])
        term[b, n - 1] = True
    return dict(obs=obs, legal={"place": legal}, actions=actions, valid=valid, lp_mu=lp_mu,
                head_out=pred["value"], rewards=rewards, term=term, seat=seat)


def _head_terms(pred, ro):
    """il_head_terms-style dicts on the rollout's actions: the placement
    categorical (legality-masked) and, where the decoder has one, the pass
    Bernoulli."""
    w = ro["valid"].double()
    place_w = (ro["valid"] & ~ro["actions"]["passed"]).double()
    terms = {"place": {"logits": pred["place_logits"], "weight": place_w,
                       "norm": place_w.sum().clamp(min=1), "mask": ro["legal"]["place"]}}
    if "pass_logits" in pred:
        terms["pass"] = {"logits": pred["pass_logits"], "kind": "bernoulli", "weight": w,
                         "norm": w.sum().clamp(min=1)}
    return terms


def _forward(net, ro):
    pred = net(ro["obs"], legal=ro["legal"], act_steps=DEPTH, frame_mask=ro["valid"])
    lp, ent, _ = net.action_decoder.log_prob_entropy(pred, ro["actions"], ro["legal"])
    return pred, lp, ent


def _per_seat_reference(r, V, gamma, lam):
    """Each seat's frames as their own chain (the outcome on its last frame)."""
    adv = torch.zeros_like(V)
    for b in range(B):
        n = int(r["valid"][b].sum())
        for s in (0, 1):
            idx = [t for t in range(n) if int(r["seat"][b, t]) == s]
            acc, nxt_v = 0.0, 0.0
            for i in reversed(range(len(idx))):
                t = idx[i]
                rew = float(r["rewards"][b, n - 1, s]) if i == len(idx) - 1 else 0.0
                delta = rew + gamma * nxt_v - float(V[b, t])
                acc = delta + gamma * lam * acc
                adv[b, t] = acc
                nxt_v = float(V[b, t])
    return adv


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
        r["adv"], r["ret"] = seat_gae(r["rewards"], V, r["seat"], r["term"], gamma=GAMMA[g],
                                      lam=0.95, valid=r["valid"])
        torch.testing.assert_close(r["adv"], _per_seat_reference(r, V, GAMMA[g], 0.95))
        # the loss trains the game's learner seats only
        learner = torch.zeros_like(r["valid"])
        for s in LEARNER_SEATS[g]:
            learner |= r["seat"] == s
        r["mask"] = r["valid"] & learner
    popart.update({g: (r["ret"], r["mask"]) for g, r in ro.items()})
    advnorm.update({g: (r["adv"], r["mask"]) for g, r in ro.items()})
    for g, r in ro.items():
        r["adv_n"] = advnorm.normalize(g, r["adv"], r["mask"])
        r["keep"] = top_q_mask(r["adv_n"], r["mask"], q=0.5, min_count=2)
        with torch.no_grad():
            pred, r["lp_old"], _ = _forward(mg[g], r)       # π_old: same weights, before epoch 1
            # PopArt preserved the critic's outputs in reward units (the
            # rollout's head outputs were in reward units: identity stats)
            torch.testing.assert_close(popart.denormalize(g, pred["value"])[r["valid"]],
                                       r["head_out"][r["valid"]], rtol=1e-5, atol=1e-5)
            assert popart.std(g) != 1.0
            ref_pred, r["lp_ref"], _ = _forward(ref[g], r)
            r["ref_terms"] = _head_terms(ref_pred, r)

    opt = torch.optim.Adam([p for p in mg.parameters() if p.requires_grad], lr=1e-3)
    core = [p for n, p in mg["othello"].named_parameters() if is_core_key(n) and p.requires_grad]
    canaries = []
    for epoch in range(2):
        pol, val, ent, kl = {}, {}, {}, {}
        for g, r in ro.items():
            pred, lp, H = _forward(mg[g], r)
            m = r["mask"]
            pol[g] = decoupled_ppo_loss(lp, r["lp_old"], r["lp_mu"], r["adv_n"], m,
                                        policy_mask=r["keep"], reduction="sum")
            val[g] = value_loss(pred["value"], popart.normalize(g, r["ret"]), m,
                                reduction="sum")
            ent[g] = entropy_loss(H, m, coef=entc[g], reduction="sum")
            kl[g] = head_terms_kl(_head_terms(pred, r), r["ref_terms"], beta=klb[g],
                                  reduction="sum")
        counts = {g: pol[g]["count"] for g in games}   # single process: global = local
        total = (combine_game_losses(pol, counts=counts)
                 + 0.5 * combine_game_losses(val, counts=counts)
                 + combine_game_losses(ent, counts=counts) + combine_game_losses(kl))
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
        assert float(kl[g]["kl"]) > 0.0       # ... away from the frozen reference

    # ---- between iterations ---------------------------------------------
    entc.update({g: float(ent[g]["entropy"]) for g in games})
    klb.update({g: float(kl[g]["kl"]) for g in games})
    sched.record({g: int(ro[g]["mask"].sum()) for g in games})
    q = sched.quotas(100)
    assert sum(q.values()) == 100 and all(v >= 20 for v in q.values())
