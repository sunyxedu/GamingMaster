"""Torch port of AverageJoe's expander magnet (train/magnet.py).

A hand-designed heuristic action prior used as the KL anchor during
self-play PPO.  It scores every (source, direction) move by what sits at
the destination — expansion onto neutral land, capturable enemy tiles,
capturable cities/generals — and boosts moves out of the player's five
largest stacks.  A softmax over those scores is the "magnet"
distribution.

Why it replaces the human-IL anchor:
  * it points at *behaviour* (expand, convert leads, keep big stacks
    moving) rather than at human strength — no ceiling to unlearn;
  * its weight never decays to zero, so the regulariser cannot silently
    disappear the way the IL anchor did before every phase-6/7 collapse;
  * through KL(magnet ‖ π) it clamps entropy from BOTH sides: a policy
    that collapses (zero mass on magnet-supported moves) or flattens
    (drifts from the shaped prior) both raise the KL.  This is what let
    AverageJoe train without any entropy ceiling/gate machinery.

Faithfulness notes vs. the JAX original:
  * direction order follows OUR env (env/generals_env.py):
    0=DOWN, 1=RIGHT, 2=UP, 3=LEFT;
  * the original used jnp.roll, which wraps destinations around board
    edges; we shift with zero padding instead (edge moves score default);
  * the original's flat (9, H, W) softmax is factorised into the four
    heads of our network: source marginal = logsumexp over directions,
    direction conditional at the teacher-forced source, uniform split
    (the original repeats identical scores for both split channels), and
    a Bernoulli pass whose odds equal the original's pass-channel mass
    over the move-channel mass;
  * army counts are decoded from the single log1p(army)/10 obs channel.
"""

import math
from typing import Dict, Optional

import torch

# Env direction order: (dr, dc) for DOWN, RIGHT, UP, LEFT
_SHIFTS = ((1, 0), (0, 1), (-1, 0), (0, -1))

# Obs grid channel indices (env/generals_env.py get_observation)
_CH_BLANK = 2          # visible neutral blank
_CH_NEUTRAL_CITY = 3
_CH_MY = (5, 6, 7)     # my land / city / general
_CH_ENEMY = (8, 9, 10)  # enemy land / city / general
_CH_ARMY = 11          # log1p(army) / 10, visible tiles only


def _dest(x: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
    """dest[..., r, c] = x[..., r+dr, c+dc]; zero outside the board."""
    out = torch.zeros_like(x)
    H, W = x.shape[-2], x.shape[-1]
    r0o, r1o = max(0, -dr), H - max(0, dr)
    c0o, c1o = max(0, -dc), W - max(0, dc)
    r0i, r1i = max(0, dr), H + min(0, dr)
    c0i, c1i = max(0, dc), W + min(0, dc)
    out[..., r0o:r1o, c0o:c1o] = x[..., r0i:r1i, c0i:c1i]
    return out


@torch.no_grad()
def expander_magnet_heads(
    grid_seq: torch.Tensor,
    source_idx: torch.Tensor,
    dir_mask: Optional[torch.Tensor] = None,
    score_pass: float = 0.2,
    score_default: float = 1.0,
    score_neutral: float = 2.0,
    score_enemy: float = 3.0,
    score_city: float = 5.0,
    score_topk: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """Magnet distributions shaped like a decoder forward (ref_pred).

    Args:
        grid_seq    [B, T, C, H, W] observation grids
        source_idx  [B, T] long — teacher-forced source cells (the
                    direction conditional is evaluated at these, exactly
                    like the frozen-reference forward it replaces)
        dir_mask    [B, T, 4, H, W] bool or None — direction validity;
                    invalid directions at the chosen source get -1e9 so
                    the KL compares masked-to-masked distributions

    Returns:
        dict(source_logits [B,T,N], direction_logits [B,T,4],
             split_logits [B,T,2], pass_logits [B,T])
    """
    B, T, C, H, W = grid_seq.shape
    g = grid_seq.reshape(B * T, C, H, W).float()

    army = torch.expm1(g[:, _CH_ARMY] * 10.0)
    own = g[:, _CH_MY[0]] + g[:, _CH_MY[1]] + g[:, _CH_MY[2]]
    enemy = g[:, _CH_ENEMY[0]] + g[:, _CH_ENEMY[1]] + g[:, _CH_ENEMY[2]]
    own_army = army * own
    enemy_army = army * enemy
    neutral_blank = g[:, _CH_BLANK]
    # Cities AND generals: capturing either is the point of the game.
    city_like = g[:, _CH_NEUTRAL_CITY] + g[:, _CH_ENEMY[1]] + g[:, _CH_ENEMY[2]]

    scores = torch.full((B * T, 4, H, W), score_default,
                        dtype=torch.float32, device=g.device)
    for d, (dr, dc) in enumerate(_SHIFTS):
        d_neutral = _dest(neutral_blank, dr, dc)
        d_enemy = _dest(enemy, dr, dc)
        d_enemy_army = _dest(enemy_army, dr, dc)
        d_city = _dest(city_like, dr, dc)
        d_own = _dest(own, dr, dc)
        d_army = _dest(army, dr, dc)

        s = scores[:, d]
        s = torch.where(d_neutral > 0, torch.full_like(s, score_neutral), s)
        can_capture_enemy = (own_army > d_enemy_army + 1) & (d_enemy > 0)
        s = torch.where(can_capture_enemy,
                        torch.full_like(s, score_enemy), s)
        can_capture_city = (d_city > 0) & (d_own == 0) & (own_army > d_army + 1)
        s = torch.where(can_capture_city,
                        torch.full_like(s, score_city), s)
        scores[:, d] = s

    # Boost every direction out of the five largest own stacks.
    flat_own = own_army.reshape(B * T, -1)
    k = min(5, flat_own.shape[1])
    top5 = flat_own.topk(k, dim=1).values[:, -1]
    boost = ((own_army >= top5.view(-1, 1, 1)) & (own_army > 0)).float()
    scores = scores + boost.unsqueeze(1) * score_topk

    # --- factorised heads ---
    source_logits = torch.logsumexp(scores, dim=1).reshape(B, T, H * W)

    idx = source_idx.reshape(-1).long().clamp(0, H * W - 1)
    sy, sx = idx // W, idx % W
    rows = torch.arange(B * T, device=g.device)
    direction_logits = scores[rows, :, sy, sx].reshape(B, T, 4)
    if dir_mask is not None:
        dm = dir_mask.reshape(B, T, 4, H * W).gather(
            -1, source_idx.reshape(B, T, 1, 1).long()
                .expand(B, T, 4, 1)).squeeze(-1).bool()
        direction_logits = direction_logits.masked_fill(~dm, -1e9)

    split_logits = torch.zeros(B, T, 2, dtype=torch.float32, device=g.device)

    # Pass odds from the original flat softmax: every cell contributes a
    # pass entry (mass H*W*e^{score_pass}) against the doubled (two split
    # channels) move mass.
    move_mass = (math.log(2.0)
                 + torch.logsumexp(scores.reshape(B * T, -1), dim=1))
    pass_mass = math.log(float(H * W)) + score_pass
    pass_logits = (pass_mass - move_mass).reshape(B, T)

    return {
        "source_logits": source_logits,
        "direction_logits": direction_logits,
        "split_logits": split_logits,
        "pass_logits": pass_logits,
    }
