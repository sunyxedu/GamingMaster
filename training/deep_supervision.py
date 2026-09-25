"""TRM deep supervision, as trained by the reference implementation.

Reference (arXiv 2510.04871 Fig. 3; SamsungSAILMontreal/TinyRecursiveModels
``pretrain.py`` + ``TinyRecursiveReasoningModel_ACTV1.forward``):

    for x, y_true in loader:
        y, z = y_init, z_init
        for step in range(N_sup):                    # N_sup = halt_max_steps
            (y, z), y_hat, q_hat = deep_recursion(x, y, z)   # carry detached
            loss = CE(y_hat, y_true) + 0.5 * BCE(q_hat, y_hat == y_true)
            loss.backward(); opt.step(); opt.zero_grad()      # EMA update
            halt examples with q_hat > 0 (training only; exploration: with
            prob 0.1 an example must run at least U{2..N_sup} steps)

Every supervision step is its own optimizer step; the latents are carried
(detached) between steps and each halted example leaves the loop.  At
evaluation every example runs all N_sup steps.

Here an "example" is one frame (the TRM ponders each frame's tokens), so
halting is per frame.  Stage 1 is recomputed at every supervision step
(the weights changed), exactly like the reference re-embeds the input.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch

#: weight of the q_halt BCE relative to the task loss (reference ACTLossHead)
Q_HALT_WEIGHT = 0.5


def halt_update(
    q_halt: torch.Tensor,
    steps: torch.Tensor,
    active: torch.Tensor,
    n_sup: int,
    exploration_prob: float,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One reference halting decision.

    Args:
        q_halt: [B, T] halt logits of the step just taken.
        steps:  [B, T] supervision steps taken BEFORE this one.
        active: [B, T] frames that took this step.
    Returns:
        (new_steps, still_active)
    """
    with torch.no_grad():
        new_steps = torch.where(active, steps + 1, steps)
        halted = new_steps >= n_sup
        if training and n_sup > 1:
            halted = halted | (q_halt > 0)
            explore = torch.rand_like(q_halt, dtype=torch.float32) < exploration_prob
            min_halt = explore.long() * torch.randint_like(
                new_steps, low=2, high=n_sup + 1)
            halted = halted & (new_steps >= min_halt)
        return new_steps, active & ~halted


def run_deep_supervision(
    n_micro: int,
    frame_masks: List[torch.Tensor],
    forward_backward: Callable[[int, Optional[Tuple[torch.Tensor, torch.Tensor]],
                                torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, float]]],
    optimizer_step: Callable[[], None],
    n_sup: int,
    exploration_prob: float,
    any_live: Optional[Callable[[bool], bool]] = None,
    carry_store: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Drive deep supervision over one optimizer window of ``n_micro``
    micro-batches (gradient accumulation).

    ``forward_backward(i, carry, active)`` runs micro-batch ``i`` from
    ``carry`` on the ``active`` frames, calls ``backward()`` on its (already
    accumulation-scaled) loss and returns ``(pred, stats)``; ``pred`` must
    hold ``q_halt`` [B, T] and ``carry``.  ``optimizer_step()`` is called
    once per supervision step.  ``any_live(local_live)`` lets distributed
    callers make the continue decision collective (every rank must run the
    same number of optimizer steps).  ``carry_store`` (e.g. CPU) parks the
    carried latents of the micro-batches not being processed — with
    gradient accumulation every micro-batch's [B*T, S, d] carry must
    survive until its next supervision step.

    Returns mean stats over all supervision steps plus ``sup_steps`` (steps
    run) and ``halt_depth`` (mean supervision steps per frame).
    """
    carries: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_micro
    active = [m.clone() for m in frame_masks]
    steps = [torch.zeros(m.shape, dtype=torch.long, device=m.device) for m in frame_masks]
    agg: Dict[str, float] = {}
    n_stats = 0
    sup_steps = 0
    for _ in range(max(1, n_sup)):
        local_live = any(bool(a.any()) for a in active)
        live = any_live(local_live) if any_live is not None else local_live
        if not live:
            break
        for i in range(n_micro):
            if not bool(active[i].any()):
                continue
            carry = carries[i]
            if carry is not None and carry_store is not None:
                dev = active[i].device
                carry = (carry[0].to(dev, non_blocking=True),
                         carry[1].to(dev, non_blocking=True))
            pred, stats = forward_backward(i, carry, active[i])
            new_carry = pred["carry"]
            if carry_store is not None:
                # blocking: a non_blocking device→host copy may still be in
                # flight when the tensor is next read
                new_carry = (new_carry[0].to(carry_store),
                             new_carry[1].to(carry_store))
            carries[i] = new_carry
            steps[i], active[i] = halt_update(
                pred["q_halt"].detach(), steps[i], active[i], n_sup,
                exploration_prob, training=True)
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            n_stats += 1
        optimizer_step()
        sup_steps += 1
    out = {k: v / max(n_stats, 1) for k, v in agg.items()}
    n_frames = sum(int(m.sum()) for m in frame_masks)
    out["sup_steps"] = float(sup_steps)
    out["halt_depth"] = (sum(int(s.sum()) for s in steps) / max(n_frames, 1))
    return out
