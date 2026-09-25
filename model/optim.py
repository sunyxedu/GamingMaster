"""Optimisation recipe of the reference TRM implementation
(SamsungSAILMontreal/TinyRecursiveModels, pretrain.py + models/ema.py):

- :class:`AdamATan2` — Adam with ``atan2`` in place of ``m / (√v + ε)``
  (no epsilon) and decoupled weight decay.  The reference imports the
  CUDA-only ``adam-atan2`` package; this replicates its kernel exactly in
  PyTorch so it also runs on CPU/MPS.
- :func:`cosine_schedule_with_warmup_lr_lambda` — linear warmup, then
  cosine towards ``min_ratio``; the reference configs use
  ``lr_min_ratio=1.0``, i.e. constant LR after warmup.
- :class:`ModelEMA` — EMA of the trainable parameters (rate 0.999),
  used for evaluation and for the saved checkpoint.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Iterable

import torch
import torch.nn as nn

#: reference hyper-parameters (config/cfg_pretrain.yaml, README commands)
TRM_LR = 1e-4
TRM_BETAS = (0.9, 0.95)
TRM_WEIGHT_DECAY = 0.1
TRM_WARMUP_STEPS = 2000
TRM_LR_MIN_RATIO = 1.0
TRM_EMA_RATE = 0.999


class AdamATan2(torch.optim.Optimizer):
    """Pure-PyTorch replica of the ``adam-atan2`` 0.0.3 CUDA kernel that
    the reference imports (csrc/adam_atan2.cu ``adam_math``), per step t:

        p   ← p · (1 − lr·wd)
        m   ← lerp(m, g, 1 − β1);   v ← lerp(v, g², 1 − β2)
        p   ← p − (lr / (1 − β1^t)) · atan2(m, √v / √(1 − β2^t))

    No extra scale factor, and the first-moment bias correction sits
    OUTSIDE the atan2 (unlike the Everett et al. / lucidrains variant).
    Math runs in fp32 for sub-fp32 parameters, like the kernel's opmath.
    """

    def __init__(self, params: Iterable, lr: float = TRM_LR,
                 betas=TRM_BETAS, weight_decay: float = TRM_WEIGHT_DECAY):
        if lr < 0:
            raise ValueError(f"invalid lr {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"invalid betas {betas}")
        if weight_decay < 0:
            raise ValueError(f"invalid weight_decay {weight_decay}")
        super().__init__(params, dict(lr=lr, betas=tuple(betas),
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group.get("weight_decay", 0.0)
            beta1, beta2 = group.get("betas", TRM_BETAS)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamATan2 does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] = int(state["step"]) + 1
                t = state["step"]
                step_size = lr / (1.0 - beta1 ** t)
                bc2_sqrt = math.sqrt(1.0 - beta2 ** t)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                low = p.dtype in (torch.float16, torch.bfloat16)
                math_dtype = torch.float32 if low else p.dtype
                param = p.to(math_dtype)
                grad = p.grad.to(math_dtype)
                m = exp_avg.to(math_dtype)
                v = exp_avg_sq.to(math_dtype)

                param.mul_(1.0 - lr * wd)
                m.lerp_(grad, 1.0 - beta1)
                v.lerp_(grad * grad, 1.0 - beta2)
                param.sub_(torch.atan2(m, v.sqrt() / bc2_sqrt), alpha=step_size)

                if param is not p:
                    p.copy_(param)
                if m is not exp_avg:
                    exp_avg.copy_(m)
                    exp_avg_sq.copy_(v)
        return loss


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int,
    num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5,
) -> float:
    """Reference pretrain.py schedule (returns the absolute LR)."""
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def trm_lr_lambda(warmup_steps: int, total_steps: int, min_ratio: float):
    """LambdaLR multiplier form of the reference schedule."""
    def fn(step: int) -> float:
        return cosine_schedule_with_warmup_lr_lambda(
            step, base_lr=1.0, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps, min_ratio=min_ratio)
    return fn


class ModelEMA:
    """Reference models/ema.py: shadow = mu·shadow + (1-mu)·param over the
    trainable parameters, updated after every optimizer step."""

    def __init__(self, model: nn.Module, mu: float = TRM_EMA_RATE):
        self.mu = mu
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad
        }
        # modules shared under several names (multi-game core): every alias
        # of a shadowed parameter maps to its canonical (first) name
        canon = {id(p): n for n, p in model.named_parameters() if n in self.shadow}
        self.alias: Dict[str, str] = {
            n: canon[id(p)] for n, p in model.named_parameters(remove_duplicate=False)
            if id(p) in canon}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.mu).add_(p.detach(), alpha=1.0 - self.mu)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])

    def ema_copy(self, model: nn.Module) -> nn.Module:
        m = copy.deepcopy(model)
        self.copy_to(m)
        return m

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        for n, v in state.items():
            if n in self.shadow and self.shadow[n].shape == v.shape:
                self.shadow[n].copy_(v)

    def ema_model_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """``model.state_dict()`` with EMA values substituted for the
        trainable parameters (buffers are taken from ``model``)."""
        sd = model.state_dict()
        out = {}
        for k, v in sd.items():
            c = self.alias.get(k)
            out[k] = self.shadow[c].detach().clone() if c is not None else v
        return out
