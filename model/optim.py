"""Optimisation recipe of the reference TRM implementation
(SamsungSAILMontreal/TinyRecursiveModels, pretrain.py + models/ema.py):

- :class:`AdamATan2` — Adam with ``atan2(m̂, b·√v̂)`` in place of
  ``m̂ / (√v̂ + ε)`` (scale-invariant, no epsilon) and decoupled weight
  decay.  The reference imports the CUDA-only ``adam_atan2`` package; this
  is a pure-PyTorch version of the same update so it also runs on CPU/MPS.
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
    """Adam-atan2 (Everett et al., 2024) with decoupled weight decay.

    update = a · atan2(m / (1-β1^t), b · sqrt(v / (1-β2^t)))
    θ ← θ · (1 - lr·wd) - lr · update
    """

    def __init__(self, params: Iterable, lr: float = TRM_LR,
                 betas=TRM_BETAS, weight_decay: float = TRM_WEIGHT_DECAY,
                 a: float = 1.27, b: float = 1.0):
        if lr < 0:
            raise ValueError(f"invalid lr {lr}")
        defaults = dict(lr=lr, betas=tuple(betas), weight_decay=weight_decay,
                        a=a, b=b)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group.get("weight_decay", 0.0)
            beta1, beta2 = group.get("betas", TRM_BETAS)
            a, b = group.get("a", 1.27), group.get("b", 1.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamATan2 does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] = int(state["step"]) + 1
                t = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if wd:
                    p.mul_(1.0 - lr * wd)
                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                den = exp_avg_sq.mul(b * b / bc2).sqrt_()
                update = exp_avg.mul(1.0 / bc1).atan2_(den)
                p.add_(update, alpha=-lr * a)
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
            out[k] = self.shadow[k].detach().clone() if k in self.shadow else v
        return out
