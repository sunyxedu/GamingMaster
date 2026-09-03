# Copyright 2026, Hasaan Ahmad.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TRM-style recursive dynamics core.

Recursion, carry and halting head follow the official TRM code
(SamsungSAILMontreal/TinyRecursiveModels, models/recursive_reasoning/trm.py,
MIT license) and its HRM ancestor (sapientinc/HRM). That code assumes a fixed
input token grid; this module replaces its inner model with a streaming version
that attends over past timesteps through KV caches.

A single shared network `net` refines two states, z_H (prediction) and z_L
(scratchpad), with the input injected by addition:

    for h in H_cycles:
        for l in L_cycles:
            z_L = net(z_L, z_H + x)
        z_H = net(z_H, z_L)

All calls share the same weights. Only the last outer cycle keeps gradients
(TRM: the earlier cycles run under no_grad). A carry (z_H, z_L) from a previous
pass can replace the fixed start values; this is how deep supervision and ACT
continue the recursion across passes. `q_head` reads a halt logit from z_H at
every timestep.

Two loop types:
- "mlp": one causal-attention context block runs once and produces x; `net` is
  a per-timestep 2-layer MLP.
- "attention": no context block; `net` is a causal-attention transformer block.
  Every call attends over the past steps at the same loop position, through its
  own KV cache.

Drop-in for modules.TransformerNetwork: forward(x, hidden, mask, return_hidden)
returns AttrDict(x, hidden, carry, q_halt).
"""

import torch
import torch.nn as nn

from nnet import modules
from nnet import structs


class ReasoningCore(nn.Module):
    """Per-timestep reasoning block: additive injection, then residual MLP layers."""

    def __init__(self, dim_model, num_layers=2, ff_ratio=2, act_fun=nn.SiLU):
        super(ReasoningCore, self).__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim_model) for _ in range(num_layers)])
        self.mlps = nn.ModuleList([nn.Sequential(
            nn.Linear(dim_model, dim_model * ff_ratio),
            act_fun(),
            nn.Linear(dim_model * ff_ratio, dim_model),
        ) for _ in range(num_layers)])

    def forward(self, z, injection):
        z = z + injection
        for norm, mlp in zip(self.norms, self.mlps):
            z = z + mlp(norm(z))
        return z


def _transformer(dim_model, num_blocks, num_heads, drop_rate, max_pos_encoding, ff_ratio, module_pre_norm):
    """The TWISTER causal transformer block stack, as used by the stock TSSM."""
    return modules.TransformerNetwork(
        dim_model=dim_model,
        num_blocks=num_blocks,
        att_params={
            "class": "RelPosMultiHeadSelfAttention",
            "params": {
                "num_heads": num_heads,
                "weight_init": "default",
                "bias_init": "default",
                "attn_drop_rate": drop_rate,
                "max_pos_encoding": max_pos_encoding,
                "causal": True
            }
        },
        emb_drop_rate=0.0,
        drop_rate=drop_rate,
        pos_embedding=None,
        mask=None,
        ff_ratio=ff_ratio,
        weight_init="default",
        bias_init="default",
        act_fun="ReLU",
        module_pre_norm=module_pre_norm
    )


class RecursiveDynamics(nn.Module):

    def __init__(
            self,
            dim_model,
            num_heads,
            drop_rate,
            max_pos_encoding,
            ff_ratio,
            module_pre_norm,
            loop="mlp",
            loop_blocks=1,
            context_blocks=1,
            H_cycles=3,
            L_cycles=4,
            reason_layers=2,
            reason_ff_ratio=2,
        ):
        super(RecursiveDynamics, self).__init__()
        assert loop in ("mlp", "attention"), "unknown loop: {}".format(loop)

        self.loop = loop
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.num_calls = H_cycles * L_cycles + H_cycles

        if loop == "attention":
            self.net = _transformer(dim_model, loop_blocks, num_heads, drop_rate, max_pos_encoding, ff_ratio, module_pre_norm)
        else:
            self.context = _transformer(dim_model, context_blocks, num_heads, drop_rate, max_pos_encoding, ff_ratio, module_pre_norm)
            self.reasoning = ReasoningCore(dim_model, num_layers=reason_layers, ff_ratio=reason_ff_ratio)

        # Fixed initial states, not trained (HRM: unit-variance buffers)
        self.register_buffer("H_init", torch.randn(dim_model))
        self.register_buffer("L_init", torch.randn(dim_model))

        # Halt head (TRM q_head, halt logit only): starts far from halting
        self.q_head = nn.Linear(dim_model, 1)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

        self.out_norm = nn.LayerNorm(dim_model)

    def initial_carry(self, x):
        """Start states broadcast to the shape of x (B, L, D)."""
        return self.H_init.to(x.dtype).expand_as(x), self.L_init.to(x.dtype).expand_as(x)

    def _recurse(self, x, net, carry=None):
        """Run the H x L schedule with `net(z, injection)`. Only the last outer cycle keeps gradients."""
        if carry is None:
            z_H, z_L = self.initial_carry(x)
        else:
            z_H, z_L = carry

        if self.H_cycles > 1:
            with torch.no_grad():
                for _ in range(self.H_cycles - 1):
                    for _ in range(self.L_cycles):
                        z_L = net(z_L, z_H + x)
                    z_H = net(z_H, z_L)

        for _ in range(self.L_cycles):
            z_L = net(z_L, z_H + x)
        z_H = net(z_H, z_L)
        return z_H, z_L

    def forward(self, x, lengths=None, hidden=None, return_hidden=False, return_att_w=False, mask=None, start_position=0, return_blocks_x=False, carry=None):

        if self.loop == "attention":
            # One cache per block per call. `hidden` is the flat list from the previous step.
            num_blocks = len(self.net.blocks)
            caches = []
            state = {"call": 0}

            def net(z, injection):
                i = state["call"]
                h = hidden[i * num_blocks:(i + 1) * num_blocks] if hidden is not None else None
                out = self.net(z + injection, lengths=lengths, hidden=h, mask=mask, return_hidden=True, start_position=start_position)
                caches.extend(out.hidden)
                state["call"] += 1
                return out.x

            z_H, z_L = self._recurse(x, net, carry)
            new_hidden = caches
            att_w = []
        else:
            ctx = self.context(x, lengths=lengths, hidden=hidden, return_hidden=return_hidden, return_att_w=return_att_w, mask=mask, start_position=start_position)
            z_H, z_L = self._recurse(ctx.x, self.reasoning, carry)
            new_hidden = ctx.hidden if return_hidden else None
            att_w = ctx.att_w if return_att_w else None

        outputs = structs.AttrDict(x=self.out_norm(z_H))
        outputs.carry = (z_H.detach(), z_L.detach())
        outputs.q_halt = self.q_head(z_H).squeeze(-1)
        if return_hidden:
            outputs.hidden = new_hidden
        if return_att_w:
            outputs.att_w = att_w
        if return_blocks_x:
            outputs.blocks_x = [outputs.x]

        return outputs
