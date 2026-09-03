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

"""Correctness tests for the recursive dynamics core in the stock TSSM.

Run from the repo root:  python -m nnet.modules.twister.test_recursive_dynamics
"""

import sys
import os
import types
import torch
import torch.nn as nn

# Path + stubs (must come before any nnet import)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

def _stub(*names):
    for name in names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

_stub("ale_py", "gymnasium", "gymnasium.wrappers")
_gym = sys.modules["gymnasium"]
_gym.wrappers = sys.modules["gymnasium.wrappers"]
_gym.register_envs = lambda *a: None

_stub("gym", "gym.envs", "gym.envs.atari", "gym.spaces", "gym.wrappers")
_oldgym = sys.modules["gym"]
_oldgym.envs = sys.modules["gym.envs"]
_oldgym.envs.atari = sys.modules["gym.envs.atari"]
_oldgym.spaces = sys.modules["gym.spaces"]
_oldgym.wrappers = sys.modules["gym.wrappers"]
_oldgym.Env = object
_oldgym.Wrapper = object
_oldgym.spaces.Box = object
_oldgym.spaces.Discrete = object
_oldgym.envs.atari.AtariEnv = object

_stub("dm_control", "dm_control.suite", "dm_control.suite.wrappers")
sys.modules["dm_control"].suite = sys.modules["dm_control.suite"]
_dmc_w = sys.modules["dm_control.suite.wrappers"]
sys.modules["dm_control.suite"].wrappers = _dmc_w
_dmc_w.pixels = types.ModuleType("pixels")

from nnet import modules
from nnet.modules.twister.recursive_dynamics import RecursiveDynamics
from nnet.modules.twister.tssm import TSSM

DIM = 64
HEADS = 4
CTX = 8


def _core(**kw):
    torch.manual_seed(0)
    return RecursiveDynamics(
        dim_model=DIM, num_heads=HEADS, drop_rate=0.0, max_pos_encoding=2048,
        ff_ratio=2, module_pre_norm=False, **kw,
    )


def _mask(seq_len, hidden_len):
    return modules.return_mask(
        seq_len=seq_len, hidden_len=hidden_len, left_context=CTX,
        right_context=0, dtype=torch.float32, device="cpu",
    )


def test_streaming_consistency():
    """Full-sequence forward equals two chunked forwards with hidden carry."""
    core = _core().eval()
    x = torch.randn(1, 6, DIM)

    full = core(x, mask=_mask(6, 0), return_hidden=True).x

    part1 = core(x[:, :3], mask=_mask(3, 0), return_hidden=True)
    part2 = core(x[:, 3:], hidden=part1.hidden, mask=_mask(3, 3), return_hidden=True)
    chunked = torch.cat([part1.x, part2.x], dim=1)

    diff = (full - chunked).abs().max().item()
    assert diff < 1e-4, "streaming mismatch: {}".format(diff)
    print("test_streaming_consistency OK (max diff {:.2e})".format(diff))


def test_gradient_flow():
    core = _core().train()
    x = torch.randn(1, 4, DIM, requires_grad=True)
    out = core(x, mask=_mask(4, 0))
    # A random projection: the plain sum of a LayerNorm output is constant. q_halt joins so q_head is covered.
    ((out.x * torch.randn_like(out.x)).sum() + out.q_halt.sum()).backward()
    missing = [n for n, p in core.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, "no gradient reaches: {}".format(missing)
    assert x.grad is not None and x.grad.abs().sum() > 0
    print("test_gradient_flow OK ({} params)".format(sum(p.numel() for p in core.parameters())))


def _tssm(dynamics_core):
    torch.manual_seed(0)
    return TSSM(
        num_actions=5, stoch_size=4, discrete=8, learn_initial=True,
        hidden_size=DIM, num_blocks=4, ff_ratio=2, num_heads=HEADS,
        drop_rate=0.0, att_context_left=CTX,
        dynamics_core=dynamics_core,
    )


def test_param_efficiency():
    base = sum(p.numel() for p in _tssm("transformer").transformer.parameters())
    trm = sum(p.numel() for p in _tssm("trm").transformer.parameters())
    assert trm < base, "recursive core ({}) not smaller than baseline ({})".format(trm, base)
    print("test_param_efficiency OK (dynamics params {} -> {}, {:.0%})".format(base, trm, trm / base))


def test_tssm_observe_and_imagine():
    """The TRM-cored stock TSSM runs the training and imagination paths end to end."""
    tssm = _tssm("trm").eval()
    B, L = 2, 6
    states = {
        "logits": torch.zeros(B, L, 4, 8),
        "stoch": torch.randn(B, L, 4, 8),
    }
    actions = torch.zeros(B, L, 5)
    is_firsts = torch.zeros(B, L)
    is_firsts[:, 0] = 1.0

    posts, priors = tssm.observe(states, actions, is_firsts)
    assert priors["deter"].shape == (B, L, DIM)
    assert priors["logits"].shape == (B, L, 4, 8)
    assert priors["stoch"].shape == (B, L, 4, 8)

    class _RPolicy(nn.Module):
        def forward(self, feat):
            dist = torch.distributions.OneHotCategorical(logits=torch.zeros(feat.shape[:-1] + (5,)))
            dist.rsample = dist.sample
            return dist

    prev_state = {k: v[:, -1:] for k, v in priors.items() if k != "hidden"}
    prev_state["hidden"] = tssm.slice_hidden(priors["hidden"])

    img = tssm.imagine(_RPolicy(), prev_state, img_steps=3)
    assert img["deter"].shape == (B, 4, DIM)
    assert img["stoch"].shape == (B, 4, 4, 8)
    print("test_tssm_observe_and_imagine OK")


if __name__ == "__main__":
    test_streaming_consistency()
    test_gradient_flow()
    test_param_efficiency()
    test_tssm_observe_and_imagine()
    print("all recursive-dynamics tests passed")
