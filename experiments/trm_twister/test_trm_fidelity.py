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

"""Fidelity tests for RecursiveDynamics against the TRM recursion (arXiv 2510.04871):

    for h in H_cycles:
        for l in L_cycles:
            z_L = net(z_L, z_H + x)
        z_H = net(z_H, z_L)

Run from the repo root:  python nnet/modules/twister/test_trm_fidelity.py
"""

import sys
import os
import types
import torch

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

DIM = 64
H_CYCLES = 3
L_CYCLES = 4


def _core():
    torch.manual_seed(7)
    return RecursiveDynamics(
        dim_model=DIM, num_heads=4, drop_rate=0.0, max_pos_encoding=2048,
        ff_ratio=2, module_pre_norm=False,
        H_cycles=H_CYCLES, L_cycles=L_CYCLES,
    )


def _mask(seq_len):
    return modules.return_mask(seq_len=seq_len, hidden_len=0, left_context=8,
                               right_context=0, dtype=torch.float32, device="cpu")


def test_matches_paper_pseudocode():
    """Module forward == naive transcription of the paper loop, exactly."""
    core = _core().eval()
    x = torch.randn(2, 5, DIM)

    with torch.no_grad():
        got = core(x, mask=_mask(5)).x

        # Naive reference using the module's own submodules
        c = core.context(x, mask=_mask(5)).x
        z_H = core.H_init.expand_as(c)
        z_L = core.L_init.expand_as(c)
        for _h in range(H_CYCLES):
            for _l in range(L_CYCLES):
                z_L = core.reasoning(z_L, z_H + c)
            z_H = core.reasoning(z_H, z_L)
        want = core.out_norm(z_H)

    diff = (got - want).abs().max().item()
    assert diff == 0.0, "module deviates from paper pseudocode: {}".format(diff)
    print("test_matches_paper_pseudocode OK (exact)")


def test_cycle_counts():
    """Exactly H*L z_L updates and H z_H updates per forward."""
    core = _core().eval()
    calls = []
    orig = core.reasoning.forward
    core.reasoning.forward = lambda z, inj: calls.append(z.shape) or orig(z, inj)
    with torch.no_grad():
        core(torch.randn(1, 3, DIM), mask=_mask(3))
    expected = H_CYCLES * L_CYCLES + H_CYCLES
    assert len(calls) == expected, "reasoning called {}x, expected {}".format(len(calls), expected)
    print("test_cycle_counts OK ({} calls)".format(len(calls)))


def test_gradient_boundary():
    """Every parameter and the input receive gradient; the init buffers do not."""
    core = _core().train()
    x = torch.randn(1, 4, DIM, requires_grad=True)
    out = core(x, mask=_mask(4))
    # A random projection: the plain sum of a LayerNorm output is constant.
    (out.x * torch.randn_like(out.x)).sum().backward()

    assert x.grad is not None and x.grad.abs().sum() > 0, "no grad to input/encoder path"
    for n, p in core.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, "no grad: {}".format(n)

    assert not core.H_init.requires_grad and not core.L_init.requires_grad
    print("test_gradient_boundary OK")


def test_no_grad_cycles_matter():
    """The H-1 no-grad cycles must change the states entering the final cycle
    (otherwise H_cycles is dead config)."""
    core = _core().eval()
    x = torch.randn(1, 4, DIM)
    with torch.no_grad():
        full = core(x, mask=_mask(4)).x
        core.H_cycles = 1
        shallow = core(x, mask=_mask(4)).x
        core.H_cycles = H_CYCLES
    assert (full - shallow).abs().max().item() > 1e-4, "extra H cycles are a no-op"
    print("test_no_grad_cycles_matter OK")


def test_baseline_state_dict_unchanged():
    """dynamics_core=transformer must produce the stock parameter names."""
    from nnet.modules.twister.tssm import TSSM
    torch.manual_seed(0)
    tssm = TSSM(num_actions=4, stoch_size=4, discrete=8, hidden_size=DIM,
                num_blocks=2, ff_ratio=2, num_heads=4, drop_rate=0.0,
                att_context_left=8, dynamics_core="transformer")
    keys = list(tssm.state_dict().keys())
    assert any(k.startswith("transformer.blocks.1.") for k in keys), "stock block keys missing"
    assert not any("reasoning" in k or "H_init" in k for k in keys), "TRM keys leaked into baseline"
    print("test_baseline_state_dict_unchanged OK")


if __name__ == "__main__":
    test_matches_paper_pseudocode()
    test_cycle_counts()
    test_gradient_boundary()
    test_no_grad_cycles_matter()
    test_baseline_state_dict_unchanged()
    print("all fidelity tests passed")
