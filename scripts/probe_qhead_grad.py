#!/usr/bin/env python3
"""Probe: under which conditions does trm.q_head lose its gradient?

Production observation (phase-2 IL, NGC 24.09 container, torch 2.5.0a0,
H200): [grad-sync] warnings show trm.q_head.{weight,bias} — and ONLY
those two parameters — missing gradients on most ranks in most
accumulation windows. On torch 2.10 CPU this never reproduces at any
ACT depth, with or without bf16 autocast.

This script replicates the exact training forward/backward on ONE GPU
and reports, for every (act_steps, autocast) combination:
  - q_halt.requires_grad and the final z_H's requires_grad
  - which parameter groups received gradients from the POLICY loss alone
    (wm loss off, so a severed final-z_H graph is visible: if the trunk
    only trains through the world-model loss, st_layers/spatial_encoder
    would show no grads here)

Run inside the production container:

    srun --gres=gpu:nvidia_h200_nvl:1 --time=00:10:00 \
        apptainer exec --nv /scratch/$USER/pytorch.sif \
        python3 ~/GamingMaster/scripts/probe_qhead_grad.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import ModelConfig
from model.losses import ImitationLoss
from model.network import GeneralsNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B, T = 1, 8

GROUPS = {
    "trm.q_head": "trm.q_head.",
    "trm.reasoning": "trm.reasoning.",
    "st_layers": "st_layers.",
    "spatial_enc": "obs_encoder.spatial_encoder.",
    "source_head": "action_decoder.source_head.",
    "value_head": "value_head.",
}


def grad_presence(net):
    out = {}
    for label, prefix in GROUPS.items():
        params = [p for n, p in net.named_parameters() if n.startswith(prefix)]
        got = sum(p.grad is not None for p in params)
        out[label] = f"{got}/{len(params)}"
    return out


def probe(net, crit, act_steps, use_autocast):
    net.zero_grad(set_to_none=True)
    torch.manual_seed(42)
    grid = torch.randn(B, T, 12, 25, 25, device=DEVICE)
    gl = torch.randn(B, T, 4, device=DEVICE)
    legal = torch.rand(B, T, 1, 25, 25, device=DEVICE) < 0.3
    tgt = {
        "count": torch.ones(B, T, dtype=torch.long, device=DEVICE),
        "src_y": torch.randint(0, 25, (B, T, 1), device=DEVICE),
        "src_x": torch.randint(0, 25, (B, T, 1), device=DEVICE),
        "dir_id": torch.randint(0, 4, (B, T, 1), device=DEVICE),
        "ratio_id": torch.randint(0, 2, (B, T, 1), device=DEVICE),
        "value": torch.rand(B, T, device=DEVICE) * 2 - 1,
    }
    fm = torch.ones(B, T, dtype=torch.bool, device=DEVICE)

    ac = torch.amp.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=use_autocast)
    with ac:
        pred = net(
            grid, gl, legal_mask=legal,
            source_y=tgt["src_y"][:, :, 0], source_x=tgt["src_x"][:, :, 0],
            frame_mask=fm, act_steps=act_steps,
        )
        # POLICY loss only — no world-model term, so trunk grads here can
        # only arrive through the TRM/decode path.
        losses = crit(pred, tgt, [T] * B)
    losses["loss"].backward()

    print(f"act_steps={act_steps} autocast={use_autocast!s:5s} "
          f"q_halt.rg={pred['q_halt'].requires_grad!s:5s}  "
          + "  ".join(f"{k}={v}" for k, v in grad_presence(net).items()),
          flush=True)


def probe_trm_direct(net, act_steps, use_autocast):
    """Replicate GameNetwork._trm_ponder on the bare TRM and inspect the
    requires_grad of everything it returns."""
    torch.manual_seed(7)
    S = net.obs_encoder.n_tokens + 2
    x = torch.randn(4, S, net.cfg.d_model, device=DEVICE, requires_grad=True)
    ac = torch.amp.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=use_autocast)
    with ac:
        carry = None
        if act_steps > 1:
            with torch.no_grad():
                for _ in range(act_steps - 1):
                    _, _, carry = net.trm(x, carry)
        z_H, q_halt, _ = net.trm(x, carry)
    print(f"  [trm-direct] act_steps={act_steps} autocast={use_autocast!s:5s} "
          f"z_H.rg={z_H.requires_grad!s:5s} q_halt.rg={q_halt.requires_grad}",
          flush=True)


def main():
    print(f"torch {torch.__version__}  device={DEVICE}  "
          f"cuda={torch.version.cuda if torch.cuda.is_available() else '-'}",
          flush=True)
    torch.manual_seed(0)
    cfg = ModelConfig(temporal_mixer="recurrent", predict_next_obs=True)
    net = GeneralsNetwork(cfg).to(DEVICE)
    net.train()
    crit = ImitationLoss(grid_w=cfg.grid_w, focal_gamma=1.5)

    print("── full forward/backward (policy loss only) ──", flush=True)
    for use_ac in (True, False):
        for k in (1, 2, 3, 5):
            probe(net, crit, k, use_ac)

    print("── bare TRM ponder ──", flush=True)
    for use_ac in (True, False):
        for k in (1, 2, 3, 5):
            probe_trm_direct(net, k, use_ac)

    print("done", flush=True)


if __name__ == "__main__":
    main()
