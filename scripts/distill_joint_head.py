"""Distill the factorized policy into the joint-action decoder.

Head-swap migration without losing strength: the trunk (encoder, ST
layers, TRM, value head) is copied verbatim and FROZEN; only the joint
decoder's arrow_head/pass_proj train, matching the teacher's factorized
distributions through the student's marginal views (per-head KL via
compute_policy_kl_heads — the same rails the magnet anchor uses).
Distillation states come from teacher self-play (vectorized collector),
so the match happens exactly on the state distribution that matters.

    python scripts/distill_joint_head.py \
        --teacher /path/rl_ema_iter2720.pt --out /path/joint_seed.pt \
        --waves 3 --envs 64 --epochs 2 --replay_dir ... --filtered_file ...
"""

import argparse
import sys

import torch

from env.generals_env import GeneralsEnv
from model.config import ModelConfig
from model.network import GeneralsNetwork
from model.rl_utils import compute_policy_kl_heads
from scripts.elo_ladder import load_net
from train_rl import collect_selfplay_episodes_vec

KL_WEIGHTS = {"kl_src": 1.0, "kl_dir": 1.0, "kl_spl": 0.5, "kl_pass": 1.0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--waves", type=int, default=3,
                   help="self-play waves of --envs games each")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rollout_steps", type=int, default=1536)
    p.add_argument("--replay_dir", type=str, default=None)
    p.add_argument("--filtered_file", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- teacher (factorized) ----
    teacher, t_cfg = load_net(args.teacher, device)  # WidthAdapter shim
    teacher.eval()
    t_net = teacher.net  # raw GeneralsNetwork for forward/state_dict

    # ---- student (joint decoder, same trunk) ----
    s_cfg_kwargs = {k: getattr(t_cfg, k)
                    for k in ModelConfig.__dataclass_fields__}
    s_cfg_kwargs["joint_decoder"] = True
    s_cfg = ModelConfig(**s_cfg_kwargs)
    student = GeneralsNetwork(s_cfg).to(device)
    t_state = t_net.state_dict()
    s_state = student.state_dict()
    copied = 0
    for k, v in t_state.items():
        if k in s_state and s_state[k].shape == v.shape:
            s_state[k] = v
            copied += 1
    student.load_state_dict(s_state)
    print(f"copied {copied}/{len(s_state)} tensors from teacher "
          f"(decoder heads stay fresh)")
    student.eval()  # BN running stats / no dropout — matches rollout regime

    head_params = list(student.action_decoder.parameters())
    for prm in student.parameters():
        prm.requires_grad = False
    for prm in head_params:
        prm.requires_grad = True
    opt = torch.optim.Adam(head_params, lr=args.lr)

    # ---- envs ----
    envs = [GeneralsEnv(grid_h=t_cfg.grid_h, grid_w=t_cfg.grid_w,
                        max_turns=args.rollout_steps)
            for _ in range(args.envs)]
    for e in envs:
        e.turn_globals = bool(getattr(t_cfg, "turn_globals", False))
    if args.replay_dir and args.filtered_file:
        n = envs[0].load_map_pool(args.replay_dir, args.filtered_file)
        for e in envs[1:]:
            e.map_pool = envs[0].map_pool
        print(f"Loaded {n} maps")

    # ---- distill ----
    for wave in range(args.waves):
        bufs, _, _ = collect_selfplay_episodes_vec(
            teacher, envs, device,
            max_steps=args.rollout_steps, gamma=1.0, gae_lambda=0.9,
            record=False, act_steps=1, legal_mask=True,
        )
        n_frames = sum(len(b) for b in bufs)
        print(f"wave {wave}: {len(bufs)} buffers, {n_frames} frames",
              flush=True)

        for epoch in range(args.epochs):
            tot = 0.0
            n_b = 0
            for buf in bufs:
                if len(buf) == 0:
                    continue
                batch = buf.to_tensor_batch(device)
                grid_seq = batch["grid_seq"]
                globals_seq = batch["globals_seq"]
                actions = batch["actions"]
                src_mask = batch["legal_mask"]
                dir_mask = batch["dir_mask"]
                source_y = actions["source_idx"] // t_cfg.grid_w
                source_x = actions["source_idx"] % t_cfg.grid_w

                with torch.no_grad():
                    t_pred = t_net(
                        grid_seq, globals_seq, legal_mask=src_mask,
                        source_y=source_y, source_x=source_x,
                        act_steps=1, direction_mask=dir_mask)
                s_pred = student(
                    grid_seq, globals_seq, legal_mask=src_mask,
                    source_y=source_y, source_x=source_x,
                    act_steps=1, direction_mask=dir_mask)

                kls = compute_policy_kl_heads(
                    s_pred, t_pred, legal_mask=src_mask,
                    passed=actions["passed"])
                loss = sum(KL_WEIGHTS[k] * v for k, v in kls.items())
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.detach())
                n_b += 1
            print(f"  wave {wave} epoch {epoch}: mean KL "
                  f"{tot / max(n_b, 1):.4f}", flush=True)

    # ---- save joint seed checkpoint ----
    out = {
        "model_state": student.state_dict(),
        "config": {k: getattr(s_cfg, k)
                   for k in ModelConfig.__dataclass_fields__},
        "iteration": 2720,
        "distilled_from": args.teacher,
    }
    torch.save(out, args.out)
    print(f"saved joint seed → {args.out}")


if __name__ == "__main__":
    sys.exit(main())
