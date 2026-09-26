"""PPO self-play fine-tuning for the placement board games.

Loads a ``training/train_il_game.py`` checkpoint (othello / gomoku), self-plays
with a single policy on both seats through
:class:`env.board_selfplay_env.BoardSelfPlayEnv`, and PPO-fine-tunes the
policy + value heads.

    uv run python -m training.train_rl_game --game othello \
        --il_checkpoint checkpoints/il_game/othello_epoch9.pt --iters 200

The effective RL action is the placement point: othello passes are
rules-forced (the game has no voluntary pass), so pass frames keep their
observation and value target but carry no policy loss, and the IL pass
head is left untouched.  Rollout log-probs and the PPO replay both go
through the fp32 ``masked_log_prob`` path at the same fixed ACT depth,
and both run in eval mode, so the ratio is exactly 1 before the first
update — same invariants as generals' training/train_rl.py.

generals keeps its dedicated ``training/train_rl.py`` (simultaneous moves,
opponent pool, IL regularisation); chess needs from→to legality
threading in FromToDecoder.sample before it can join.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Dict, List

import torch
import torch.nn.functional as F

from env.board_selfplay_env import BoardSelfPlayEnv
from model.components.core import update_moe_balance
from model.games.base import categorical_entropy, masked_log_prob
from model.rl_utils import compute_gae, freeze_encoder, optim_param_groups
from training.train_il_game import build_net_tolerant, pick_device, resume_arch_overrides


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True, choices=("othello", "gomoku"))
    p.add_argument("--il_checkpoint", required=True)
    p.add_argument("--board_size", type=int, default=None,
                   help="gomoku only; must match the checkpoint's board_h")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--games_per_iter", type=int, default=8)
    p.add_argument("--ppo_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--moe_balance_rate", type=float, default=1e-3,
                   help="γ of the MoE aux-loss-free balance-bias update, once per "
                        "optimizer step (MoE cores only; as the IL trainers)")
    p.add_argument("--act_steps", type=int, default=0,
                   help="fixed TRM depth for rollout AND update "
                        "(0 = the model's full depth cfg.halt_max_steps)")
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--save_dir", default="checkpoints/rl_game")
    p.add_argument("--save_interval", type=int, default=25)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ----------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------

def play_game(
    net, game: str, board_size, act_steps: int, device: torch.device,
) -> List[Dict]:
    """Self-play one game; returns one trajectory per seat.

    Each seat's trajectory holds only that seat's decision frames
    (mover-POV, like an IL sample): obs [T,...], place [T], legal [T,N],
    log_prob [T], value [T], policy_mask [T] (False on forced-pass
    frames), and the seat's terminal reward.
    """
    env = BoardSelfPlayEnv(game, board_size)
    seats: Dict[int, Dict[str, list]] = {
        1: {"obs": [], "place": [], "legal": [], "lp": [], "value": [],
            "pmask": []},
        2: {"obs": [], "place": [], "legal": [], "lp": [], "value": [],
            "pmask": []},
    }
    caches = {1: None, 2: None}

    with torch.no_grad():
        while not env.done:
            seat = env.mover
            tr = seats[seat]
            obs = {k: v.unsqueeze(0).to(device) for k, v in env.obs().items()}
            legal = env.legal_mask().to(device)
            forced_pass = not bool(legal.any())

            out = net.step(
                obs,
                legal=None if forced_pass else {"place": legal.unsqueeze(0)},
                st_cache=caches[seat],
                step_idx=len(tr["place"]),
                act_steps=act_steps,
            )
            caches[seat] = out["st_cache"]

            if forced_pass:
                place, lp = 0, 0.0
                env.step(None)
            else:
                place = int(out["action_place"].item())
                # place-only log-prob: the pass component is rules-forced,
                # deterministic, and excluded from the policy (see module
                # docstring) — action_log_prob would include the Bernoulli.
                lp = float(masked_log_prob(
                    out["place_logits"].float(),
                    out["action_place"],
                    legal.unsqueeze(0),
                ).item())
                env.step(place)

            tr["obs"].append({k: v.squeeze(0).cpu() for k, v in obs.items()})
            tr["place"].append(place)
            tr["legal"].append(legal.cpu())
            tr["lp"].append(lp)
            tr["value"].append(float(out["value"].item()))
            tr["pmask"].append(not forced_pass)

    def reward(seat: int) -> float:
        if env.winner == 0:
            return 0.0
        return 1.0 if env.winner == seat else -1.0

    out_trajs = []
    for seat, tr in seats.items():
        if not tr["place"]:
            continue
        out_trajs.append({
            "obs": tr["obs"],
            "place": torch.tensor(tr["place"], dtype=torch.long),
            "legal": torch.stack(tr["legal"]),
            "lp": torch.tensor(tr["lp"], dtype=torch.float32),
            "value": torch.tensor(tr["value"], dtype=torch.float32),
            "pmask": torch.tensor(tr["pmask"], dtype=torch.bool),
            "reward": reward(seat),
            "winner": env.winner,
            "plies": env.ply,
        })
    return out_trajs


def collect(net, game, board_size, n_games, act_steps, device):
    trajs = []
    for _ in range(n_games):
        trajs.extend(play_game(net, game, board_size, act_steps, device))
    return trajs


# ----------------------------------------------------------------------
# PPO update
# ----------------------------------------------------------------------

def batch_trajectories(
    trajs: List[Dict], device: torch.device,
    gamma: float = 0.99, lam: float = 0.95,
) -> Dict:
    """Tail-pad per-seat trajectories to [B, T_max, ...] and attach
    GAE advantages/returns (terminal reward only)."""
    B = len(trajs)
    T_max = max(len(t["place"]) for t in trajs)
    obs_keys = trajs[0]["obs"][0].keys()

    def pad(x: torch.Tensor, T: int) -> torch.Tensor:
        out = torch.zeros(T_max, *x.shape[1:], dtype=x.dtype)
        out[:T] = x
        return out

    obs = {
        k: torch.stack([
            pad(torch.stack([f[k] for f in t["obs"]]), len(t["obs"]))
            for t in trajs
        ]).to(device)
        for k in obs_keys
    }
    frame_mask = torch.zeros(B, T_max, dtype=torch.bool)
    adv = torch.zeros(B, T_max)
    ret = torch.zeros(B, T_max)
    for i, t in enumerate(trajs):
        T = len(t["place"])
        frame_mask[i, :T] = True
        rewards = torch.zeros(T)
        rewards[-1] = t["reward"]
        dones = torch.zeros(T, dtype=torch.bool)
        dones[-1] = True
        a, r = compute_gae(rewards, t["value"], dones, gamma=gamma, lam=lam)
        adv[i, :T] = a
        ret[i, :T] = r
    return {
        "obs": obs,
        "place": torch.stack([pad(t["place"], len(t["place"])) for t in trajs]).to(device),
        "legal": torch.stack([pad(t["legal"], len(t["legal"])) for t in trajs]).to(device),
        "old_lp": torch.stack([pad(t["lp"], len(t["lp"])) for t in trajs]).to(device),
        "pmask": (torch.stack([pad(t["pmask"], len(t["pmask"])) for t in trajs])
                  & frame_mask).to(device),
        "frame_mask": frame_mask.to(device),
        "adv": adv.to(device),
        "ret": ret.to(device),
    }


def ppo_update(
    net, batch: Dict, optimizer, *, ppo_epochs: int, clip: float,
    value_coef: float, entropy_coef: float, clip_grad: float,
    act_steps: int, moe_balance_rate: float = 0.0,
) -> Dict[str, float]:
    pmask = batch["pmask"]
    n_pol = pmask.float().sum().clamp(min=1.0)
    n_frames = batch["frame_mask"].float().sum().clamp(min=1.0)

    adv = batch["adv"]
    a_mean = (adv * pmask).sum() / n_pol
    a_std = (((adv - a_mean) ** 2 * pmask).sum() / n_pol).sqrt().clamp(min=1e-6)
    adv = (adv - a_mean) / a_std

    # All-False legal rows (forced pass) are policy-masked out, but the
    # entropy term still evaluates them — give those rows a uniform mask
    # so no NaN enters the graph.
    safe_legal = batch["legal"].clone()
    safe_legal[~safe_legal.any(-1)] = True

    stats = {}
    for _ in range(ppo_epochs):
        pred = net(batch["obs"], act_steps=act_steps,
                   frame_mask=batch["frame_mask"])
        new_lp = masked_log_prob(
            pred["place_logits"].float(), batch["place"], batch["legal"],
        )
        ent = categorical_entropy(
            pred["place_logits"].float().masked_fill(~safe_legal, float("-inf")),
        )

        ratio = torch.exp(new_lp - batch["old_lp"])
        clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
        pol_loss = (-torch.min(ratio * adv, clipped * adv) * pmask).sum() / n_pol
        v_loss = (F.mse_loss(pred["value"].float(), batch["ret"],
                             reduction="none") * batch["frame_mask"]).sum() / n_frames
        ent_bonus = (ent * pmask).sum() / n_pol
        loss = pol_loss + value_coef * v_loss - entropy_coef * ent_bonus

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in net.parameters() if p.requires_grad], clip_grad,
        )
        optimizer.step()
        # MoE routing balance from this step's load (resets the tally; no-op
        # without MoE layers)
        update_moe_balance(net, moe_balance_rate)

        stats = {
            "loss": float(loss.detach()),
            "policy_loss": float(pol_loss.detach()),
            "value_loss": float(v_loss.detach()),
            "entropy": float(ent_bonus.detach()),
            "ratio_dev": float((((ratio - 1.0).abs() * pmask).sum() / n_pol).detach()),
        }
    return stats


# ----------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()

    ckpt = torch.load(args.il_checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("game") != args.game:
        raise ValueError(
            f"checkpoint is for game {ckpt.get('game')!r}, not {args.game!r}"
        )
    arch = resume_arch_overrides(ckpt, {})
    net, dropped = build_net_tolerant(args.game, arch)
    if dropped:
        print(f"[warn] checkpoint config fields unknown to current code, "
              f"ignored: {dropped}")
    net.load_state_dict(ckpt["model_state"], strict=True)
    net = net.to(device)
    cfg = net.cfg
    if args.act_steps <= 0:
        args.act_steps = net.default_depth
    board_size = args.board_size or cfg.board_h
    if (cfg.board_h, cfg.board_w) != (board_size, board_size):
        raise ValueError(
            f"--board_size {board_size} but the checkpoint is "
            f"{cfg.board_h}×{cfg.board_w}"
        )

    # Eval mode for rollout AND update: masked BatchNorm statistics must
    # not drift between the two or the PPO ratio is not exact.
    net.eval()
    if args.freeze_encoder:
        freeze_encoder(net)
    optimizer = torch.optim.AdamW(
        optim_param_groups(net, args.weight_decay), lr=args.lr,
    )
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[init] game={args.game} device={device.type} "
          f"board={board_size} trainable={n_params:,} qk_norm={cfg.qk_norm}")

    os.makedirs(args.save_dir, exist_ok=True)
    for it in range(args.iters):
        t0 = time.time()
        trajs = collect(net, args.game, board_size, args.games_per_iter,
                        args.act_steps, device)
        batch = batch_trajectories(trajs, device,
                                   gamma=args.gamma, lam=args.gae_lambda)
        stats = ppo_update(
            net, batch, optimizer, ppo_epochs=args.ppo_epochs,
            clip=args.clip, value_coef=args.value_coef,
            entropy_coef=args.entropy_coef, clip_grad=args.clip_grad,
            act_steps=args.act_steps, moe_balance_rate=args.moe_balance_rate,
        )
        if it % args.log_every == 0:
            n_games = args.games_per_iter
            p1_wins = sum(1 for t in trajs if t["winner"] == 1) / 2  # 2 trajs/game
            draws = sum(1 for t in trajs if t["winner"] == 0) / 2
            plies = sum(t["plies"] for t in trajs) / max(len(trajs), 1)
            head = " ".join(f"{k}={v:.4f}" for k, v in sorted(stats.items()))
            print(f"[it{it}] {head} p1_win={p1_wins / n_games:.2f} "
                  f"draw={draws / n_games:.2f} plies={plies:.0f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
        if (it + 1) % args.save_interval == 0 or (it + 1) == args.iters:
            path = os.path.join(args.save_dir, f"{args.game}_rl_iter{it + 1}.pt")
            torch.save({
                "game": args.game,
                "iter": it + 1,
                "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg.__dict__,
                "config_overrides": arch,
            }, path)
            print(f"[it{it}] saved {path}")


if __name__ == "__main__":
    main()
