"""PPO self-play fine-tuning for the placement board games.

Loads a ``training/train_il_game.py`` checkpoint (othello / gomoku), plays
through :class:`env.board_selfplay_env.BoardSelfPlayEnv`, and PPO-fine-tunes
the policy + value heads.

    uv run python -m training.train_rl_game --game othello \
        --il_checkpoint checkpoints/il_game/othello_epoch9.pt --iters 200

Board stream convention (shared with IL)
----------------------------------------
One game = ONE interleaved sequence: every ply is a frame, both movers'
frames in play order, each observed from the side to move (mover-POV
planes, mover-POV value).  That is the layout every board IL converter
writes (chess, go, othello, gomoku, shogi) and IL trains on all of its
frames.  RL streams the game the same way — the learner net steps every
ply through ONE Stage-1 cache (``step_idx`` = ply index), so its context
holds both sides' positions and the PPO replay (full ``forward`` over the
[plies] sequence) sees exactly the rollout's context — and puts the
policy/value loss on the learner seats' frames only:

  * ``learner_seats=(1, 2)`` (default): self-play, both seats learn;
  * ``learner_seats=(1,)`` / ``(2,)``: the other seat is played by the
    learner net itself, or by a frozen ``opponent`` net that also steps
    every ply on its own cache (so it keeps the full interleaved history
    too) and acts on its seats.  The learner net still steps on the
    opponent's frames (its replay context must equal the rollout's), but
    those frames carry no loss.

GAE runs per seat over that seat's subsequence of frames (reward 0 except
the seat's last frame = its terminal reward), scattered back into the
[games, plies] batch.  The joint multi-game RL trainer and future pgx
go / chess / shogi environments follow the same convention: one sequence
per game, both movers' frames, loss on the learner seats.

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
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from env.board_selfplay_env import BoardSelfPlayEnv
from model.components.core import update_moe_balance
from model.games.base import categorical_entropy, masked_log_prob
from model.rl_utils import compute_gae, freeze_encoder, optim_param_groups
from training.train_il_game import build_net_tolerant, pick_device, resume_arch_overrides

SEATS = (1, 2)


def parse_seats(spec: str) -> tuple:
    """'1,2' → (1, 2); every seat must be 1 or 2, at least one."""
    seats = tuple(sorted({int(s) for s in spec.split(",") if s.strip()}))
    if not seats or any(s not in SEATS for s in seats):
        raise argparse.ArgumentTypeError(
            f"learner seats must be a non-empty subset of 1,2 (got {spec!r})")
    return seats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True, choices=("othello", "gomoku"))
    p.add_argument("--il_checkpoint", required=True)
    p.add_argument("--board_size", type=int, default=None,
                   help="gomoku only; must match the checkpoint's board_h")
    p.add_argument("--learner_seats", type=parse_seats, default=SEATS,
                   help="seats whose frames carry the PPO loss: '1,2' = "
                        "self-play (default), '1' / '2' = the other seat is "
                        "the opponent (--opponent_checkpoint, else the "
                        "learner net itself)")
    p.add_argument("--opponent_checkpoint", default=None,
                   help="frozen opponent for the non-learner seat (a "
                        "train_il_game / train_rl_game checkpoint of the same "
                        "game); needs a single --learner_seats seat")
    p.add_argument("--opponent_act_steps", type=int, default=0,
                   help="opponent TRM depth (0 = its full depth)")
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
    learner_seats: Sequence[int] = SEATS, opponent=None,
    opponent_act_steps: Optional[int] = None,
) -> Dict:
    """Play one game as ONE interleaved mover-POV sequence (module docstring).

    Returns a dict over the game's T plies (forced passes included):
    ``obs`` (list of per-ply mover-POV obs), ``place`` [T] (0 on forced
    passes), ``legal`` [T,N], ``lp`` [T] (place-only masked log-prob of
    the executed placement on learner, non-forced frames, else 0),
    ``value`` [T] (the learner net's value, mover-POV), ``seat`` [T]
    (mover 1/2), ``pmask`` [T] (learner seat and not a forced pass),
    ``vmask`` [T] (learner seat), ``reward`` {1: r1, 2: r2}, ``winner``,
    ``plies``.

    Non-learner seats act with ``opponent``'s sample when given (it steps
    every ply on its own cache at ``opponent_act_steps``, default
    ``act_steps``), else with the learner net's own sample.
    """
    learners = frozenset(int(s) for s in learner_seats)
    if not learners or not learners <= set(SEATS):
        raise ValueError(f"learner_seats must be a non-empty subset of {SEATS}")
    if opponent is not None and learners == set(SEATS):
        raise ValueError("an opponent needs a non-learner seat to play "
                         "(learner_seats covers both seats)")
    opp_steps = act_steps if opponent_act_steps is None else opponent_act_steps

    env = BoardSelfPlayEnv(game, board_size)
    n_points = env.size * env.size
    rec: Dict[str, list] = {k: [] for k in (
        "obs", "place", "legal", "lp", "value", "seat", "pmask", "vmask")}
    cache = opp_cache = None

    with torch.no_grad():
        while not env.done:
            ply, seat = env.ply, env.to_play
            obs = {k: v.unsqueeze(0).to(device) for k, v in env.obs().items()}
            env_legal = env.legal()
            forced_pass = env_legal is None
            legal = (torch.zeros(n_points, dtype=torch.bool) if forced_pass
                     else env_legal["place"]).to(device)
            step_legal = None if forced_pass else {"place": legal.unsqueeze(0)}

            # the learner streams every frame (its replay context = rollout's)
            out = net.step(obs, legal=step_legal, st_cache=cache,
                           step_idx=ply, act_steps=act_steps)
            cache = out["st_cache"]
            actor = out
            if opponent is not None:
                opp_out = opponent.step(obs, legal=step_legal, st_cache=opp_cache,
                                        step_idx=ply, act_steps=opp_steps)
                opp_cache = opp_out["st_cache"]
                if seat not in learners:
                    actor = opp_out
            learner = seat in learners

            if forced_pass:
                place, lp = 0, 0.0
                env.step(None)
            else:
                action = actor["action_place"]
                place = int(action.item())
                # place-only log-prob: the pass component is rules-forced,
                # deterministic, and excluded from the policy (see module
                # docstring) — action_log_prob would include the Bernoulli.
                lp = float(masked_log_prob(
                    out["place_logits"].float(), action, legal.unsqueeze(0),
                ).item()) if learner else 0.0
                env.step(place)

            rec["obs"].append({k: v.squeeze(0).cpu() for k, v in obs.items()})
            rec["place"].append(place)
            rec["legal"].append(legal.cpu())
            rec["lp"].append(lp)
            rec["value"].append(float(out["value"].item()))
            rec["seat"].append(seat)
            rec["pmask"].append(learner and not forced_pass)
            rec["vmask"].append(learner)

    def reward(s: int) -> float:
        if env.winner == 0:
            return 0.0
        return 1.0 if env.winner == s else -1.0

    return {
        "obs": rec["obs"],
        "place": torch.tensor(rec["place"], dtype=torch.long),
        "legal": torch.stack(rec["legal"]),
        "lp": torch.tensor(rec["lp"], dtype=torch.float32),
        "value": torch.tensor(rec["value"], dtype=torch.float32),
        "seat": torch.tensor(rec["seat"], dtype=torch.long),
        "pmask": torch.tensor(rec["pmask"], dtype=torch.bool),
        "vmask": torch.tensor(rec["vmask"], dtype=torch.bool),
        "reward": {s: reward(s) for s in SEATS},
        "winner": env.winner,
        "plies": env.ply,
    }


def collect(net, game, board_size, n_games, act_steps, device,
            learner_seats: Sequence[int] = SEATS, opponent=None,
            opponent_act_steps: Optional[int] = None) -> List[Dict]:
    """``n_games`` games → one :func:`play_game` dict per game."""
    return [play_game(net, game, board_size, act_steps, device,
                      learner_seats=learner_seats, opponent=opponent,
                      opponent_act_steps=opponent_act_steps)
            for _ in range(n_games)]


# ----------------------------------------------------------------------
# PPO update
# ----------------------------------------------------------------------

def seat_gae(traj: Dict, gamma: float, lam: float):
    """Per-seat GAE of one game: each seat's frames form their own
    trajectory (values = that seat's frames, reward 0 except its last
    frame = the seat's terminal reward, done there).  Returns [T] adv/ret
    with every frame filled from its own seat's computation."""
    T = len(traj["place"])
    adv, ret = torch.zeros(T), torch.zeros(T)
    for s in SEATS:
        idx = (traj["seat"] == s).nonzero().squeeze(1)
        n = idx.numel()
        if n == 0:
            continue
        rewards = torch.zeros(n)
        rewards[-1] = traj["reward"][s]
        dones = torch.zeros(n, dtype=torch.bool)
        dones[-1] = True
        a, r = compute_gae(rewards, traj["value"][idx], dones, gamma=gamma, lam=lam)
        adv[idx] = a
        ret[idx] = r
    return adv, ret


def batch_trajectories(
    trajs: List[Dict], device: torch.device,
    gamma: float = 0.99, lam: float = 0.95,
) -> Dict:
    """Tail-pad per-game sequences to [B=games, T=max plies, ...] and
    attach per-seat GAE advantages/returns (terminal reward only)."""
    B = len(trajs)
    T_max = max(len(t["place"]) for t in trajs)
    obs_keys = trajs[0]["obs"][0].keys()

    def pad(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(T_max, *x.shape[1:], dtype=x.dtype)
        out[:x.shape[0]] = x
        return out

    def stack(key: str) -> torch.Tensor:
        return torch.stack([pad(t[key]) for t in trajs])

    obs = {
        k: torch.stack([pad(torch.stack([f[k] for f in t["obs"]])) for t in trajs]).to(device)
        for k in obs_keys
    }
    frame_mask = torch.zeros(B, T_max, dtype=torch.bool)
    adv = torch.zeros(B, T_max)
    ret = torch.zeros(B, T_max)
    for i, t in enumerate(trajs):
        T = len(t["place"])
        frame_mask[i, :T] = True
        adv[i, :T], ret[i, :T] = seat_gae(t, gamma, lam)
    return {
        "obs": obs,
        "place": stack("place").to(device),
        "legal": stack("legal").to(device),
        "old_lp": stack("lp").to(device),
        "pmask": (stack("pmask") & frame_mask).to(device),
        "vmask": (stack("vmask") & frame_mask).to(device),
        "frame_mask": frame_mask.to(device),
        "adv": adv.to(device),
        "ret": ret.to(device),
        "seat": stack("seat").to(device),
    }


def ppo_update(
    net, batch: Dict, optimizer, *, ppo_epochs: int, clip: float,
    value_coef: float, entropy_coef: float, clip_grad: float,
    act_steps: int, moe_balance_rate: float = 0.0,
) -> Dict[str, float]:
    pmask = batch["pmask"]
    vmask = batch.get("vmask", batch["frame_mask"])
    n_pol = pmask.float().sum().clamp(min=1.0)
    n_val = vmask.float().sum().clamp(min=1.0)

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
                             reduction="none") * vmask).sum() / n_val
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

def load_game_net(path: str, game: str):
    """(net, checkpoint, arch) rebuilt from a train_il_game / train_rl_game
    checkpoint of ``game`` (saved config + legacy-field upgrades)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("game") != game:
        raise ValueError(
            f"{path}: checkpoint is for game {ckpt.get('game')!r}, not {game!r}"
        )
    arch = resume_arch_overrides(ckpt, {})
    net, dropped = build_net_tolerant(game, arch)
    if dropped:
        print(f"[warn] {path}: checkpoint config fields unknown to current "
              f"code, ignored: {dropped}")
    net.load_state_dict(ckpt["model_state"], strict=True)
    return net, ckpt, arch


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()

    net, _, arch = load_game_net(args.il_checkpoint, args.game)
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

    opponent = None
    if args.opponent_checkpoint:
        if set(args.learner_seats) == set(SEATS):
            raise ValueError("--opponent_checkpoint needs a single "
                             "--learner_seats seat (1 or 2)")
        opponent, _, _ = load_game_net(args.opponent_checkpoint, args.game)
        ocfg = opponent.cfg
        if (ocfg.board_h, ocfg.board_w) != (board_size, board_size):
            raise ValueError(f"opponent is {ocfg.board_h}×{ocfg.board_w}, "
                             f"the learner {board_size}×{board_size}")
        opponent = opponent.to(device).eval().requires_grad_(False)
        if args.opponent_act_steps <= 0:
            args.opponent_act_steps = opponent.default_depth

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
          f"board={board_size} trainable={n_params:,} qk_norm={cfg.qk_norm} "
          f"learner_seats={','.join(map(str, args.learner_seats))} "
          f"opponent={args.opponent_checkpoint or 'self'}")

    os.makedirs(args.save_dir, exist_ok=True)
    for it in range(args.iters):
        t0 = time.time()
        trajs = collect(net, args.game, board_size, args.games_per_iter,
                        args.act_steps, device, learner_seats=args.learner_seats,
                        opponent=opponent,
                        opponent_act_steps=args.opponent_act_steps or None)
        batch = batch_trajectories(trajs, device,
                                   gamma=args.gamma, lam=args.gae_lambda)
        stats = ppo_update(
            net, batch, optimizer, ppo_epochs=args.ppo_epochs,
            clip=args.clip, value_coef=args.value_coef,
            entropy_coef=args.entropy_coef, clip_grad=args.clip_grad,
            act_steps=args.act_steps, moe_balance_rate=args.moe_balance_rate,
        )
        if it % args.log_every == 0:
            n_games = max(len(trajs), 1)                  # one sequence per game
            p1_win = sum(1 for t in trajs if t["winner"] == 1) / n_games
            draw = sum(1 for t in trajs if t["winner"] == 0) / n_games
            plies = sum(t["plies"] for t in trajs) / n_games
            learner = ""
            if len(args.learner_seats) == 1:
                s = args.learner_seats[0]
                learner = (" learner_win="
                           f"{sum(t['reward'][s] > 0 for t in trajs) / n_games:.2f}")
            head = " ".join(f"{k}={v:.4f}" for k, v in sorted(stats.items()))
            print(f"[it{it}] {head} p1_win={p1_win:.2f} draw={draw:.2f}"
                  f"{learner} plies={plies:.0f} ({time.time() - t0:.1f}s)",
                  flush=True)
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
