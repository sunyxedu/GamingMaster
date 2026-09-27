"""
Imitation Learning training script for the generals.io bot.

Usage:
    # Single GPU:
    python3 -m training.train_il --data_dir <path_to_gio_data> [options]

    # Multi-GPU (torchrun + manual grad sync):
    torchrun --nproc_per_node=2 --module training.train_il --data_dir <path_to_gio_data> [options]
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_processing.dataset_seq import SeqReplayDataset, collate_variable_length_sequences
from model.config import ModelConfig, checkpoint_config
from model.games.generals import apply_memory_vision
from model.losses import ImitationLoss, next_obs_loss
from model.network import GeneralsNetwork
from model.optim import (
    TRM_BETAS,
    TRM_EMA_RATE,
    TRM_LR,
    TRM_LR_MIN_RATIO,
    TRM_WARMUP_STEPS,
    TRM_WEIGHT_DECAY,
    AdamATan2,
    ModelEMA,
    trm_lr_lambda,
)
from model.rl_utils import load_model_state, sync_gradients
from model.components.core import allreduce_moe_loads, update_moe_balance
from training.deep_supervision import BatchCarry, split_rows


def parse_args():
    p = argparse.ArgumentParser(description="IL training for generals.io bot")
    p.add_argument("--data_dir", type=str, nargs="+", required=True, help="Directory(s) with .json training data")
    p.add_argument(
        "--output_dir", type=str, default="checkpoints", help="Checkpoint output dir"
    )
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Per-GPU micro-batch (full-game sequences)")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps; effective batch = "
                        "batch_size x world_size x grad_accum")
    # Optimisation defaults = reference TRM recipe (model/optim.py):
    # AdamATan2, lr 1e-4, betas (0.9, 0.95), wd 0.1, 2000 warmup steps,
    # constant LR afterwards (lr_min_ratio 1.0), EMA 0.999.
    p.add_argument("--lr", type=float, default=TRM_LR)
    p.add_argument("--warmup_steps", type=int, default=TRM_WARMUP_STEPS,
                   help="Linear LR warmup (optimizer steps)")
    p.add_argument("--weight_decay", type=float, default=TRM_WEIGHT_DECAY)
    p.add_argument("--beta1", type=float, default=TRM_BETAS[0])
    p.add_argument("--beta2", type=float, default=TRM_BETAS[1])
    p.add_argument("--lr_min_ratio", type=float, default=TRM_LR_MIN_RATIO,
                   help="cosine floor after warmup; 1.0 = constant LR (reference)")
    p.add_argument("--optimizer", choices=["adam_atan2", "adamw"], default="adam_atan2")
    p.add_argument("--moe_balance_rate", type=float, default=1e-3,
                   help="γ of the MoE aux-loss-free balance-bias update "
                        "(DeepSeek-V3), applied once per optimizer step")
    p.add_argument("--ema_rate", type=float, default=TRM_EMA_RATE,
                   help="EMA of the weights (evaluation + saved model_state); 0 = off")
    p.add_argument("--max_seq_len", type=int, default=1000, help="Max sequence length per sample")
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--trm_H_cycles", type=int, default=3)
    p.add_argument("--trm_L_cycles", type=int, default=6)
    p.add_argument("--trm_block_layers", type=int, default=2)
    p.add_argument("--log_interval", type=int, default=10, help="Log every N steps")
    p.add_argument("--save_interval", type=int, default=1, help="Save every N epochs")
    p.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps within epoch")
    p.add_argument("--grad_clip", type=float, default=0.0,
                   help="max grad norm; 0 = no clipping (the reference TRM "
                        "does not clip; the non-finite check runs either way)")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--tf_start", type=float, default=0.0, help="Teacher forcing ratio at epoch 1")
    p.add_argument("--tf_end", type=float, default=0.0, help="Teacher forcing ratio at last epoch")
    p.add_argument("--focal_gamma", type=float, default=1.5, help="Focal loss gamma for source CE")
    p.add_argument("--reset_scheduler", action="store_true", help="Don't load scheduler state from checkpoint")
    p.add_argument("--reinit", nargs="*", default=[],
                   help="Submodule prefixes to re-initialize when loading "
                        "--finetune_from (e.g. 'direction_head.' after an "
                        "architecture change); other weights load strictly.")
    p.add_argument("--finetune_from", type=str, default=None,
                   help="Load model WEIGHTS + architecture from this checkpoint "
                        "but start a fresh run (epoch 1, new optimizer/schedule). "
                        "Use for phase-2 fine-tuning; distinct from --resume, "
                        "which continues an interrupted run.")
    p.add_argument(
        "--halt_max_steps", type=int, default=0,
        help="N_sup: max deep-supervision steps per frame (0 = model config, "
             "reference 16). Inference runs all of them.",
    )
    p.add_argument(
        "--temporal_mixer", type=str, default="attention",
        choices=["attention", "recurrent"],
        help="Temporal mixer: full-history causal attention (KV cache) or "
             "gated linear recurrence (fixed-size neural memory, full BPTT)",
    )
    p.add_argument(
        "--world_model_weight", type=float, default=0.0,
        help="Weight of the self-supervised next-observation loss (memory pressure "
             "on the temporal state; 0 = off). Adds a next_obs head to the model — "
             "checkpoints trained without it cannot be resumed with it enabled.",
    )
    return p.parse_args()


PART_KEYS = ["source", "direction", "split", "pass", "value", "q_halt", "wm"]
ACC_KEYS = ["src_acc", "dir_acc", "full_acc", "complete_acc"]


def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, args,
                rank0=True, save_fn=None, start_step=0, tf_ratio=1.0, distributed=False,
                ema=None, carries=None):
    """One epoch in the reference ``pretrain.py`` layout.

    Every loader batch is ONE TRM supervision (ACT) step and ONE optimizer
    step: ``carries`` (one training.deep_supervision.BatchCarry per
    accumulation slot = the reference ``train_state.carry``, persisting
    across epochs) swaps halted rows for the new batch's rows, runs one
    deep recursion from the carried latents, and halts frames by q_halt
    (+ exploration). ``grad_accum`` consecutive loader batches feed the
    ``grad_accum`` slots of one optimizer step.
    """
    model.train()
    from collections import deque
    window = args.log_interval
    recent_loss = deque(maxlen=window)
    recent_parts = {k: deque(maxlen=window) for k in PART_KEYS}
    recent_acc = {k: deque(maxlen=window) for k in ACC_KEYS}
    recent_halt = deque(maxlen=window)
    epoch_loss_sum = 0.0
    epoch_parts_sum = {k: 0.0 for k in PART_KEYS}
    epoch_acc_sum = {k: 0.0 for k in ACC_KEYS}
    epoch_steps = 0
    use_amp = device.type == "cuda"
    accum = max(1, getattr(args, "grad_accum", 1))
    optimizer.zero_grad(set_to_none=True)

    def prepare_rows(batch):
        """Batch → per-row data; the teacher-forcing draw is part of the
        example (kept for all of its supervision steps)."""
        tgt = {k: v.to(device) for k, v in batch["tgt"].items()}
        obs = {k: v.to(device) for k, v in batch["obs"].items()}
        if getattr(model.cfg, "memory_vision", False):
            # the observation a memory_vision network plays on, derived
            # causally from the records' pure-fog frames
            obs["grid_seq"] = apply_memory_vision(obs["grid_seq"])
        aux = {}
        if tf_ratio > 0:
            source_y = tgt["src_y"][:, :, 0]
            source_x = tgt["src_x"][:, :, 0]
            move_mask = tgt["count"] > 0
            tf_mask = torch.rand_like(source_y, dtype=torch.float32) < tf_ratio
            use_gt = tf_mask & move_mask
            aux = {"source_y": torch.where(use_gt, source_y, torch.full_like(source_y, -1)),
                   "source_x": torch.where(use_gt, source_x, torch.full_like(source_x, -1)),
                   "use_gt": use_gt}
        return split_rows({"obs": obs, "tgt": tgt, "aux": aux,
                           "meta": {"lengths": batch["meta"]["lengths"]}})

    def forward_backward(mb, carry, active, reset, weight):
        aux = mb.get("aux", {})
        grid_seq = mb["obs"]["grid_seq"]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred = model(
                grid_seq, mb["obs"]["globals_seq"],
                legal_mask=mb["obs"]["legal_source_mask_seq"],
                source_y=aux.get("source_y"), source_x=aux.get("source_x"),
                frame_mask=mb["meta"]["frame_mask"],
                carry=carry, active=active, reset=reset, return_carry=True,
            )
            losses = criterion(pred, mb["tgt"], tf_mask=aux.get("use_gt"), frame_mask=active)
            if args.world_model_weight > 0 and "next_state_logits" in pred:
                wm = next_obs_loss(pred, grid_seq, active)
                wm_total = wm["state_loss"] + wm["army_loss"]
                losses["loss"] = losses["loss"] + args.world_model_weight * wm_total
                losses["wm_loss"] = wm_total.detach()
            else:
                losses["wm_loss"] = losses["loss"].new_tensor(0.0)
        loss = losses["loss"]
        # A non-finite micro-batch is dropped rank-locally; the gradient
        # sync below stays collective (one per optimizer step on every rank).
        if torch.isfinite(loss.detach()):
            # weight = n_active / n_frames: the reference's fixed
            # global-batch normalisation (training/deep_supervision.py)
            (loss * (weight / accum)).backward()
            stats = {"loss": loss.item()}
            stats.update({f"{k}_loss": losses[f"{k}_loss"].item() for k in PART_KEYS})
            stats.update({k: losses[k].item() for k in ACC_KEYS})
        else:
            stats = {}
            if rank0:
                print(f"  [NaN] dropped a micro-batch (T={grid_seq.shape[1]})")
        return pred, stats

    def optimizer_step():
        if distributed:
            # replaces DDP's bucketed allreduce; grads are averaged so every
            # rank steps identically. sync_gradients keeps the collective
            # schedule identical even when a rank has no gradient for a
            # parameter.
            sync_gradients(model, dist.get_world_size())
        # max_norm=inf only MEASURES the norm (clip coefficient clamps to 1),
        # so the non-finite guard below works with clipping off too
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf"))
        if not torch.isfinite(grad_norm):
            # grads are bitwise-equal across ranks after the sync, so every
            # rank takes this branch together
            optimizer.zero_grad(set_to_none=True)
            if rank0:
                print(f"  [NaN] dropped an optimizer step (grad_norm={grad_norm.item():.4f})")
            return
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)
        # MoE aux-loss-free balancing over the whole step (all micro-batches)
        # on the load summed over the ranks: buffers are broadcast only at
        # startup, so rank-local loads would let balance_bias drift apart
        # (no-op single-process / without MoE layers)
        allreduce_moe_loads(model)
        update_moe_balance(model, args.moe_balance_rate)

    n_batches = len(loader)
    slot = 0
    window_stats = []
    for step, batch in enumerate(loader):
        if step < start_step:
            continue
        T_cur = batch["obs"]["grid_seq"].shape[1]
        try:
            st = carries[slot].step(prepare_rows(batch), forward_backward)
        except torch.cuda.OutOfMemoryError:
            if distributed:
                # A rank-local skip desynchronizes the gradient allreduce.
                raise
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            slot = 0
            window_stats = []
            if rank0:
                print(f"  [OOM] Skipped step {step+1} (T={T_cur})")
            continue
        if st is not None and "loss" in st:
            window_stats.append(st)
        slot += 1
        if slot < accum and step + 1 < n_batches:
            continue
        slot = 0
        optimizer_step()
        if not window_stats:
            continue
        stats = {k: sum(w[k] for w in window_stats) / len(window_stats)
                 for k in window_stats[0] if k != "halt_depth"}
        depths = [w["halt_depth"] for w in window_stats if w["halt_depth"] == w["halt_depth"]]
        window_stats = []

        recent_loss.append(stats["loss"])
        if depths:
            recent_halt.append(sum(depths) / len(depths))
        epoch_loss_sum += stats["loss"]
        epoch_steps += 1
        for k in recent_parts:
            v = stats[f"{k}_loss"]
            recent_parts[k].append(v)
            epoch_parts_sum[k] += v
        for k in recent_acc:
            v = stats[k]
            recent_acc[k].append(v)
            epoch_acc_sum[k] += v

        if rank0 and epoch_steps % max(1, args.log_interval // accum) == 0:
            avg = sum(recent_loss) / len(recent_loss)
            parts = {k: sum(v) / len(v) for k, v in recent_parts.items()}
            acc = {k: sum(v) / len(v) for k, v in recent_acc.items()}
            ep_avg = epoch_loss_sum / epoch_steps
            ep_parts = {k: epoch_parts_sum[k] / epoch_steps for k in epoch_parts_sum}
            ep_acc = {k: epoch_acc_sum[k] / epoch_steps for k in epoch_acc_sum}
            depth = (sum(recent_halt) / len(recent_halt)) if recent_halt else float("nan")
            print(
                f"  Epoch {epoch} Step {step+1}/{n_batches}  T={T_cur}  loss={avg:.4f}[{ep_avg:.4f}]  "
                f"active={stats['active_frac']:.2f} halt_depth={depth:.2f}:\n"
                f"    src={parts['source']:.4f}[{ep_parts['source']:.4f}]  "
                f"dir={parts['direction']:.4f}[{ep_parts['direction']:.4f}]  split={parts['split']:.4f}[{ep_parts['split']:.4f}]  "
                f"pass={parts['pass']:.4f}[{ep_parts['pass']:.4f}]  val={parts['value']:.4f}[{ep_parts['value']:.4f}]  "
                f"qhalt={parts['q_halt']:.4f}[{ep_parts['q_halt']:.4f}]  wm={parts['wm']:.4f}[{ep_parts['wm']:.4f}]\n"
                f"    acc: src={acc['src_acc']:.1%}[{ep_acc['src_acc']:.1%}]  "
                f"dir={acc['dir_acc']:.1%}[{ep_acc['dir_acc']:.1%}]  "
                f"full={acc['full_acc']:.1%}[{ep_acc['full_acc']:.1%}]  "
                f"complete={acc['complete_acc']:.1%}[{ep_acc['complete_acc']:.1%}]"
            )

        if save_fn and rank0 and (step + 1) // args.save_steps != (step + 1 - accum) // args.save_steps:
            save_fn(epoch, step + 1)

    avg_loss = epoch_loss_sum / max(epoch_steps, 1)
    return avg_loss


def main():
    args = parse_args()

    # Distributed setup (auto-detected via torchrun)
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        device = (torch.device(f"cuda:{local_rank}")
                  if torch.cuda.is_available() else torch.device("cpu"))
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    rank0 = not distributed or dist.get_rank() == 0
    if rank0:
        suffix = f" (world_size={dist.get_world_size()})" if distributed else ""
        print(f"Device: {device}{suffix}")

    if args.resume and args.finetune_from:
        raise SystemExit("use either --resume or --finetune_from, not both")

    # A resume/finetune checkpoint defines the architecture — rebuilding from
    # CLI defaults would silently drop heads or switch the temporal mixer.
    resume_ckpt = None
    arch_ckpt_path = args.resume or args.finetune_from
    if arch_ckpt_path:
        resume_ckpt = torch.load(arch_ckpt_path, map_location="cpu", weights_only=False)

    if resume_ckpt is not None and "config" in resume_ckpt:
        cfg = checkpoint_config(resume_ckpt)
        # Enabling the world-model loss on a lineage without the head fails
        # loudly at weight load (documented in --world_model_weight help).
        cfg.predict_next_obs = cfg.predict_next_obs or (args.world_model_weight > 0)
        if rank0:
            print(f"Architecture from checkpoint {arch_ckpt_path}")
            if cfg.temporal_mixer != args.temporal_mixer:
                print(f"  [note] --temporal_mixer {args.temporal_mixer} ignored; "
                      f"checkpoint uses {cfg.temporal_mixer!r}")
    else:
        cfg = ModelConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            trm_H_cycles=args.trm_H_cycles,
            trm_L_cycles=args.trm_L_cycles,
            trm_block_layers=args.trm_block_layers,
            temporal_mixer=args.temporal_mixer,
            predict_next_obs=(args.world_model_weight > 0),
        )
    if args.halt_max_steps > 0:
        cfg.halt_max_steps = args.halt_max_steps
    if rank0:
        print(f"Model config: d={cfg.d_model}, heads={cfg.n_heads}, core={cfg.core_type}, "
              f"moe_experts={cfg.moe_experts}, qk_norm={cfg.qk_norm}")

    model = GeneralsNetwork(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if distributed:
        # Manual gradient sync instead of DDP: IL trains at a RANDOM ACT
        # depth per iteration (deep supervision), which DDP's reducer rejects
        # (static_graph) or trips over (checkpoint recompute alignment). For a
        # 16M model one grads-allreduce per optimizer step costs microseconds
        # and is correct for ANY graph. Replicate DDP's initial sync here:
        import itertools
        for t in itertools.chain(model.parameters(), model.buffers()):
            dist.broadcast(t.data, src=0)
    if rank0:
        print(f"Model parameters: {n_params:,}")

    # q_halt is a TRM-only head (other cores emit a constant 0)
    criterion = ImitationLoss(grid_w=cfg.grid_w, focal_gamma=args.focal_gamma,
                              q_halt_weight=0.5 if model.core_type == "trm" else 0.0)
    from model.rl_utils import optim_param_groups
    groups = optim_param_groups(model, args.weight_decay)
    if args.optimizer == "adam_atan2":
        optimizer = AdamATan2(groups, lr=args.lr, betas=(args.beta1, args.beta2),
                              weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2))

    if len(args.data_dir) == 1:
        dataset = SeqReplayDataset(args.data_dir[0], max_sequence_length=args.max_seq_len)
    else:
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset([
            SeqReplayDataset(d, max_sequence_length=args.max_seq_len)
            for d in args.data_dir
        ])
    if rank0:
        print(f"Dataset: {len(dataset)} replays from {args.data_dir}")

    world_size = dist.get_world_size() if distributed else 1
    # Scheduler horizon counts OPTIMIZER steps (one per accumulation window
    # of loader batches — reference total_steps = epochs * N / global_batch)
    steps_per_epoch = len(dataset) // (args.batch_size * world_size * max(1, args.grad_accum))
    if rank0 and args.grad_accum > 1:
        print(f"Effective batch: {args.batch_size} x {world_size} GPUs x "
              f"{args.grad_accum} accum = {args.batch_size * world_size * args.grad_accum} sequences")

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_variable_length_sequences,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        # Keep workers alive across epochs: skips per-epoch respawn AND the
        # multiprocessing temp-dir teardown that spams harmless NFS
        # "Device or resource busy" tracebacks at every epoch boundary.
        persistent_workers=(args.num_workers > 0),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    start_epoch = 1
    start_step = 0
    if args.finetune_from:
        # Load weights only; everything else (epoch counter, optimizer state,
        # LR schedule) starts fresh — this is a NEW run seeded from phase 1.
        raw_model = model
        load_model_state(raw_model, resume_ckpt["model_state"], ctx="finetune checkpoint",
                         reinit_prefixes=tuple(args.reinit))
        if rank0:
            print(f"Fine-tuning from {args.finetune_from} "
                  f"(fresh epoch/optimizer/schedule)")
    elif args.resume:
        ckpt = resume_ckpt  # loaded above, where it defined the architecture
        raw_model = model
        load_model_state(raw_model, ckpt["model_state"], ctx="resume checkpoint")
        # Only reuse optimizer state from IL checkpoints whose trainable set
        # matches — RL checkpoints also carry optimizer_state, but it targets
        # a different objective and (in the frozen-encoder phase) a parameter
        # subset, which would crash or mislead AdamW here. Optimizer state is
        # matched by POSITION, so a checkpoint saved before the game-adapter
        # refactor (different registration order, same counts) must be
        # rejected too, not just count mismatches.
        if "optimizer_state" in ckpt:
            from model.rl_utils import is_legacy_state_dict, optim_state_mismatch
            saved = ckpt["optimizer_state"]
            if is_legacy_state_dict(ckpt.get("model_state", {}), raw_model):
                reason = "pre-refactor checkpoint (parameter order changed)"
            else:
                reason = optim_state_mismatch(optimizer, saved)
            if reason is None and ckpt.get("optimizer_name", "adamw") != args.optimizer:
                reason = (f"saved by {ckpt.get('optimizer_name', 'adamw')}, "
                          f"now --optimizer {args.optimizer}")
            if "epoch" in ckpt and reason is None:
                optimizer.load_state_dict(saved)
            elif rank0:
                print(f"  [warn] optimizer state skipped "
                      f"({reason or 'non-IL checkpoint'}) — starting fresh")
        start_step = ckpt.get("step", 0)
        if "epoch" in ckpt:
            if start_step > 0:
                start_epoch = ckpt["epoch"]
            else:
                start_epoch = ckpt["epoch"] + 1
        if rank0:
            if "epoch" in ckpt:
                msg = f"Resumed from {args.resume} (epoch {ckpt['epoch']}, loss={ckpt.get('loss', 0):.4f}"
                if start_step > 0:
                    msg += f", step {start_step}"
                print(msg + ")")
            else:
                # RL checkpoint — just report iteration if available
                it = ckpt.get("iteration", "?")
                print(f"Loaded weights from RL checkpoint {args.resume} (iteration {it})")

    # EMA of the weights (reference ema=True, rate 0.999): evaluation and the
    # saved ``model_state`` use it; ``train_model_state`` keeps the raw
    # weights so --resume continues the exact optimisation.
    ema = ModelEMA(model, mu=args.ema_rate) if args.ema_rate > 0 else None
    if args.resume and resume_ckpt is not None and "train_model_state" in resume_ckpt:
        if ema is not None:
            ema.load_state_dict({k: v.to(device) for k, v in resume_ckpt["model_state"].items()})
        load_model_state(model, resume_ckpt["train_model_state"], ctx="resume (raw weights)")

    # Build LR scheduler after resume so we know start_epoch
    remaining_epochs = args.epochs - start_epoch + 1
    if args.reset_scheduler or not args.resume:
        # Cosine covers only the remaining epochs
        total_steps = steps_per_epoch * remaining_epochs
    else:
        total_steps = steps_per_epoch * args.epochs
    warmup_steps = args.warmup_steps
    # Reference schedule: linear warmup, then cosine towards lr_min_ratio
    # (1.0 = constant, the reference default).
    lr_lambda = trm_lr_lambda(warmup_steps, total_steps, args.lr_min_ratio)

    # Force optimizer to use the CLI lr (load_state_dict restores old initial_lr)
    if args.reset_scheduler:
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr
            pg["initial_lr"] = args.lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.resume and not args.reset_scheduler:
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])

    if rank0:
        print(f"LR schedule: {remaining_epochs} epochs, {total_steps} steps, "
              f"warmup={warmup_steps}, reset={args.reset_scheduler}")

    def save_checkpoint(epoch, step=0, loss=0.0):
        raw = model
        ckpt = {
            "epoch": epoch,
            "step": step,
            # deployment weights = EMA (reference saves the EMA copy)
            "model_state": (ema.ema_model_state(raw) if ema is not None
                            else raw.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "optimizer_name": args.optimizer,
            "scheduler_state": scheduler.state_dict(),
            "config": cfg.__dict__,
            "loss": loss,
        }
        if ema is not None:
            ckpt["train_model_state"] = raw.state_dict()
        if step > 0:
            path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}_step{step}.pt")
        else:
            path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(ckpt, path)
        print(f"  Saved {path}")

    # Reference train_state.carry: one per accumulation slot, persisting
    # across batches AND epochs. With accumulation the other slots' TRM
    # latents are parked on the host.
    store = (torch.device("cpu")
             if max(1, args.grad_accum) > 1 and device.type == "cuda" else None)
    carries = [BatchCarry(model.supervision_steps, cfg.halt_exploration_prob, store=store)
               for _ in range(max(1, args.grad_accum))]

    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        # Linear decay: tf_start at epoch 1 → tf_end at last epoch
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        tf_ratio = args.tf_start + (args.tf_end - args.tf_start) * progress
        t0 = time.time()
        avg_loss = train_epoch(
            model, loader, criterion, optimizer, scheduler, device, epoch, args,
            rank0=rank0, save_fn=save_checkpoint, start_step=start_step,
            tf_ratio=tf_ratio, distributed=distributed, ema=ema,
            carries=carries,
        )
        start_step = 0  # only skip steps for the first epoch after resume
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        if rank0:
            print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  lr={lr:.6f}  "
                  f"tf={tf_ratio:.2f}  time={elapsed:.1f}s")

        if epoch % args.save_interval == 0 and rank0:
            save_checkpoint(epoch, loss=avg_loss)

    if rank0:
        print("Training complete.")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
