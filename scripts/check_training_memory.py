"""Diagnose attention backend and training-step GPU memory for long sequences.

Examples:
    python3 scripts/check_training_memory.py sdpa --batch-size 3 --seq-len 1000
    python3 scripts/check_training_memory.py model --batch-size 3 --seq-len 1000
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import sys
import time
from typing import Iterator

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.config import ModelConfig
from model.losses import ImitationLoss
from model.network import GeneralsNetwork

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover
    from torch.backends.cuda import SDPBackend, sdp_kernel as sdpa_kernel


def gib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024 ** 3)


def fmt_gib(num_bytes: int | float) -> str:
    return f"{gib(num_bytes):.2f} GiB"


def print_cuda_info() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Total VRAM: {fmt_gib(props.total_memory)}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"Mem-efficient SDP enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"Math SDP enabled: {torch.backends.cuda.math_sdp_enabled()}")


def reset_cuda_state() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


@contextlib.contextmanager
def force_backend(name: str) -> Iterator[None]:
    if name == "default":
        yield
        return
    backend = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }[name]
    # set_priority is a torch>=2.6 kwarg; the backend list alone forces the
    # kernel on older torch (the training container ships 2.5). Probe the
    # signature once so a body exception can't be mistaken for it.
    import inspect
    if "set_priority" in inspect.signature(sdpa_kernel).parameters:
        cm = sdpa_kernel([backend], set_priority=True)
    else:
        cm = sdpa_kernel([backend])
    with cm:
        yield


def alloc_training_batch(cfg: ModelConfig, batch_size: int, seq_len: int, device: torch.device):
    H, W = cfg.grid_h, cfg.grid_w
    B, T = batch_size, seq_len
    K = 4

    grid_seq = torch.randn(
        B, T, cfg.grid_channels, H, W, device=device, dtype=torch.float32,
    )
    globals_seq = torch.randn(
        B, T, cfg.global_features, device=device, dtype=torch.float32,
    )
    legal_mask = torch.ones(B, T, 1, H, W, device=device, dtype=torch.bool)

    count = torch.ones(B, T, device=device, dtype=torch.long)
    src_y = torch.zeros(B, T, K, device=device, dtype=torch.long)
    src_x = torch.zeros(B, T, K, device=device, dtype=torch.long)
    dir_id = torch.zeros(B, T, K, device=device, dtype=torch.long)
    ratio_id = torch.zeros(B, T, K, device=device, dtype=torch.long)
    value = torch.zeros(B, T, device=device, dtype=torch.float32)

    # Mirror training/train_il.py: these targets are moved to GPU even though IL loss
    # currently does not use them.
    enemy_heatmap = torch.zeros(B, T, 1, H, W, device=device, dtype=torch.float32)
    next_visibility = torch.zeros(B, T, 1, H, W, device=device, dtype=torch.float32)

    tgt = {
        "count": count,
        "src_y": src_y,
        "src_x": src_x,
        "dir_id": dir_id,
        "ratio_id": ratio_id,
        "value": value,
        "enemy_heatmap": enemy_heatmap,
        "next_visibility": next_visibility,
    }
    lengths = [T] * B
    source_y = src_y[:, :, 0]
    source_x = src_x[:, :, 0]
    return grid_seq, globals_seq, legal_mask, tgt, lengths, source_y, source_x


def run_sdpa_case(name: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> None:
    print(f"\n[{name}]")
    try:
        reset_cuda_state()
        start = time.time()
        with force_backend(name):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
                loss = out.square().mean()
            loss.backward()
        torch.cuda.synchronize()
        print(f"status: ok")
        print(f"peak allocated: {fmt_gib(torch.cuda.max_memory_allocated())}")
        print(f"peak reserved : {fmt_gib(torch.cuda.max_memory_reserved())}")
        print(f"elapsed       : {time.time() - start:.2f}s")
    except torch.cuda.OutOfMemoryError as exc:
        print(f"status: oom")
        print(f"error: {exc}")
    except Exception as exc:  # backend unsupported, dtype mismatch, etc.
        print(f"status: error")
        print(f"error: {type(exc).__name__}: {exc}")
    finally:
        reset_cuda_state()


def cmd_sdpa(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for sdpa diagnostics.")

    device = torch.device("cuda")
    B = args.batch_size
    T = args.seq_len
    H = args.grid_h
    W = args.grid_w
    S = H * W + 2
    heads = args.n_heads
    d_model = args.d_model
    d_head = d_model // heads

    print_cuda_info()
    print(f"\nshape summary: B={B}, T={T}, H={H}, W={W}, S={S}, d={d_model}, heads={heads}")

    spatial_batch = B * T
    temporal_batch = B * S

    print("\nspatial attention shape: "
          f"q/k/v = [{spatial_batch}, {heads}, {S}, {d_head}]")
    q = torch.randn(spatial_batch, heads, S, d_head, device=device, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    for name in args.backends:
        run_sdpa_case(name, q, k, v, is_causal=False)
    del q, k, v
    reset_cuda_state()

    print("\ntemporal attention shape: "
          f"q/k/v = [{temporal_batch}, {heads}, {T}, {d_head}]")
    q = torch.randn(temporal_batch, heads, T, d_head, device=device, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    for name in args.backends:
        run_sdpa_case(name, q, k, v, is_causal=True)
    del q, k, v
    reset_cuda_state()


def cmd_model(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for model diagnostics.")

    device = torch.device("cuda")
    cfg = ModelConfig(
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_st_layers=args.n_st_layers,
        trm_H_cycles=args.trm_H_cycles,
        trm_L_cycles=args.trm_L_cycles,
        trm_block_layers=args.trm_block_layers,
        encoder_channels=args.encoder_channels,
        temporal_type=args.temporal_type,
        temporal_mixer=args.temporal_mixer,
        predict_next_obs=args.predict_next_obs,
        dropout=args.dropout,
    )

    print_cuda_info()
    print("\nmodel config:")
    print(f"  temporal_type   = {cfg.temporal_type}")
    print(f"  temporal_mixer  = {cfg.temporal_mixer}")
    print(f"  predict_next_obs= {cfg.predict_next_obs}")
    print(f"  d_model         = {cfg.d_model}")
    print(f"  n_heads         = {cfg.n_heads}")
    print(f"  d_ff            = {cfg.d_ff}")
    print(f"  n_st_layers     = {cfg.n_st_layers}")
    print(f"  trm_H_cycles    = {cfg.trm_H_cycles}")
    print(f"  trm_L_cycles    = {cfg.trm_L_cycles}")
    print(f"  trm_block_layers= {cfg.trm_block_layers}")
    print(f"  encoder_channels= {cfg.encoder_channels}")
    print(f"  dropout         = {cfg.dropout}")
    print(f"  batch_size      = {args.batch_size}")
    print(f"  seq_len         = {args.seq_len}")

    for backend in args.backends:
        print(f"\n[backend={backend}]")
        model = None
        criterion = None
        optimizer = None
        try:
            reset_cuda_state()
            model = GeneralsNetwork(cfg).to(device)
            criterion = ImitationLoss(grid_w=cfg.grid_w).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"params          : {n_params:,}")
            print(f"after init alloc: {fmt_gib(torch.cuda.memory_allocated())}")
            print(f"after init reserv: {fmt_gib(torch.cuda.memory_reserved())}")

            batch = alloc_training_batch(cfg, args.batch_size, args.seq_len, device)
            print(f"after batch alloc: {fmt_gib(torch.cuda.memory_allocated())}")

            grid_seq, globals_seq, legal_mask, tgt, lengths, source_y, source_x = batch
            optimizer.zero_grad(set_to_none=True)

            with force_backend(backend):
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(
                        grid_seq, globals_seq,
                        legal_mask=legal_mask,
                        source_y=source_y,
                        source_x=source_x,
                    )
                    loss = criterion(pred, tgt, lengths)["loss"]
                torch.cuda.synchronize()
                print(f"after forward alloc : {fmt_gib(torch.cuda.memory_allocated())}")
                print(f"after forward peak  : {fmt_gib(torch.cuda.max_memory_allocated())}")
                print(f"pred dtype source   : {pred['source_logits'].dtype}")
                print(f"pred dtype value    : {pred['value'].dtype}")

                loss.backward()
                torch.cuda.synchronize()
                print(f"after backward alloc: {fmt_gib(torch.cuda.memory_allocated())}")
                print(f"after backward peak : {fmt_gib(torch.cuda.max_memory_allocated())}")

                optimizer.step()
                torch.cuda.synchronize()
                print(f"after step alloc    : {fmt_gib(torch.cuda.memory_allocated())}")
                print(f"after step peak     : {fmt_gib(torch.cuda.max_memory_allocated())}")
                print(f"peak reserved       : {fmt_gib(torch.cuda.max_memory_reserved())}")
        except torch.cuda.OutOfMemoryError as exc:
            print("status: oom")
            print(f"error : {exc}")
            print(f"peak allocated: {fmt_gib(torch.cuda.max_memory_allocated())}")
            print(f"peak reserved : {fmt_gib(torch.cuda.max_memory_reserved())}")
        except Exception as exc:
            print("status: error")
            print(f"error : {type(exc).__name__}: {exc}")
        finally:
            del optimizer, criterion, model
            reset_cuda_state()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--batch-size", type=int, default=3)
    common.add_argument("--seq-len", type=int, default=1000)
    common.add_argument("--grid-h", type=int, default=25)
    common.add_argument("--grid-w", type=int, default=25)
    # Defaults mirror ModelConfig so the probe measures the real model size
    common.add_argument("--d-model", type=int, default=384)
    common.add_argument("--n-heads", type=int, default=12)
    common.add_argument(
        "--backends",
        nargs="+",
        default=["default", "flash", "efficient", "math"],
        choices=["default", "flash", "efficient", "math", "cudnn"],
    )

    p_sdpa = sub.add_parser("sdpa", parents=[common], help="Test SDPA memory for model-sized shapes")
    p_sdpa.set_defaults(func=cmd_sdpa)

    p_model = sub.add_parser("model", parents=[common], help="Run one model+loss+backward step")
    p_model.add_argument("--d-ff", type=int, default=1024)
    p_model.add_argument("--n-st-layers", type=int, default=3)
    p_model.add_argument("--trm-H-cycles", type=int, default=3)
    p_model.add_argument("--trm-L-cycles", type=int, default=6)
    p_model.add_argument("--trm-block-layers", type=int, default=3)
    p_model.add_argument("--encoder-channels", type=int, default=192)
    p_model.add_argument("--dropout", type=float, default=0.1)
    p_model.add_argument(
        "--temporal-type",
        type=str,
        default="spatiotemporal",
        choices=["spatiotemporal", "gru"],
    )
    p_model.add_argument(
        "--temporal-mixer",
        type=str,
        default="attention",
        choices=["attention", "recurrent"],
    )
    p_model.add_argument("--predict-next-obs", action="store_true")
    p_model.set_defaults(func=cmd_model)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
