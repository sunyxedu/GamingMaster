"""Check which SDPA backends are available on this machine."""

import torch
import torch.nn.functional as F

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("No CUDA — Flash Attention not available")
    raise SystemExit()

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
print(f"Mem-eff SDP enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
print(f"Math SDP enabled: {torch.backends.cuda.math_sdp_enabled()}")

from torch.backends.cuda import SDPBackend, sdp_kernel

B, S, H, D = 100, 627, 8, 32

with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    q = torch.randn(B, H, S, D, device="cuda")
    k = torch.randn(B, H, S, D, device="cuda")
    v = torch.randn(B, H, S, D, device="cuda")

    for name, be in [
        ("Flash", SDPBackend.FLASH_ATTENTION),
        ("Mem-efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("Math", SDPBackend.MATH),
    ]:
        try:
            with sdp_kernel(
                enable_flash=(be == SDPBackend.FLASH_ATTENTION),
                enable_mem_efficient=(be == SDPBackend.EFFICIENT_ATTENTION),
                enable_math=(be == SDPBackend.MATH),
            ):
                F.scaled_dot_product_attention(q, k, v)
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
