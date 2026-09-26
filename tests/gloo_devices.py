"""Devices of the 2-rank gloo equivalence tests: always the CPU, plus CUDA
tensors over gloo when CUDA is available — the MIG-slice data-parallel mode
(training.distributed): inside an allocation of >= 2 MIG instances every
rank pins itself to its own slice, on one GPU both ranks share it (gloo, unlike
NCCL, allows that), with several GPUs each rank takes its own."""

import multiprocessing
import os

import pytest
import torch

from training.distributed import is_mig, pin_rank_device, visible_devices, visible_mig


def _has_cuda() -> bool:
    """Only the pytest process decides what to skip.  A spawned rank
    re-imports the test modules (and so this one), and
    torch.cuda.is_available() there would run cuInit on the unpinned MIG
    list before the rank pins its slice (pin_rank_device then refuses)."""
    if multiprocessing.parent_process() is not None:
        return False
    return torch.cuda.is_available()


HAS_CUDA = _has_cuda()
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA (gloo on CUDA tensors)")
DEVICES = ["cpu", pytest.param("cuda", marks=needs_cuda)]


def local_ranks(world: int, kind: str):
    """LOCAL_RANK of each of ``world`` spawned ranks (parent side): the rank
    itself when there is a CUDA device / MIG slice per rank, else 0 (all on
    the first one)."""
    if kind != "cuda":
        return list(range(world))
    entries = visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    n = len(entries) if is_mig(entries) else torch.cuda.device_count()
    return [r if n >= world else 0 for r in range(world)]


def worker_device(kind: str, local_rank: int) -> torch.device:
    """In a spawned rank, BEFORE anything touches CUDA: export LOCAL_RANK, pin
    this rank's MIG slice (training.distributed.pin_rank_device) and return
    its device (made current)."""
    os.environ["LOCAL_RANK"] = str(local_rank)
    if kind == "cpu":
        return torch.device("cpu")
    pin_rank_device()
    device = torch.device("cuda", 0 if visible_mig() else local_rank)
    torch.cuda.set_device(device)
    return device
