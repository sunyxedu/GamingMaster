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


#: set by the pytest process once it has probed CUDA; every rank it spawns
#: afterwards inherits it
_SPAWNED_ENV = "GM_GLOO_TEST_SPAWNED"


def _has_cuda() -> bool:
    """Only the pytest process decides what to skip.  A spawned rank
    re-imports the test modules (and so this one), and
    torch.cuda.is_available() there would run cuInit on the unpinned MIG
    list before the rank pins its slice (pin_rank_device then refuses).
    multiprocessing.parent_process() alone cannot tell: a spawn child
    unpickles its target — importing the test module — BEFORE
    Process._bootstrap sets the parent, so the parent marks its children
    through the environment instead."""
    if os.environ.get(_SPAWNED_ENV) == "1" or multiprocessing.parent_process() is not None:
        return False
    has = torch.cuda.is_available()
    os.environ[_SPAWNED_ENV] = "1"
    return has


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


def hide_gpus_if_cpu(kind: str) -> None:
    """In a spawned CPU-variant rank: hide every GPU before anything can
    touch CUDA, so the trainer's MIG pin (training.distributed) never sees a
    MIG allocation it would have to split (LOCAL_RANK may exceed the slice
    count when a 3-rank CPU test runs inside a 2-slice allocation)."""
    if kind == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def worker_device(kind: str, local_rank: int) -> torch.device:
    """In a spawned rank, BEFORE anything touches CUDA: export LOCAL_RANK, pin
    this rank's MIG slice (training.distributed.pin_rank_device) and return
    its device (made current)."""
    os.environ["LOCAL_RANK"] = str(local_rank)
    if kind == "cpu":
        hide_gpus_if_cpu(kind)
        return torch.device("cpu")
    pin_rank_device()
    device = torch.device("cuda", 0 if visible_mig() else local_rank)
    torch.cuda.set_device(device)
    return device
