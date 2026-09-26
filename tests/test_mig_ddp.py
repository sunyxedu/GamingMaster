"""Data parallelism over MIG slices (training/distributed.py).

- the per-rank device mapping as pure functions: a CUDA_VISIBLE_DEVICES list
  of MIG instances is narrowed to entry LOCAL_RANK; single MIG instances,
  plain GPU indices, GPU UUIDs, empty / unset values and non-torchrun
  processes are left alone; a LOCAL_RANK outside the list is a clear error;
- the backend choice (--dist_backend auto: gloo on MIG slices and the CPU,
  NCCL on full GPUs);
- the trainers pin BEFORE anything touches CUDA (pick_device's
  torch.cuda.is_available() initialises the driver, after which the pin
  would be silently ignored — so a late pin raises), and importing them
  initialises nothing;
- two real ranks with a (fake) MIG list pin themselves to their own entries
  and pick gloo;
- every collective the trainers post works under gloo, on CPU tensors and —
  where CUDA is available — on CUDA tensors, with the exact expected values.
"""

import math
import os
import socket
import subprocess
import sys
import time
from multiprocessing.connection import wait
from pathlib import Path

import pytest
import torch

from tests.gloo_devices import DEVICES, local_ranks, needs_cuda
from training import distributed as D

ROOT = Path(__file__).resolve().parents[1]
WORLD = 2
TIMEOUT_S = 180
MIG_A = "MIG-4f1a6c1e-8a42-5a1f-9d3e-2b7f0c1d2e3f"
MIG_B = "MIG-9b2c7d3e-1f4a-5b6c-8d7e-0a1b2c3d4e5f"
MIG_LEGACY = "MIG-GPU-0e2c5c8a-6a1b-4e2f-9c3d-5e6f7a8b9c0d/1/0"


# ----------------------------------------------------------------------
# device mapping (pure)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("cvd, local_rank, want", [
    (f"{MIG_A},{MIG_B}", "0", MIG_A),                     # a MIG list: the rank's own entry
    (f"{MIG_A},{MIG_B}", "1", MIG_B),
    (f" {MIG_A} , {MIG_B} ", "1", MIG_B),                 # whitespace tolerated
    (f"{MIG_A},{MIG_LEGACY}", "1", MIG_LEGACY),           # the legacy MIG-GPU-<uuid>/gi/ci form
    (MIG_A, "0", None),                                   # a single instance: nothing to narrow
    (MIG_B, "1", None),                                   # already pinned (idempotent)
    ("0,1,2,3", "2", None),                               # plain GPU indices: cuda:LOCAL_RANK
    ("GPU-6c3a1b2d-aaaa-bbbb-cccc-000000000000,GPU-7d4b2c3e-aaaa-bbbb-cccc-111111111111",
     "1", None),                                          # full-GPU UUIDs
    ("", "0", None),                                      # empty: no device
    (None, "0", None),                                    # unset: every device
    (f"{MIG_A},{MIG_B}", None, None),                     # not a torchrun rank
    (f"{MIG_A},{MIG_B}", "", None),
])
def test_rank_visible_devices(cvd, local_rank, want):
    assert D.rank_visible_devices(cvd, local_rank) == want


@pytest.mark.parametrize("local_rank", ["2", "7", "-1"])
def test_local_rank_outside_the_mig_list_is_a_clear_error(local_rank):
    with pytest.raises(ValueError, match=r"LOCAL_RANK=-?\d+ but CUDA_VISIBLE_DEVICES lists 2 "
                                         r"device\(s\).*at most 2 rank"):
        D.rank_visible_devices(f"{MIG_A},{MIG_B}", local_rank)
    with pytest.raises(ValueError, match="not an integer"):
        D.rank_visible_devices(f"{MIG_A},{MIG_B}", "one")


def test_visible_devices_and_mig_detection():
    assert D.visible_devices(None) == D.visible_devices("") == D.visible_devices(" , ") == []
    assert D.visible_devices("0, 1,,2") == ["0", "1", "2"]
    assert D.is_mig([MIG_A]) and D.is_mig(["0", MIG_B]) and not D.is_mig(["0", "GPU-x"])
    assert D.visible_mig({"CUDA_VISIBLE_DEVICES": MIG_A})
    assert not D.visible_mig({"CUDA_VISIBLE_DEVICES": "0,1"}) and not D.visible_mig({})


def test_pin_rank_device_rewrites_once_and_keeps_the_list():
    env = {"CUDA_VISIBLE_DEVICES": f"{MIG_A},{MIG_B}", "LOCAL_RANK": "1"}
    assert D.pin_rank_device(env) == MIG_B
    assert env == {"CUDA_VISIBLE_DEVICES": MIG_B, "LOCAL_RANK": "1",
                   D.MIG_ALL_DEVICES_ENV: f"{MIG_A},{MIG_B}"}
    assert D.pin_rank_device(env) is None and env["CUDA_VISIBLE_DEVICES"] == MIG_B
    plain = {"CUDA_VISIBLE_DEVICES": "0,1", "LOCAL_RANK": "1"}
    assert D.pin_rank_device(plain) is None and plain == {"CUDA_VISIBLE_DEVICES": "0,1",
                                                          "LOCAL_RANK": "1"}
    with pytest.raises(SystemExit, match="LOCAL_RANK=2 but"):
        D.pin_rank_device({"CUDA_VISIBLE_DEVICES": f"{MIG_A},{MIG_B}", "LOCAL_RANK": "2"})


def test_pin_rank_device_on_the_process_environment(monkeypatch):
    monkeypatch.setattr(D, "cuda_initialised", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{MIG_A},{MIG_B}")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv(D.MIG_ALL_DEVICES_ENV, "")      # restored (unset) afterwards
    assert D.pin_rank_device() == MIG_A
    assert os.environ["CUDA_VISIBLE_DEVICES"] == MIG_A
    assert os.environ[D.MIG_ALL_DEVICES_ENV] == f"{MIG_A},{MIG_B}"


def test_a_pin_after_cuda_initialised_raises(monkeypatch):
    """Too late the rewrite would be ignored (every rank on the first slice):
    raise instead — but only when there is something to rewrite."""
    monkeypatch.setattr(D, "cuda_initialised", lambda: "torch.cuda is already initialised")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{MIG_A},{MIG_B}")
    monkeypatch.setenv("LOCAL_RANK", "1")
    with pytest.raises(RuntimeError, match="cannot pin LOCAL_RANK=1.*already initialised"):
        D.pin_rank_device()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == f"{MIG_A},{MIG_B}"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", MIG_B)
    assert D.pin_rank_device() is None


@pytest.mark.parametrize("requested, device, mig, want", [
    ("auto", "cuda", False, "nccl"),
    ("auto", "cuda", True, "gloo"),          # NCCL refuses two ranks on one physical GPU
    ("auto", "cpu", False, "gloo"),
    ("auto", "cpu", True, "gloo"),
    ("gloo", "cuda", False, "gloo"),         # gloo on CUDA tensors
    ("gloo", "cpu", False, "gloo"),
    ("nccl", "cuda", True, "nccl"),          # explicit: the user's call
])
def test_resolve_backend(requested, device, mig, want):
    assert D.resolve_backend(requested, device, mig) == want


def test_resolve_backend_refusals():
    with pytest.raises(SystemExit, match="needs CUDA"):
        D.resolve_backend("nccl", "cpu", False)
    with pytest.raises(SystemExit, match="not mps"):
        D.resolve_backend("auto", "mps", False)
    with pytest.raises(SystemExit, match="expected one of"):
        D.resolve_backend("mpi", "cuda", False)


def test_single_process_needs_no_group(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    env, device = D.init_distributed(torch.device("cpu"), backend="gloo")
    assert not env.enabled and device == torch.device("cpu")


def test_forced_nccl_on_mig_warns_before_it_fails(monkeypatch, capsys):
    """--dist_backend nccl on MIG slices: NCCL connects eagerly inside
    init_process_group (device_id is passed) and raises "Duplicate GPU
    detected" there, so the hint must be printed BEFORE that call."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", MIG_B)           # pinned already
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    seen = {}

    def init_process_group(backend, timeout, device_id):
        seen.update(backend=backend, device_id=device_id,
                    printed=capsys.readouterr().out)
        raise RuntimeError("ncclInvalidUsage: Duplicate GPU detected")

    monkeypatch.setattr(D.dist, "init_process_group", init_process_group)
    with pytest.raises(RuntimeError, match="Duplicate GPU"):
        D.init_distributed(torch.device("cuda"), backend="nccl")
    assert seen["backend"] == "nccl" and seen["device_id"] == torch.device("cuda", 0)
    assert "WARNING: NCCL on MIG" in seen["printed"]


# ----------------------------------------------------------------------
# the trainers pin before anything touches CUDA
# ----------------------------------------------------------------------

class _Stop(Exception):
    pass


def _record_order(monkeypatch, module, calls):
    monkeypatch.setattr(D, "cuda_initialised", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{MIG_A},{MIG_B}")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv(D.MIG_ALL_DEVICES_ENV, "")      # restored (unset) afterwards

    def pick_device():
        calls.append(("pick_device", os.environ["CUDA_VISIBLE_DEVICES"]))
        return torch.device("cpu")

    def init_distributed(device, timeout, backend):
        calls.append(("init_distributed", backend))
        raise _Stop

    monkeypatch.setattr(module, "pick_device", pick_device)
    monkeypatch.setattr(module, "init_distributed", init_distributed)


def test_train_multi_game_pins_before_picking_the_device(monkeypatch):
    from training import train_multi_game as tmg

    calls = []
    _record_order(monkeypatch, tmg, calls)
    with pytest.raises(_Stop):
        tmg.main(["--data", "othello=unused", "--dist_backend", "gloo"])
    assert calls == [("pick_device", MIG_B), ("init_distributed", "gloo")]


def test_train_rl_joint_pins_before_picking_the_device(monkeypatch):
    from training import train_rl_joint as trj

    calls = []
    _record_order(monkeypatch, trj, calls)
    with pytest.raises(_Stop):
        trj.JointRLTrainer(trj.parse_args(["--init", "unused.pt", "--rl", "othello"]))
    assert calls == [("pick_device", MIG_B), ("init_distributed", "auto")]


def _python(code: str, **env) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                          text=True, timeout=120,
                          env={**os.environ, "PYTHONPATH": str(ROOT), **env})


def test_importing_the_trainers_initialises_no_cuda():
    """The pin runs at the top of main(); nothing on the import path may have
    initialised CUDA before it (checked at the driver level where there is
    one: torch.cuda.is_available() runs cuInit without torch's lazy init)."""
    res = _python("import training.train_multi_game, training.train_rl_joint\n"
                  "from training.distributed import cuda_initialised\n"
                  "why = cuda_initialised()\n"
                  "assert why is None, why\n")
    assert res.returncode == 0, res.stderr


def _import_probe(queue) -> None:
    import tests.test_grad_sync  # noqa: F401
    import tests.test_mig_ddp  # noqa: F401
    import tests.test_multi_game_ddp  # noqa: F401
    import tests.test_rl_joint_dist  # noqa: F401
    import tests.test_rl_joint_trainer  # noqa: F401
    import tests.test_zero1  # noqa: F401
    from training.distributed import cuda_initialised

    queue.put(cuda_initialised())


def test_spawned_ranks_import_the_test_modules_without_cuda():
    """A spawned rank re-imports its test module before it pins its MIG
    slice: no module on that path (tests/gloo_devices.py's skip marker
    included) may initialise CUDA."""
    ctx = torch.multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(target=_import_probe, args=(queue,))
    p.start()
    try:
        why = queue.get(timeout=TIMEOUT_S)
    finally:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    assert why is None, why


@needs_cuda
def test_the_driver_probe_sees_cuinit():
    """Positive path of the late-pin diagnostic (CUDA machines): after
    cudaGetDeviceCount (what torch.cuda.is_available() runs, unless
    PYTORCH_NVML_BASED_CUDA_CHECK=1 on non-MIG devices) a MIG pin raises
    instead of being ignored, although torch.cuda is not initialised."""
    res = _python("import os, torch\n"
                  "from training.distributed import cuda_initialised, pin_rank_device\n"
                  "assert cuda_initialised() is None\n"
                  "assert torch._C._cuda_getDeviceCount() > 0\n"
                  "assert not torch.cuda.is_initialized()\n"
                  "assert cuda_initialised() is not None\n"
                  f"os.environ.update(CUDA_VISIBLE_DEVICES='{MIG_A},{MIG_B}', LOCAL_RANK='1')\n"
                  "try:\n"
                  "    pin_rank_device()\n"
                  "except RuntimeError as e:\n"
                  "    assert 'already initialised' in str(e)\n"
                  "else:\n"
                  "    raise SystemExit('late pin did not raise')\n")
    assert res.returncode == 0, res.stderr


# ----------------------------------------------------------------------
# spawned ranks
# ----------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(target, *args):
    ctx = torch.multiprocessing.get_context("spawn")
    port = _free_port()
    procs = [ctx.Process(target=target, args=(r, port, *args)) for r in range(WORLD)]
    for p in procs:
        p.start()
    deadline = time.monotonic() + TIMEOUT_S
    try:
        while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
            wait([p.sentinel for p in procs if p.is_alive()], timeout=1.0)
            bad = [p.exitcode for p in procs if p.exitcode not in (None, 0)]
            assert not bad, f"rank failed with exit code {bad[0]}"
        assert not any(p.is_alive() for p in procs), "ranks hung (collective desync?)"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=10)


def _torchrun_env(rank: int, port: int, **extra) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(WORLD), **extra)
    torch.set_num_threads(1)


def _fake_mig_worker(rank: int, port: int, root: str) -> None:
    _torchrun_env(rank, port, LOCAL_RANK=str(rank), CUDA_VISIBLE_DEVICES=f"{MIG_A},{MIG_B}")
    import torch.distributed as dist

    from training.distributed import init_distributed

    env, device = init_distributed(torch.device("cpu"), 1.0, "auto")
    try:
        t = torch.tensor([float(rank + 1)])
        dist.all_reduce(t)
        torch.save({"cvd": os.environ["CUDA_VISIBLE_DEVICES"],
                    "all": os.environ[D.MIG_ALL_DEVICES_ENV], "backend": dist.get_backend(),
                    "device": str(device), "sum": float(t)},
                   os.path.join(root, f"mig{rank}.pt"))
    finally:
        env.destroy()


def test_ranks_on_a_mig_list_pin_their_own_slice_and_use_gloo(tmp_path):
    """init_distributed on a (fake, CPU-only) 2-slice allocation: rank r keeps
    entry r, the full list is kept aside, auto picks gloo."""
    _spawn(_fake_mig_worker, str(tmp_path))
    for r, want in enumerate((MIG_A, MIG_B)):
        out = torch.load(tmp_path / f"mig{r}.pt")
        assert out == {"cvd": want, "all": f"{MIG_A},{MIG_B}", "backend": "gloo",
                       "device": "cpu", "sum": 3.0}


def _collectives_worker(rank: int, port: int, root: str, kind: str, lrs) -> None:
    """Every collective call shape of train_multi_game / train_rl_joint, on
    ``kind`` tensors over gloo (``lrs``: every rank's LOCAL_RANK)."""
    _torchrun_env(rank, port)
    from tests.gloo_devices import worker_device

    want_device = worker_device(kind, lrs[rank])           # pins the MIG slice first
    import torch.distributed as dist

    from model.components.core import allreduce_moe_loads, moe_layers
    from model.multi_game import build_multi_game
    from model.rl_utils import sync_gradients
    from tests.test_core_moe import CORE
    from training.distributed import (
        BatchNormStatSync,
        broadcast_module_state,
        collective_clip_grad_norm_,
        init_distributed,
    )

    env, dev = init_distributed(torch.device(kind), 2.0, "gloo")
    out = {"device_ok": dev == want_device, "backend": dist.get_backend()}
    try:
        red = {}
        for dt in (torch.float32, torch.float64, torch.int32, torch.int64):
            for op in ("SUM", "MAX", "MIN"):
                x = (torch.arange(6, device=dev) * (1 + rank) - 3 * rank).to(dt)
                dist.all_reduce(x, op=getattr(dist.ReduceOp, op))
                red[str(dt), op] = (x.device == dev, x.cpu())
        out["reduce"] = red
        scalar = torch.tensor(rank + 0.5, device=dev)           # 0-d (the clip norm)
        dist.all_reduce(scalar, op=dist.ReduceOp.MAX)
        out["scalar"] = float(scalar)
        bc = {}
        for dt in (torch.float32, torch.float64, torch.int64, torch.bool):
            t = (torch.arange(4, device=dev) + 7 * rank)
            t = (t % 2 == rank) if dt == torch.bool else t.to(dt)
            dist.broadcast(t, src=0)
            bc[str(dt)] = t.cpu()
        out["broadcast"] = bc
        flag = torch.tensor([float(rank == 0)], device=dev)     # _want_checkpoint's flag
        dist.broadcast(flag, src=0)
        out["flag"] = float(flag)
        objs = [None] * WORLD
        dist.all_gather_object(objs, {"rank": rank, "sums": ({"loss": rank + 0.25}, rank)})
        lst = [{"pool": [rank] * 2} if rank == 0 else None]
        dist.broadcast_object_list(lst, src=0)
        out["objects"] = (objs, lst)
        dist.barrier()

        # a MoE multi-game network, initialised differently per rank
        torch.manual_seed(100 + rank)
        model = build_multi_game(["othello", "chess"], CORE).to(dev)
        out["pre"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        broadcast_module_state(model, src=0)
        out["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for i, m in enumerate(moe_layers(model)):
            m.load.fill_(float(rank + 1 + i))
        allreduce_moe_loads(model)
        out["loads"] = [m.load.cpu().clone() for m in moe_layers(model)]

        # gradients on one game's adapter only on rank 1, a parameter no rank
        # touches, then the bucketed sync and the collective clip
        named = list(model.named_parameters())
        for i, (n, p) in enumerate(named):
            if n.startswith("nets.chess.") and rank == 0:
                continue
            if i == len(named) - 1:
                continue
            p.grad = (rank + 1) * torch.arange(p.numel(), dtype=p.dtype,
                                               device=dev).view_as(p) + i
        sync_gradients(model, WORLD, warn_partial=False, bucket_mb=0.001)
        out["grads"] = {n: None if p.grad is None else p.grad.cpu().clone() for n, p in named}
        params = [p for _, p in named]
        norm = collective_clip_grad_norm_(params, 1e9, dev)
        out["norm"] = (norm.device == dev, float(norm))
        for p in params[:3]:
            p.grad = torch.full_like(p, float("nan") if rank == 1 else 1.0)
        out["bad_norm"] = float(collective_clip_grad_norm_(params[:3], 1.0, dev))

        bns = torch.nn.ModuleList([torch.nn.BatchNorm2d(3), torch.nn.BatchNorm2d(3)]).to(dev)
        sync = BatchNormStatSync(bns)
        if rank == 1:
            bns[0](torch.randn(4, 3, 2, 2, generator=torch.Generator().manual_seed(1)).to(dev))
        out["bn_local"] = [(m.running_mean.cpu().clone(), m.running_var.cpu().clone())
                           for m in bns]
        sync()
        out["bn"] = [(m.running_mean.cpu().clone(), m.running_var.cpu().clone(),
                      int(m.num_batches_tracked)) for m in bns]
        torch.save(out, os.path.join(root, f"coll{rank}.pt"))
    finally:
        env.destroy()


@pytest.mark.parametrize("kind", DEVICES)
def test_every_trainer_collective_is_exact_under_gloo(tmp_path, kind):
    _spawn(_collectives_worker, str(tmp_path), kind, local_ranks(WORLD, kind))
    r0, r1 = (torch.load(tmp_path / f"coll{r}.pt", weights_only=False) for r in range(WORLD))
    xs = [torch.arange(6) * (1 + r) - 3 * r for r in range(WORLD)]
    want = {"SUM": xs[0] + xs[1], "MAX": torch.maximum(*xs), "MIN": torch.minimum(*xs)}
    for r in (r0, r1):
        assert r["device_ok"] and r["backend"] == "gloo"
        for (dt, op), (on_device, got) in r["reduce"].items():
            assert on_device and str(got.dtype) == dt
            assert torch.equal(got, want[op].to(got.dtype)), (dt, op)
        assert r["scalar"] == 1.5
        assert r["flag"] == 1.0
        for dt, got in r["broadcast"].items():
            base = torch.arange(4)
            assert torch.equal(got, base % 2 == 0 if dt == "torch.bool" else base.to(got.dtype))
        objs, lst = r["objects"]
        assert objs == [{"rank": 0, "sums": ({"loss": 0.25}, 0)},
                        {"rank": 1, "sums": ({"loss": 1.25}, 1)}]
        assert lst == [{"pool": [0, 0]}]
        assert [ld.tolist() for ld in r["loads"]] == [
            [float(3 + 2 * i)] * len(ld) for i, ld in enumerate(r["loads"])]
        assert math.isinf(r["bad_norm"])
    # module state: rank 0's everywhere (and it was different before)
    assert any(not torch.equal(r0["pre"][k], r1["pre"][k]) for k in r0["pre"])
    for k, v in r0["state"].items():
        assert torch.equal(v, r0["pre"][k]) and torch.equal(v, r1["state"][k]), k
    # gradients: the rank mean, element by element; rank 0 had no chess
    # gradient (zeros); the untouched parameter stays None on both ranks
    names = list(r0["grads"])
    for i, n in enumerate(names):
        g0, g1 = r0["grads"][n], r1["grads"][n]
        if i == len(names) - 1:
            assert g0 is None and g1 is None
            continue
        assert torch.equal(g0, g1), n
        idx = torch.arange(g0.numel(), dtype=g0.dtype).view_as(g0)
        mine = [(r + 1) * idx + i for r in range(WORLD)]
        want_g = (mine[1] / 2 if n.startswith("nets.chess.")
                  else (mine[0] + mine[1]) / 2)
        assert torch.equal(g0, want_g), n
    assert r0["norm"] == r1["norm"] and r0["norm"][0]
    assert r0["norm"][1] == pytest.approx(
        math.sqrt(sum(float(g.double().square().sum()) for g in r0["grads"].values()
                      if g is not None)), rel=1e-5)
    # BatchNorm: only rank 1 updated layer 0 -> exactly its statistics; layer 1 untouched
    for a, b in zip(r0["bn"], r1["bn"]):
        assert all(torch.equal(x, y) for x, y in zip(a[:2], b[:2])) and a[2] == b[2]
    assert torch.equal(r0["bn"][0][0], r1["bn_local"][0][0])
    assert torch.equal(r0["bn"][0][1], r1["bn_local"][0][1])
    assert (r0["bn"][0][2], r0["bn"][1][2]) == (1, 0)
    assert torch.equal(r0["bn"][1][0], torch.zeros(3))


def test_grad_sync_benchmark_runs_under_torchrun():
    """scripts/bench_grad_sync.py (the cluster's MIG throughput check) on 2
    CPU ranks: torchrun -> pin -> gloo -> one timing line per bucket size."""
    res = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
         str(ROOT / "scripts" / "bench_grad_sync.py"), "--device", "cpu", "--params", "2e5",
         "--width", "64", "--small", "5", "--bucket_mb", "0", "0.5", "--reps", "1"],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
        env={**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1"})
    assert res.returncode == 0, res.stderr[-2000:]
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("[bench] bucket_mb=")]
    assert [ln.split()[1] for ln in lines] == ["bucket_mb=0", "bucket_mb=0.5"]
    assert "backend=gloo" in res.stdout


def test_spawned_children_never_probe_cuda(monkeypatch):
    """tests/gloo_devices.py: once the pytest process has probed CUDA it
    marks the environment, so a spawn child — which imports the test module
    while unpickling its target, before multiprocessing sets its parent —
    skips torch.cuda.is_available() (cuInit on the unpinned MIG list)."""
    import tests.gloo_devices as gd

    assert os.environ.get(gd._SPAWNED_ENV) == "1"         # set when the module was imported

    def boom():
        raise AssertionError("a spawned rank must not probe CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    assert gd._has_cuda() is False
