"""Per-decision inference captured in one CUDA graph.

At batch 1 the step is launch-bound: hundreds of small kernels, each paying
the CPU launch latency.  :class:`GraphedStep` records ``GameNetwork.step``
once into a ``torch.cuda.CUDAGraph`` and replays it per frame, with

* static input buffers (the new observation / legal masks are copied in),
* a static step-index tensor (the decoder's time feature reads it at run
  time — see ``model.components.transformer.sinusoidal_pe_rows``),
* a static Stage-1 recurrent state, updated IN PLACE by the graph itself
  (the recurrent temporal mixer only: its state has a fixed size, whereas
  the attention mixer's KV cache grows every frame),
* warm-up on a side stream before capture, as the PyTorch docs require.

    net = build_game_network("generals", ...).cuda().eval()
    g = GraphedStep(net, {"grid_seq": grid, "globals_seq": glob}, act_steps=8,
                    autocast_dtype=torch.bfloat16)
    g.reset()                              # new episode
    for t, obs in enumerate(frames):
        out = g(obs, t)                    # same keys as step() minus st_cache

The graph replays exactly the eager step: same kernels on the same
addresses, identical outputs (sampled actions use the default CUDA
generator, which advances on every replay).  What capture freezes: the
batch size, the obs / ``legal`` key sets and shapes (checked per call),
``act_steps``, the precision, and ``action_decoder.greedy`` — rebuild the
wrapper to change any of them.  Output tensors are overwritten by the next
call; clone what you keep.

Weights are read by address: an in-place update (``load_state_dict``
without ``assign``, an optimizer step, a learner→actor ``copy_``) is picked
up by the next replay, and :meth:`GraphedStep.reset` re-reads the live
h_init.  Anything that reallocates parameters (``.to()``, ``assign=True``)
needs a new wrapper.

Requires a sync-free step (eval mode, non-TRM core); see
``GameNetwork.step``.  MoE layers are captured on their dense
(all-experts, sync-free) path at any batch size — the sparse routed path
cannot be captured — so a large captured batch pays E/k × the routed
expert FLOPs and ~N·E·(3·inter + 2·d) elements of graph-pool memory.
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional

import torch

from model.components.core import MoEFFN
from model.game_network import GameNetwork


class GraphedStep:
    """``GameNetwork.step`` at a fixed batch size, replayed from a CUDA graph."""

    def __init__(
        self,
        net: GameNetwork,
        example_obs: Dict[str, torch.Tensor],
        example_legal: Optional[Dict[str, torch.Tensor]] = None,
        act_steps: Optional[int] = None,
        autocast_dtype: Optional[torch.dtype] = None,
        warmup_iters: int = 3,
        record_routes: bool = False,
    ):
        """
        Args:
            net:            a ``GameNetwork`` on a CUDA device, in eval mode.
            example_obs:    one frame in ``GameNetwork.step``'s obs format,
                            [B, ...] (generals: grid_seq [B, C, H, W],
                            globals_seq [B, G]); fixes the shapes.
            example_legal:  legal masks in step()'s format, or None to
                            capture the unmasked (RL) path.
            act_steps:      core depth, fixed at capture.
            autocast_dtype: e.g. torch.bfloat16 to capture under autocast
                            (with the weight-cast cache disabled, as graph
                            capture requires).
            record_routes:  also output ``moe_routes`` (the MoE routing of
                            the frame, ``GameNetwork.step(record_routes=True)``).
        """
        device = next(net.parameters()).device
        if device.type != "cuda":
            raise ValueError(f"GraphedStep needs the network on CUDA (got {device})")
        if net.training:
            raise ValueError("call net.eval() first: training-mode MoE routing and "
                             "recurrent-depth sampling sync with the host")
        self.net = net
        self.device = device
        self.act_steps = act_steps
        self.record_routes = bool(record_routes)
        self.batch = next(iter(example_obs.values())).shape[0]
        with torch.no_grad():
            self.obs = {k: v.detach().to(device).clone() for k, v in example_obs.items()}
            self.legal = ({k: v.detach().to(device).clone() for k, v in example_legal.items()}
                          if example_legal else None)
            self.step_idx = torch.zeros((), dtype=torch.long, device=device)
            self.st_cache: List[torch.Tensor] = net.initial_st_cache(self.batch)
        net.prepare_inference(device)
        self._autocast = (
            torch.autocast("cuda", dtype=autocast_dtype, cache_enabled=False)
            if autocast_dtype is not None else contextlib.nullcontext())

        # Warm-up on a side stream (lazy cuBLAS/cuDNN init, allocator), then
        # capture — both on the NETWORK's device (torch.cuda.graph's default
        # capture stream is created once, on whichever device is current at
        # first use), with every MoE layer forced onto the dense path so
        # warm-up and capture run the same kernels.  Capture only records
        # kernels — the state it reads is restored after warm-up, before the
        # first real frame.
        with torch.cuda.device(device), MoEFFN.inference_path("dense"):
            side = torch.cuda.Stream(device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                for _ in range(max(1, warmup_iters)):
                    self._run()
            torch.cuda.current_stream(device).wait_stream(side)
            self.reset()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=torch.cuda.Stream(device)):
                self.out = self._run()
        self.reset()

    def _run(self) -> Dict[str, torch.Tensor]:
        with torch.no_grad(), self._autocast:
            out = GameNetwork.step(self.net, self.obs, legal=self.legal,
                                   st_cache=self.st_cache, step_idx=self.step_idx,
                                   act_steps=self.act_steps,
                                   record_routes=self.record_routes)
            for static, new in zip(self.st_cache, out.pop("st_cache")):
                static.copy_(new)
        return out

    def reset(self) -> None:
        """Start a new episode: Stage-1 state := its learned initial value,
        read from the live h_init parameters (so weights loaded in place
        since capture are honoured)."""
        with torch.no_grad():
            for static, init in zip(self.st_cache, self.net.initial_st_cache(self.batch)):
                static.copy_(init)

    def __call__(
        self,
        obs: Dict[str, torch.Tensor],
        step_idx: int,
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Advance one frame.  ``step_idx`` is the absolute frame index (as
        for ``step()``); returns the static output dict (overwritten by the
        next call) — the Stage-1 state is kept internally."""
        if (legal is None) != (self.legal is None):
            raise ValueError("legal masks must be given iff the graph was "
                             "captured with example_legal")
        # Host-side metadata checks only (no sync): copy_ would silently
        # broadcast a mis-shaped input, and a key absent at capture would be
        # silently ignored (e.g. a direction mask → unmasked directions).
        self._check_inputs("obs", obs, self.obs)
        if legal is not None:
            self._check_inputs("legal", legal, self.legal)
        with torch.no_grad():
            for k, buf in self.obs.items():
                buf.copy_(obs[k], non_blocking=True)
            if legal is not None:
                for k, buf in self.legal.items():
                    buf.copy_(legal[k], non_blocking=True)
            self.step_idx.fill_(step_idx)
        self.graph.replay()
        return self.out

    @staticmethod
    def _check_inputs(what: str, given: Dict[str, torch.Tensor],
                      static: Dict[str, torch.Tensor]) -> None:
        if given.keys() != static.keys():
            raise ValueError(f"{what} keys {sorted(given)} != the keys captured "
                             f"{sorted(static)}; rebuild the GraphedStep")
        for k, buf in static.items():
            if given[k].shape != buf.shape:
                raise ValueError(f"{what}[{k!r}] has shape {tuple(given[k].shape)}, "
                                 f"captured {tuple(buf.shape)}")
