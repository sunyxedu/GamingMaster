"""TRM deep supervision, written the way the reference trains.

Reference: SamsungSAILMontreal/TinyRecursiveModels ``pretrain.py``
(``train_batch``) + ``TinyRecursiveReasoningModel_ACTV1.forward``:

    train_state.carry = None                      # persists across batches/epochs
    for batch in loader:                          # one batch = ONE ACT step
        if carry is None: carry = initial_carry(batch)   # everything "halted"
        # rows that halted last step: reset latents to (H_init, L_init),
        # steps to 0, and take THIS batch's row i as their data; rows still
        # running keep their old data (this batch's row i is dropped)
        carry, loss = model(carry, batch)          # one deep recursion
        (loss / global_batch_size).backward()
        optimizer.step(); zero_grad(); ema.update()
        # halting (training only): halted = steps >= N_sup | q_halt > 0,
        # exploration: w.p. 0.1 an example must reach U{2..N_sup} steps
    total_steps = epochs * N / global_batch_size   # fixed

:class:`BatchCarry` is ``train_state.carry``.  Our "example" is a frame
(the TRM ponders each frame's tokens); a batch row is one game sequence,
because Stage 1 encodes the whole sequence.  Frames halt individually —
a halted frame's latents are frozen and it leaves the loss — and a row is
swapped for new data once all its frames have halted (or reached N_sup).

Loss normalisation: the reference divides the summed per-example loss by
the fixed global batch size.  Our loss functions return the mean over the
active frames, so each step multiplies it by ``n_active / n_frames`` (all
real frames currently in the batch): every (frame, step) pair weighs
1 / n_frames regardless of how many frames are still pondering.

Evaluation (reference ``evaluate``): every example runs all N_sup steps —
``net(..., act_steps=halt_max_steps)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

#: weight of the q_halt BCE relative to the task loss (reference ACTLossHead)
Q_HALT_WEIGHT = 0.5

#: batch groups whose tensors are [B, T, ...] per-frame data
FRAME_GROUPS = ("obs", "tgt", "aux")


def halt_update(
    q_halt: torch.Tensor,
    steps: torch.Tensor,
    active: torch.Tensor,
    n_sup: int,
    exploration_prob: float,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One reference halting decision.

    Args:
        q_halt: [B, T] halt logits of the step just taken.
        steps:  [B, T] supervision steps taken BEFORE this one.
        active: [B, T] frames that took this step.
    Returns:
        (new_steps, still_active)
    """
    with torch.no_grad():
        new_steps = torch.where(active, steps + 1, steps)
        halted = new_steps >= n_sup
        if training and n_sup > 1:
            halted = halted | (q_halt > 0)
            explore = torch.rand_like(q_halt, dtype=torch.float32) < exploration_prob
            min_halt = explore.long() * torch.randint_like(
                new_steps, low=2, high=n_sup + 1)
            halted = halted & (new_steps >= min_halt)
        return new_steps, active & ~halted


# ---------------------------------------------------------------------------
# row <-> batch
# ---------------------------------------------------------------------------

def split_rows(batch: Dict) -> List[Dict]:
    """Collated batch → per-row data, every [B, T, ...] tensor of the
    ``obs`` / ``tgt`` / ``aux`` groups trimmed to the row's length."""
    lengths = [int(n) for n in batch["meta"]["lengths"]]
    rows = []
    for i, L in enumerate(lengths):
        row = {"T": L}
        for g in FRAME_GROUPS:
            if g in batch:
                row[g] = {k: v[i, :L] for k, v in batch[g].items()}
        rows.append(row)
    return rows


def merge_rows(rows: List[Dict]) -> Dict:
    """Per-row data → zero-padded batch (the padding both dataset collates
    use), with ``meta.lengths`` and ``meta.frame_mask``."""
    lengths = [r["T"] for r in rows]
    B, T_max = len(rows), max(lengths)
    out: Dict = {}
    for g in FRAME_GROUPS:
        if g not in rows[0]:
            continue
        grp = {}
        for k, ref in rows[0][g].items():
            t = ref.new_zeros((B, T_max, *ref.shape[1:]))
            for i, r in enumerate(rows):
                t[i, : r["T"]] = r[g][k]
            grp[k] = t
        out[g] = grp
    dev = next(iter(rows[0]["obs"].values())).device
    frame_mask = torch.zeros(B, T_max, dtype=torch.bool, device=dev)
    for i, L in enumerate(lengths):
        frame_mask[i, :L] = True
    out["meta"] = {"lengths": lengths, "frame_mask": frame_mask}
    return out


@dataclass
class _Row:
    data: Dict
    steps: torch.Tensor                   # [T] long
    active: torch.Tensor                  # [T] bool
    carry: Optional[Tuple[torch.Tensor, torch.Tensor]] = None   # [T, S, d] x 2


ForwardBackward = Callable[
    [Dict, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor,
     torch.Tensor, float],
    Tuple[Dict[str, torch.Tensor], Dict[str, float]],
]


class BatchCarry:
    """The reference ``train_state.carry`` for one (micro-)batch slot.

    Call :meth:`step` once per loader batch.  It swaps halted rows for the
    new batch's rows at the same positions, runs ``forward_backward`` once
    (one ACT step for every row) and applies the halting rule.
    """

    def __init__(self, n_sup: int, exploration_prob: float,
                 store: Optional[torch.device] = None):
        self.n_sup = max(1, n_sup)
        self.exploration_prob = exploration_prob
        self.store = store              # park latents here between steps
        self.rows: List[Optional[_Row]] = []

    def step(self, new_rows: List[Dict], forward_backward: ForwardBackward
             ) -> Optional[Dict[str, float]]:
        """``new_rows``: this batch's per-row data (:func:`split_rows`, after
        any per-example preparation such as teacher-forcing draws).
        ``forward_backward(batch, carry, active, reset, weight)`` must run
        the model with those arguments, ``backward()`` on
        ``weight * active_mean_loss`` (divided by any accumulation factor)
        and return ``(pred, stats)`` with ``pred["q_halt"]`` / ``["carry"]``.
        """
        # reference: new_current_data = where(halted, batch, current_data)
        n = max(len(self.rows), len(new_rows))
        self.rows += [None] * (n - len(self.rows))
        for i in range(n):
            r = self.rows[i]
            if r is None or not bool(r.active.any()):
                if i < len(new_rows):
                    d = new_rows[i]
                    dev = next(iter(d["obs"].values())).device
                    self.rows[i] = _Row(
                        data=d,
                        steps=torch.zeros(d["T"], dtype=torch.long, device=dev),
                        active=torch.ones(d["T"], dtype=torch.bool, device=dev))
                else:
                    self.rows[i] = None   # short final batch: slot stays empty
        live = [r for r in self.rows if r is not None]
        if not live:
            return None

        batch = merge_rows([r.data for r in live])
        frame_mask = batch["meta"]["frame_mask"]
        B, T_max = frame_mask.shape
        dev = frame_mask.device
        active = torch.zeros(B, T_max, dtype=torch.bool, device=dev)
        steps = torch.zeros(B, T_max, dtype=torch.long, device=dev)
        for i, r in enumerate(live):
            active[i, : r.data["T"]] = r.active
            steps[i, : r.data["T"]] = r.steps
        # reference reset_carry: fresh rows start from (H_init, L_init)
        reset = frame_mask & (steps == 0)
        carry = None
        ref = next((r.carry for r in live if r.carry is not None), None)
        if ref is not None:
            S, d = ref[0].shape[1:]
            zc = [torch.zeros(B, T_max, S, d, dtype=ref[j].dtype, device=dev)
                  for j in range(2)]
            for i, r in enumerate(live):
                if r.carry is not None:
                    for j in range(2):
                        zc[j][i, : r.data["T"]] = r.carry[j].to(dev)
            carry = (zc[0].reshape(B * T_max, S, d), zc[1].reshape(B * T_max, S, d))

        n_frames = int(frame_mask.sum())
        weight = int(active.sum()) / max(n_frames, 1)
        pred, stats = forward_backward(batch, carry, active, reset, weight)

        new_steps, new_active = halt_update(
            pred["q_halt"].detach(), steps, active, self.n_sup,
            self.exploration_prob, training=True)
        z = pred["carry"]
        S, d = z[0].shape[1:]
        zH = z[0].reshape(B, T_max, S, d)
        zL = z[1].reshape(B, T_max, S, d)
        halted_now = active & ~new_active
        for i, r in enumerate(live):
            T = r.data["T"]
            c = (zH[i, :T], zL[i, :T])
            if self.store is not None:
                # blocking copy: a non_blocking device→host copy may still
                # be in flight when the tensor is next read
                c = (c[0].to(self.store), c[1].to(self.store))
            r.carry = c
            r.steps = new_steps[i, :T]
            r.active = new_active[i, :T]

        stats = dict(stats)
        n_halt = int(halted_now.sum())
        stats["active_frac"] = weight
        stats["halted_frames"] = float(n_halt)
        stats["halt_depth"] = (float(new_steps[halted_now].float().mean())
                               if n_halt else float("nan"))
        return stats
