"""Conv observation-encoder building blocks shared by every grid adapter: the
generals ``SpatialEncoder``, the board / shogi / lux / clash_royale stack
(:func:`conv_stack`, the same layout), the starcraft minimap stack and their
``ResBlock``s.

Normalisation (``cfg.encoder_norm``, ``cfg.encoder_norm_groups``).  Every
encoder norm layer comes from ONE factory, :func:`make_encoder_norm`
(adapters hold the config's choice as an :class:`EncoderNorm`):

  "group"  per-frame GroupNorm (:class:`FrameGroupNorm`, eps GN_EPS) with
           ``encoder_norm_groups`` groups, by default 1: LayerNorm over
           (C, H, W) with a per-channel affine.  Each frame is normalised with
           the statistics of that frame alone, the same function in train and
           eval mode.  The trainers flatten B sequences of T frames into one
           [B·T, C, H, W] conv batch, and a per-frame norm keeps frame t's
           features a function of frame t: no statistics of the sequence's
           later frames, of the other samples or of padded frames reach it.
           The ``frame_mask`` exclusion of :func:`encode_frames` is not needed
           and is skipped.
  "batch"  BatchNorm2d, the layout of every checkpoint written before the field
           existed (``model.config.LEGACY_CONFIG_DEFAULTS``), kept bit-exact.
           In train mode its statistics pool all B·T frames of the batch, so
           frame t of an IL / PPO sequence sees (aggregated) statistics of its
           own future frames and of the other samples, and training normalises
           differently from inference (running statistics).  Padded frames are
           kept out of those statistics (:func:`encode_frames` with a
           ``frame_mask``), and data-parallel runs merge the running statistics
           (``training.distributed.BatchNormStatSync``).

Why one group by default.  A per-frame norm subtracts and divides out
statistics of the frame: G groups take 2·G numbers per frame and layer out of
the features.  Board encodings carry game state in exactly such totals (disc
counts, material, how full the board is), and a group of similarly-responding
channels loses them; one group loses only the frame's overall offset and
scale.  Small groups are also where near-degenerate statistics live (see
GN_EPS).  On a tiny othello + chess joint IL run (d_model 64, 32 encoder
channels; validation loss after 600 steps, mean of 2 seeds) one group matched
BatchNorm on othello (2.024 vs 2.004) and beat it on chess (4.72 vs 4.85),
while 4-channel groups gave 2.026 / 4.91 with a far larger spread between
seeds.  Other group counts stay available (``encoder_norm_groups``, e.g. 32
at production widths) for ablations at scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig

#: ``cfg.encoder_norm`` values (see the module docstring)
ENCODER_NORMS = ("group", "batch")

#: "group" epsilon.  Per-frame statistics of sparse board encodings can be
#: near-degenerate: features almost constant over one frame (an almost empty
#: board, channels silenced by the previous ReLU; far likelier within a small
#: group than over the whole frame).  The norm divides their deviations by
#: sqrt(var + eps), so with PyTorch's 1e-5 it amplifies them, rounding noise
#: included, by up to ~300x.  The network then becomes hypersensitive to fp32
#: kernel noise (convs block differently at different batch sizes: the RL
#: actor's per-step batch vs the learner's sequences) and to bf16 rounding;
#: measured on a Stage-1 othello net with 4-channel groups, actor vs learner
#: log-probs differed by 4e-4 (BatchNorm: 5e-7).  1e-3 (Keras' normalisation
#: default) caps the gain at ~32x and brings the mismatch back to 7e-7, while
#: staying <= 1% of the O(0.1-1) variance of the kaiming-initialised conv
#: features.  BatchNorm is immune: its statistics pool the batch (train) or
#: come from running averages (eval).  Changing it changes what "group"
#: checkpoints compute.
GN_EPS = 1e-3


class FrameGroupNorm(nn.GroupNorm):
    """``nn.GroupNorm`` of a batch of frames [N, C, H, W] (each row normalised
    on its own) that keeps the input dtype under autocast.

    CUDA autocast runs ``group_norm`` in float32 (it is on autocast's fp32 op
    list), while the BatchNorm2d this layer replaces runs in the input dtype, as
    GroupNorm itself does under CPU autocast.  A float32 output would double
    the saved activations of the encoder (the 625-token generals grid above
    all) and change the dtype flow into the core.  Here a bf16 / fp16 input
    under autocast is normalised in its own dtype: ATen's group_norm kernels
    accumulate the statistics of low-precision inputs in float32, so only the
    output is rounded, as BatchNorm's is (the affine parameters are cast like
    autocast casts conv weights).  Everywhere else it is exactly
    ``nn.GroupNorm``.  Pure tensor ops, no host sync: capturable in a CUDA
    graph."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.float16, torch.bfloat16) and torch.is_autocast_enabled(x.device.type):
            w = None if self.weight is None else self.weight.to(x.dtype)
            b = None if self.bias is None else self.bias.to(x.dtype)
            with torch.autocast(x.device.type, enabled=False):
                return F.group_norm(x, self.num_groups, w, b, self.eps)
        return super().forward(x)


def make_encoder_norm(kind: str, channels: int, groups: int = 1) -> nn.Module:
    """THE normalisation layer of the conv observation encoders, for
    ``cfg.encoder_norm`` = ``kind`` (see the module docstring):
    "group" → :class:`FrameGroupNorm` with ``groups`` groups (eps GN_EPS;
    ``groups`` must divide ``channels``), "batch" → ``nn.BatchNorm2d``
    (legacy checkpoints; ``groups`` unused)."""
    if kind == "group":
        if groups < 1 or channels % groups:
            raise ValueError(
                f"encoder_norm_groups={groups} does not divide a {channels}-channel "
                f"conv-encoder layer: it must divide encoder_channels and d_model")
        return FrameGroupNorm(groups, channels, eps=GN_EPS)
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    raise ValueError(f"unknown encoder_norm {kind!r} (expected one of {ENCODER_NORMS})")


@dataclass(frozen=True)
class EncoderNorm:
    """A config's encoder normalisation (``encoder_norm``,
    ``encoder_norm_groups``); calling it with a channel count builds the layer
    through :func:`make_encoder_norm`."""

    kind: str = "group"
    groups: int = 1

    def __call__(self, channels: int) -> nn.Module:
        return make_encoder_norm(self.kind, channels, self.groups)

    @property
    def batch_stats(self) -> bool:
        """Pools statistics over the batch (BatchNorm), i.e. needs
        :func:`encode_frames`' padded-frame exclusion."""
        return self.kind == "batch"


def encoder_norm_of(cfg) -> EncoderNorm:
    """The :class:`EncoderNorm` of ``cfg``; a config object without the
    field is legacy ("batch", like ``model.config.LEGACY_CONFIG_DEFAULTS``)."""
    return EncoderNorm(getattr(cfg, "encoder_norm", "batch"),
                       getattr(cfg, "encoder_norm_groups", 1))


def uses_batch_stats(cfg) -> bool:
    """Whether cfg's encoder norm pools statistics over the batch (BatchNorm)."""
    return encoder_norm_of(cfg).batch_stats


NormSpec = Union[str, EncoderNorm]


def _as_norm(norm: NormSpec) -> EncoderNorm:
    return norm if isinstance(norm, EncoderNorm) else EncoderNorm(norm)


class ResBlock(nn.Module):
    """Pre-activation residual block for the spatial encoders
    (norm → ReLU → conv3×3, twice).  ``norm``: an :class:`EncoderNorm`
    (:func:`encoder_norm_of` a config) or an ``encoder_norm`` value."""

    def __init__(self, channels: int, norm: NormSpec = "batch"):
        super().__init__()
        norm = _as_norm(norm)
        self.net = nn.Sequential(
            norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def conv_stack(in_channels: int, mid_channels: int, out_channels: int,
               norm: NormSpec) -> nn.Sequential:
    """conv3×3 → norm → ReLU → ResBlock → conv3×3 → norm → ReLU → ResBlock at
    full resolution: the SpatialEncoder layout, shared by the board, shogi,
    lux and clash_royale encoders.  Layers are built in this order, which fixes
    the Sequential indices (hence every adapter's checkpoint keys) and the
    random-init stream."""
    norm = _as_norm(norm)
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
        norm(mid_channels),
        nn.ReLU(inplace=True),
        ResBlock(mid_channels, norm),
        nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
        norm(out_channels),
        nn.ReLU(inplace=True),
        ResBlock(out_channels, norm),
    )


def encode_frames(stack: nn.Module, frames: torch.Tensor, B: int, T: int,
                  frame_mask: Optional[torch.Tensor], batch_stats: bool) -> torch.Tensor:
    """Run a conv ``stack`` over B·T flattened frames [B·T, C, H, W] and
    return [B, T, C', H', W'].

    ``batch_stats`` (a BatchNorm stack) with a ``frame_mask`` ([B, T] bool,
    True = real frame) that holds padded frames: the stack runs on the real
    frames only, so the padded ones stay out of the batch statistics, and the
    padded frames get zero features.  They cannot influence real frames
    because collate pads only at the sequence TAIL and both temporal mixers are
    causal (attention masks the future; the recurrence only carries state
    forward).  A per-frame norm needs no exclusion: padded frames cannot reach
    a real frame's features, so the stack runs on every frame (no host sync,
    static shapes)."""
    if batch_stats and frame_mask is not None and not bool(frame_mask.all()):
        fm = frame_mask.reshape(B * T)
        enc = stack(frames[fm])
        feat = enc.new_zeros(B * T, *enc.shape[1:])
        feat[fm] = enc
        return feat.reshape(B, T, *feat.shape[1:])
    enc = stack(frames)
    return enc.reshape(B, T, *enc.shape[1:])


class SpatialEncoder(nn.Module):
    """
    Create Embeddings — converts raw grid channels into per-cell feature vectors.

    Input : [B, C_in, H, W]  (C_in = 12 = 11 type one-hot + 1 army)
    Output: [B, d_model, H, W]
    """

    def __init__(self, cfg: ModelConfig, in_channels: Optional[int] = None):
        super().__init__()
        if in_channels is None:
            in_channels = cfg.grid_channels
        self.net = conv_stack(in_channels, cfg.encoder_channels, cfg.d_model,
                              encoder_norm_of(cfg))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
