"""Digest helpers for the converter golden files (tests/fixtures/golden/*.json).

A golden file pins, per sample, the SHA-256 of every obs/tgt tensor (dtype, shape and
bytes) plus selected meta fields. They were generated from the retired Python reference
converters on the committed fixtures, so the Rust converters stay bit-identical to them.
"""
from __future__ import annotations

import hashlib
import json

import torch


def tensor_digest(t: torch.Tensor) -> str:
    t = t.contiguous()
    h = hashlib.sha256()
    h.update(f"{t.dtype}|{tuple(t.shape)}|".encode())
    h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool else t.to(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _norm(v):
    if isinstance(v, tuple):
        return [_norm(x) for x in v]
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def sample_digests(samples, meta_keys=()):
    out = []
    for s in samples:
        out.append({
            "obs": {k: tensor_digest(v) for k, v in sorted(s["obs"].items())},
            "tgt": {k: tensor_digest(v) for k, v in sorted(s["tgt"].items())},
            "meta": {k: _norm(s["meta"].get(k)) for k in meta_keys},
        })
    return out


def write(path, digests, shard_meta, converter_version):
    json.dump({"converter_version": converter_version, "shard_meta": shard_meta, "samples": digests},
              open(path, "w"), indent=0)
