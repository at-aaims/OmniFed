# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""Hybrid gradient accumulation mode: mean-batch (Path A) vs sample-weighted (Path B)."""

from __future__ import annotations

from typing import Literal, Union

from omegaconf import DictConfig, OmegaConf

GradAccumulation = Literal["mean_batch", "sample_weighted"]

GRAD_ACCUMULATION_MEAN_BATCH: GradAccumulation = "mean_batch"
GRAD_ACCUMULATION_SAMPLE_WEIGHTED: GradAccumulation = "sample_weighted"
_VALID = frozenset({GRAD_ACCUMULATION_MEAN_BATCH, GRAD_ACCUMULATION_SAMPLE_WEIGHTED})


def normalize_grad_accumulation(raw: object) -> GradAccumulation:
    if raw is None:
        return GRAD_ACCUMULATION_MEAN_BATCH
    key = str(raw).strip().lower()
    if key in ("mean_batch", "mean-batch", "path_a", "a"):
        return GRAD_ACCUMULATION_MEAN_BATCH
    if key in ("sample_weighted", "sample-weighted", "path_b", "b", "weighted"):
        return GRAD_ACCUMULATION_SAMPLE_WEIGHTED
    raise ValueError(
        "engine.hierarchical.grad_accumulation must be 'mean_batch' or 'sample_weighted', "
        f"got {raw!r}"
    )


def hybrid_grad_accumulation_from_cfg(cfg: Union[DictConfig, object]) -> GradAccumulation:
    raw = OmegaConf.select(cfg, "engine.hierarchical.grad_accumulation", default="mean_batch")
    return normalize_grad_accumulation(raw)


def format_hybrid_grad_policy(cfg: Union[DictConfig, object]) -> str:
    mode = hybrid_grad_accumulation_from_cfg(cfg)
    return f"grad_accumulation={mode!r}"


__all__ = [
    "GRAD_ACCUMULATION_MEAN_BATCH",
    "GRAD_ACCUMULATION_SAMPLE_WEIGHTED",
    "GradAccumulation",
    "format_hybrid_grad_policy",
    "hybrid_grad_accumulation_from_cfg",
    "normalize_grad_accumulation",
]
