# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""Gradient-aggregation training helpers for hybrid Slurm (Phase 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Tuple

import torch
from torch import nn

from src.omnifed.algorithm import utils

if TYPE_CHECKING:
    from torch.optim import Optimizer


def missing_grad_param_names(model: nn.Module) -> Tuple[str, ...]:
    return tuple(
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    )


def require_model_grads(model: nn.Module) -> None:
    missing = missing_grad_param_names(model)
    if missing:
        raise RuntimeError(
            "Gradient aggregation requires param.grad on every trainable parameter "
            f"before sync (missing {len(missing)} tensors, e.g. {list(missing[:3])})."
        )


def grad_l2_norm(model: nn.Module) -> float:
    norms: Iterable[torch.Tensor] = (
        p.grad.detach().norm()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    )
    stacked = [t for t in norms]
    if not stacked:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(stacked)).item())


def normalize_accumulated_grads(model: nn.Module, num_batches: int) -> None:
    """Path A: scale summed batch grads to the mean batch grad (FedSGD before sync)."""
    if num_batches < 1:
        raise ValueError(
            f"normalize_accumulated_grads requires num_batches >= 1, got {num_batches}"
        )
    utils.scale_grads(model, 1.0 / num_batches)


def scale_accumulated_grads_by_batch_size(model: nn.Module, batch_size: int) -> None:
    """Path B: after backward on a fresh microbatch grad, scale mean → sum (g_b <- g_b * n_b)."""
    if batch_size < 1:
        raise ValueError(
            f"scale_accumulated_grads_by_batch_size requires batch_size >= 1, got {batch_size}"
        )
    utils.scale_grads(model, float(batch_size))


def init_weighted_grad_accum() -> dict[str, torch.Tensor]:
    return {}


def add_weighted_batch_grad_to_accum(
    model: nn.Module,
    batch_size: int,
    accum: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Add ``grad × batch_size`` for the current microbatch into ``accum``."""
    scale_accumulated_grads_by_batch_size(model, batch_size)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        chunk = param.grad.detach()
        if name in accum:
            accum[name] = accum[name] + chunk
        else:
            accum[name] = chunk.clone()
    return accum


def load_weighted_grad_accum_into_model(
    model: nn.Module, accum: dict[str, torch.Tensor]
) -> None:
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in accum:
            param.grad = accum[name].clone()
        else:
            param.grad = None


def clear_weighted_grad_accum(accum: dict[str, torch.Tensor]) -> None:
    accum.clear()


def normalize_accumulated_grads_by_samples(model: nn.Module, num_samples: int) -> None:
    """Path B: after local MPI SUM of weighted grad sums, convert G_fac to mean (÷ S_fac)."""
    if num_samples < 1:
        raise ValueError(
            f"normalize_accumulated_grads_by_samples requires num_samples >= 1, "
            f"got {num_samples}"
        )
    utils.scale_grads(model, 1.0 / num_samples)


def apply_optimizer_grads(model: nn.Module, optimizer: Optimizer) -> None:
    """Apply sample-weighted averaged gradients already stored in ``param.grad``."""
    require_model_grads(model)
    optimizer.step()


def clear_model_grads(model: nn.Module, *, optimizer: Optimizer | None = None) -> None:
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        return
    with torch.no_grad():
        for param in model.parameters():
            param.grad = None


__all__ = [
    "add_weighted_batch_grad_to_accum",
    "apply_optimizer_grads",
    "clear_model_grads",
    "clear_weighted_grad_accum",
    "grad_l2_norm",
    "init_weighted_grad_accum",
    "load_weighted_grad_accum_into_model",
    "missing_grad_param_names",
    "normalize_accumulated_grads",
    "normalize_accumulated_grads_by_samples",
    "require_model_grads",
    "scale_accumulated_grads_by_batch_size",
]
