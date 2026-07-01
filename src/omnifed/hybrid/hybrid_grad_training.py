# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""Gradient-aggregation training helpers for hybrid Slurm (Phase 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Tuple

import torch
from torch import nn

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
    "apply_optimizer_grads",
    "clear_model_grads",
    "grad_l2_norm",
    "missing_grad_param_names",
    "require_model_grads",
]
