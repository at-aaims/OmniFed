# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""Pure sync scheduling logic for hybrid PR-B (no training hooks until later phases)."""

from __future__ import annotations

from typing import List, Tuple

from src.omnifed.hierarchical.sync_config import (
    LOCAL_SYNC_UNIT_EPOCH,
    LOCAL_SYNC_UNIT_ITERATION,
    HybridSyncConfig,
)

__all__ = [
    "epochs_per_global_sync",
    "plan_epoch_sync_for_round",
    "plan_iteration_sync_for_round",
    "should_global_sync",
    "should_local_sync_at_epoch_end",
    "should_local_sync_at_iteration",
]


def epochs_per_global_sync(cfg: HybridSyncConfig) -> int | None:
    """Epochs between global syncs when ``local.unit`` is ``epoch``."""
    return cfg.epochs_per_global_sync


def should_local_sync_at_epoch_end(epoch_idx: int, cfg: HybridSyncConfig) -> bool:
    """
    Whether to run a **local** sync at the end of this epoch.

    ``epoch_idx`` is 0-based within the outer round (matches ``BaseAlgorithm.epoch_idx``).
    """
    if cfg.local_unit != LOCAL_SYNC_UNIT_EPOCH:
        return False
    if epoch_idx < 0:
        raise ValueError(f"epoch_idx must be >= 0, got {epoch_idx}")
    return (epoch_idx + 1) % cfg.local_every == 0


def should_local_sync_at_iteration(iteration_idx: int, cfg: HybridSyncConfig) -> bool:
    """
    Whether to run a **local** sync after this training iteration (minibatch).

    ``iteration_idx`` is 0-based batches since outer round start.
    """
    if cfg.local_unit != LOCAL_SYNC_UNIT_ITERATION:
        return False
    if iteration_idx < 0:
        raise ValueError(f"iteration_idx must be >= 0, got {iteration_idx}")
    return (iteration_idx + 1) % cfg.local_every == 0


def should_global_sync(local_sync_count: int, cfg: HybridSyncConfig) -> bool:
    """Whether the **local_sync_count**-th local sync should also trigger global sync."""
    if local_sync_count < 1:
        return False
    return local_sync_count % cfg.global_every_local == 0


def plan_epoch_sync_for_round(
    num_epochs: int,
    cfg: HybridSyncConfig,
) -> Tuple[List[int], List[int]]:
    """
    Simulate epoch-end scheduling for one outer round.

    Returns:
        ``local_epoch_indices``: 0-based epoch indices where local sync runs.
        ``global_epoch_indices``: 0-based epoch indices where global sync runs.
    """
    if num_epochs < 0:
        raise ValueError(f"num_epochs must be >= 0, got {num_epochs}")

    local_epoch_indices: List[int] = []
    global_epoch_indices: List[int] = []
    local_sync_count = 0

    for epoch_idx in range(num_epochs):
        if not should_local_sync_at_epoch_end(epoch_idx, cfg):
            continue
        local_sync_count += 1
        local_epoch_indices.append(epoch_idx)
        if should_global_sync(local_sync_count, cfg):
            global_epoch_indices.append(epoch_idx)

    return local_epoch_indices, global_epoch_indices


def plan_iteration_sync_for_round(
    num_iterations: int,
    cfg: HybridSyncConfig,
) -> Tuple[List[int], List[int]]:
    """Same as :func:`plan_epoch_sync_for_round` but for iteration-based local sync."""
    if num_iterations < 0:
        raise ValueError(f"num_iterations must be >= 0, got {num_iterations}")

    local_iteration_indices: List[int] = []
    global_iteration_indices: List[int] = []
    local_sync_count = 0

    for iteration_idx in range(num_iterations):
        if not should_local_sync_at_iteration(iteration_idx, cfg):
            continue
        local_sync_count += 1
        local_iteration_indices.append(iteration_idx)
        if should_global_sync(local_sync_count, cfg):
            global_iteration_indices.append(iteration_idx)

    return local_iteration_indices, global_iteration_indices
