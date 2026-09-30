# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""Hybrid Slurm local / global sync frequency (PR-B config only; wiring in later phases)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from omegaconf import DictConfig, OmegaConf

LocalSyncUnit = Literal["epoch", "iteration"]

LOCAL_SYNC_UNIT_EPOCH: LocalSyncUnit = "epoch"
LOCAL_SYNC_UNIT_ITERATION: LocalSyncUnit = "iteration"
_VALID_UNITS = frozenset({LOCAL_SYNC_UNIT_EPOCH, LOCAL_SYNC_UNIT_ITERATION})

DEFAULT_LOCAL_SYNC_EVERY = 4
DEFAULT_GLOBAL_EVERY_LOCAL = 1


@dataclass(frozen=True)
class HybridSyncConfig:
    """Resolved sync policy from ``engine.hierarchical.sync`` (see design spec)."""

    local_unit: LocalSyncUnit
    local_every: int
    global_every_local: int

    @property
    def epochs_per_global_sync(self) -> int | None:
        """Defined when ``local_unit`` is ``epoch``; else ``None`` (iteration-based)."""
        if self.local_unit != LOCAL_SYNC_UNIT_EPOCH:
            return None
        return self.local_every * self.global_every_local


def normalize_local_sync_unit(raw: object) -> LocalSyncUnit:
    if raw is None:
        return LOCAL_SYNC_UNIT_EPOCH
    key = str(raw).strip().lower()
    if key in ("epoch", "epochs"):
        return LOCAL_SYNC_UNIT_EPOCH
    if key in ("iteration", "iterations", "step", "steps", "batch", "batches"):
        return LOCAL_SYNC_UNIT_ITERATION
    raise ValueError(
        "engine.hierarchical.sync.local.unit must be 'epoch' or 'iteration', "
        f"got {raw!r}"
    )


def normalize_positive_int(
    raw: object,
    *,
    name: str,
    default: int,
) -> int:
    if raw is None:
        value = default
    else:
        value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def normalize_sync_config(
    local_unit: object = None,
    local_every: object = None,
    global_every_local: object = None,
) -> HybridSyncConfig:
    """Parse raw knob values; apply defaults matching current hybrid behavior."""
    return HybridSyncConfig(
        local_unit=normalize_local_sync_unit(local_unit),
        local_every=normalize_positive_int(
            local_every,
            name="engine.hierarchical.sync.local.every",
            default=DEFAULT_LOCAL_SYNC_EVERY,
        ),
        global_every_local=normalize_positive_int(
            global_every_local,
            name="engine.hierarchical.sync.global_every_local",
            default=DEFAULT_GLOBAL_EVERY_LOCAL,
        ),
    )


def hybrid_sync_from_cfg(cfg: Union[DictConfig, object]) -> HybridSyncConfig:
    """Read ``engine.hierarchical.sync`` from a composed Hydra config."""
    return normalize_sync_config(
        local_unit=OmegaConf.select(cfg, "engine.hierarchical.sync.local.unit", default=None),
        local_every=OmegaConf.select(cfg, "engine.hierarchical.sync.local.every", default=None),
        global_every_local=OmegaConf.select(
            cfg, "engine.hierarchical.sync.global_every_local", default=None
        ),
    )


def format_hybrid_sync_policy(cfg: Union[DictConfig, object]) -> str:
    """Single-line summary for startup logs (PR-B P2; scheduling wired in later phases)."""
    sync = hybrid_sync_from_cfg(cfg)
    if sync.epochs_per_global_sync is not None:
        extra = f" epochs_per_global_sync={sync.epochs_per_global_sync}"
    else:
        extra = ""
    return (
        f"sync.local.unit={sync.local_unit!r} sync.local.every={sync.local_every} "
        f"sync.global_every_local={sync.global_every_local}{extra}"
    )


__all__ = [
    "DEFAULT_GLOBAL_EVERY_LOCAL",
    "DEFAULT_LOCAL_SYNC_EVERY",
    "LOCAL_SYNC_UNIT_EPOCH",
    "LOCAL_SYNC_UNIT_ITERATION",
    "HybridSyncConfig",
    "LocalSyncUnit",
    "format_hybrid_sync_policy",
    "hybrid_sync_from_cfg",
    "normalize_sync_config",
    "normalize_local_sync_unit",
]
