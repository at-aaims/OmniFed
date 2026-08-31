"""
Universal per-iteration summary recording.

Detects pipeline mode from config and installs the appropriate recorder.
Classic is implemented today; hybrid will plug in here later.
"""

from __future__ import annotations

from typing import Any, Optional

from src.omnifed.summary.per_iteration_classic import (
    ClassicIterationRecorder,
    install_classic_iteration_recorder,
)
from src.omnifed.summary.pipeline import summary_mode_from_cfg

__all__ = [
    "ClassicIterationRecorder",
    "accumulate_iter_comm",
    "accumulate_classic_iter_comm",
    "get_gpu_memory_snapshot_mb",
    "install_iteration_recorder",
    "install_classic_iteration_recorder",
]


def _iter_comm_bucket(logger: Any) -> Optional[dict]:
    bucket = getattr(logger, "_summary_iter_comm", None)
    if bucket is not None:
        return bucket
    legacy = getattr(logger, "_classic_iter_comm", None)
    if legacy is not None:
        return legacy
    return None


def accumulate_iter_comm(logger: Any, key: str, duration_sec: float) -> None:
    """Sum wire timings into the active iteration bucket (when recorder enabled)."""
    bucket = _iter_comm_bucket(logger)
    if bucket is None:
        return
    bucket[key] = float(bucket.get(key, 0.0)) + float(duration_sec)


def accumulate_classic_iter_comm(logger: Any, key: str, duration_sec: float) -> None:
    """Backward-compatible alias for ``accumulate_iter_comm``."""
    accumulate_iter_comm(logger, key, duration_sec)


def get_gpu_memory_snapshot_mb():
    from src.omnifed.summary.per_iteration_classic import get_gpu_memory_snapshot_mb as _fn

    return _fn()


def install_iteration_recorder(
    cfg: Any,
    algorithm: Any,
    *,
    rank: int,
    log_dir: str,
    comm_backend: str = "grpc",
    aggregate_payload: str = "gradients",
) -> Optional[ClassicIterationRecorder]:
    """Install per-iteration CSV hooks for the pipeline implied by ``cfg``."""
    mode = summary_mode_from_cfg(cfg)
    if mode == "classic":
        return install_classic_iteration_recorder(
            algorithm,
            rank=rank,
            log_dir=log_dir,
            comm_backend=comm_backend,
            aggregate_payload=aggregate_payload,
            mode=mode,
        )
    # Hybrid per-iteration CSV: not implemented yet.
    return None
