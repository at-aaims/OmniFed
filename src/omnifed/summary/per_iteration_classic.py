# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""
Classic centralized per-iteration CSV recorder.

Used by the universal ``summary.per_iteration`` dispatcher when pipeline mode is ``classic``.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Any, Dict, Optional

import torch

_ITER_CSV_COLUMNS = [
    "round_idx",
    "epoch_idx",
    "batch_idx",
    "global_step",
    "rank",
    "batch_time_total_s",
    "batch_time_data_s",
    "batch_time_compute_s",
    "train_loss",
    "train_grad_norm",
    "sync_time_total_s",
    "grpc_agg_sample_s",
    "grpc_agg_grad_s",
    "grpc_agg_bn_s",
    "grad_apply_s",
    "grpc_upstream_s",
    "grpc_downstream_s",
    "grpc_compress_s",
    "grpc_decompress_s",
    "gpu_allocated_mb",
    "gpu_reserved_mb",
    "gpu_max_allocated_mb",
    "gpu_max_reserved_mb",
    "gpu_device_used_mb",
    "gpu_device_total_mb",
    "gpu_device_util_pct",
    "comm_backend",
    "aggregate_payload",
]

_COMM_KEYS = (
    "grpc_upstream_s",
    "grpc_downstream_s",
    "grpc_compress_s",
    "grpc_decompress_s",
)

_SYNC_KEYS = (
    "grpc_agg_sample_time",
    "grpc_agg_grad_time",
    "grpc_agg_bn_time",
    "grad_apply_time",
)


def per_iteration_csv_basename(rank: int, *, mode: str = "classic") -> str:
    """Filename for per-rank per-iteration CSV (classic keeps historical name)."""
    if mode == "classic":
        return f"rank{int(rank)}_classic_per_iteration_summary.csv"
    return f"rank{int(rank)}_per_iteration_summary.csv"


def get_gpu_memory_snapshot_mb() -> Dict[str, Optional[float]]:
    """GPU memory in MiB: PyTorch allocator stats + device-wide ``mem_get_info``."""
    empty: Dict[str, Optional[float]] = {
        "gpu_allocated_mb": None,
        "gpu_reserved_mb": None,
        "gpu_max_allocated_mb": None,
        "gpu_max_reserved_mb": None,
        "gpu_device_used_mb": None,
        "gpu_device_total_mb": None,
        "gpu_device_util_pct": None,
    }
    if not torch.cuda.is_available():
        return empty
    try:
        device_idx = torch.cuda.current_device()
        free_b, total_b = torch.cuda.mem_get_info(device_idx)
        used_b = total_b - free_b
        util_pct = (100.0 * used_b / total_b) if total_b > 0 else None
        return {
            "gpu_allocated_mb": torch.cuda.memory_allocated(device_idx) / (1024 ** 2),
            "gpu_reserved_mb": torch.cuda.memory_reserved(device_idx) / (1024 ** 2),
            "gpu_max_allocated_mb": torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2),
            "gpu_max_reserved_mb": torch.cuda.max_memory_reserved(device_idx) / (1024 ** 2),
            "gpu_device_used_mb": used_b / (1024 ** 2),
            "gpu_device_total_mb": total_b / (1024 ** 2),
            "gpu_device_util_pct": util_pct,
        }
    except Exception:
        return empty


class ClassicIterationRecorder:
    """Append one CSV row per training batch on each rank (classic centralized)."""

    def __init__(
        self,
        *,
        rank: int,
        log_dir: str,
        comm_backend: str = "grpc",
        aggregate_payload: str = "gradients",
        mode: str = "classic",
    ) -> None:
        self.rank = int(rank)
        self.log_dir = log_dir
        self.comm_backend = str(comm_backend)
        self.aggregate_payload = str(aggregate_payload)
        self.mode = str(mode)
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(
            log_dir, per_iteration_csv_basename(self.rank, mode=self.mode)
        )
        self._header_written = os.path.isfile(self.csv_path) and os.path.getsize(
            self.csv_path
        ) > 0

    def reset_batch(self, algorithm: Any) -> None:
        algorithm._summary_iter_comm = {k: 0.0 for k in _COMM_KEYS}
        algorithm._summary_iter_sync = {}
        algorithm._summary_iter_batch_t0 = time.perf_counter()
        algorithm._summary_iter_train = {}

    def record_after_batch(self, algorithm: Any) -> None:
        """Write one row; call from ``_train_batch_end`` after sync."""
        batch_t0 = getattr(algorithm, "_summary_iter_batch_t0", None)
        batch_time_total_s = (
            time.perf_counter() - batch_t0 if batch_t0 is not None else None
        )

        peek = getattr(algorithm, "peek_metric", None)
        batch_time_data_s = (
            peek("batch_time_data", agg_context="train") if peek else None
        )
        batch_time_compute_s = (
            peek("batch_time_compute", agg_context="train") if peek else None
        )

        train_state = getattr(algorithm, "_summary_iter_train", {}) or {}
        train_loss = train_state.get("loss")
        train_grad_norm = train_state.get("grad_norm")

        sync_times: Dict[str, float] = getattr(algorithm, "_summary_iter_sync", {}) or {}
        comm_times: Dict[str, float] = getattr(algorithm, "_summary_iter_comm", {}) or {}

        sync_parts = [float(sync_times.get(k, 0.0) or 0.0) for k in _SYNC_KEYS]
        sync_time_total_s = sum(sync_parts) if any(sync_parts) else None

        gpu = get_gpu_memory_snapshot_mb()

        row = {
            "round_idx": int(getattr(algorithm, "round_idx", -1)),
            "epoch_idx": int(getattr(algorithm, "epoch_idx", -1)),
            "batch_idx": int(getattr(algorithm, "batch_idx", -1)),
            "global_step": int(getattr(algorithm, "experiment_batch_idx", -1)),
            "rank": self.rank,
            "batch_time_total_s": _fmt(batch_time_total_s),
            "batch_time_data_s": _fmt(batch_time_data_s),
            "batch_time_compute_s": _fmt(batch_time_compute_s),
            "train_loss": _fmt(train_loss),
            "train_grad_norm": _fmt(train_grad_norm),
            "sync_time_total_s": _fmt(sync_time_total_s),
            "grpc_agg_sample_s": _fmt(sync_times.get("grpc_agg_sample_time")),
            "grpc_agg_grad_s": _fmt(sync_times.get("grpc_agg_grad_time")),
            "grpc_agg_bn_s": _fmt(sync_times.get("grpc_agg_bn_time")),
            "grad_apply_s": _fmt(sync_times.get("grad_apply_time")),
            "grpc_upstream_s": _fmt(comm_times.get("grpc_upstream_s")),
            "grpc_downstream_s": _fmt(comm_times.get("grpc_downstream_s")),
            "grpc_compress_s": _fmt(comm_times.get("grpc_compress_s")),
            "grpc_decompress_s": _fmt(comm_times.get("grpc_decompress_s")),
            **{k: _fmt(v) for k, v in gpu.items()},
            "comm_backend": self.comm_backend,
            "aggregate_payload": self.aggregate_payload,
        }

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_ITER_CSV_COLUMNS)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


def _fmt(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


def install_classic_iteration_recorder(
    algorithm: Any,
    *,
    rank: int,
    log_dir: str,
    comm_backend: str = "grpc",
    aggregate_payload: str = "gradients",
    mode: str = "classic",
) -> ClassicIterationRecorder:
    """Attach batch hooks and store recorder on ``algorithm``."""
    recorder = ClassicIterationRecorder(
        rank=rank,
        log_dir=log_dir,
        comm_backend=comm_backend,
        aggregate_payload=aggregate_payload,
        mode=mode,
    )
    algorithm._summary_iteration_recorder = recorder

    base_batch_start = algorithm._train_batch_start
    base_batch_end = algorithm._train_batch_end

    def _train_batch_start(self) -> None:
        recorder.reset_batch(self)
        base_batch_start()

    def _train_batch_end(self) -> None:
        base_batch_end()
        recorder.record_after_batch(self)

    algorithm._train_batch_start = _train_batch_start.__get__(algorithm, type(algorithm))
    algorithm._train_batch_end = _train_batch_end.__get__(algorithm, type(algorithm))

    return recorder
