# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""One-shot model load / device-placement timings (job start, not per-iteration)."""

from __future__ import annotations

import csv
import os
import time
from typing import Any

import torch
from hydra.utils import instantiate
from torch import nn


_STARTUP_COLUMNS = (
    "rank",
    "model_load_s",
    "model_to_device_s",
    "model_startup_total_s",
)


def instantiate_model_timed(model_cfg: Any) -> tuple[nn.Module, float]:
    """Hydra-instantiate the model (disk / HF load). Returns ``(model, seconds)``."""
    t0 = time.perf_counter()
    model = instantiate(model_cfg)
    return model, time.perf_counter() - t0


def move_model_to_device_timed(
    model: nn.Module,
    device: torch.device | str,
    *,
    non_blocking: bool = True,
) -> tuple[nn.Module, float]:
    """``model.to(device)``; CUDA is synchronized so the timer is wall-clock true."""
    device = torch.device(device)
    t0 = time.perf_counter()
    model = model.to(device, non_blocking=non_blocking)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return model, time.perf_counter() - t0


def startup_csv_path(log_dir: str, rank: int) -> str:
    return os.path.join(log_dir, f"rank{int(rank)}_startup_summary.csv")


def emit_model_startup(
    *,
    rank: int,
    log_dir: str,
    model_load_s: float,
    model_to_device_s: float,
    algorithm: Any = None,
    write_csv: bool = True,
) -> str:
    """Print, optionally write CSV, optionally ``log_metric`` on the algorithm."""
    total_s = float(model_load_s) + float(model_to_device_s)
    path = startup_csv_path(log_dir, rank)
    if write_csv:
        print(
            f"[startup] rank={int(rank)} model_load_s={model_load_s:.3f} "
            f"model_to_device_s={model_to_device_s:.3f} "
            f"model_startup_total_s={total_s:.3f}",
            flush=True,
        )
        os.makedirs(log_dir, exist_ok=True)
        write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_STARTUP_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "rank": int(rank),
                    "model_load_s": f"{float(model_load_s):.6f}",
                    "model_to_device_s": f"{float(model_to_device_s):.6f}",
                    "model_startup_total_s": f"{total_s:.6f}",
                }
            )
    if algorithm is not None:
        log_metric = getattr(algorithm, "log_metric", None)
        if callable(log_metric):
            log_metric("model_load_s", float(model_load_s))
            log_metric("model_to_device_s", float(model_to_device_s))
            log_metric("model_startup_total_s", total_s)
    return path
