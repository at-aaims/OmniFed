"""Per-process Slurm device selection (classic path; hybrid can reuse later).

``model_device`` is where the local module lives.
``agg_device`` is where SUM/compress run when they fit (OOM → CPU).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def is_cuda_oom(exc: BaseException) -> bool:
    """True for CUDA/HIP allocator failures (Frontier reports HIP OOM)."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", ())
    if oom_cls and isinstance(exc, oom_cls):
        return True
    msg = str(exc).lower()
    if "out of memory" not in msg:
        return False
    return any(tag in msg for tag in ("cuda", "hip", "rocm"))


def mapping_to_device(data: Any, device: torch.device | str) -> Any:
    """Move tensors in a mapping (or a single tensor) onto ``device``."""
    device = torch.device(device)
    if isinstance(data, dict):
        return {key: mapping_to_device(value, device) for key, value in data.items()}
    if torch.is_tensor(data):
        if data.device == device:
            return data
        return data.to(device, non_blocking=False)
    return data


def try_mapping_to_device(data: Any, device: torch.device | str) -> tuple[Any, torch.device]:
    """Move onto ``device``; on GPU OOM return CPU tensors instead."""
    wanted = torch.device(device)
    try:
        return mapping_to_device(data, wanted), wanted
    except Exception as exc:
        if wanted.type != "cuda" or not is_cuda_oom(exc):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return mapping_to_device(data, "cpu"), torch.device("cpu")


@dataclass(frozen=True)
class SlurmDevices:
    model_device: torch.device
    agg_device: torch.device


def is_classic_grpc_server(local_comm: Any) -> bool:
    """True only for classic ``GrpcCommunicator`` rank-0 (``is_server`` is a bool)."""
    if not hasattr(local_comm, "is_server"):
        return False
    val = local_comm.is_server
    return isinstance(val, bool) and val


def cuda_device_for_local_rank(local_rank: int) -> torch.device:
    """``cuda:{local_rank % device_count}``, or CPU if this process sees no GPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    n = int(torch.cuda.device_count())
    if n <= 0:
        return torch.device("cpu")
    return torch.device("cuda", int(local_rank) % n)


def resolve_slurm_devices(
    node_cfg: Any,
    *,
    rank: int,
    local_rank: int,
    local_comm: Any,
) -> SlurmDevices:
    """Pick model vs aggregation devices for one Slurm task.

    1. ``topology.overrides.<rank>.device_hint`` (not ``auto``) wins for both.
    2. Classic gRPC server: model stays CPU; ``agg_device`` is GPU if visible.
    3. Trainers (clients, TorchDist ranks): GPU when communicator backend is
       NCCL and CUDA is available, else CPU. ``agg_device`` matches the model.
    """
    hint = getattr(node_cfg, "device_hint", "auto")
    if hint is None:
        hint = "auto"
    hint_s = str(hint).strip()
    if hint_s.lower() != "auto":
        device = torch.device(hint_s)
        print(
            f"[device_resolver] rank={rank} explicit device_hint={hint_s}",
            flush=True,
        )
        return SlurmDevices(model_device=device, agg_device=device)

    gpu = cuda_device_for_local_rank(local_rank)
    backend = str(getattr(local_comm, "backend", "gloo") or "gloo").lower()
    trainer_on_gpu = backend == "nccl" and gpu.type == "cuda"

    if is_classic_grpc_server(local_comm):
        model_device = torch.device("cpu")
        print(
            f"[device_resolver] rank={rank} classic gRPC server: "
            f"model_device={model_device} agg_device={gpu} "
            f"(SUM/compress uses agg_device, OOM falls back to CPU)",
            flush=True,
        )
        return SlurmDevices(model_device=model_device, agg_device=gpu)

    model_device = gpu if trainer_on_gpu else torch.device("cpu")
    print(
        f"[device_resolver] rank={rank} trainer: "
        f"model_device={model_device} agg_device={model_device}",
        flush=True,
    )
    return SlurmDevices(model_device=model_device, agg_device=model_device)
