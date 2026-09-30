# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""TorchDist collective helpers for compressed aggregation (Top-K, QSGD)."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.distributed as dist

from src.omnifed.summary.per_iteration import accumulate_iter_comm

from .quantization import QSGDQuantCompression
from .sparsification import TopKCompression


def _sync_cuda_for_timing() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            return


def _record_comm_time(logger: Any, key: str, t0: float) -> None:
    """Accumulate codec time (pack/unpack). Collectives stay in grpc_agg_grad_s."""
    _sync_cuda_for_timing()
    if logger is not None:
        accumulate_iter_comm(logger, key, time.perf_counter() - t0)


def _all_gather_sparse_payloads(
    values: torch.Tensor,
    indices: torch.Tensor,
    *,
    world_size: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
    """Pad + ``all_gather`` Top-K payloads. Collective time belongs in grpc_agg_grad_s."""
    tensor_size = int(values.numel())
    size_tensor = torch.tensor([tensor_size], device=device, dtype=torch.long)
    size_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(size_list, size_tensor)
    sizes = [int(s.item()) for s in size_list]
    max_size = max(sizes) if sizes else 0
    if max_size == 0:
        return [], [], sizes

    val_buf = values
    ix_buf = indices.to(device=device, dtype=torch.long)
    if tensor_size < max_size:
        val_pad = torch.zeros(max_size - tensor_size, dtype=values.dtype, device=device)
        ix_pad = torch.zeros(max_size - tensor_size, dtype=torch.long, device=device)
        val_buf = torch.cat((val_buf, val_pad), dim=0)
        ix_buf = torch.cat((ix_buf, ix_pad), dim=0)

    val_list = [
        torch.zeros(max_size, dtype=values.dtype, device=device) for _ in range(world_size)
    ]
    ix_list = [
        torch.zeros(max_size, dtype=torch.long, device=device) for _ in range(world_size)
    ]
    dist.all_gather(val_list, val_buf)
    dist.all_gather(ix_list, ix_buf)
    return val_list, ix_list, sizes


def _sparse_gathered_to_dense(
    val_list: list[torch.Tensor],
    ix_list: list[torch.Tensor],
    sizes: list[int],
    *,
    numel: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    out = torch.zeros(numel, dtype=dtype, device=device)
    for r, n in enumerate(sizes):
        if n <= 0:
            continue
        out.index_add_(0, ix_list[r][:n], val_list[r][:n].to(dtype=dtype))
    return out


def all_gather_sparse_sum(
    values: torch.Tensor,
    indices: torch.Tensor,
    *,
    numel: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pad + ``all_gather`` Top-K payloads, then SUM into a dense vector."""
    val_list, ix_list, sizes = _all_gather_sparse_payloads(
        values, indices, world_size=world_size, device=device
    )
    if not sizes or max(sizes) == 0:
        return torch.zeros(numel, dtype=dtype, device=device)
    return _sparse_gathered_to_dense(
        val_list, ix_list, sizes, numel=numel, dtype=dtype, device=device
    )


def aggregate_topk_tensor(
    compressor: TopKCompression,
    tensor: torch.Tensor,
    *,
    name: str,
    world_size: int,
    logger: Any = None,
) -> torch.Tensor:
    """Top-K compress on device, sparse ``all_gather``, SUM (matches gRPC server SUM)."""
    device = tensor.device
    _sync_cuda_for_timing()
    t0 = time.perf_counter()
    (values, indices), ctx = compressor.compress(tensor, name)
    _record_comm_time(logger, "grpc_pack_s", t0)

    numel, shape = ctx
    val_list, ix_list, sizes = _all_gather_sparse_payloads(
        values, indices, world_size=world_size, device=device
    )
    _sync_cuda_for_timing()
    t0 = time.perf_counter()
    if not sizes or max(sizes) == 0:
        dense = torch.zeros(numel, dtype=tensor.dtype, device=device)
    else:
        dense = _sparse_gathered_to_dense(
            val_list,
            ix_list,
            sizes,
            numel=numel,
            dtype=tensor.dtype,
            device=device,
        )
    dense = dense.view(shape)
    _record_comm_time(logger, "grpc_unpack_s", t0)
    return dense


def aggregate_qsgd_tensor(
    compressor: QSGDQuantCompression,
    tensor: torch.Tensor,
    *,
    name: str,
    op: dist.ReduceOp,
    logger: Any = None,
) -> torch.Tensor:
    """QSGD quantize on device, dequantize locally, dense ``all_reduce`` SUM."""
    _sync_cuda_for_timing()
    t0 = time.perf_counter()
    signed, norm, width, levels = compressor.compress(tensor, name)
    _record_comm_time(logger, "grpc_pack_s", t0)

    if width == -1 or levels == -1 or norm == -1:
        dist.all_reduce(tensor, op=op)
        return tensor

    _sync_cuda_for_timing()
    t0 = time.perf_counter()
    dense = QSGDQuantCompression.decompress_quantized(
        signed, norm, levels, tensor.shape
    ).to(device=tensor.device, dtype=tensor.dtype)
    _record_comm_time(logger, "grpc_unpack_s", t0)

    dist.all_reduce(dense, op=op)
    return dense


def is_topk_compressor(compressor: Any) -> bool:
    return isinstance(compressor, TopKCompression)


def is_qsgd_compressor(compressor: Any) -> bool:
    return isinstance(compressor, QSGDQuantCompression)


__all__ = [
    "aggregate_qsgd_tensor",
    "aggregate_topk_tensor",
    "all_gather_sparse_sum",
    "is_qsgd_compressor",
    "is_topk_compressor",
]
