# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""TopK encode/decode helpers for hybrid global ``LayerState`` gRPC messages."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import src.omnifed.hybrid.communicator.global_grpc_pb2 as global_grpc_pb2
from src.omnifed.hybrid.compression.topk import TOPK_COMPRESSION_NAME, TopKCompression


def build_topk_compressor(
    *,
    enabled: bool,
    scheme: str = "topk",
    compress_ratio: float = 0.01,
    device: torch.device | str = "cpu",
) -> Optional[TopKCompression]:
    if not enabled:
        return None
    if scheme != "topk":
        raise ValueError(f"Unsupported global_compression.scheme={scheme!r}; only 'topk' in v1")
    return TopKCompression(device=device, compress_ratio=float(compress_ratio))


def hybrid_global_compressor_from_cfg(
    cfg: DictConfig,
    device: torch.device | str = "cpu",
) -> Optional[TopKCompression]:
    enabled = bool(OmegaConf.select(cfg, "engine.hybrid.global_compression.enabled", default=False))
    scheme = str(OmegaConf.select(cfg, "engine.hybrid.global_compression.scheme", default="topk"))
    ratio = float(
        OmegaConf.select(cfg, "engine.hybrid.global_compression.compress_ratio", default=0.01)
    )
    return build_topk_compressor(
        enabled=enabled,
        scheme=scheme,
        compress_ratio=ratio,
        device=device,
    )


def encode_layer_state(
    name: str,
    tensor: torch.Tensor,
    compressor: Optional[TopKCompression],
) -> global_grpc_pb2.LayerState:
    layer = global_grpc_pb2.LayerState(layer_name=name)
    t = tensor.detach().cpu()

    if compressor is None:
        layer.param_shape.extend(list(t.shape))
        layer.param_update.extend(t.flatten().tolist())
        return layer

    (values, indices), _ctx = compressor.compress(t, name=name)
    values_np = values.detach().cpu().numpy().astype(np.float32, copy=False)
    indices_np = indices.detach().cpu().numpy().astype(np.int64, copy=False)
    layer.compression_type = TOPK_COMPRESSION_NAME
    layer.values_data = values_np.tobytes()
    layer.indices_data = indices_np.tobytes()
    layer.values_dtype = "torch.float32"
    layer.indices_dtype = "torch.int64"
    layer.original_shape.extend(list(t.shape))
    return layer


def decode_layer_tensor(
    layer: global_grpc_pb2.LayerState,
    *,
    base_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    compression_type = layer.compression_type or None

    if compression_type is None:
        if not layer.param_shape:
            raise ValueError(f"Dense layer {layer.layer_name!r} missing param_shape")
        arr = np.array(layer.param_update, dtype=np.float32).reshape(tuple(layer.param_shape))
        return torch.from_numpy(arr.copy())

    if compression_type != TOPK_COMPRESSION_NAME:
        raise ValueError(f"Unsupported compression_type={compression_type!r}")

    if not layer.values_data or not layer.indices_data:
        raise ValueError(f"Compressed layer {layer.layer_name!r} missing values/indices")

    values = np.frombuffer(layer.values_data, dtype=np.float32)
    indices = np.frombuffer(layer.indices_data, dtype=np.int64)
    original_shape = tuple(layer.original_shape)
    numel = int(np.prod(original_shape))

    if base_tensor is not None:
        flat = base_tensor.detach().cpu().numpy().reshape(-1).copy()
        flat[indices] = values
        return torch.from_numpy(flat.reshape(original_shape).copy())

    dense = np.zeros(numel, dtype=np.float32)
    dense[indices] = values
    return torch.from_numpy(dense.reshape(original_shape).copy())


def encode_updates_dict(
    updates: Dict[str, torch.Tensor],
    compressor: Optional[TopKCompression],
) -> list:
    return [encode_layer_state(name, tensor, compressor) for name, tensor in updates.items()]


def decode_updates_dict(
    proto_layers,
    *,
    base_updates: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for layer in proto_layers:
        base = None if base_updates is None else base_updates.get(layer.layer_name)
        out[layer.layer_name] = decode_layer_tensor(layer, base_tensor=base)
    return out
