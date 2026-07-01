# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""TopK and QSGD encode/decode for hybrid global ``LayerState`` gRPC messages."""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import src.omnifed.hybrid.communicator.global_grpc_pb2 as global_grpc_pb2
from src.omnifed.hybrid.compression.qsgd import QSGD_COMPRESSION_NAME, QSGDQuantCompression
from src.omnifed.hybrid.compression.topk import TOPK_COMPRESSION_NAME, TopKCompression

GlobalHybridCompressor = Union[TopKCompression, QSGDQuantCompression]

_QSGD_NUMPY_DTYPES = {
    8: np.int8,
    32: np.int32,
}


def compression_mode_name(compressor: Optional[GlobalHybridCompressor]) -> str:
    if compressor is None:
        return "dense"
    if isinstance(compressor, TopKCompression):
        return "TopK"
    if isinstance(compressor, QSGDQuantCompression):
        return "QSGD"
    return type(compressor).__name__


def build_global_compressor(
    *,
    enabled: bool,
    scheme: str = "topk",
    compress_ratio: float = 0.01,
    bit_width: int = 8,
    device: torch.device | str = "cpu",
) -> Optional[GlobalHybridCompressor]:
    if not enabled:
        return None
    scheme_norm = str(scheme).lower()
    if scheme_norm == "topk":
        return TopKCompression(device=device, compress_ratio=float(compress_ratio))
    if scheme_norm == "qsgd":
        return QSGDQuantCompression(bit_width=int(bit_width), device=device)
    raise ValueError(
        f"Unsupported global_compression.scheme={scheme!r}; expected 'topk' or 'qsgd'"
    )


def hybrid_global_compressor_from_cfg(
    cfg: DictConfig,
    device: torch.device | str = "cpu",
) -> Optional[GlobalHybridCompressor]:
    enabled = bool(OmegaConf.select(cfg, "engine.hybrid.global_compression.enabled", default=False))
    scheme = str(OmegaConf.select(cfg, "engine.hybrid.global_compression.scheme", default="topk"))
    ratio = float(
        OmegaConf.select(cfg, "engine.hybrid.global_compression.compress_ratio", default=0.01)
    )
    bit_width = int(
        OmegaConf.select(cfg, "engine.hybrid.global_compression.bit_width", default=8)
    )
    return build_global_compressor(
        enabled=enabled,
        scheme=scheme,
        compress_ratio=ratio,
        bit_width=bit_width,
        device=device,
    )


def _encode_dense_layer(name: str, tensor: torch.Tensor) -> global_grpc_pb2.LayerState:
    layer = global_grpc_pb2.LayerState(layer_name=name)
    t = tensor.detach().cpu()
    layer.param_shape.extend(list(t.shape))
    layer.param_update.extend(t.flatten().tolist())
    return layer


def _encode_topk_layer(
    name: str, tensor: torch.Tensor, compressor: TopKCompression
) -> global_grpc_pb2.LayerState:
    layer = global_grpc_pb2.LayerState(layer_name=name)
    t = tensor.detach().cpu()
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


def _encode_qsgd_layer(
    name: str, tensor: torch.Tensor, compressor: QSGDQuantCompression
) -> global_grpc_pb2.LayerState:
    layer = global_grpc_pb2.LayerState(layer_name=name)
    t = tensor.detach().cpu()
    signed_levels, norm, width, levels = compressor.compress(t, name=name)

    if width == -1 or levels == -1 or norm == -1:
        return _encode_dense_layer(name, t)

    np_dtype = _QSGD_NUMPY_DTYPES[width]
    levels_np = signed_levels.detach().cpu().numpy().astype(np_dtype, copy=False)
    norm_np = np.array([float(norm)], dtype=np.float32)

    layer.compression_type = QSGD_COMPRESSION_NAME
    layer.values_data = levels_np.tobytes()
    layer.values_dtype = f"torch.int{width}"
    layer.original_shape.extend(list(t.shape))
    layer.meta_tensor = norm_np.tobytes()
    layer.meta_tensor_dtype = "torch.float32"
    layer.width = int(width)
    layer.level = int(levels)
    return layer


def encode_layer_state(
    name: str,
    tensor: torch.Tensor,
    compressor: Optional[GlobalHybridCompressor],
) -> global_grpc_pb2.LayerState:
    if compressor is None:
        return _encode_dense_layer(name, tensor)
    if isinstance(compressor, TopKCompression):
        return _encode_topk_layer(name, tensor, compressor)
    if isinstance(compressor, QSGDQuantCompression):
        return _encode_qsgd_layer(name, tensor, compressor)
    raise TypeError(f"Unsupported compressor type: {type(compressor)!r}")


def _decode_topk_layer(
    layer: global_grpc_pb2.LayerState,
    *,
    base_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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


def _decode_qsgd_layer(layer: global_grpc_pb2.LayerState) -> torch.Tensor:
    if not layer.values_data:
        raise ValueError(f"QSGD layer {layer.layer_name!r} missing values_data")
    if not layer.meta_tensor:
        raise ValueError(f"QSGD layer {layer.layer_name!r} missing meta_tensor (norm)")
    if layer.width not in _QSGD_NUMPY_DTYPES:
        raise ValueError(f"QSGD layer {layer.layer_name!r} has unsupported width={layer.width}")
    if layer.level <= 0:
        raise ValueError(f"QSGD layer {layer.layer_name!r} has invalid level={layer.level}")

    np_dtype = _QSGD_NUMPY_DTYPES[layer.width]
    signed_levels = np.frombuffer(layer.values_data, dtype=np_dtype).reshape(tuple(layer.original_shape))
    norm = float(np.frombuffer(layer.meta_tensor, dtype=np.float32).reshape(-1)[0])
    restored = QSGDQuantCompression.decompress_quantized(
        torch.from_numpy(signed_levels.copy()),
        norm,
        int(layer.level),
        tuple(layer.original_shape),
    )
    return restored


def decode_layer_tensor(
    layer: global_grpc_pb2.LayerState,
    *,
    base_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    compression_type = layer.compression_type or None

    if compression_type is None or compression_type == "":
        if not layer.param_shape:
            raise ValueError(f"Dense layer {layer.layer_name!r} missing param_shape")
        arr = np.array(layer.param_update, dtype=np.float32).reshape(tuple(layer.param_shape))
        return torch.from_numpy(arr.copy())

    if compression_type == TOPK_COMPRESSION_NAME:
        return _decode_topk_layer(layer, base_tensor=base_tensor)

    if compression_type == QSGD_COMPRESSION_NAME:
        return _decode_qsgd_layer(layer)

    raise ValueError(f"Unsupported compression_type={compression_type!r}")


def encode_updates_dict(
    updates: Dict[str, torch.Tensor],
    compressor: Optional[GlobalHybridCompressor],
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
