# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional, Union
import importlib

import numpy as np
import torch
import torch.nn as nn

from src.omnifed.device_resolver import is_cuda_oom, try_mapping_to_device

from . import grpc_pb2
from collections import defaultdict
from .compression.sparsification import TopKCompression
from .compression.quantization import QSGDQuantCompression

_QSGD_NUMPY_DTYPES = {
    8: np.int8,
    32: np.int32,
}


def get_class_from_str(path: str):
    module_name, class_name = path.rsplit(".", 1)  # split module vs class
    module = importlib.import_module(module_name)  # import the module
    cls = getattr(module, class_name)  # get class by name
    return cls


def aggregation_metric_for_communicate_params(communicate_params: bool) -> str:
    """Payload kind for compress/extract: ``grad`` or ``param``."""
    return "param" if communicate_params else "grad"


def compressor_proto_name(compressor) -> Optional[str]:
    if compressor is None:
        return None
    return compressor.__class__.__name__


def extract_tensordict(msg, aggregation_metric):
    """
    Returns dict[str, Tensor] with stable keys.
    """
    if isinstance(msg, torch.Tensor):
        return {"__tensor__": msg}

    elif isinstance(msg, dict):
        tensordict = {}
        for name, value in msg.items():
            if isinstance(value, torch.Tensor):
                tensordict[name] = value
                continue
            if aggregation_metric == "grad":
                if value.grad is not None:
                    tensordict[name] = value.grad
                else:
                    tensordict[name] = value.data
            elif aggregation_metric == "param":
                tensordict[name] = value.data
            else:
                raise ValueError(
                    f"Unsupported aggregation_metric: {aggregation_metric}"
                )
        return tensordict

    elif isinstance(msg, nn.Module):
        # assume dict[str, Tensor]
        tensordict = {}
        print(msg)
        for name, param in msg.named_parameters():
            # print(f"name = {name} , param = {param}")
            if aggregation_metric == "grad":
                if param.grad is not None:
                    tensordict[name] = param.grad
            elif aggregation_metric == "param":
                tensordict[name] = param.data
            else:
                raise ValueError(
                    f"Unsupported aggregation_metric: {aggregation_metric}"
                )
        return tensordict

    else:
        raise TypeError("Unsupported msg type")


def _compress_single_tensor(compressor, tensor: torch.Tensor, key: str):
    if isinstance(compressor, TopKCompression):
        (values, indices), ctx = compressor.compress(tensor=tensor, name=key)
        return {
            "values": values,
            "indices": indices,
            "original_shape": tensor.shape,
            "ctx": ctx,
        }
    if isinstance(compressor, QSGDQuantCompression):
        signed_levels, norm, width, levels = compressor.compress(tensor, name=key)
        if width == -1 or levels == -1 or norm == -1:
            return tensor
        return {
            "signed_levels": signed_levels,
            "norm": norm,
            "width": width,
            "levels": levels,
            "original_shape": tensor.shape,
            "original_device": str(tensor.device),
        }
    raise TypeError(f"Unsupported compressor type: {type(compressor)!r}")


def compress_message_tensors(msg, compressor, aggregation_metric):
    """
    Returns a compressed representation with 1-1 key correspondence.
    """
    if compressor is None:
        return msg

    if isinstance(msg, torch.Tensor):
        return msg

    tensordict = extract_tensordict(msg, aggregation_metric)

    compressed = {}
    with torch.no_grad():
        for key, tensor in tensordict.items():
            compressed[key] = _compress_single_tensor(compressor, tensor, key)

    return compressed




def tensordict_to_proto(
    tensordict: Dict[str, torch.Tensor], compression_type=None
) -> grpc_pb2.TensorDict:
    """
    Convert tensor dictionary to protobuf format for gRPC transmission.

    Serializes PyTorch tensors to byte format with metadata for exact reconstruction.
    Includes device information and data type preservation.

    Args:
        tensordict: Dictionary mapping parameter names to tensor values,
        compression_type: The compression used


    Returns:
        TensorDict protobuf message ready for gRPC transmission
    """
    entries = []

    for key, item in tensordict.items():

        # ----------------------------------------------------
        # CASE 1: COMPRESSED ENTRY (Top-K)
        # ----------------------------------------------------
        if (
            isinstance(item, dict)
            and compression_type == TopKCompression.__name__
            and "indices" in item
            and item["indices"].numel() > 0
        ):
            values = item["values"]
            indices = item["indices"]
            numel, shape = item["ctx"]
            original_shape = item["original_shape"]

            original_device = str(values.device)

            values_cpu = values.cpu()
            indices_cpu = indices.cpu()

            data_bytes = values_cpu.numpy().tobytes()
            index_bytes = indices_cpu.numpy().tobytes()

            entry = grpc_pb2.TensorEntry(
                key=key,
                data=data_bytes,
                shape=list(values_cpu.shape),
                dtype=str(values_cpu.dtype),
                device=original_device,
                data_size=len(data_bytes),
                compression_type=TopKCompression.__name__,
                index=index_bytes,
                index_shape=list(indices_cpu.shape),
                index_dtype=str(indices_cpu.dtype),
                original_shape=original_shape,
            )

        # ----------------------------------------------------
        # CASE 2: COMPRESSED ENTRY (QSGD)
        # ----------------------------------------------------
        elif (
            isinstance(item, dict)
            and compression_type == QSGDQuantCompression.__name__
            and "signed_levels" in item
        ):
            signed_levels = item["signed_levels"].cpu()
            width = int(item["width"])
            levels = int(item["levels"])
            np_dtype = _QSGD_NUMPY_DTYPES[width]
            levels_np = signed_levels.numpy().astype(np_dtype, copy=False)
            data_bytes = levels_np.tobytes()
            norm_np = np.array([float(item["norm"])], dtype=np.float32)

            entry = grpc_pb2.TensorEntry(
                key=key,
                data=data_bytes,
                shape=list(levels_np.shape),
                dtype=f"torch.int{width}",
                device=item.get("original_device", str(signed_levels.device)),
                data_size=len(data_bytes),
                compression_type=QSGDQuantCompression.__name__,
                original_shape=list(item["original_shape"]),
                meta_tensor=norm_np.tobytes(),
                meta_dtype="torch.float32",
                width=width,
                level=levels,
            )

        # ----------------------------------------------------
        # CASE 3: UNCOMPRESSED DENSE TENSOR
        # ----------------------------------------------------
        else:
            tensor = item

            original_device = str(tensor.device)
            tensor_cpu = tensor.cpu()

            data_bytes = tensor_cpu.numpy().tobytes()

            entry = grpc_pb2.TensorEntry(
                key=key,
                data=data_bytes,
                shape=list(tensor_cpu.shape),
                dtype=str(tensor_cpu.dtype),
                device=original_device,
                data_size=len(data_bytes),
            )

        entries.append(entry)


    return grpc_pb2.TensorDict(entries=entries)



from typing import Dict
import numpy as np
import torch

def proto_to_tensordict(
    proto_tensordict,
) -> Dict[str, torch.Tensor]:
    """
    Convert protobuf TensorDict back to PyTorch tensors.
    """
    tensordict = {}

    dtype_mapping = {
        "torch.float32": np.float32,
        "torch.float64": np.float64,
        "torch.int32": np.int32,
        "torch.int64": np.int64,
        "torch.int8": np.int8,
        "torch.bool": np.bool_,
    }

    for entry in proto_tensordict.entries:
        # ----------------------------
        # Validate dtype
        # ----------------------------
        if entry.dtype not in dtype_mapping:
            raise ValueError(
                f"Unsupported dtype: {entry.dtype}. "
                f"Supported: {list(dtype_mapping.keys())}"
            )

        numpy_dtype = dtype_mapping[entry.dtype]

        # Validate data size (dense case only)
        if len(entry.data) != entry.data_size:
            raise ValueError(
                f"Data size mismatch for tensor {entry.key}: "
                f"expected {entry.data_size}, got {len(entry.data)}"
            )

        numpy_array = np.frombuffer(entry.data, dtype=numpy_dtype)
        # print(f"No compression; numpy_array.shape = {numpy_array.shape}, entry.shape = {entry.shape}")
        numpy_array = numpy_array.reshape(tuple(entry.shape))


        # ----------------------------
        # Convert to torch.Tensor
        # ----------------------------
        # .copy() because frombuffer gives a read-only view
        tensor = torch.from_numpy(numpy_array.copy()).to(entry.device)
        tensordict[entry.key] = tensor

    return tensordict


def _place_cpu_tensor(tensor: torch.Tensor, compute_device: Optional[Any]) -> torch.Tensor:
    """Protobuf unpack is CPU; optionally move to ``compute_device`` (OOM → CPU)."""
    if compute_device is None:
        return tensor
    placed, _used = try_mapping_to_device(tensor, compute_device)
    return placed


def _topk_dense_from_numpy(
    values: np.ndarray,
    indices: np.ndarray,
    numel: int,
    original_shape,
    numpy_dtype,
    compute_device: Optional[Any],
) -> torch.Tensor:
    """Zero-fill + scatter. Try GPU when ``compute_device`` is CUDA; OOM → CPU."""
    wanted = torch.device(compute_device) if compute_device is not None else torch.device("cpu")
    if wanted.type == "cuda":
        try:
            values_t = torch.from_numpy(np.ascontiguousarray(values)).to(wanted)
            indices_t = torch.from_numpy(
                np.ascontiguousarray(indices, dtype=np.int64)
            ).to(wanted)
            dense = torch.zeros(numel, dtype=values_t.dtype, device=wanted)
            dense.scatter_(0, indices_t, values_t)
            return dense.view(tuple(original_shape))
        except Exception as exc:
            if not is_cuda_oom(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    dense_np = np.zeros(numel, dtype=numpy_dtype)
    dense_np[indices] = values
    return torch.from_numpy(dense_np.reshape(tuple(original_shape)).copy())


def proto_to_tensordict_extended(
    proto_tensordict,
    overlay_base: Optional[Any] = None,
    compute_device: Optional[Any] = None,
) -> tuple[Dict[str, torch.Tensor], bool]:
    """
    Convert protobuf TensorDict back to PyTorch tensors.
    Supports uncompressed, Top-K, and QSGD compressed tensors.

    For sync grad aggregation, pass ``overlay_base=None`` so Top-K entries
    zero-fill then scatter (full sparse message). Pass a tensor dict only when
    intentionally overlaying sparse values onto an existing base (legacy paths).

    Unpack is always from CPU bytes. Pass ``compute_device`` (client/server
    ``agg_device``) to run decompress on GPU when it fits.
    """
    tensordict = {}

    is_model_communicated = False

    dtype_mapping = {
        "torch.float32": np.float32,
        "torch.float64": np.float64,
        "torch.int32": np.int32,
        "torch.int64": np.int64,
        "torch.int8": np.int8,
        "torch.bool": np.bool_,
    }

    overlay_lookup = None
    if overlay_base is not None:
        if isinstance(overlay_base, dict):
            overlay_lookup = overlay_base
        elif isinstance(overlay_base, nn.Module):
            overlay_lookup = {
                name: param.data for name, param in overlay_base.named_parameters()
            }

    for entry in proto_tensordict.entries:
        if entry.dtype not in dtype_mapping:
            raise ValueError(
                f"Unsupported dtype: {entry.dtype}. "
                f"Supported: {list(dtype_mapping.keys())}"
            )

        numpy_dtype = dtype_mapping[entry.dtype]
        compression_type = entry.compression_type or None

        if compression_type == TopKCompression.__name__:
            if not entry.index:
                raise ValueError(
                    f"Missing indices for compressed tensor {entry.key}"
                )

            index_dtype = dtype_mapping[entry.index_dtype]
            values = np.frombuffer(entry.data, dtype=numpy_dtype)
            indices = np.frombuffer(entry.index, dtype=index_dtype)
            numel = int(np.prod(entry.original_shape))

            if len(indices) != len(values):
                raise RuntimeError(
                    f"Mismatch: indices ({len(indices)}) != values ({len(values)})"
                )

            if indices.size == 0:
                raise RuntimeError(
                    "proto_to_tensordict -> Index array is empty for TopKCompression"
                )

            if overlay_lookup is not None and entry.key in overlay_lookup:
                base = overlay_lookup[entry.key]
                flat = base.detach().cpu().numpy().reshape(-1).copy()
                flat[indices] = values
                dense = flat.reshape(entry.original_shape)
                is_model_communicated = True
                numpy_array = dense.reshape(tuple(entry.original_shape))
            else:
                tensordict[entry.key] = _topk_dense_from_numpy(
                    values,
                    indices,
                    numel,
                    entry.original_shape,
                    numpy_dtype,
                    compute_device,
                )
                continue

        elif compression_type == QSGDQuantCompression.__name__:
            if not entry.meta_tensor:
                raise ValueError(
                    f"Missing meta_tensor (norm) for QSGD tensor {entry.key}"
                )
            if entry.width not in _QSGD_NUMPY_DTYPES:
                raise ValueError(
                    f"QSGD tensor {entry.key} has unsupported width={entry.width}"
                )
            if entry.level <= 0:
                raise ValueError(
                    f"QSGD tensor {entry.key} has invalid level={entry.level}"
                )
            qsgd_dtype = _QSGD_NUMPY_DTYPES[entry.width]
            signed_levels = np.frombuffer(entry.data, dtype=qsgd_dtype).reshape(
                tuple(entry.original_shape)
            )
            norm = float(np.frombuffer(entry.meta_tensor, dtype=np.float32)[0])
            signed_t = torch.from_numpy(np.array(signed_levels, copy=True))
            if compute_device is not None:
                signed_t, _used = try_mapping_to_device(signed_t, compute_device)
            restored = QSGDQuantCompression.decompress_quantized(
                signed_t,
                norm,
                int(entry.level),
                tuple(entry.original_shape),
            )
            tensordict[entry.key] = restored
            continue

        elif compression_type is None:
            if len(entry.data) != entry.data_size:
                raise ValueError(
                    f"Data size mismatch for tensor {entry.key}: "
                    f"expected {entry.data_size}, got {len(entry.data)}"
                )

            numpy_array = np.frombuffer(entry.data, dtype=numpy_dtype)
            numpy_array = numpy_array.reshape(tuple(entry.shape))

        else:
            raise ValueError(
                f"Unsupported compression type: {compression_type}, the type is {type(compression_type)}"
            )

        tensor = torch.from_numpy(numpy_array.copy())
        tensordict[entry.key] = _place_cpu_tensor(tensor, compute_device)

    return tensordict, is_model_communicated


# def proto_to_tensordict(
#     proto_tensordict: grpc_pb2.TensorDict,
# ) -> Dict[str, torch.Tensor]:
#     """
#     Convert protobuf tensor dictionary back to PyTorch tensors.

#     Deserializes byte data back to PyTorch tensors with original shapes,
#     data types, and device placement preserved.

#     Args:
#         proto_tensordict: TensorDict protobuf message from gRPC

#     Returns:
#         Dictionary mapping parameter names to reconstructed tensors

#     Raises:
#         ValueError: If data size mismatch or unsupported dtype
#     """
#     tensordict = {}
#     for entry in proto_tensordict.entries:
#         # Validate data size
#         if len(entry.data) != entry.data_size:
#             raise ValueError(
#                 f"Data size mismatch for tensor {entry.key}: expected {entry.data_size}, got {len(entry.data)}"
#             )

#         # Dtype mapping: string -> numpy_dtype
#         dtype_mapping = {
#             "torch.float32": np.float32,
#             "torch.float64": np.float64,
#             "torch.int32": np.int32,
#             "torch.int64": np.int64,
#             "torch.bool": np.bool_,
#         }

#         if entry.dtype not in dtype_mapping:
#             supported_dtypes = list(dtype_mapping.keys())
#             raise ValueError(
#                 f"Unsupported dtype: {entry.dtype}. Supported: {supported_dtypes}"
#             )

#         numpy_dtype = dtype_mapping[entry.dtype]

#         # Reconstruct tensor from serialized bytes
#         numpy_array = np.frombuffer(entry.data, dtype=numpy_dtype)
#         numpy_array = numpy_array.reshape(tuple(entry.shape))

#         # Create tensor and restore to original device
#         # Note: .copy() needed because np.frombuffer creates read-only arrays
#         tensor = torch.from_numpy(numpy_array.copy()).to(entry.device)

#         tensordict[entry.key] = tensor

#     return tensordict


def get_msg_info(
    msg: Union[torch.Tensor, nn.Module, Dict[str, Any], Any],
) -> Dict[str, Any]:
    """
    Extract metadata from message for logging and debugging.

    Provides structured information about tensors, models, or dictionaries
    for communication operation logging and troubleshooting.

    Args:
        msg: Message to analyze (tensor, model, or dict)

    Returns:
        Dictionary with type, shape, device, and size information

    Raises:
        TypeError: If message type is not supported
    """
    info: Dict[str, Any] = {
        "type": type(msg).__name__,
    }

    # Extract tensor metadata for logging
    if isinstance(msg, torch.Tensor):
        info.update(
            {
                "shape": list(msg.shape),
                "numel": msg.numel(),
                "dtype": str(msg.dtype),
                "device": str(msg.device),
            }
        )
    elif isinstance(msg, nn.Module):
        info["params"] = sum(p.numel() for p in msg.parameters())
        tensor = next(msg.parameters(), None)
        if tensor is not None:
            info.update(
                {
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                }
            )
    elif isinstance(msg, dict):
        keys = list(msg.keys())
        # Limit keys output to prevent log overflow
        if len(keys) <= 5:
            info["keys"] = keys
        else:
            info["keys"] = keys[:3] + ["...", f"({len(keys)} total)"]

        # Add tensor-specific information if dictionary contains tensors
        tensor_values = [v for v in msg.values() if isinstance(v, torch.Tensor)]
        if tensor_values:
            info["tensors"] = len(tensor_values)
            info["total_params"] = sum(t.numel() for t in tensor_values)

            # Use first tensor for dtype/device info
            first_tensor = tensor_values[0]
            info.update(
                {
                    "dtype": str(first_tensor.dtype),
                    "device": str(first_tensor.device),
                }
            )
    else:
        raise TypeError(f"Unsupported message type: {type(msg)}")

    return info
