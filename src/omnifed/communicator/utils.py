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

from typing import Any, Dict, Union
import importlib

import numpy as np
import torch
import torch.nn as nn

from . import grpc_pb2
from .compression.sparsification import _sparse_compression_
from .compression.quantization import QSGDQuantCompression


DTYPE_MAPPING = {
    "torch.float32": np.float32,
    "torch.float64": np.float64,
    "torch.int8": np.int8,
    "torch.int16": np.int16,
    "torch.int32": np.int32,
    "torch.int64": np.int64,
    "torch.bool": np.bool_,
}


def _numpy_dtype(dtype):
    if dtype not in DTYPE_MAPPING:
        raise ValueError(
            f"Unsupported dtype: {dtype}. Supported: {list(DTYPE_MAPPING.keys())}"
        )
    return DTYPE_MAPPING[dtype]


def _tensor_entry(
    key,
    tensor,
    compression_type=None,
    meta_tensor=None,
    original_shape=None,
    width=0,
    level=0,
):
    assert isinstance(tensor, torch.Tensor), (
        f"Expected torch.Tensor, but got {type(tensor)}. value={tensor}"
    )

    tensor_cpu = tensor.detach().cpu().contiguous()
    data_bytes = tensor_cpu.numpy().tobytes()

    kwargs = {
        "key": key,
        "data": data_bytes,
        "shape": list(tensor_cpu.shape),
        "dtype": str(tensor_cpu.dtype),
        "device": str(tensor.device),
        "data_size": len(data_bytes),
        "compression_type": compression_type or "",
    }

    if original_shape is not None:
        kwargs["original_shape"] = list(original_shape)
    if meta_tensor is not None:
        meta_tensor_cpu = torch.as_tensor(meta_tensor).detach().cpu().contiguous()
        kwargs.update(
            {
                "meta_tensor": meta_tensor_cpu.numpy().tobytes(),
                "meta_tensor_shape": list(meta_tensor_cpu.shape),
                "meta_tensor_dtype": str(meta_tensor_cpu.dtype),
            }
        )
    if width:
        kwargs["width"] = width
    if level:
        kwargs["level"] = level

    return grpc_pb2.TensorEntry(**kwargs)


def _dense_entry(key, item):
    tensor = item["values"] if isinstance(item, dict) and "values" in item else item
    return _tensor_entry(key, tensor)


def _dense_numpy_array(entry):
    if len(entry.data) != entry.data_size:
        raise ValueError(
            f"Data size mismatch for tensor {entry.key}: "
            f"expected {entry.data_size}, got {len(entry.data)}"
        )
    return np.frombuffer(entry.data, dtype=_numpy_dtype(entry.dtype)).reshape(
        tuple(entry.shape)
    )


class SparseTensorCodec:
    @staticmethod
    def compress(compressor, tensor, name):
        (values, indices), ctx = compressor.compress(tensor=tensor, name=name)
        return {
            "values": values,
            "indices": indices,
            "original_shape": tensor.shape,
            "ctx": ctx,
        }

    @staticmethod
    def can_serialize(item):
        return isinstance(item, dict) and "values" in item and "indices" in item

    @staticmethod
    def to_entry(key, item, compression_type):
        indices = item["indices"]
        if indices.numel() == 0:
            return _dense_entry(key, item)

        return _tensor_entry(
            key,
            item["values"],
            compression_type=compression_type,
            meta_tensor=indices,
            original_shape=item["original_shape"],
        )

    @staticmethod
    def from_entry(entry, server_model):
        numpy_dtype = _numpy_dtype(entry.dtype)
        if not entry.meta_tensor:
            raise ValueError(f"Missing meta_tensor for compressed tensor {entry.key}")

        values = np.frombuffer(entry.data, dtype=numpy_dtype)
        indices = np.frombuffer(
            entry.meta_tensor, dtype=_numpy_dtype(entry.meta_tensor_dtype)
        )
        if len(indices) != len(values):
            raise RuntimeError(
                f"Mismatch: meta_tensor ({len(indices)}) != values ({len(values)})"
            )
        if indices.size == 0:
            raise RuntimeError(
                f"proto_to_tensordict -> meta_tensor array is empty for {entry.compression_type}"
            )

        numel = int(np.prod(entry.original_shape))
        dense = np.zeros(numel, dtype=numpy_dtype)
        is_model_communicated = False

        try:
            flat = server_model[entry.key].detach().cpu().numpy().reshape(-1).copy()
            flat[indices] = values
            dense = flat.reshape(entry.original_shape)
            is_model_communicated = True
        except Exception:
            dense[indices] = values

        return dense.reshape(tuple(entry.original_shape)), is_model_communicated


class QSGDTensorCodec:
    _STORAGE_DTYPES = {
        8: torch.int8,
        16: torch.int16,
        32: torch.int32,
    }
    _NUMPY_STORAGE_DTYPES = {
        8: np.int8,
        16: np.int16,
        32: np.int32,
    }

    @classmethod
    def _assert_reduced_tensor(cls, values, width, level, norm, where):
        assert width in cls._STORAGE_DTYPES, (
            f"{where}: QSGD width must be one of {sorted(cls._STORAGE_DTYPES)}, "
            f"got {width}"
        )
        assert level > 0, f"{where}: QSGD level must be positive, got {level}"
        assert norm is not None, f"{where}: QSGD norm is missing"
        assert isinstance(values, torch.Tensor), (
            f"{where}: QSGD values must be a torch.Tensor, got {type(values)}"
        )
        assert values.dtype == cls._STORAGE_DTYPES[width], (
            f"{where}: QSGD values must be reduced integer levels with dtype "
            f"{cls._STORAGE_DTYPES[width]}, got {values.dtype}"
        )

    @classmethod
    def _assert_reduced_entry(cls, entry):
        assert entry.compression_type == QSGDQuantCompression.__name__, (
            f"QSGD proto entry has wrong compression_type={entry.compression_type!r}"
        )
        assert entry.width in cls._NUMPY_STORAGE_DTYPES, (
            f"QSGD proto width must be one of {sorted(cls._NUMPY_STORAGE_DTYPES)}, "
            f"got {entry.width}"
        )
        assert entry.level > 0, f"QSGD proto level must be positive, got {entry.level}"
        assert entry.meta_tensor, f"QSGD proto entry {entry.key} is missing norm"
        expected_dtype = cls._NUMPY_STORAGE_DTYPES[entry.width]
        assert _numpy_dtype(entry.dtype) == expected_dtype, (
            f"QSGD proto entry {entry.key} must store reduced integer levels as "
            f"{expected_dtype}, got {entry.dtype}"
        )
        expected_size = int(np.prod(entry.shape)) * np.dtype(expected_dtype).itemsize
        assert len(entry.data) == expected_size, (
            f"QSGD proto entry {entry.key} data size mismatch for reduced payload: "
            f"expected {expected_size}, got {len(entry.data)}"
        )

    @staticmethod
    def compress(compressor, tensor, name):
        values, norm, width, levels = compressor.compress(tensor=tensor, name=name)
        if width == -1 or levels == -1 or norm is None:
            values = tensor
        else:
            QSGDTensorCodec._assert_reduced_tensor(
                values, width, levels, norm, f"QSGD compress({name})"
            )
        return {
            "values": values,
            "norm": norm,
            "width": width,
            "level": levels,
            "original_shape": tensor.shape,
        }

    @staticmethod
    def can_serialize(item):
        return isinstance(item, dict) and "values" in item and "width" in item

    @staticmethod
    def to_entry(key, item, compression_type):
        width = item.get("width", -1)
        level = item.get("level", -1)
        norm = item.get("norm")

        if width == -1 or level == -1 or norm is None:
            return _dense_entry(key, item)

        QSGDTensorCodec._assert_reduced_tensor(
            item["values"], width, level, norm, f"QSGD serialize({key})"
        )
        return _tensor_entry(
            key,
            item["values"],
            compression_type=compression_type,
            meta_tensor=torch.as_tensor(norm, dtype=torch.float32),
            original_shape=item["original_shape"],
            width=width,
            level=level,
        )

    @staticmethod
    def from_entry(entry, server_model):
        if entry.width == -1 or entry.level == -1:
            return _dense_numpy_array(entry), False

        QSGDTensorCodec._assert_reduced_entry(entry)
        qsgd_numpy_dtype = QSGDTensorCodec._NUMPY_STORAGE_DTYPES[entry.width]

        encoded_array = np.frombuffer(entry.data, dtype=qsgd_numpy_dtype).reshape(
            tuple(entry.shape)
        )
        norm_array = np.frombuffer(
            entry.meta_tensor, dtype=_numpy_dtype(entry.meta_tensor_dtype)
        ).reshape(tuple(entry.meta_tensor_shape))
        numpy_array = (
            float(norm_array.reshape(-1)[0])
            * encoded_array.astype(np.float32)
            / entry.level
        )
        if entry.original_shape:
            numpy_array = numpy_array.reshape(tuple(entry.original_shape))
        return numpy_array, False


_COMPRESSION_CODECS = {
    **{name: SparseTensorCodec for name in _sparse_compression_},
    QSGDQuantCompression.__name__: QSGDTensorCodec,
}


def _codec_for_compressor(compressor):
    compression_type = compressor.__class__.__name__
    try:
        return compression_type, _COMPRESSION_CODECS[compression_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported compressor: {compression_type}. "
            f"Register a codec in _COMPRESSION_CODECS."
        ) from exc


def get_class_from_str(path: str):
    module_name, class_name = path.rsplit(".", 1)  # split module vs class
    module = importlib.import_module(module_name)  # import the module
    cls = getattr(module, class_name)  # get class by name
    return cls


def extract_tensordict(msg, aggregation_metric):
    """
    Returns dict[str, Tensor] with stable keys.
    """
    if isinstance(msg, torch.Tensor):
        return {"__tensor__": msg}

    elif isinstance(msg, dict):
        # assume dict[str, Tensor]
        tensordict = {}
        # print(msg)
        for name, param in msg.items():
            # print(f"name = {name} , param = {param}")
            if aggregation_metric == "grad":
                if param.grad is not None:
                    tensordict[name] = param.grad
                else:
                    tensordict[name] = param.data
            elif aggregation_metric == "param":
                tensordict[name] = param.data
            else:
                raise ValueError(
                    f"Unsupported aggregation_metric: {aggregation_metric}"
                )
        return tensordict

    elif isinstance(msg, nn.Module):
        # assume dict[str, Tensor]
        tensordict = {}
        # print(msg)
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


def compress_message_tensors(msg, compressor, aggregation_metric):
    """
    Returns a compressed representation with 1-1 key correspondence.
    """
    if compressor is None or isinstance(msg, torch.Tensor):
        return msg

    compression_type, codec = _codec_for_compressor(compressor)
    tensordict = extract_tensordict(msg, aggregation_metric)
    compressed = {}

    with torch.no_grad():
        for key, tensor in tensordict.items():
            compressed[key] = codec.compress(compressor, tensor, key)
            compressed[key]["compression_type"] = compression_type
    return compressed




def tensordict_to_proto_extended(
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
        item_compression_type = (
            item.get("compression_type", compression_type)
            if isinstance(item, dict)
            else compression_type
        )
        codec = _COMPRESSION_CODECS.get(item_compression_type)

        if codec is not None and codec.can_serialize(item):
            entry = codec.to_entry(key, item, item_compression_type)
        else:
            entry = _dense_entry(key, item)

        entries.append(entry)

    return grpc_pb2.TensorDict(entries=entries)


def tensordict_to_proto(
    tensordict: Dict[str, torch.Tensor], compression_type=None
) -> grpc_pb2.TensorDict:
    return tensordict_to_proto_extended(tensordict, compression_type)



def proto_to_tensordict(
    proto_tensordict,
) -> Dict[str, torch.Tensor]:
    """
    Convert protobuf TensorDict back to PyTorch tensors.
    """
    tensordict = {}

    for entry in proto_tensordict.entries:
        numpy_array = _dense_numpy_array(entry)
        tensor = torch.from_numpy(numpy_array.copy()).to(entry.device)
        tensordict[entry.key] = tensor

    return tensordict

def proto_to_tensordict_extended(
    proto_tensordict,
    server_model
) -> Dict[str, torch.Tensor]:
    """
    Convert protobuf TensorDict back to PyTorch tensors.
    Supports registered compression codecs and uncompressed dense tensors.
    """
    tensordict = {}
    is_model_communicated = False

    for entry in proto_tensordict.entries:
        compression_type = entry.compression_type or None
        codec = _COMPRESSION_CODECS.get(compression_type)
        if compression_type is None:
            numpy_array = _dense_numpy_array(entry)
        elif codec is not None:
            numpy_array, communicated_model = codec.from_entry(entry, server_model)
            is_model_communicated = is_model_communicated or communicated_model
        else:
            raise ValueError(
                f"Unsupported compression type: {compression_type}, "
                f"the type is {type(compression_type)}"
            )

        tensor = torch.from_numpy(numpy_array.copy()).to(entry.device)
        tensordict[entry.key] = tensor

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
