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
from collections import defaultdict
from .compression.sparsification import TopKCompression


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


def compress_message_tensors(msg, compressor, aggregation_metric):
    """
    Returns a compressed representation with 1-1 key correspondence.
    """
    compressed = {}

    if compressor is None:
        return msg

    if isinstance(msg, torch.Tensor):
        return msg

    

    tensordict = extract_tensordict(msg, aggregation_metric)

    with torch.no_grad():
        for key, tensor in tensordict.items():
            # print(f"key = {key}")
            (values, indices), ctx = compressor.compress(
                tensor=tensor,
                name=key,
            )

            # print(f"Inside compressor: key = {key}, values = {values}, value shape = {values.shape}, indices = {indices}, index shape = {indices.shape}")
            compressed[key] = {
                "values": values,
                "indices": indices,
                "original_shape": tensor.shape,
                "ctx": ctx,  # (numel, shape)
            }

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

        indices_empty = True
        try:
            indices_empty = item["indices"].numel() == 0
        except Exception as e:
            indices_empty = True

        # ----------------------------------------------------
        # CASE 1: COMPRESSED ENTRY (Top-K)
        # ----------------------------------------------------
        if isinstance(item, dict) and compression_type == TopKCompression.__name__ and not indices_empty:
            values  = item["values"]
            indices = item["indices"]
            numel, shape = item["ctx"]
            original_shape = item["original_shape"]

            original_device = str(values.device)

            # print(f"TopKCompression, client submitting indices = {indices}, shape = {indices.shape}")
            # print(f"TopKCompression, client submitting indices with type {type(indices)} and values with type {type(values)}")

            values_cpu  = values.cpu()
            indices_cpu = indices.cpu()

            data_bytes = values_cpu.numpy().tobytes()
            index_bytes = indices_cpu.numpy().tobytes()

            # print(f"tensordict_to_proto =>  TopKCompression, client submitting indices_cpu = {indices_cpu}, shape = {indices_cpu.shape}")
            # print(f"tensordict_to_proto => TopKCompression, client submitting index_bytes = {index_bytes}, shape = {indices_cpu.shape}")
            # print(f"tensordict_to_proto => TopKCompression, client submitting index_bytes = {index_bytes}, shape = {list(indices_cpu.shape)}")
            # print(f"tensordict_to_proto => TopKCompression, client submitting data = {values_cpu}")
            # print(f"tensordict_to_proto => TopKCompression, client submitting data_bytes = {data_bytes}")
            # print(f"tensordict_to_proto => TopKCompression, client submitting data_shape = {values_cpu.shape}")
            compression_type = "TopKCompression" if len(indices_cpu) != 0 else None

            # print(f"TopKCompression, index dtype is {indices_cpu.dtype}")
            entry = grpc_pb2.TensorEntry(
                key=key,
                data=data_bytes,                 # VALUES
                shape=list(values_cpu.shape),               # ORIGINAL tensor shape
                dtype=str(values_cpu.dtype),
                device=original_device,
                data_size=len(data_bytes),
                compression_type=compression_type,
                index=index_bytes,               # INDICES
                index_shape=list(indices_cpu.shape),  # usually [k]
                index_dtype=str(indices_cpu.dtype),
                original_shape=original_shape
            )

        # ----------------------------------------------------
        # CASE 2: UNCOMPRESSED DENSE TENSOR
        # ----------------------------------------------------
        else:
            # print(f"Inside tensordict_to_proto: No compressor found, compressor = {compression_type}")
            tensor = item

            # print(f"Tensor type = {type(tensor)}, tensor = {tensor}")

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
                # index omitted → defaults to b""
                # idx_shape omitted → defaults to []
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

def proto_to_tensordict_extended(
    proto_tensordict,
    server_model
) -> Dict[str, torch.Tensor]:
    """
    Convert protobuf TensorDict back to PyTorch tensors.
    Supports both uncompressed and Top-K compressed tensors.
    """
    tensordict = {}

    is_model_communicated = False

    # print(f"Model is {server_model}")

    dtype_mapping = {
        "torch.float32": np.float32,
        "torch.float64": np.float64,
        "torch.int32": np.int32,
        "torch.int64": np.int64,
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
        # print(f"entry.key = {entry.key}")
        # Normalize compression type (proto3 default is "")
        compression_type = entry.compression_type or None

        # print(f"Compression = {compression_type}, type = {type(compression_type)}")

        # print(f"TopKCompression.__class__.__name__ = {TopKCompression.__name__}")
        # ----------------------------
        # CASE 1: Top-K compressed
        # ----------------------------
        if compression_type == TopKCompression.__name__:
            # print(f"Data is compressed. Decompressing the data")
            # Sanity checks
            if not entry.index:
                raise ValueError(
                    f"Missing indices for compressed tensor {entry.key}"
                )

            index_dtype = dtype_mapping[entry.index_dtype]
            # Decode values
            values = np.frombuffer(entry.data, dtype=numpy_dtype)

            # Decode indices
            indices = np.frombuffer(entry.index, dtype=index_dtype)
            # indices = indices.reshape(entry.idx_shape)
            numel = int(np.prod(entry.original_shape))
            dense = np.zeros(numel, dtype=numpy_dtype)

            # print("proto_to_tensordict => indices.shape:", indices.shape)
            # print("proto_to_tensordict =>  values.shape:", values.shape)
            # print("proto_to_tensordict =>  indices:", indices)
            # print("proto_to_tensordict =>  values:", values)

            if len(indices) != len(values):
                # print(f"Error: protodict_to_tensordict => {entry.dtype}")
                raise RuntimeError(
                    f"Mismatch: indices ({len(indices)}) != values ({len(values)})"
                )



            if(indices.size == 0):
                raise RuntimeError("proto_to_tensordict -> Index array is empty for TopKCompression")
            try:
                # Reconstruct dense tensor
                # Get server tensor
                server_tensor = server_model[entry.key]


                # print(f"entry.key = {entry.key} is contained in the server_model")

                # Move to CPU if needed and flatten
                flat = server_tensor.detach().cpu().numpy().reshape(-1).copy()

                # Overwrite only transmitted indices
                flat[indices] = values

                # Reshape back
                dense = flat.reshape(entry.original_shape)
                is_model_communicated = True
                # print(f"Compressed data communicated is the model itself")
            except Exception as e:
                # print(f"Inside extended the error is {e}")
                dense[indices] = values

            # print(f"TopKCompression; dense.shape = {dense.shape}, entry.shape = {entry.shape}")
            numpy_array = dense.reshape(tuple(entry.original_shape))

        # ----------------------------
        # CASE 2: Uncompressed (dense)
        # ----------------------------
        elif compression_type is None:
            # Validate data size (dense case only)
            if len(entry.data) != entry.data_size:
                raise ValueError(
                    f"Data size mismatch for tensor {entry.key}: "
                    f"expected {entry.data_size}, got {len(entry.data)}"
                )

            numpy_array = np.frombuffer(entry.data, dtype=numpy_dtype)
            # print(f"No compression; numpy_array.shape = {numpy_array.shape}, entry.shape = {entry.shape}")
            numpy_array = numpy_array.reshape(tuple(entry.shape))

        else:
            raise ValueError(
                f"Unsupported compression type: {compression_type}, the type is {type(compression_type)}"
            )

        # ----------------------------
        # Convert to torch.Tensor
        # ----------------------------
        # .copy() because frombuffer gives a read-only view
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
