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

from typing import Dict

import numpy as np
import torch

from . import grpc_communicator_pb2 as grpc_communicator_pb2


def tensordict_to_proto(
    tensordict: Dict[str, torch.Tensor],
) -> grpc_communicator_pb2.TensorDict:
    """Convert tensor dictionary to protobuf format with exact reconstruction support"""
    entries = []

    for key, tensor in tensordict.items():
        # Store original device before CPU conversion
        original_device = str(tensor.device)

        # Convert to CPU for serialization
        tensor_cpu = tensor.cpu()

        # Serialize to bytes for exact reconstruction
        tensor_bytes = tensor_cpu.numpy().tobytes()

        entry = grpc_communicator_pb2.TensorEntry(
            key=key,
            data=tensor_bytes,
            shape=list(tensor_cpu.shape),
            dtype=str(tensor_cpu.dtype),
            device=original_device,
            data_size=len(tensor_bytes),
        )
        entries.append(entry)

    return grpc_communicator_pb2.TensorDict(entries=entries)


def proto_to_tensordict(
    proto_tensordict: grpc_communicator_pb2.TensorDict,
) -> Dict[str, torch.Tensor]:
    """Convert protobuf tensor dictionary to native tensor dictionary with exact reconstruction"""
    tensordict = {}
    for entry in proto_tensordict.entries:
        # Validate data size
        if len(entry.data) != entry.data_size:
            raise ValueError(
                f"Data size mismatch for tensor {entry.key}: expected {entry.data_size}, got {len(entry.data)}"
            )

        # Dtype mapping: string -> numpy_dtype
        dtype_mapping = {
            "torch.float32": np.float32,
            "torch.float64": np.float64,
            "torch.int32": np.int32,
            "torch.int64": np.int64,
            "torch.bool": np.bool_,
        }

        if entry.dtype not in dtype_mapping:
            supported_dtypes = list(dtype_mapping.keys())
            raise ValueError(
                f"Unsupported dtype: {entry.dtype}. Supported: {supported_dtypes}"
            )

        numpy_dtype = dtype_mapping[entry.dtype]

        # Reconstruct tensor from bytes
        numpy_array = np.frombuffer(entry.data, dtype=numpy_dtype)
        numpy_array = numpy_array.reshape(tuple(entry.shape))

        # Create tensor and restore to original device
        # Note: .copy() needed because np.frombuffer creates read-only arrays
        tensor = torch.from_numpy(numpy_array.copy()).to(entry.device)

        tensordict[entry.key] = tensor

    return tensordict
