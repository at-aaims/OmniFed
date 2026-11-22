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

"""Encoder for the secure proto's ``TensorDict``.

``utils.tensordict_to_proto`` builds ``grpc_pb2.TensorEntry`` objects, and protobuf refuses to
put those inside a ``secure_grpc_pb2`` message (they are different generated classes). The
function below is that encoder verbatim, with the generated module passed in as ``pb2``.

The *decoders* (``utils.proto_to_tensordict`` / ``proto_to_tensordict_extended``) only read
fields by name, so they work on either module's messages and are reused unchanged.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..compression.quantization import QSGDQuantCompression
from ..compression.sparsification import TopKCompression
from ..utils import _QSGD_NUMPY_DTYPES


def tensordict_to_proto(
    pb2, tensordict: Dict[str, torch.Tensor], compression_type=None
):
    """
    Convert tensor dictionary to protobuf format for gRPC transmission.

    Serializes PyTorch tensors to byte format with metadata for exact reconstruction.
    Includes device information and data type preservation.

    Args:
        pb2: the generated secure_grpc_pb2 module (its TensorEntry/TensorDict are used)
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

            entry = pb2.TensorEntry(
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

            entry = pb2.TensorEntry(
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

            entry = pb2.TensorEntry(
                key=key,
                data=data_bytes,
                shape=list(tensor_cpu.shape),
                dtype=str(tensor_cpu.dtype),
                device=original_device,
                data_size=len(data_bytes),
            )

        entries.append(entry)


    return pb2.TensorDict(entries=entries)


def _is_repeated(field) -> bool:
    # FieldDescriptor.label is deprecated (protobuf >= 6.31) and gone in 7.x.
    flag = getattr(field, "is_repeated", None)
    return bool(flag) if flag is not None else field.label == field.LABEL_REPEATED


def _fields(message_cls):
    return sorted(
        (f.name, f.number, f.type, _is_repeated(f)) for f in message_cls.DESCRIPTOR.fields
    )


def check_wire_compat(pb2) -> None:
    """Fail early if secure_grpc.proto's tensor messages drifted from grpc.proto's.

    The shared decoders read ``TensorEntry`` fields by name; if a field was added to
    grpc.proto and not here (or vice versa) this says so at start-up instead of failing
    in the middle of a round.
    """
    from .. import grpc_pb2 as base_pb2

    for name in ("TensorEntry", "TensorDict"):
        ours, theirs = _fields(getattr(pb2, name)), _fields(getattr(base_pb2, name))
        if ours != theirs:
            raise RuntimeError(
                f"secure_grpc.proto {name} differs from grpc.proto {name}; keep them identical.\n"
                f"  only in secure: {sorted(set(ours) - set(theirs))}\n"
                f"  only in grpc:   {sorted(set(theirs) - set(ours))}"
            )
