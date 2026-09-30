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

"""Split / stitch gRPC payloads so each message stays under the unary cap.

The gate is packed size (after compression), not model name or codec type.
Callers compress first, then ``should_chunk_payload`` / ``iter_payload_chunks``.
"""

from __future__ import annotations

from math import prod
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

import torch

from .grpc_limits import (
    GRPC_CHUNK_PAYLOAD_BYTES,
    GRPC_CHUNK_THRESHOLD_BYTES,
    GRPC_MAX_MESSAGE_BYTES,
)

# Transport-only keys. Must not appear in real parameter names.
_TENSOR_SLICE_SEPARATOR = ".__omnifed_slice__."
_ENTRY_OVERHEAD_BYTES = 64
_MESSAGE_OVERHEAD_BYTES = 32


def estimate_entry_payload_bytes(value: Any) -> int:
    """Raw tensor bytes that would land in protobuf ``data`` / ``index`` / meta."""
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if not isinstance(value, dict):
        return 0
    total = 0
    values = value.get("values")
    indices = value.get("indices")
    if torch.is_tensor(values):
        total += int(values.numel() * values.element_size())
    if torch.is_tensor(indices):
        total += int(indices.numel() * indices.element_size())
    signed_levels = value.get("signed_levels")
    if torch.is_tensor(signed_levels):
        width = int(value.get("width") or 8)
        # Packed QSGD uses int8/int32 on the wire, not necessarily the tensor dtype.
        packed_width = 8 if width <= 8 else 32
        total += int(signed_levels.numel() * (packed_width // 8))
        total += 4  # float32 norm
    return total


def estimate_packed_bytes(tensordict: Mapping[str, Any] | None) -> int:
    """Conservative protobuf size: payload + per-entry/key overhead."""
    if not tensordict:
        return _MESSAGE_OVERHEAD_BYTES
    total = _MESSAGE_OVERHEAD_BYTES
    for key, value in tensordict.items():
        total += estimate_entry_payload_bytes(value)
        total += _ENTRY_OVERHEAD_BYTES + len(str(key).encode("utf-8"))
    return int(total)


def should_chunk_payload(
    tensordict: Mapping[str, Any] | None,
    threshold_bytes: int = GRPC_CHUNK_THRESHOLD_BYTES,
) -> bool:
    """True when a single unary would exceed ``threshold_bytes``."""
    if threshold_bytes <= 0:
        return True
    return estimate_packed_bytes(tensordict) > int(threshold_bytes)


def packed_message_bytes(message: Any) -> int:
    """Exact serialized size of a protobuf message (``ByteSize()``)."""
    return int(message.ByteSize())


def _make_slice_key(name: str, start: int, shape: Sequence[int]) -> str:
    shape_text = "x".join(str(int(d)) for d in shape)
    return f"{name}{_TENSOR_SLICE_SEPARATOR}{int(start)}:{shape_text}"


def parse_slice_key(key: str) -> tuple[str, int, tuple[int, ...]] | None:
    """Return ``(name, start, shape)`` for a slice key, else ``None``."""
    if _TENSOR_SLICE_SEPARATOR not in key:
        return None
    name, rest = key.rsplit(_TENSOR_SLICE_SEPARATOR, 1)
    start_text, _, shape_text = rest.partition(":")
    if not start_text or not shape_text:
        raise ValueError(f"Invalid slice transport key: {key}")
    shape = tuple(int(p) for p in shape_text.split("x") if p)
    return name, int(start_text), shape


def _flush_chunk(chunk: dict[str, Any]) -> dict[str, Any] | None:
    if not chunk:
        return None
    return chunk


def iter_payload_chunks(
    tensordict: Mapping[str, Any],
    max_payload_bytes: int = GRPC_CHUNK_PAYLOAD_BYTES,
) -> Iterator[dict[str, Any]]:
    """Yield tensordicts whose estimated packed size is ≤ ``max_payload_bytes``.

    Dense tensors larger than the limit are flattened and sliced. Compressed
    dict entries stay atomic (Top-K / QSGD) and occupy their own chunk if they
    do not fit beside other entries.
    """
    limit = int(max_payload_bytes)
    if limit <= 0:
        raise ValueError(f"max_payload_bytes must be positive, got {max_payload_bytes}")

    if not should_chunk_payload(tensordict, threshold_bytes=limit):
        yield dict(tensordict)
        return

    chunk: dict[str, Any] = {}
    chunk_bytes = 0

    def flush() -> dict[str, Any] | None:
        nonlocal chunk, chunk_bytes
        ready = _flush_chunk(chunk)
        chunk = {}
        chunk_bytes = 0
        return ready

    for name, value in tensordict.items():
        if _TENSOR_SLICE_SEPARATOR in str(name):
            raise ValueError(f"Parameter name contains reserved slice separator: {name}")
        entry_bytes = estimate_entry_payload_bytes(value)
        if entry_bytes <= 0 and not torch.is_tensor(value) and not isinstance(value, dict):
            continue

        # Compressed (or other dict) entries are never sliced.
        if isinstance(value, dict) or (
            torch.is_tensor(value) and entry_bytes <= limit
        ):
            if chunk and chunk_bytes + entry_bytes > limit:
                ready = flush()
                if ready is not None:
                    yield ready
            if entry_bytes > GRPC_MAX_MESSAGE_BYTES:
                raise ValueError(
                    f"Entry {name!r} estimate {entry_bytes} bytes exceeds "
                    f"GRPC_MAX_MESSAGE_BYTES={GRPC_MAX_MESSAGE_BYTES}"
                )
            if entry_bytes > limit:
                ready = flush()
                if ready is not None:
                    yield ready
                yield {name: value}
                continue
            chunk[name] = value
            chunk_bytes += entry_bytes
            continue

        if not torch.is_tensor(value):
            raise TypeError(f"Unsupported payload entry {name!r}: {type(value)!r}")

        ready = flush()
        if ready is not None:
            yield ready

        tensor = value.detach().contiguous()
        elem_size = int(tensor.element_size())
        elements_per_slice = max(1, limit // max(elem_size, 1))
        flat = tensor.reshape(-1)
        original_shape = tuple(int(d) for d in tensor.shape)
        for start in range(0, flat.numel(), elements_per_slice):
            end = min(start + elements_per_slice, flat.numel())
            yield {_make_slice_key(name, start, original_shape): flat[start:end].clone()}

    ready = flush()
    if ready is not None:
        yield ready


class PayloadChunkAssembler:
    """Stitch transport chunks back into the original tensordict keys."""

    def __init__(self) -> None:
        self._output: dict[str, Any] = {}
        self._slice_parts: dict[str, dict[str, Any]] = {}

    def add_chunk(self, transport_chunk: Mapping[str, Any]) -> None:
        for key, value in transport_chunk.items():
            parsed = parse_slice_key(str(key))
            if parsed is None:
                self._output[str(key)] = value
                continue
            name, start, shape = parsed
            if not torch.is_tensor(value):
                raise TypeError(f"Slice {key} must be a tensor, got {type(value)!r}")
            flat = value.detach().reshape(-1)
            rec = self._slice_parts.setdefault(
                name,
                {
                    "shape": shape,
                    "dtype": flat.dtype,
                    "parts": [],
                    "numel": int(prod(shape)) if shape else int(flat.numel()),
                },
            )
            if rec["shape"] != shape:
                raise ValueError(
                    f"Shape mismatch for sliced {name}: {rec['shape']} vs {shape}"
                )
            rec["parts"].append((start, flat.clone()))

    def finalize(self) -> dict[str, Any]:
        for name, rec in self._slice_parts.items():
            numel = int(rec["numel"])
            out = torch.empty(numel, dtype=rec["dtype"])
            covered = 0
            for start, part in sorted(rec["parts"], key=lambda p: p[0]):
                end = start + int(part.numel())
                if start < 0 or end > numel:
                    raise ValueError(
                        f"Invalid slice for {name}: [{start}, {end}) numel={numel}"
                    )
                out[start:end] = part
                covered += int(part.numel())
            if covered != numel:
                raise ValueError(
                    f"Incomplete slices for {name}: got {covered} of {numel} elements"
                )
            self._output[name] = out.view(rec["shape"])
        return self._output


def assemble_payload_chunks(
    chunks: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    assembler = PayloadChunkAssembler()
    for chunk in chunks:
        assembler.add_chunk(chunk)
    return assembler.finalize()


def chunk_count(
    tensordict: Mapping[str, Any],
    max_payload_bytes: int = GRPC_CHUNK_PAYLOAD_BYTES,
) -> int:
    return sum(1 for _ in iter_payload_chunks(tensordict, max_payload_bytes))
