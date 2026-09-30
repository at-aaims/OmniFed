from __future__ import annotations

from collections.abc import Iterator, Mapping

import torch


_TENSOR_SLICE_SEPARATOR = (
    ".__omnifed_slice__."
)


def make_tensor_slice_key(name: str, start: int) -> str:
    return f"{name}{_TENSOR_SLICE_SEPARATOR}{start}"


def parse_tensor_slice_key(key: str) -> tuple[str, int] | None:
    if _TENSOR_SLICE_SEPARATOR not in key:
        return None

    name, start_text = key.rsplit(_TENSOR_SLICE_SEPARATOR, 1)
    return name, int(start_text)


def iter_model_state_chunks(
    state_dict: Mapping[str, torch.Tensor],
    chunk_size_mb: int,
) -> Iterator[dict[str, torch.Tensor]]:
    """
    Yield transport chunks smaller than approximately chunk_size_mb.

    Small parameters retain their original names.

    A parameter larger than the chunk limit is flattened and represented as:

        lm_head.weight.__omnifed_slice__.0
        lm_head.weight.__omnifed_slice__.16777216
        lm_head.weight.__omnifed_slice__.33554432
        ...

    The number at the end is the starting element offset in the flattened
    original tensor.
    """
    limit_bytes = int(chunk_size_mb) * 1024 * 1024

    if limit_bytes <= 0:
        raise ValueError(
            f"chunk_size_mb must be positive, got {chunk_size_mb}"
        )

    chunk: dict[str, torch.Tensor] = {}
    chunk_bytes = 0

    def flush_chunk():
        nonlocal chunk, chunk_bytes

        if not chunk:
            return None

        result = chunk
        chunk = {}
        chunk_bytes = 0
        return result

    for name in sorted(state_dict):
        if _TENSOR_SLICE_SEPARATOR in name:
            raise ValueError(
                f"Parameter name contains reserved slice separator: {name}"
            )

        tensor = state_dict[name].detach().cpu()

        # Your FedAvg calculations currently use float32.
        if tensor.is_floating_point():
            tensor = tensor.float()

        tensor = tensor.contiguous()
        tensor_bytes = tensor.numel() * tensor.element_size()

        # Normal parameter: pack the complete parameter into the current chunk.
        if tensor_bytes <= limit_bytes:
            if chunk and chunk_bytes + tensor_bytes > limit_bytes:
                ready = flush_chunk()
                if ready is not None:
                    yield ready

            chunk[name] = tensor
            chunk_bytes += tensor_bytes
            continue

        # Oversized parameter: flush previous parameters first.
        ready = flush_chunk()
        if ready is not None:
            yield ready

        # Divide the oversized tensor into flat slices.
        elements_per_slice = max(
            1,
            limit_bytes // tensor.element_size(),
        )

        flat_tensor = tensor.reshape(-1)

        for start in range(0, flat_tensor.numel(), elements_per_slice):
            end = min(
                start + elements_per_slice,
                flat_tensor.numel(),
            )

            # clone() gives the slice independent contiguous storage.
            tensor_slice = flat_tensor[start:end].clone()
            slice_key = make_tensor_slice_key(name, start)

            # Each large slice is its own transport chunk. This simplifies
            # ordering and prevents protobuf overhead from pushing a packed
            # chunk over the intended limit.
            yield {slice_key: tensor_slice}

    ready = flush_chunk()
    if ready is not None:
        yield ready


class ModelStateChunkAssembler:
    """
    Reconstruct a model state dictionary from aggregated transport chunks.

    Unsliced parameters are stored directly. Sliced parameters are copied
    into preallocated tensors using offsets encoded in their transport keys.
    """

    def __init__(
        self,
        reference_state: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> None:
        self.reference_state = reference_state
        self.output: dict[str, torch.Tensor] = {}

        # Records received [start, end) ranges for each sliced parameter.
        self.received_ranges: dict[str, list[tuple[int, int]]] = {}

    def add_chunk(self, transport_chunk):
        for transport_key, value in transport_chunk.items():
            parsed = parse_tensor_slice_key(transport_key)

            # This is a normal, unsliced parameter.
            if parsed is None:
                if transport_key not in self.reference_state:
                    raise KeyError(
                        f"Received unknown parameter: {transport_key}"
                    )

                expected_shape = tuple(
                    self.reference_state[transport_key].shape
                )

                if tuple(value.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for {transport_key}: "
                        f"received={tuple(value.shape)}, "
                        f"expected={expected_shape}"
                    )

                self.output[transport_key] = value.detach().cpu()
                continue

            # This is one slice of an oversized parameter.
            original_name, start = parsed

            if original_name not in self.reference_state:
                raise KeyError(
                    f"Received slice for unknown parameter: {original_name}"
                )

            reference_tensor = self.reference_state[original_name]
            expected_numel = reference_tensor.numel()

            flat_slice = value.detach().cpu().reshape(-1)
            end = start + flat_slice.numel()

            if start < 0 or end > expected_numel:
                raise ValueError(
                    f"Invalid slice for {original_name}: "
                    f"start={start}, end={end}, "
                    f"expected_numel={expected_numel}"
                )

            if original_name not in self.output:
                self.output[original_name] = torch.empty(
                    tuple(reference_tensor.shape),
                    dtype=flat_slice.dtype,
                    device="cpu",
                )
                self.received_ranges[original_name] = []

            destination = self.output[original_name].view(-1)
            destination[start:end].copy_(flat_slice)

            self.received_ranges[original_name].append((start, end))

    def finish(self):
        """
        Validate that all parameters and all tensor slices were received.
        """
        missing_parameters = (
            set(self.reference_state) - set(self.output)
        )

        if missing_parameters:
            preview = sorted(missing_parameters)[:10]
            raise RuntimeError(
                "Aggregation did not return all model parameters. "
                f"Missing examples: {preview}; "
                f"total_missing={len(missing_parameters)}"
            )

        for name, ranges in self.received_ranges.items():
            ranges = sorted(ranges)
            expected_numel = self.reference_state[name].numel()

            expected_start = 0
            for start, end in ranges:
                if start != expected_start:
                    raise RuntimeError(
                        f"Missing or overlapping slice for {name}: "
                        f"expected_start={expected_start}, "
                        f"received_start={start}"
                    )
                expected_start = end

            if expected_start != expected_numel:
                raise RuntimeError(
                    f"Incomplete tensor reconstruction for {name}: "
                    f"received_elements={expected_start}, "
                    f"expected_elements={expected_numel}"
                )

        return self.output