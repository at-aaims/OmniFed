# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""QSGD quantization for hybrid global gRPC (ported from PR #86)."""

from __future__ import annotations

import torch

from src.omnifed.hybrid.compression.core import Compression

QSGD_COMPRESSION_NAME = "QSGDQuantCompression"


def should_compress_tensor(x: torch.Tensor) -> bool:
    return isinstance(x, torch.Tensor) and x.is_floating_point() and x.numel() > 0


def choose_qsgd_storage_width(levels: int) -> tuple[int, torch.dtype]:
    if levels <= torch.iinfo(torch.int8).max:
        return 8, torch.int8
    return 32, torch.int32


class QSGDQuantCompression(Compression):
    """
    QSGD implementation based on:
    QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding
    (Alistarh et al., 2017)
    """

    def __init__(self, bit_width: int = 8, device: torch.device | str = "cpu"):
        super().__init__()
        self.s = int(bit_width)
        self.device = torch.device(device)

    def quantize_vector(self, v: torch.Tensor):
        """
        QSGD quantization function Q_s(v) as defined in Algorithm 1 of the paper.

        Returns:
            signed_levels, norm_v, width, levels
        """
        if v.numel() == 0:
            return v, -1, -1, -1

        norm_v = torch.norm(v).item()
        if norm_v == 0:
            return torch.zeros_like(v), -1, -1, -1

        v_normalized = v / norm_v
        signs = torch.sign(v_normalized)
        abs_v = torch.abs(v_normalized)
        levels = 2**self.s

        scaled_abs = abs_v * levels
        lower = torch.floor(scaled_abs).long()
        prob_round_up = scaled_abs - lower.float()
        round_up = (torch.rand_like(prob_round_up) < prob_round_up).long()
        quantized_levels = torch.clamp(lower + round_up, 0, levels)

        signed_levels = (signs.long() * quantized_levels)
        width, storage_dtype = choose_qsgd_storage_width(levels)
        signed_levels = signed_levels.to(storage_dtype)
        return signed_levels, norm_v, width, levels

    def _do_compress(self, tensor: torch.Tensor):
        flat = tensor.flatten()
        signed_levels, norm, width, levels = self.quantize_vector(flat)
        signed_levels = signed_levels.reshape(tensor.shape).to(tensor.device)
        return signed_levels, norm, width, levels

    def compress(self, tensor: torch.Tensor, name: str = ""):
        """
        Compress a tensor with QSGD.

        Returns:
            (signed_levels, norm, width, levels) where width/levels are -1 for dense passthrough.
        """
        del name  # reserved for error-feedback in a later phase
        if not should_compress_tensor(tensor):
            return tensor, -1, -1, -1
        return self._do_compress(tensor)

    @staticmethod
    def decompress_quantized(
        signed_levels: torch.Tensor,
        norm: float,
        levels: int,
        shape: torch.Size | tuple[int, ...],
    ) -> torch.Tensor:
        """Inverse of QSGD encode: norm * signed_level / levels."""
        if levels <= 0 or norm is None or norm == -1:
            return signed_levels
        flat = signed_levels.float().reshape(-1)
        restored = float(norm) * flat / float(levels)
        return restored.reshape(shape)

    def decompress(self, tensors, ctx):
        """
        Decompress QSGD payload.

        ``tensors``: signed integer levels (same layout as compress output).
        ``ctx``: (norm, width, levels, shape) — width is unused here but kept for wire parity with PR #86.
        """
        norm, _width, levels, shape = ctx
        signed_levels = tensors[0] if isinstance(tensors, (tuple, list)) else tensors
        return self.decompress_quantized(signed_levels, norm, levels, shape)
