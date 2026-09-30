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

"""QSGD quantization for classic centralized gRPC (sync grad path)."""

from __future__ import annotations

import torch

from src.omnifed.device_resolver import is_cuda_oom

from . import Compression

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
        Stochastic QSGD Q_s(v). Returns signed_levels, norm_v, width, levels.
        width/levels/norm are -1 when passthrough (empty or zero norm).
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

        signed_levels = signs.long() * quantized_levels
        width, storage_dtype = choose_qsgd_storage_width(levels)
        signed_levels = signed_levels.to(storage_dtype)
        return signed_levels, norm_v, width, levels

    def _do_compress(self, tensor: torch.Tensor):
        flat = tensor.flatten()
        signed_levels, norm, width, levels = self.quantize_vector(flat)
        if width == -1 or levels == -1 or norm == -1:
            return tensor, -1, -1, -1
        signed_levels = signed_levels.reshape(tensor.shape).to(tensor.device)
        return signed_levels, norm, width, levels

    def compress(self, tensor: torch.Tensor, name: str = ""):
        del name
        if not should_compress_tensor(tensor):
            return tensor, -1, -1, -1
        try:
            return self._do_compress(tensor)
        except Exception as exc:
            if not (tensor.is_cuda and is_cuda_oom(exc)):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._do_compress(tensor.detach().cpu())

    @staticmethod
    def decompress_quantized(
        signed_levels: torch.Tensor,
        norm: float,
        levels: int,
        shape: torch.Size | tuple[int, ...],
    ) -> torch.Tensor:
        """Inverse of QSGD encode: norm * signed_level / levels."""
        if levels <= 0 or norm is None or norm == -1:
            return signed_levels.float() if signed_levels.is_floating_point() else signed_levels
        flat = signed_levels.float().reshape(-1)
        restored = float(norm) * flat / float(levels)
        return restored.reshape(shape)

    def decompress(self, tensors, ctx):
        norm, _width, levels, shape = ctx
        signed_levels = tensors[0] if isinstance(tensors, (tuple, list)) else tensors
        return self.decompress_quantized(signed_levels, norm, levels, shape)

    def aggregate_torchdist(
        self,
        tensor,
        *,
        name: str,
        world_size: int,
        op,
        logger=None,
    ):
        del world_size
        from .torchdist_collectives import aggregate_qsgd_tensor

        return aggregate_qsgd_tensor(
            self, tensor, name=name, op=op, logger=logger
        )


_quantized_compression_ = [QSGD_COMPRESSION_NAME]
