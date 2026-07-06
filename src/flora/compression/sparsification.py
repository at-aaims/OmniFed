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

import torch

from src.flora.compression._core import Compression, ResidualUpdates

TOPK_COMPRESSION_NAME = "TopKCompression"


def topk_sparse(tensor: torch.Tensor, compress_ratio: float):
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))
    _, indices = torch.topk(tensor.abs(), k, sorted=False)
    values = torch.gather(tensor, 0, indices)
    return values, indices


def topk_desparse(values: torch.Tensor, indices: torch.Tensor, numel: int, device: torch.device):
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=device)
    tensor_decompressed.scatter_(0, indices.to(device), values.to(device))
    return tensor_decompressed


class TopKCompression(Compression):
    """Top-k sparsification with error feedback (largest-magnitude elements)."""

    def __init__(self, device: torch.device | str = "cpu", compress_ratio: float = 0.01):
        super().__init__()
        self.residual = ResidualUpdates()
        self.device = torch.device(device)
        self.compress_ratio = float(compress_ratio)

    def compress(self, tensor: torch.Tensor, name: str):
        tensor = tensor.to(self.device)
        tensor = self.residual.compensate(tensor, name)
        numel = tensor.numel()
        shape = tensor.size()
        values, indices = topk_sparse(tensor, self.compress_ratio)
        ctx = (numel, shape)
        self.residual.update(tensor, name, self, (values, indices), ctx)
        return (values, indices), ctx

    def decompress(self, tensors, ctx):
        numel, shape = ctx
        values, indices = tensors
        tensor_decompressed = topk_desparse(values, indices, numel, self.device)
        return tensor_decompressed.view(shape)
