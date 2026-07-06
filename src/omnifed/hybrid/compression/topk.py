# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

import torch

from src.omnifed.hybrid.compression.core import Compression, ResidualUpdates

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
