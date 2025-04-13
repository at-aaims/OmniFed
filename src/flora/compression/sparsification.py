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

import math
import torch

from src.flora.compression import Compression, ResidualUpdates


def topk_sparse(tensor, compress_ratio):
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))

    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    return values, indices

    # k = max(1, int(tensor.numel() * compress_ratio))
    # values, indexes = tensor.abs().sort(descending=True)
    #
    # return values[:k], indexes[:k]


def topk_desparse(tensors, numel, device):
    values, indices = tensors
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=device)
    tensor_decompressed.scatter_(0, indices, values)

    return tensor_decompressed


class TopKCompression(Compression):
    """Implementation of Top-k lossy sparsification where largest k updates (in magnitude) are selected"""
    def __init__(self, device, compress_ratio):
        super().__init__()
        self.residual = ResidualUpdates()
        self.device = device
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        tensor = tensor.to(self.device)

        tensor = self.residual.compensate(tensor, name)
        numel = tensor.numel()
        shape = tensor.size()
        tensors = topk_sparse(tensor, self.compress_ratio)
        ctx = numel, shape
        self.residual.update(tensor, name, self, tensors, ctx)

        return tensors, ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        tensor_decompressed = topk_desparse(tensors, numel, self.device)

        return tensor_decompressed.view(shape)


class DGCCompression(Compression):
    """Implementation of Deep Gradient Compression (DGC) lossy sparsification"""
    def __init__(self, device, compress_ratio):
        super().__init__(is_tensor_size_same=False)
        self.residual = ResidualUpdates()
        self.device = device
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        tensor = tensor.to(self.device)
        tensor = self.residual.compensate(tensor, name)
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        sample_shape = [max(1, int(numel * 0.01))]
        sample_index = torch.empty(sample_shape).uniform_(0, numel).type(torch.long)
        sample_tensor = tensor[sample_index]

        # k = max(1, int(numel * compress_ratio * 0.01))
        # vals, indices = torch.topk(sample_tensor.abs(), k)

        k = max(1, int(numel * self.compress_ratio * 0.01))
        vals, indices = sample_tensor.abs().sort(descending=True)
        vals, indices = vals[:k], indices[:k]

        thr = vals.min()
        mask = (tensor.abs() >= thr)
        selected = mask.sum()

        for _ in range(10):
            if selected > 1.3 * numel * self.compress_ratio:
                thr = 1.3 * thr
            elif selected < 0.7 * numel * self.compress_ratio:
                thr = 0.7 * thr
            else:
                break

            mask = (tensor.abs() >= thr)
            selected = mask.sum()

        indices, = torch.where(mask)
        values = tensor[indices]

        tensor_compressed = values, indices
        ctx = numel, shape
        self.residual.update(tensor.view(shape), name, self, tensor_compressed, ctx)

        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx):
        values, indices = tensor_compressed
        numel, shape = ctx
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=self.device)
        tensor_decompressed.scatter_(0, indices, values)

        return tensor_decompressed.view(shape)


class RedsyncCompression(Compression):
    """Implementation of Redsync lossy sparsification"""
    def __init__(self, device, compress_ratio):
        super().__init__(is_tensor_size_same=False)
        self.device = device
        self.residual = ResidualUpdates()
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        tensor = tensor.to(self.device)
        tensor = self.residual.compensate(tensor, name)
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()
        k = max(int(numel * self.compress_ratio), 1)

        l = 0.0
        r = 1.0
        thres = 0.0
        eps = 0.2
        abs_tensor = torch.abs(tensor)
        mean_val = torch.mean(abs_tensor)
        max_val = torch.max(abs_tensor)

        while r - l > eps:
            tmp_ratio = l + (r - l) / 2
            thres = mean_val + tmp_ratio * (max_val - mean_val)
            one_indexes = (abs_tensor > thres)
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            nnz = indexes.numel()
            if nnz > k and 2 * k > nnz:
                break
            elif nnz < k / 2:
                r = tmp_ratio
            else:
                l = tmp_ratio

        values = tensor.data[indexes]
        tensors = values, indexes
        ctx = numel, shape
        self.residual.update(tensor.view(shape), name, self, tensors, ctx)

        return tensors, ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        values, indices = tensors
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=self.device)
        tensor_decompressed.scatter_(0, indices, values).to(self.device)

        return tensor_decompressed.view(shape)


class SIDCoCompression(Compression):

    def __init__(self, num_stages, device, compress_ratio):
        """Implementation of Sparsity-induced lossy sparsification SIDCo based on double-exponential distribution"""
        super().__init__(is_tensor_size_same=False)
        self.num_stages = num_stages
        self.device = device
        self.compress_ratio = compress_ratio
        self.residual = ResidualUpdates()
        self.first_ratio = 0.75
        self.i_ratio = 0.25

    def compress(self, tensor, name):
        tensor = tensor.to(self.device)
        tensor = self.residual.compensate(tensor, name)
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()
        t_norm = tensor.norm(2)
        abs_norm_tensor = (tensor.abs() / t_norm)
        abs_norm_tensor_cpy = abs_norm_tensor.clone()
        t_mean = torch.mean(abs_norm_tensor)
        if self.num_stages == 1 or self.compress_ratio >= self.first_ratio:
            threshold = -t_mean * math.log(self.compress_ratio)
        else:
            threshold = -t_mean * math.log(self.first_ratio)

        r_ratio = self.compress_ratio / self.first_ratio
        if self.num_stages > 1 or self.num_stages == 0:
            if self.num_stages == 0:
                loop = (math.ceil(math.log(r_ratio) / math.log(self.i_ratio)))
            else:
                self.i_ratio = (math.pow(r_ratio, 1.0 / (self.num_stages - 1)))
                loop = self.num_stages - 1
            i = loop
            while i > 0:
                one_indexes = (abs_norm_tensor > threshold).to(self.device)
                indexes = one_indexes.nonzero().data.squeeze().view(-1).to(self.device)
                abs_norm_tensor = abs_norm_tensor.data[indexes].to(self.device)

                # to handle failure when # stages renders abs_norm_tensor to be empty
                if abs_norm_tensor.size()[0] > 0:
                    t_min = abs_norm_tensor.min()
                    t_mean = torch.mean(abs_norm_tensor)

                    threshold = (-(t_mean - t_min) * math.log(self.i_ratio) + t_min)
                    if i == 1 and self.num_stages == 0:
                        threshold = (-(t_mean - t_min) * math.log(r_ratio / math.pow(self.i_ratio, loop - 1)) + t_min)
                    i -= 1
                else:
                    break

        one_indexes = (abs_norm_tensor_cpy > threshold).to(self.device)
        indexes = one_indexes.nonzero().data.squeeze().view(-1).to(self.device)
        values = tensor.data[indexes].to(self.device)
        tensors = values, indexes
        ctx = numel, shape
        self.residual.update(tensor.view(shape), name, self, tensors, ctx)
        return tensors, ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        values, indices = tensors
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=self.device)
        tensor_decompressed.scatter_(0, indices, values).to(self.device)
        return tensor_decompressed.view(shape)


def randomk_sparse(tensor, compress_ratio, device):
    tensor = tensor.flatten()
    numel = tensor.numel()
    k = max(1, int(numel * compress_ratio))

    indices = torch.randint(low=0, high=numel-1, size=(k,)).to(device)
    values = tensor[indices].to(device)

    return values, indices


def randomk_desparse(tensors, numel, device):
    values, indices = tensors
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=device)
    tensor_decompressed.scatter_(0, indices, values).to(device)

    return tensor_decompressed


class RandomKCompression(Compression):
    """Implementation of random-k lossy sparsification where k-portion of updates are chosen randomly"""
    def __init__(self, device, compress_ratio):
        super().__init__()
        self.global_step = 0
        self.device = device
        self.residual = ResidualUpdates()
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        tensor = tensor.to(self.device)

        tensor = self.residual.compensate(tensor, name)
        numel = tensor.numel()
        shape = tensor.size()
        tensors = randomk_sparse(tensor, self.compress_ratio, self.device)
        ctx = numel, shape
        self.residual.update(tensor, name, self, tensors, ctx)
        return tensors, ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        tensor_decompressed = randomk_desparse(tensors, numel, self.device)
        return tensor_decompressed.view(shape)