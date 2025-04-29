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

from abc import ABC, abstractmethod

class ResidualMemory(ABC):
    @abstractmethod
    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        raise NotImplemented("compensate was not implemented.")

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass


class ResidualUpdates(ResidualMemory):
    def __init__(self, beta=1.0, gamma=1.0):
        self.residuals = {}
        self.beta = beta
        self.gamma = gamma
        self.layer_decompress = {}

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        if name in self.residuals:
            tensor = self.beta * self.residuals[name] + self.gamma * tensor
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        self.layer_decompress[name] = tensor_decompressed
        residual = tensor - tensor_decompressed
        self.residuals[name] = residual


class Compression():
    """Interface for compressing and decompressing a given tensor."""
    def __init__(self, average=True, is_tensor_size_same=True):
        self.average = average
        self.is_tensor_size_same = is_tensor_size_same

    def compress(self, tensor, **kwargs):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compress not implemented.")

    def decompress(self, **kwargs):
        """Decompress the tensor with the given context."""
        raise NotImplemented("decompress not implemented.")

    # def aggregate(self, tensors):
    #     """Aggregate a list of tensors."""
    #     return sum(tensors)

    def loss_scaling(self, loss):
        raise NotImplemented("loss_scaling not implemented.")

    def gradient_unscaling(self, **kwargs):
        raise NotImplemented("gradient_unscaling not implemented.")