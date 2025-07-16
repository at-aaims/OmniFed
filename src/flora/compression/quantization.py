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

from src.flora.compression import Compression

# TODO: check loss scaling is correctly applied in AMPCompression for mixed-precision training


class QSGDCompression(Compression):
    """Implementation of quantized SGD or QSGD lossy quantization-based compression"""

    def __init__(self, device, bit_width):
        super().__init__()
        self.device = device
        self.bit_width = bit_width
        self.scale = (2**self.bit_width) - 1

    def get_compress_minmax(self, tensor):
        tensor = tensor.to(self.device)
        min_val, max_val = tensor.min(), tensor.max()

        return min_val, max_val


    def compress(self, tensor, min_val, max_val):
        tensor = tensor.to(self.device)
        # Scale the tensor to [0, scale]
        tensor = (tensor - min_val) / (max_val - min_val) * self.scale
        # Round and clamp to [0, scale]
        tensor = torch.round(tensor).clamp(0, self.scale)

        # return quantized tensor, min and max val for communication.
        # In MPI, call allreduce on first, and allgather on latter two
        return tensor

    def decompress(self, tensor, max_val, min_val):
        # Transform tensors back to its original range
        return min_val + (tensor.float() / self.scale) * (max_val - min_val)


class AMPCompression(Compression):
    """Implementation of Automatic Mixed-Precision (AMP) training, where gradients are compressed to 16-bit"""

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.loss_scale_factor = (2**16) - 1

    def compress(self, tensor):
        # Convert tensors to 16-bit
        return tensor.to(self.device).half()

    def decompress(self, tensor):
        # Convert tensors back to 32-bit
        return tensor.float()

    def loss_scaling(self, loss):
        return loss * self.loss_scale_factor

    def gradient_unscaling(self, model):
        if isinstance(model, torch.nn.Module):
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data /= self.loss_scale_factor

            return model
        else:
            raise TypeError(
                "gradient_unscaling fn needs torch.nn.Module type for model argument"
            )
