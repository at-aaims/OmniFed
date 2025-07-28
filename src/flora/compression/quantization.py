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

from src.flora.compression import Compression, ResidualUpdates


class QSGDCompression(Compression):
    """Implementation of quantized SGD or QSGD lossy quantization-based compression with error feedback"""

    def __init__(self, device, bit_width):
        super().__init__()
        self.device = device
        self.bit_width = bit_width
        self.scale = (2 ** self.bit_width) - 1
        self.residual = ResidualUpdates()  # Add error feedback

    def get_compress_minmax(self, tensor):
        tensor = tensor.to(self.device)
        min_val, max_val = tensor.min(), tensor.max()
        return min_val, max_val

    def compress(self, tensor, name, min_val=None, max_val=None):
        tensor = tensor.to(self.device)

        # Store the ORIGINAL tensor before any compensation
        original_tensor = tensor.clone()

        # Apply error feedback compensation AFTER storing original
        compensated_tensor = self.residual.compensate(tensor, name)

        # Get min/max from the compensated tensor (this is important!)
        if min_val is None or max_val is None:
            min_val, max_val = self.get_compress_minmax(compensated_tensor)

        # Quantize the compensated tensor
        if torch.abs(max_val - min_val) < 1e-8:  # Better numerical stability check
            quantized_tensor = torch.zeros_like(compensated_tensor)
        else:
            # Scale the compensated tensor to [0, scale]
            scaled_tensor = (compensated_tensor - min_val) / (max_val - min_val) * self.scale
            # Round and clamp to [0, scale]
            quantized_tensor = torch.round(scaled_tensor).clamp(0, self.scale)

        # Create context for decompression
        ctx = (min_val, max_val, tensor.shape)

        # CRITICAL: Update residuals using the ORIGINAL tensor, not compensated
        self.residual.update(original_tensor, name, self, quantized_tensor, ctx)

        return quantized_tensor, ctx

    def decompress(self, tensor, ctx):
        min_val, max_val, shape = ctx

        # Avoid division by zero
        if torch.abs(max_val - min_val) < 1e-8:
            return torch.full(shape, min_val.item(), dtype=torch.float32, device=self.device)

        # Transform tensors back to its original range
        decompressed = min_val + (tensor.float() / self.scale) * (max_val - min_val)
        return decompressed.view(shape)