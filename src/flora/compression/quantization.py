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

from typing import List, Dict, Tuple

import torch

from src.flora.compression import Compression


class QSGDQuantCompression(Compression):
    def __init__(self, device, bit_width: int = 8):
        super().__init__()
        self.device = device
        self.bit_width = bit_width
        self.levels = 2 ** bit_width - 1

    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor to specified bit width
        Returns: (quantized_tensor, scale, zero_point)
        """
        if tensor.numel() == 0:
            return tensor, torch.tensor(1.0), torch.tensor(0.0)

        # Calculate min/max for uniform quantization
        min_val = tensor.min()
        max_val = tensor.max()

        # Handle edge case where all values are the same
        if min_val == max_val:
            return torch.zeros_like(tensor, dtype=torch.uint8), torch.tensor(1.0), min_val

        # Calculate scale and zero point
        scale = (max_val - min_val) / self.levels
        zero_point = torch.round(-min_val / scale).clamp(0, self.levels)

        # Quantize
        quantized = torch.round(tensor / scale + zero_point).clamp(0, self.levels)

        return quantized.to(torch.uint8), scale, zero_point

    def compress(self, gradients: List[torch.Tensor], tensor=None) -> Dict:
        quantized_data = {
            'tensors': [],
            'scales': [],
            'zero_points': [],
            'shapes': []
        }

        for grad in gradients:
            if grad is not None:
                q_tensor, scale, zero_point = self.quantize_tensor(grad.flatten())
                quantized_data['tensors'].append(q_tensor)
                quantized_data['scales'].append(scale)
                quantized_data['zero_points'].append(zero_point)
                quantized_data['shapes'].append(grad.shape)
            else:
                # Handle None gradients
                quantized_data['tensors'].append(None)
                quantized_data['scales'].append(None)
                quantized_data['zero_points'].append(None)
                quantized_data['shapes'].append(None)

        return quantized_data

    def dequantize_tensor(self, quantized: torch.Tensor, scale: torch.Tensor,
                          zero_point: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        """Dequantize tensor back to float"""
        if quantized.numel() == 0:
            return torch.zeros(original_shape)

        dequantized = scale * (quantized.float() - zero_point)
        return dequantized.reshape(original_shape)

    def decompress(self, quantized_data: Dict) -> List[torch.Tensor]:
        """Dequantize back to gradient tensors"""
        gradients = []
        for i in range(len(quantized_data['tensors'])):
            if quantized_data['tensors'][i] is not None:
                grad = self.dequantize_tensor(
                    quantized_data['tensors'][i],
                    quantized_data['scales'][i],
                    quantized_data['zero_points'][i],
                    quantized_data['shapes'][i]
                )
                gradients.append(grad)
            else:
                gradients.append(None)

        return gradients
