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


class QSGDQuantCompression:
    """
    Correct QSGD implementation based on the paper:
    "QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding"
    """

    def __init__(self, bit_width=8, device='cpu'):
        self.s = bit_width  # Number of quantization levels = 2^s
        self.device = device

    def quantize_vector(self, v):
        """
        QSGD quantization function Q_s(v) as defined in Algorithm 1 of the paper

        Args:
            v: Input vector to quantize

        Returns:
            Quantized vector with same expected value as input
        """
        if v.numel() == 0:
            return v

        # Step 1: Compute the norm
        norm_v = torch.norm(v).item()

        if norm_v == 0:
            return torch.zeros_like(v)

        # Step 2: Normalize the vector
        v_normalized = v / norm_v

        # Step 3: Component-wise quantization
        # For each component v_i, compute sign and magnitude
        signs = torch.sign(v_normalized)
        abs_v = torch.abs(v_normalized)

        # Quantization levels from 0 to s
        levels = 2 ** self.s

        # Stochastic quantization
        # ξ_i = l/s where l is chosen such that l/s ≤ |v_i| ≤ (l+1)/s
        # with probability proportional to the distance
        scaled_abs = abs_v * levels
        l = torch.floor(scaled_abs).long()  # Lower quantization level

        # Stochastic rounding: probability of rounding up
        prob_round_up = scaled_abs - l.float()
        random_vals = torch.rand_like(prob_round_up)
        round_up = (random_vals < prob_round_up).long()

        # Final quantization levels
        quantized_levels = l + round_up
        quantized_levels = torch.clamp(quantized_levels, 0, levels)

        # Convert back to quantized values
        quantized_abs = quantized_levels.float() / levels
        quantized_normalized = signs * quantized_abs

        # Scale back by the norm
        quantized = norm_v * quantized_normalized

        return quantized, norm_v

    def compress(self, gradients):
        """
        Compress gradients using proper QSGD
        """
        quantized_grads = []
        norms = []

        for grad in gradients:
            if grad is None:
                quantized_grads.append(None)
                norms.append(None)
            else:
                flat_grad = grad.flatten()
                q_grad, norm = self.quantize_vector(flat_grad)
                quantized_grads.append(q_grad.reshape(grad.shape))
                norms.append(norm)

        return quantized_grads, norms