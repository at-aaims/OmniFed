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

from . import Compression

# ======================================================================================

def should_compress_tensor(x):
    return (
        isinstance(x, torch.Tensor)
        and x.is_floating_point()
        and x.numel() > 0
    )

def choose_qsgd_storage_width(levels):
    if levels <= torch.iinfo(torch.int8).max:
        return 8, torch.int8
    return 32, torch.int32

class QSGDQuantCompression(Compression):
    """
    QSGD implementation based on the paper:
    [QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding](https://arxiv.org/abs/1610.02132)
    | Alistarh et al. | 2017
    """

    def __init__(self, bit_width=8, device="cpu"):
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
            return v, v, -1, -1

        # Step 1: Compute the norm
        norm_v = torch.norm(v).item()

        if norm_v == 0:
            return torch.zeros_like(v), torch.zeros_like(v), -1, -1

        # Step 2: Normalize the vector
        v_normalized = v / norm_v

        # Step 3: Component-wise quantization
        # For each component v_i, compute sign and magnitude
        signs = torch.sign(v_normalized)
        abs_v = torch.abs(v_normalized)

        # Quantization levels from 0 to s
        levels = 2**self.s

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

        signed_levels = signs.long() * quantized_levels
        # width = 8
        # if levels <= 127:
        #     signed_levels = signed_levels.to(torch.int8)
        # else:
        #     signed_levels = signed_levels.to(torch.int16)
        #     width = 16
        width, storage_dtype = choose_qsgd_storage_width(levels)

        signed_levels = signed_levels.to(storage_dtype)

        return signed_levels, norm_v, width, levels

        # # Convert back to quantized values
        # quantized_abs = quantized_levels.float() / levels
        # quantized_normalized = signs * quantized_abs

        # # Scale back by the norm
        # quantized = norm_v * quantized_normalized

        # return quantized, norm_v
    
    def __do_compress(self, gradients):
        # quantized_grads = []
        # norms = []

        # for grad in gradients:
        if gradients is None:
            return None, None, -1, -1
        else:
            flat_grad = gradients.flatten()
            q_grad, norm, width, levels = self.quantize_vector(flat_grad)
            

            # print("[qsgd] after quantize_vector")
            # print(f"  flat_grad type: {type(flat_grad)}")
            # print(f"  flat_grad dtype: {getattr(flat_grad, 'dtype', None)}")
            # print(f"  flat_grad device: {getattr(flat_grad, 'device', None)}")
            # print(f"  flat_grad shape: {getattr(flat_grad, 'shape', None)}")
            # print(f"  flat_grad numel: {flat_grad.numel() if isinstance(flat_grad, torch.Tensor) else 'N/A'}")
            # print(f"  flat_grad element_size: {flat_grad.element_size() if isinstance(flat_grad, torch.Tensor) else 'N/A'}")

            # print(f"  q_grad type: {type(q_grad)}")
            # print(f"  q_grad dtype: {getattr(q_grad, 'dtype', None)}")
            # print(f"  q_grad device: {getattr(q_grad, 'device', None)}")
            # print(f"  q_grad shape: {getattr(q_grad, 'shape', None)}")
            # print(f"  q_grad numel: {q_grad.numel() if isinstance(q_grad, torch.Tensor) else 'N/A'}")
            # print(f"  q_grad element_size: {q_grad.element_size() if isinstance(q_grad, torch.Tensor) else 'N/A'}")

            # print(f"  norm type: {type(norm)}")
            # print(f"  norm dtype: {getattr(norm, 'dtype', None)}")
            # print(f"  norm device: {getattr(norm, 'device', None)}")
            # print(f"  norm shape: {getattr(norm, 'shape', None)}")

            # if isinstance(q_grad, torch.Tensor):
            #     print(f"  q_grad min: {q_grad.min().item() if q_grad.numel() > 0 else 'empty'}")
            #     print(f"  q_grad max: {q_grad.max().item() if q_grad.numel() > 0 else 'empty'}")
            #     print(f"  q_grad expected bytes: {q_grad.numel() * q_grad.element_size()}")

            q_grad = q_grad.reshape(gradients.shape).to(gradients.device)

            return q_grad, norm, width, levels

            # print("[qsgd] after reshape + device move")
            # print(f"  gradients shape: {gradients.shape}")
            # print(f"  gradients dtype: {gradients.dtype}")
            # print(f"  gradients device: {gradients.device}")
            # print(f"  q_grad dtype: {q_grad.dtype}")
            # print(f"  q_grad device: {q_grad.device}")
            # print(f"  q_grad shape: {q_grad.shape}")
            # print(f"  q_grad element_size: {q_grad.element_size()}")
            # print(f"  q_grad expected bytes: {q_grad.numel() * q_grad.element_size()}")
            # norms.append(norm)

        # print(f"quantized_grads = {quantized_grads}")
        # # print(f"quantized_grads type = {type(quantized_grads)}")
        # print(f"norms = {norms}")

    def compress(self, tensor, name=''):
        """
        Compress tensors using proper QSGD
        """
        # print(f"tensor = {tensor}")
        if not should_compress_tensor(tensor):
            return tensor, -1, -1, -1
        return self.__do_compress(tensor)



_quantized_compression_ = ["QSGDQuantCompression"]
