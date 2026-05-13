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
import logging

import torch

from src.flora.compression import Compression, ResidualUpdates


class QSGDCompressionDebug(Compression):
    """Debug version of QSGD with extensive logging"""

    def __init__(self, device, bit_width):
        super().__init__()
        self.device = device
        self.bit_width = bit_width
        self.scale = (2**self.bit_width) - 1
        self.residual = ResidualUpdates()
        self.step_count = 0

    def get_compress_minmax(self, tensor):
        tensor = tensor.to(self.device)
        min_val, max_val = tensor.min(), tensor.max()
        return min_val, max_val

    def compress(self, tensor, name, min_val=None, max_val=None):
        tensor = tensor.to(self.device)
        self.step_count += 1

        # Store original tensor before compensation
        original_tensor = tensor.clone()
        original_norm = original_tensor.norm().item()

        # Apply error feedback compensation
        compensated_tensor = self.residual.compensate(tensor, name)
        compensated_norm = compensated_tensor.norm().item()

        # Debug logging every 50 steps
        if self.step_count % 50 == 0:
            logging.info(
                f"DEBUG {name}: original_norm={original_norm:.6f}, compensated_norm={compensated_norm:.6f}"
            )
            if name in self.residual.residuals:
                residual_norm = self.residual.residuals[name].norm().item()
                logging.info(f"DEBUG {name}: residual_norm={residual_norm:.6f}")

        # Get min/max from compensated tensor
        if min_val is None or max_val is None:
            min_val, max_val = self.get_compress_minmax(compensated_tensor)

        range_val = (max_val - min_val).item()

        # Debug logging for problematic ranges
        if range_val < 1e-6 or range_val > 1000:
            logging.info(
                f"WARNING {name}: range={range_val:.8f}, min={min_val:.6f}, max={max_val:.6f}"
            )

        # Quantize the compensated tensor
        if torch.abs(max_val - min_val) < 1e-8:
            quantized_tensor = torch.zeros_like(compensated_tensor)
            logging.info(f"WARNING {name}: Zero range, setting to zeros")
        else:
            # Scale tensor to [0, scale]
            scaled_tensor = (
                (compensated_tensor - min_val) / (max_val - min_val) * self.scale
            )
            quantized_tensor = torch.round(scaled_tensor).clamp(0, self.scale)

        # Create context
        ctx = (min_val, max_val, tensor.shape)

        # Update residuals using original tensor
        self.residual.update(original_tensor, name, self, quantized_tensor, ctx)

        return quantized_tensor, ctx

    def decompress(self, tensor, ctx):
        min_val, max_val, shape = ctx

        if torch.abs(max_val - min_val) < 1e-8:
            return torch.full(
                shape, min_val.item(), dtype=torch.float32, device=self.device
            )

        # Decompress back to original range
        decompressed = min_val + (tensor.float() / self.scale) * (max_val - min_val)

        # Debug logging
        if self.step_count % 50 == 0:
            decompressed_norm = decompressed.norm().item()
            quantization_error = (
                (decompressed.view(-1) - tensor.float().view(-1)).norm().item()
            )
            logging.info(
                f"DEBUG decompress: norm={decompressed_norm:.6f}, quant_error={quantization_error:.6f}"
            )

        return decompressed.view(shape)


class QSGDCompression(Compression):
    """Implementation of quantized SGD with proper distributed training support"""

    def __init__(self, device, bit_width):
        super().__init__()
        self.device = device
        self.bit_width = bit_width
        self.scale = (2**self.bit_width) - 1
        self.residual = ResidualUpdates()

    def get_compress_minmax(self, tensor):
        tensor = tensor.to(self.device)
        min_val, max_val = tensor.min(), tensor.max()
        return min_val, max_val

    def compress(self, tensor, name, min_val=None, max_val=None):
        tensor = tensor.to(self.device)

        # Store original tensor before compensation
        original_tensor = tensor.clone()

        # Apply error feedback compensation
        compensated_tensor = self.residual.compensate(tensor, name)

        # CRITICAL: Get min/max from compensated tensor, not original
        if min_val is None or max_val is None:
            min_val, max_val = self.get_compress_minmax(compensated_tensor)

        # Quantize the compensated tensor
        if torch.abs(max_val - min_val) < 1e-8:
            quantized_tensor = torch.zeros_like(compensated_tensor)
        else:
            # Scale tensor to [0, scale]
            scaled_tensor = (
                (compensated_tensor - min_val) / (max_val - min_val) * self.scale
            )
            quantized_tensor = torch.round(scaled_tensor).clamp(0, self.scale)

        # Create context with per-tensor min/max
        ctx = (min_val, max_val, tensor.shape)

        # Update residuals using original tensor
        self.residual.update(original_tensor, name, self, quantized_tensor, ctx)

        return quantized_tensor, ctx

    def decompress(self, tensor, ctx):
        min_val, max_val, shape = ctx

        if torch.abs(max_val - min_val) < 1e-8:
            return torch.full(
                shape, min_val.item(), dtype=torch.float32, device=self.device
            )

        # Decompress back to original range
        decompressed = min_val + (tensor.float() / self.scale) * (max_val - min_val)
        return decompressed.view(shape)


class AMPCompression(Compression):
    """Implementation of Automatic Mixed-Precision (AMP) training with proper error feedback"""

    def __init__(self, device, use_loss_scaling=True):
        super().__init__()
        self.device = device
        self.use_loss_scaling = use_loss_scaling
        self.loss_scale_factor = (2**16) - 1 if use_loss_scaling else 1.0
        self.residual = ResidualUpdates()  # Add error feedback

    def compress(self, tensor, name):
        tensor = tensor.to(self.device)

        # Store the ORIGINAL tensor before any compensation
        original_tensor = tensor.clone()

        # Apply error feedback compensation AFTER storing original
        compensated_tensor = self.residual.compensate(tensor, name)

        # Convert compensated tensor to 16-bit
        compressed_tensor = compensated_tensor.half()

        # Create context for decompression
        ctx = tensor.shape

        # CRITICAL: Update residuals using the ORIGINAL tensor, not compensated
        self.residual.update(original_tensor, name, self, compressed_tensor, ctx)

        return compressed_tensor, ctx

    def decompress(self, tensor, ctx):
        shape = ctx
        # Convert tensors back to 32-bit
        return tensor.float().view(shape)

    def loss_scaling(self, loss):
        """Scale loss to prevent gradient underflow in FP16"""
        if self.use_loss_scaling:
            return loss * self.loss_scale_factor
        return loss

    def gradient_unscaling(self, model):
        """Unscale gradients after backward pass, integrating with residual updates"""
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                "gradient_unscaling fn needs torch.nn.Module type for model argument"
            )

        if self.use_loss_scaling:
            # Unscale gradients and handle residuals properly
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Unscale the gradient
                    param.grad.data /= self.loss_scale_factor

                    # Also unscale residuals if they exist
                    if name in self.residual.residuals:
                        self.residual.residuals[name] /= self.loss_scale_factor

        return model

    def check_grad_overflow(self, model):
        """Check for gradient overflow (infinity/NaN) which can happen with loss scaling"""
        for param in model.parameters():
            if param.grad is not None:
                if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    return True
        return False

    def handle_overflow(self):
        """Reset residuals on overflow to prevent error accumulation"""
        if self.use_loss_scaling:
            # Clear residuals on overflow
            self.residual.residuals.clear()
            # Optionally reduce loss scale factor
            self.loss_scale_factor = max(self.loss_scale_factor / 2.0, 1.0)
