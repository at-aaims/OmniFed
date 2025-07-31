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
import numpy as np


class QSGDQuantCompression:
    def __init__(self, bit_width=8, device="cpu"):
        """
        QSGD (Quantized Stochastic Gradient Descent) compression

        Args:
            bit_width: Number of bits for quantization (default: 8)
            device: Device to perform computations on
        """
        self.bit_width = bit_width
        self.device = device
        self.levels = 2**bit_width - 1  # Number of quantization levels

    def compress(self, gradients):
        """
        Compress gradients using QSGD quantization

        Args:
            gradients: List of gradient tensors

        Returns:
            Dictionary containing quantized tensors, scales, and zero points
        """
        quantized_tensors = []
        scales = []
        zero_points = []

        for grad in gradients:
            if grad is None:
                quantized_tensors.append(None)
                scales.append(None)
                zero_points.append(None)
                continue

            # Flatten gradient for easier processing
            flat_grad = grad.flatten()

            # Calculate quantization parameters
            grad_min = flat_grad.min()
            grad_max = flat_grad.max()

            # Handle edge case where all gradients are the same
            if grad_max == grad_min:
                scale = 1.0
                zero_point = 0.0
                quantized = torch.zeros_like(flat_grad, dtype=torch.int32)
            else:
                # Calculate scale and zero point for symmetric quantization
                scale = (grad_max - grad_min) / self.levels
                zero_point = grad_min / scale

                # Quantize: q = round((x - min) / scale)
                quantized = torch.round((flat_grad - grad_min) / scale)
                quantized = torch.clamp(quantized, 0, self.levels).int()

            quantized_tensors.append(quantized)
            scales.append(scale)
            zero_points.append(zero_point)

        return {
            "tensors": quantized_tensors,
            "scales": scales,
            "zero_points": zero_points,
        }

    def decompress(self, quantized_data):
        """
        Decompress quantized gradients back to float tensors

        Args:
            quantized_data: Dictionary from compress() method

        Returns:
            List of decompressed gradient tensors
        """
        decompressed_gradients = []

        for i, (q_tensor, scale, zero_point) in enumerate(
            zip(
                quantized_data["tensors"],
                quantized_data["scales"],
                quantized_data["zero_points"],
            )
        ):
            if q_tensor is None:
                decompressed_gradients.append(None)
                continue

            # Dequantize: x = scale * q + min_val
            # Since we used zero_point = min_val / scale, min_val = scale * zero_point
            min_val = scale * zero_point
            decompressed = scale * q_tensor.float() + min_val

            decompressed_gradients.append(decompressed)

        return decompressed_gradients


class ImprovedQSGDCompressTraining:
    """
    Improved QSGD training with better compression and aggregation
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        test_data: torch.utils.data.DataLoader,
        communicator,
        client_id: int,
        total_clients: int,
        train_params,
        compression: QSGDQuantCompression,
    ):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.communicator = communicator
        self.client_id = client_id
        self.total_clients = total_clients
        self.train_params = train_params
        self.compression = compression
        self.optimizer = self.train_params.get_optimizer()
        self.comm_freq = self.train_params.get_comm_freq()
        self.loss = self.train_params.get_loss()
        self.lr_scheduler = self.train_params.get_lr_scheduler()
        self.epochs = self.train_params.get_epochs()
        self.local_step = 0
        self.training_samples = 0

        dev_id = self.client_id % 4
        self.device = torch.device(
            "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

    def broadcast_model(self, model):
        model = self.communicator.broadcast(msg=model, id=0)
        return model

    def aggregate_compressed_gradients(self):
        """
        Aggregate gradients using QSGD compression
        """
        # Extract gradients
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone())
            else:
                gradients.append(None)

        # Compress gradients locally
        quantized_data = self.compression.compress(gradients)

        # Method 1: Proper QSGD - dequantize locally then aggregate
        decompressed_grads = self.compression.decompress(quantized_data)

        # Update model gradients with decompressed versions
        param_idx = 0
        for param in self.model.parameters():
            if param.grad is not None:
                if decompressed_grads[param_idx] is not None:
                    param.grad.data = decompressed_grads[param_idx].reshape(
                        param.grad.shape
                    )
                param_idx += 1

        # Now aggregate the decompressed gradients
        self.communicator.aggregate(
            self.model, communicate_params=False, compute_mean=True
        )

    def train_loop(self, epoch):
        """Main training loop with improved QSGD"""
        for batch_idx, (inputs, labels) in enumerate(self.train_data):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            pred = self.model(inputs)
            loss = self.loss(pred, labels)

            # Backward pass
            loss.backward()

            # Aggregate gradients with compression every comm_freq steps
            if (self.local_step + 1) % self.comm_freq == 0:
                self.aggregate_compressed_gradients()

            # Update model
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.local_step += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def train(self):
        print("Broadcasting initial model...")
        self.model = self.broadcast_model(model=self.model)

        for epoch in range(self.epochs):
            print(f"Starting epoch {epoch + 1}/{self.epochs}")
            self.train_loop(epoch)

            # Test accuracy
            self.evaluate(epoch)
