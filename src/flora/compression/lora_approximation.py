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

from typing import Dict, Tuple, Optional
from time import perf_counter_ns

import torch

from src.flora.compression import Compression
from src.flora.communicator import Communicator

nanosec_to_millisec = 1e6


# class PowerSGDCompressionOLD(Compression):
#     def __init__(self, device, compress_rank=1, power_itr=1):
#         """Implementation of PowerSGD, a low-rank approximation technique for compression model updates.
#         If T is (n x m), and rank is 'r', matrix P has shape (n x r) and Q is (m x r). So,
#         T = P @ Q.T
#         P = T @ Q
#         Q = T.T @ P
#         """
#         super().__init__()
#         self.device = device
#         self.compress_rank = compress_rank
#         self.power_itr = power_itr
#
#     # def compress(self, tensor):
#     #     u, s, v = (tensor.view(-1, tensor.numel()), some=False)
#     #     if self.compress_rank <= u.shape[1]:
#     #         u = u[:, :self.compress_rank]
#     #         s = s[: self.compress_rank]
#     #         v = v[:, :self.compress_rank]
#     #
#     #     return (u @ torch.diag(s) @ v.t()).view_as(tensor)
#
#     def compress(self, tensor):
#         n, m = tensor.shape
#         tensor = tensor.view(tensor.shape[0], -1)
#         Q = torch.randn(m, self.compress_rank, device=self.device)
#         Q, _ = torch.linalg.qr(Q)
#         # Power iteration
#         for _ in range(self.power_itr):
#             # compute P = tensor @ Q
#             P = torch.matmul(tensor, Q)
#             # Orthonormalize
#             P, _ = torch.linalg.qr(P)
#
#             # update Q = tensor.T @ P
#             Q = torch.matmul(tensor.T, P)
#             # Orthonormalize
#             Q, _ = torch.linalg.qr(Q)
#
#         return P, Q
#
#     def decompress(self, P, Q):
#         return torch.matmul(P, Q.T)

# class PowerSGDCompressionVersion1(Compression):
#     def __init__(self, device, compress_rank=1, power_itr=2, min_compression_rate=2):
#         """Implementation of PowerSGD with error feedback.
#         Args:
#             device: Device to run computations on
#             compress_rank: Rank for low-rank approximation
#             power_itr: Number of power iterations
#             min_compression_rate: Minimum compression rate to apply PowerSGD
#         """
#         super().__init__()
#         self.device = device
#         self.compress_rank = compress_rank
#         self.power_itr = power_itr
#         self.min_compression_rate = min_compression_rate
#
#         # Error feedback buffers for each parameter
#         self.error_dict: Dict[str, torch.Tensor] = {}
#
#         # Q matrices for each parameter (maintained across iterations)
#         self.q_memory: Dict[str, torch.Tensor] = {}
#
#     def _should_compress(self, tensor: torch.Tensor) -> bool:
#         """Check if tensor should be compressed based on compression rate."""
#         numel = tensor.numel()
#         if len(tensor.shape) < 2:
#             return False
#
#         # For matrix of shape (m, n), compressed size is rank * (m + n)
#         # For higher dimensional tensors, we reshape to 2D
#         if len(tensor.shape) == 2:
#             m, n = tensor.shape
#         else:
#             m = tensor.shape[0]
#             n = tensor.numel() // m
#
#         compressed_size = self.compress_rank * (m + n)
#         compression_rate = numel / compressed_size
#
#         return compression_rate >= self.min_compression_rate
#
#     def _reshape_for_compression(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
#         """Reshape tensor to 2D for compression, return reshaped tensor and original shape."""
#         original_shape = tensor.shape
#
#         if len(tensor.shape) == 1:
#             # Vector case - no compression needed
#             # print(f'@@@@@@@@@@@@@@@@@@@@@ no compression {tensor.shape} og_shape {original_shape}')
#             return tensor, original_shape
#         elif len(tensor.shape) == 2:
#             # Already 2D
#             # print(f'^^^^^^^^^^^^^^^^^^^^ shape {tensor.shape} og_shape {original_shape}')
#             return tensor, original_shape
#         else:
#             # Higher dimensional tensor - reshape to 2D
#             # Keep first dimension, flatten the rest
#             reshaped = tensor.view(tensor.shape[0], -1)
#             # print(f'&&&&&&&&&&&&&&&&&&&&& reshaped higher dimension! {reshaped.shape} og_shape {original_shape}')
#             return reshaped, original_shape
#
#     def _update_Q(self, param_Q: torch.Tensor, param_name: str):
#         self.q_memory[param_name] = param_Q
#
#     def compress(self, tensor: torch.Tensor, param_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
#         """Compress tensor using PowerSGD low-rank approximation.
#
#                 Args:
#                     tensor: Input tensor (gradient)
#                     param_name: Parameter name for error feedback tracking
#
#                 Returns:
#                     P matrix, Q matrix, original shape, whether compression was applied
#                 """
#         # Add error feedback
#         if param_name in self.error_dict:
#             tensor = tensor + self.error_dict[param_name]
#
#             # Check if we should compress
#         if not self._should_compress(tensor):
#             return tensor, None, tensor.shape
#
#             # Reshape for compression
#         matrix, original_shape = self._reshape_for_compression(tensor)
#         m, n = matrix.shape
#
#         # Initialize or reuse Q matrix
#         if param_name not in self.q_memory:
#             Q = torch.randn(n, self.compress_rank, device=self.device, dtype=matrix.dtype)
#             Q, _ = torch.linalg.qr(Q)
#             self.q_memory[param_name] = Q
#         else:
#             Q = self.q_memory[param_name]
#
#         # Power iteration
#         for _ in range(self.power_itr):
#             # P = matrix @ Q
#             P = torch.matmul(matrix, Q)
#             # Orthonormalize P
#             P, _ = torch.linalg.qr(P)
#
#             # Q = matrix.T @ P
#             Q = torch.matmul(matrix.T, P)
#             # Orthonormalize Q
#             Q, _ = torch.linalg.qr(Q)
#
#         return P, Q, original_shape
#
#     def decompress(self, P: torch.Tensor, Q: torch.Tensor, original_shape: torch.Size):
#         """Decompress P and Q matrices back to original tensor.
#         P: P matrix from compression
#         Q: Q matrix from compression
#         original_shape: Original tensor shape
#         Returns:
#             Decompressed tensor
#         """
#         return torch.matmul(P, Q.T).view(original_shape)
#
#     def update_error_feedback(self, original_grad: torch.Tensor, compressed_grad: torch.Tensor, param_name: str):
#         """Update error feedback buffer.
#         Args:
#             original_grad: Original gradient before compression
#             compressed_grad: Gradient after compression and decompression
#             param_name: Parameter name for tracking
#         """
#         error = original_grad - compressed_grad
#
#         if param_name in self.error_dict:
#             self.error_dict[param_name] = error
#         else:
#             self.error_dict[param_name] = error.clone()

class PowerSGDCompression(Compression):
    def __init__(self, device, compress_rank=1, power_itr=2, min_compression_rate=2):
        """Implementation of PowerSGD with error feedback.
        Args:
            device: Device to run computations on
            compress_rank: Rank for low-rank approximation
            power_itr: Number of power iterations
            min_compression_rate: Minimum compression rate to apply PowerSGD
        """
        super().__init__()
        self.device = device
        self.compress_rank = compress_rank
        self.power_itr = power_itr
        self.min_compression_rate = min_compression_rate

        # Error feedback buffers for each parameter
        self.error_dict: Dict[str, torch.Tensor] = {}

    def _should_compress(self, tensor: torch.Tensor) -> bool:
        """Check if tensor should be compressed based on compression rate."""
        numel = tensor.numel()
        if len(tensor.shape) < 2:
            return False

        # For matrix of shape (m, n), compressed size is rank * (m + n)
        # For higher dimensional tensors, we reshape to 2D
        if len(tensor.shape) == 2:
            m, n = tensor.shape
        else:
            m = tensor.shape[0]
            n = tensor.numel() // m

        compressed_size = self.compress_rank * (m + n)
        compression_rate = numel / compressed_size

        return compression_rate >= self.min_compression_rate

    def _reshape_for_compression(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
        """Reshape tensor to 2D for compression, return reshaped tensor and original shape."""
        original_shape = tensor.shape

        if len(tensor.shape) == 1:
            # Vector case - no compression needed
            return tensor, original_shape
        elif len(tensor.shape) == 2:
            # Already 2D
            return tensor, original_shape
        else:
            # Higher dimensional tensor - reshape to 2D
            # Keep first dimension, flatten the rest
            reshaped = tensor.view(tensor.shape[0], -1)
            return reshaped, original_shape

    def _restore_shape(self, tensor: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        """Restore tensor to original shape."""
        return tensor.view(original_shape)

    def compress(self, tensor: torch.Tensor, param_name: str) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Size, bool]:
        """Compress tensor using PowerSGD low-rank approximation.
            Args:
                tensor: Input tensor (gradient)
                param_name: Parameter name for error feedback tracking
            Returns:
                P matrix (or original tensor if not compressed), Q matrix, original shape, whether compression was applied
        """
        # Add error feedback
        if param_name in self.error_dict:
            tensor = tensor + self.error_dict[param_name]

        # Check if we should compress
        if not self._should_compress(tensor):
            # For uncompressed tensors, we still need to aggregate across workers
            # if dist.is_initialized():
            #     dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            #     tensor = tensor / dist.get_world_size()
            # return tensor, None, tensor.shape, False
            return tensor, None, tensor.shape, False

        # Reshape for compression
        matrix, original_shape = self._reshape_for_compression(tensor)
        m, n = matrix.shape

        # Initialize Q matrix fresh each time (as per paper)
        Q = torch.randn(n, self.compress_rank, device=self.device, dtype=matrix.dtype)
        Q, _ = torch.linalg.qr(Q)

        # Power iteration (local computation on each worker)
        for _ in range(self.power_itr):
            # P = matrix @ Q
            P = torch.matmul(matrix, Q)
            # Orthonormalize P
            P, _ = torch.linalg.qr(P)

        # Communication: Only all-reduce P matrices (as per paper)
        # if dist.is_initialized():
        #     dist.all_reduce(P, op=dist.ReduceOp.SUM)
        #     P = P / dist.get_world_size()

        return P, matrix, original_shape, True

    def _update_Q(self, P, matrix):
        # Recompute Q using the averaged P (as per paper)
        Q = torch.matmul(matrix.T, P)
        Q, _ = torch.linalg.qr(Q)
        return Q

    def decompress(self, P: torch.Tensor, Q: Optional[torch.Tensor], original_shape: torch.Size,
                   was_compressed: bool) -> torch.Tensor:
        """Decompress P and Q matrices back to original tensor.

        Args:
            P: P matrix from compression (or original tensor if not compressed)
            Q: Q matrix from compression (None if not compressed)
            original_shape: Original tensor shape
            was_compressed: Whether compression was actually applied

        Returns:
            Decompressed tensor
        """
        if not was_compressed:
            # P contains the already aggregated original tensor
            return P

        # Reconstruct matrix from P and Q
        reconstructed = torch.matmul(P, Q.T)

        # Restore original shape
        return self._restore_shape(reconstructed, original_shape)

    def update_error_feedback(self, original_grad: torch.Tensor, compressed_grad: torch.Tensor,
                              param_name: str):
        """Update error feedback buffer.

        Args:
            original_grad: Original gradient before compression
            compressed_grad: Gradient after compression and decompression
            param_name: Parameter name for tracking
        """
        error = original_grad - compressed_grad

        if param_name in self.error_dict:
            self.error_dict[param_name] = error
        else:
            self.error_dict[param_name] = error.clone()
