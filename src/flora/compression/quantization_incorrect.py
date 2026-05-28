# # Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
#
# import torch
#
# from src.flora.compression import Compression, ResidualUpdates
#
#
# class QSGDCompression(Compression):
#     """Implementation of quantized SGD or QSGD lossy quantization-based compression with error feedback"""
#
#     def __init__(self, device, bit_width):
#         super().__init__()
#         self.device = device
#         self.bit_width = bit_width
#         self.scale = (2 ** self.bit_width) - 1
#         self.residual = ResidualUpdates()  # Add error feedback
#
#     def get_compress_minmax(self, tensor):
#         tensor = tensor.to(self.device)
#         min_val, max_val = tensor.min(), tensor.max()
#         return min_val, max_val
#
#     def compress(self, tensor, name, min_val=None, max_val=None):
#         tensor = tensor.to(self.device)
#
#         # Apply error feedback compensation
#         tensor = self.residual.compensate(tensor, name)
#
#         # Get min/max if not provided
#         if min_val is None or max_val is None:
#             min_val, max_val = self.get_compress_minmax(tensor)
#
#         # Store original for residual calculation
#         original_tensor = tensor.clone()
#
#         # Avoid division by zero
#         if max_val == min_val:
#             quantized_tensor = torch.zeros_like(tensor)
#         else:
#             # Scale the tensor to [0, scale]
#             scaled_tensor = (tensor - min_val) / (max_val - min_val) * self.scale
#             # Round and clamp to [0, scale]
#             quantized_tensor = torch.round(scaled_tensor).clamp(0, self.scale)
#
#         # Create context for decompression
#         ctx = (min_val, max_val, tensor.shape)
#
#         # Update residuals with quantization error
#         self.residual.update(original_tensor, name, self, quantized_tensor, ctx)
#
#         return quantized_tensor, ctx
#
#     def decompress(self, tensor, ctx):
#         min_val, max_val, shape = ctx
#
#         # Avoid division by zero
#         if max_val == min_val:
#             return torch.full(shape, min_val.item(), dtype=torch.float32, device=self.device)
#
#         # Transform tensors back to its original range
#         decompressed = min_val + (tensor.float() / self.scale) * (max_val - min_val)
#         return decompressed.view(shape)
#
#
# class AMPCompression(Compression):
#     """Implementation of Automatic Mixed-Precision (AMP) training with error feedback"""
#
#     def __init__(self, device):
#         super().__init__()
#         self.device = device
#         self.loss_scale_factor = (2 ** 16) - 1
#         self.residual = ResidualUpdates()  # Add error feedback
#
#     def compress(self, tensor, name):
#         tensor = tensor.to(self.device)
#
#         # Apply error feedback compensation
#         tensor = self.residual.compensate(tensor, name)
#
#         # Store original for residual calculation
#         original_tensor = tensor.clone()
#
#         # Convert tensors to 16-bit
#         compressed_tensor = tensor.half()
#
#         # Create context for decompression
#         ctx = tensor.shape
#
#         # Update residuals with precision loss error
#         self.residual.update(original_tensor, name, self, compressed_tensor, ctx)
#
#         return compressed_tensor, ctx
#
#     def decompress(self, tensor, ctx):
#         shape = ctx
#         # Convert tensors back to 32-bit
#         return tensor.float().view(shape)
#
#     def loss_scaling(self, loss):
#         return loss * self.loss_scale_factor
#
#     def gradient_unscaling(self, model):
#         if isinstance(model, torch.nn.Module):
#             for p in model.parameters():
#                 if p.grad is not None:
#                     p.grad.data /= self.loss_scale_factor
#             return model
#         else:
#             raise TypeError(
#                 "gradient_unscaling fn needs torch.nn.Module type for model argument"
#             )
