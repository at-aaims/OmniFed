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

# TODO: implement error-feedback in PowerSGD low-rank approximation


class PowerSGDCompression(Compression):
    def __init__(self, device, compress_rank=1, power_itr=1):
        """Implementation of PowerSGD, a low-rank approximation technique for compression model updates.
        If T is (n x m), and rank is 'r', matrix P has shape (n x r) and Q is (m x r). So,
        T = P @ Q.T
        P = T @ Q
        Q = T.T @ P
        """
        super().__init__()
        self.device = device
        self.compress_rank = compress_rank
        self.power_itr = power_itr

    # def compress(self, tensor):
    #     u, s, v = (tensor.view(-1, tensor.numel()), some=False)
    #     if self.compress_rank <= u.shape[1]:
    #         u = u[:, :self.compress_rank]
    #         s = s[: self.compress_rank]
    #         v = v[:, :self.compress_rank]
    #
    #     return (u @ torch.diag(s) @ v.t()).view_as(tensor)

    def compress(self, tensor):
        n, m = tensor.shape
        tensor = tensor.view(tensor.shape[0], -1)
        Q = torch.randn(m, self.compress_rank, device=self.device)
        Q, _ = torch.linalg.qr(Q)
        # Power iteration
        for _ in range(self.power_itr):
            # compute P = tensor @ Q
            P = torch.matmul(tensor, Q)
            # Orthonormalize
            P, _ = torch.linalg.qr(P)

            # update Q = tensor.T @ P
            Q = torch.matmul(tensor.T, P)
            # Orthonormalize
            Q, _ = torch.linalg.qr(Q)

        return P, Q

    def decompress(self, P, Q):
        return torch.matmul(P, Q.T)
