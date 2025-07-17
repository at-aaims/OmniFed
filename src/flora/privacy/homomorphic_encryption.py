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

import tenseal as ts
import torch


class HomomorphicEncryption:
    def __init__(self):
        self.context = context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.global_scale = 2 ** 40
        self.context.generate_galois_keys()

    def encrypt(self, model: torch.nn.Module, encrypt_grads=True):
        """
        model: pytorch model to encrypt
        encrypt_grads: whether to encrypt gradients or model parameters
        """
        encrypted_updates = {}
        for name, param in model.named_parameters():
