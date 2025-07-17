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
from typing import Dict


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
            if encrypt_grads:
                encrypted_updates[name] = ts.ckks_vector(self.context, param.grad)
            else:
                encrypted_updates[name] = ts.ckks_vector(self.context, param.data)

        return encrypted_updates

    def decrypt(self, model: torch.nn.Module, encrypted_updates: Dict, encrypt_grads=True):
        for (name1, param), (name2, encrypt_data) in zip(model.named_parameters(), encrypted_updates.items()):
            assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
            if encrypt_grads:
                param.grad = torch.tensor(encrypt_data.decrypt(), dtype=torch.float32).view(param.shape)
            else:
                param.data = torch.tensor(encrypt_data.decrypt(), dtype=torch.float32).view(param.shape)

        return model
