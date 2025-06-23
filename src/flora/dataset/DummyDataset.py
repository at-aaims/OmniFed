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

import rich.repr
import torch
import torch.utils.data as td

# ======================================================================================


@rich.repr.auto
class DummyDataset(td.TensorDataset):
    """
    Synthetic dataset generator for federated learning experimentation and testing.
    """

    def __init__(
        self, num_samples: int = 200, input_dim: int = 10, num_classes: int = 2
    ):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        X = torch.randn(num_samples, input_dim)
        y = torch.randint(0, num_classes, (num_samples,))
        super().__init__(X, y)
