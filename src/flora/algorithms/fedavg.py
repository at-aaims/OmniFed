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

from typing import Any, Tuple

import rich.repr
import torch
from torch import nn


from ..communicator import ReductionType
from . import utils
from .BaseAlgorithm import BaseAlgorithm

# ======================================================================================


@rich.repr.auto
class FedAvg(BaseAlgorithm):
    """
    Federated Averaging (FedAvg) algorithm implementation.

    FedAvg performs standard federated learning by averaging model parameters across clients after local training rounds.
    Only model parameters are aggregated; all clients synchronize with the global model at the start of each round.

    [FedAvg](https://arxiv.org/abs/1602.05629) | H. Brendan McMahan | 2016-02-17
    """

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer for local updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

    def _batch_compute(self, batch: Any) -> Tuple[torch.Tensor, int]:
        """
        Forward pass and compute the cross-entropy loss for a batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        batch_size = inputs.size(0)
        return loss, batch_size

    def _aggregate(self) -> nn.Module:
        """
        Aggregate model parameters across clients and update the local model with the weighted average.
        """

        # Aggregate local sample counts to compute federation total
        global_samples = self.local_comm.aggregate(
            torch.tensor([self.num_samples_trained], dtype=torch.float32),
            reduction=ReductionType.SUM,
        ).item()

        # Calculate this client's data proportion for weighted aggregation
        data_proportion = self.num_samples_trained / max(global_samples, 1)

        # All nodes participate regardless of sample count
        utils.scale_params(self.local_model, data_proportion)

        # Aggregate models across all clients
        aggregated_model = self.local_comm.aggregate(
            self.local_model,
            reduction=ReductionType.SUM,
        )

        return aggregated_model
