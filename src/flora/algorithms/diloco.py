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

import copy
from typing import Any, Dict, Tuple

import rich.repr
import torch
from torch import nn

from src.flora.helper.node_config import NodeConfig
from src.flora.helper.training_params import DiLocoTrainingParameters

from ..communicator import BaseCommunicator, ReductionType
from . import utils
from .BaseAlgorithm import BaseAlgorithm


class DiLoCo:
    """
    NOTE: Original implementation kept for reference purposes
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: BaseCommunicator,
        total_clients: int,
        train_params: DiLocoTrainingParameters,
    ):
        """
        :param model: model to train
        :param data: data to train
        :param communicator: communicator object
        :param total_clients: total number of clients / world size
        :param train_params: training hyperparameters
        """
        self.model = model
        self.train_data = train_data
        self.communicator = communicator
        self.total_clients = total_clients
        self.train_params = train_params
        self.optimizer = self.train_params.get_optimizer()
        self.comm_freq = self.train_params.get_comm_freq()
        self.loss = self.train_params.get_loss()
        self.epochs = self.train_params.get_epochs()
        self.outer_lr = self.train_params.get_outer_lr()
        self.outer_momentum = self.train_params.get_outer_momentum()
        self.local_step = 0

        dev_id = NodeConfig().get_gpus() % self.total_clients
        self.device = torch.device(
            "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.diff_params = copy.deepcopy(self.model)
        self.global_model, self.diff_params = (
            self.global_model.to(self.device),
            self.diff_params.to(self.device),
        )
        self.velocity = {
            name: torch.zeros_like(param.data)
            for name, param in self.model.named_parameters()
        }

    def broadcast_model(self, model):
        # broadcast model from central server with id 0
        model = self.communicator.broadcast(msg=model, id=0)
        return model

    def aggregate_updates(self):
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                self.global_model.named_parameters(), self.model.named_parameters()
            ):
                assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
                target_param = dict(self.diff_params.named_parameters())[name1]
                target_param.copy_(param1 - param2)

        self.diff_params = self.communicator.aggregate(
            msg=self.diff_params, compute_mean=True
        )

    def _zero_velocity(self):
        for v in self.velocity.values():
            v.zero_()

    def _outer_step(self):
        with torch.no_grad():
            for (name, param), (_, param_delta) in zip(
                self.global_model.named_parameters(),
                self.diff_params.named_parameters(),
            ):
                v = self.velocity[name]
                # Momentum update rule
                v.mul_(self.outer_momentum).add_(param_delta.data, alpha=self.outer_lr)
                # Update model parameters
                param.data.add_(v)

        return self.global_model

    def train_loop(self):
        for inputs, labels in self.train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            pred = self.model(inputs)
            loss = self.loss(pred, labels)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.local_step += 1
            if self.local_step % self.comm_freq == 0:
                self.aggregate_updates()
                self.global_model = self._outer_step()
                self.model.load_state_dict(self.global_model.state_dict())

    def train(self):
        self.model = self.broadcast_model(model=self.model)
        if self.epochs is not None and isinstance(self.epochs, int) and self.epochs > 0:
            for epoch in range(self.epochs):
                self.train_loop()
        else:
            while True:
                self.train_loop()


# ======================================================================================


@rich.repr.auto
class DiLoCoNew(BaseAlgorithm):
    """
    Implementation of DiLoCo (Distributed Low-Communication).

    DiLoCo combines local SGD with server-side momentum updates to
    reduce communication frequency while maintaining convergence properties.

    [DiLoCo](https://arxiv.org/abs/2311.08105) | Arthur Douillard | 2023-11-14
    """

    def __init__(
        self,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        **kwargs,
    ):
        """Initialize DiLoCo algorithm with distributed optimizer parameters."""
        super().__init__(**kwargs)
        self.outer_lr = outer_lr
        self.outer_momentum = outer_momentum

    def _setup(self, device: torch.device) -> None:
        """
        DiLoCo-specific setup: initialize global model and velocity.
        """
        super()._setup(device=device)
        # Deep-copy retains requires_grad state from local_model
        self.global_model = copy.deepcopy(self.local_model)
        # Global model is reference-only for delta computation, set eval mode and disable gradients
        self.global_model.eval()  # eval() does NOT turn off gradient tracking.
        for param in self.global_model.parameters():
            param.requires_grad = False

        # Initialize velocity (server-side momentum)
        # all zero-initialized tensors based on param.data have requires_grad=False by default
        self.velocity: Dict[str, torch.Tensor] = {}
        for name, param in self.local_model.named_parameters():
            if param.requires_grad:
                self.velocity[name] = torch.zeros_like(param.data)

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        Configure SGD optimizer for DiLoCo local updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

    def _train_step(self, batch: Any) -> tuple[torch.Tensor, int]:
        """
        Forward pass and compute cross-entropy loss for a batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        return loss, inputs.size(0)

    def _aggregate(self) -> nn.Module:
        """
        DiLoCo aggregation: distributed low-communication with server-side momentum.
        """
        # Compute local model update (delta from global model)
        local_deltas: Dict[str, torch.Tensor] = {}
        for local_pname, local_pval in self.local_model.named_parameters():
            if local_pval.requires_grad and local_pname in self.velocity:
                global_pval = dict(self.global_model.named_parameters())[local_pname]
                local_deltas[local_pname] = local_pval.data - global_pval.data

        # DiLoCo uses mean aggregation rather than weighted aggregation
        aggregated_deltas = self.local_comm.aggregate(
            msg=local_deltas,
            reduction=ReductionType.MEAN,
        )

        # Apply DiLoCo outer step with momentum using aggregated deltas
        with torch.no_grad():
            for local_pname, global_pval in self.global_model.named_parameters():
                if (
                    global_pval.requires_grad
                    and local_pname in self.velocity
                    and local_pname in aggregated_deltas
                ):
                    # Update velocity with momentum (v = momentum * v + lr_outer * delta)
                    self.velocity[local_pname].mul_(self.outer_momentum).add_(
                        aggregated_deltas[local_pname], alpha=self.outer_lr
                    )
                    # Update global model parameters (param += v)
                    global_pval.data.add_(self.velocity[local_pname])

        # Return updated global model as the new local model for next training period
        return copy.deepcopy(self.global_model)
