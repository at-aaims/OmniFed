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

import torch
from torch import nn

from src.flora.communicator import Communicator
from src.flora.helper.node_config import NodeConfig
from src.flora.helper.training_params import DiLocoTrainingParameters

from . import utils
from .BaseAlgorithm import Algorithm


class DiLoCo:
    """
    NOTE: Original implementation kept for reference purposes
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: Communicator,
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
            msg=self.diff_params, communicate_params=True, compute_mean=True
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


class DiLoCoNew(Algorithm):
    """
    Implementation of DiLoCo (Distributed Low-Communication).

    DiLoCo combines local SGD with server-side momentum updates to
    reduce communication frequency while maintaining convergence properties.
    """

    def __init__(
        self,
        local_model: nn.Module,
        comm: Communicator,
        lr: float = 0.01,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        inner_steps: int = 5,
    ):
        super().__init__(local_model, comm)
        self.lr = lr
        self.outer_lr = outer_lr
        self.outer_momentum = outer_momentum
        self.inner_steps = inner_steps

        # ---
        self.global_model = copy.deepcopy(local_model)

        self.velocity: Dict[str, torch.Tensor] = {}
        for name, param in local_model.named_parameters():
            if param.requires_grad:
                self.velocity[name] = torch.zeros_like(param.data)

    def configure_optimizer(self) -> torch.optim.Optimizer:
        """
        Configure SGD optimizer for DiLoCo local updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=self.lr)

    def train_step(self, batch: Any, batch_idx: int) -> Tuple[torch.Tensor, int]:
        """
        Forward pass and compute cross-entropy loss for a batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        return loss, inputs.size(0)

    def round_start(self, round_idx: int) -> None:
        """
        Synchronize the local model with the global model at the start of each round.
        """
        # Receive the latest global model from the server
        self.local_model = self.comm.broadcast(self.local_model, src=0)
        self.global_model.load_state_dict(self.local_model.state_dict())

    def round_end(self, round_idx: int) -> None:
        """
        Apply DiLoCo outer step with momentum aggregation.

        Steps:
        1. Aggregate local model updates (delta from global model)
        2. Apply server-side momentum update to global model
        3. Broadcast updated global model to all clients
        """
        # Compute local model update (delta from global model)
        local_deltas: Dict[str, torch.Tensor] = {}
        for name, param in self.local_model.named_parameters():
            if param.requires_grad and name in self.velocity:
                global_param = dict(self.global_model.named_parameters())[name]
                local_deltas[name] = param.data - global_param.data

        # Aggregate sample counts to compute global total
        total_samples = self.comm.aggregate(
            torch.tensor([self.round_total_samples], dtype=torch.float32),
            communicate_params=False,
            compute_mean=False,
        ).item()

        if total_samples <= 0:
            print(
                "WARN: No samples processed in this round... possible client failure or aggregation error?"
            )
            return

        # Aggregate local deltas across all clients
        aggregated_deltas = self.comm.aggregate(msg=local_deltas, compute_mean=True)

        # Apply DiLoCo outer step with momentum using aggregated deltas
        for name, param in self.global_model.named_parameters():
            if (
                param.requires_grad
                and name in self.velocity
                and name in aggregated_deltas
            ):
                # Update velocity with momentum (v = momentum * v + lr_outer * delta)
                self.velocity[name].mul_(self.outer_momentum).add_(
                    aggregated_deltas[name], alpha=self.outer_lr
                )
                # Update global model parameters (param += v)
                param.data.add_(self.velocity[name])

        # Update local model to match global model
        self.local_model = copy.deepcopy(self.global_model)
