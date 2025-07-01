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
import torch.nn as nn

from src.flora.communicator import Communicator
from src.flora.helper.node_config import NodeConfig
from src.flora.helper.training_params import FedNovaTrainingParameters

from . import utils
from .BaseAlgorithm import Algorithm


class FedNova:
    """
    Implementation of Federated Normalized Averaging or FedNova

    NOTE: Original implementation kept for reference purposes
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: Communicator,
        total_clients: int,
        train_params: FedNovaTrainingParameters,
    ):
        """
        Initialize FedNova algorithm instance.

        Args:
            model (torch.nn.Module): Model to train.
            train_data (torch.utils.data.DataLoader): Local training data.
            communicator (Communicator): Communication interface.
            total_clients (int): Number of clients in the federation.
            train_params (FedNovaTrainingParameters): Training hyperparameters.
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
        self.weight_decay = self.train_params.get_weight_decay()
        self.local_step = 0
        self.training_samples = 0
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

    def broadcast_model(self, model):
        # broadcast model from central server with id 0
        model = self.communicator.broadcast(msg=model, id=0)
        return model

    def compute_alpha(self, lr):
        momentum_term = 1 - lr * self.weight_decay
        alpha = 0.0
        for j in range(self.comm_freq):
            alpha += momentum_term**j
        alpha *= lr
        return alpha

    def normalized_update(self, weight_scaling: float):
        """sends normalized updates and receives scaled, aggregated update"""
        lr = self.optimizer.param_groups[0]["lr"]
        alpha = self.compute_alpha(lr)
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                self.global_model.named_parameters(), self.model.named_parameters()
            ):
                assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
                target_param = dict(self.diff_params.named_parameters())[name1]
                target_param.copy_((weight_scaling * (param1 - param2)) / alpha)

        self.diff_params = self.communicator.aggregate(
            msg=self.diff_params, communicate_params=True, compute_mean=False
        )

    def model_update(self):
        lr = self.optimizer.param_groups[0]["lr"]
        with torch.no_grad():
            for (name1, param1), (name2, param_delta) in zip(
                self.global_model.named_parameters(),
                self.diff_params.named_parameters(),
            ):
                assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
                param1 -= lr * param_delta

    def train_loop(self):
        for inputs, labels in self.train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            pred = self.model(inputs)
            loss = self.loss(pred, labels)
            self.training_samples += inputs.size(0)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.local_step += 1
            if self.local_step % self.comm_freq == 0:
                # total samples processed across all clients
                total_samples = self.communicator.aggregate(
                    msg=torch.Tensor([self.training_samples]), compute_mean=False
                )
                weight_scaling = self.training_samples / total_samples.item()
                self.normalized_update(weight_scaling)
                self.model_update()
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


class FedNovaNew(Algorithm):
    """
    Implementation of Federated Normalized Averaging (FedNova).

    FedNova normalizes local updates to address objective inconsistency in federated learning,
    accounting for varying numbers of local steps and learning dynamics across clients.
    """

    def __init__(
        self,
        local_model: nn.Module,
        comm: Communicator,
        lr: float = 0.01,
        weight_decay: float = 0.0,
    ):
        super().__init__(local_model, comm)
        self.lr = lr
        self.weight_decay = weight_decay

        # ---
        self.global_model = copy.deepcopy(local_model)
        self.local_steps_this_round: int = 0

    def configure_optimizer(self) -> torch.optim.Optimizer:
        """
        SGD optimizer with weight decay.
        """
        return torch.optim.SGD(
            self.local_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def train_step(self, batch: Any, batch_idx: int) -> Tuple[torch.Tensor, int]:
        """
        Perform a forward pass and compute cross-entropy loss for a batch.
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
        self.local_steps_this_round = 0

    def optimizer_step(self, batch_idx: int) -> None:
        """
        Perform an optimizer step and increment the local step counter for normalization.
        """
        self.optimizer.step()
        self.local_steps_this_round += 1

    def _compute_alpha(self, lr: float, local_steps: int) -> float:
        """
        Compute the normalization coefficient alpha.
        """
        if local_steps <= 0:
            return lr
        momentum_term = 1 - lr * self.weight_decay
        alpha = 0.0
        for j in range(local_steps):
            alpha += momentum_term**j
        alpha *= lr
        return alpha

    def round_end(self, round_idx: int) -> None:
        """
        Apply FedNova normalized averaging and update the global model with aggregated updates.
        """
        lr = self.optimizer.param_groups[0]["lr"]
        alpha = self._compute_alpha(lr, self.local_steps_this_round)

        # Compute normalized parameter deltas for each trainable parameter
        normalized_deltas: Dict[str, torch.Tensor] = {}
        for name, param in self.local_model.named_parameters():
            if param.requires_grad:
                global_param = dict(self.global_model.named_parameters())[name]
                normalized_deltas[name] = (global_param.data - param.data) / alpha

        # Aggregate sample counts from all clients to determine total data processed
        total_samples = self.comm.aggregate(
            torch.tensor([self.total_samples], dtype=torch.float32),
            communicate_params=False,
            compute_mean=False,
        ).item()

        if total_samples <= 0:
            print(
                "WARN: No samples processed in this round... possible client failure or aggregation error?"
            )
            return

        # Calculate the proportion of data this client contributed
        data_proportion = self.total_samples / total_samples

        # Scale normalized deltas by the data proportion for weighted aggregation
        for name, delta in normalized_deltas.items():
            delta.mul_(data_proportion)

        # Aggregate normalized deltas across all clients
        aggregated_deltas = self.comm.aggregate(
            msg=normalized_deltas, compute_mean=False
        )

        # Apply the aggregated normalized updates to the global model parameters
        utils.apply_model_delta(self.global_model, aggregated_deltas, scale=lr)

        # Update the local model to match the updated global model
        self.local_model = copy.deepcopy(self.global_model)
