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
from typing import Any

import torch
import torch.nn as nn

from src.flora.communicator import Communicator
from src.flora.helper.node_config import NodeConfig
from src.flora.helper.training_params import DittoTrainingParameters

from . import utils
from .BaseAlgorithm import Algorithm


class Ditto:
    """Implementation of Ditto federated learning for lightweight personalization where clients train a global model
    and custom local models tailored to their private data"""

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: Communicator,
        total_clients: int,
        train_params: DittoTrainingParameters,
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
        self.ditto_regularizer = self.train_params.get_ditto_regularizer()
        self.global_loss = self.train_params.get_global_loss()
        self.global_optimizer = self.train_params.get_global_optimizer()
        self.local_step = 0
        self.training_samples = 0

        dev_id = NodeConfig().get_gpus() % self.total_clients
        self.device = torch.device(
            "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.global_model = self.global_model.to(self.device)
        self.diff_params = {
            name: torch.zeros_like(param.data)
            for name, param in self.model.named_parameters()
        }

    def broadcast_model(self, model):
        # broadcast model from central server with id 0
        model = self.communicator.broadcast(msg=model, id=0)
        return model

    def train_loop(self):
        for inputs, labels in self.train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # calculate loss over global model and update global model
            global_pred = self.global_model(inputs)
            global_loss = self.global_loss(global_pred, labels)
            global_loss.backward()
            self.global_optimizer.step()
            self.global_optimizer.zero_grad()
            self.training_samples += inputs.size(0)

            # calculate loss over local model and update local model
            pred = self.model(inputs)
            loss = self.loss(pred, labels)
            proximal_regularizer = 0.0
            for param1, param2 in zip(
                self.model.parameters(), self.global_model.parameters()
            ):
                proximal_regularizer += torch.sum((param1 - param2.detach()) ** 2)

            loss += 0.5 * self.ditto_regularizer * proximal_regularizer
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.local_step % self.comm_freq == 0:
                # total samples processed across all clients
                total_samples = self.communicator.aggregate(
                    msg=torch.Tensor([self.training_samples]), compute_mean=False
                )
                weight_scaling = self.training_samples / total_samples.item()
                for _, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue
                    param.data *= weight_scaling

                self.global_model = self.communicator.aggregate(
                    msg=self.global_model, communicate_params=True, compute_mean=False
                )
                self.training_samples = 0

    def train(self):
        self.model.train()
        self.global_model.train()
        self.model = self.broadcast_model(model=self.model)
        if self.epochs is not None and isinstance(self.epochs, int) and self.epochs > 0:
            for epoch in range(self.epochs):
                self.train_loop()
        else:
            while True:
                self.train_loop()


# ======================================================================================


class DittoNew(Algorithm):
    """
    Ditto algorithm implementation for personalized federated learning.

    Ditto maintains both a global model (shared across clients) and a local personal model.
    The personal model is trained with a proximal regularization term to stay close to the global model.
    """

    def __init__(
        self,
        local_model: nn.Module,
        comm: Communicator,
        lr: float = 0.01,
        global_lr: float = 0.01,
        ditto_lambda: float = 0.1,
    ):
        super().__init__(local_model, comm)
        self.lr = lr
        self.global_lr = global_lr
        self.ditto_lambda = ditto_lambda

        # ---
        self.global_model = copy.deepcopy(local_model)

        self.global_optimizer = torch.optim.SGD(
            self.global_model.parameters(), lr=self.global_lr
        )

    def configure_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        SGD optimizer for local updates.
        """
        return torch.optim.SGD(model.parameters(), lr=self.lr)

    def train_step(self, batch: Any, batch_idx: int) -> tuple[torch.Tensor, int]:
        """
        Perform dual training: update both global model and personal model.
        """
        inputs, targets = batch
        batch_size = inputs.size(0)

        # Step 1: Update global model with standard loss
        global_outputs = self.global_model(inputs)
        global_loss = nn.functional.cross_entropy(global_outputs, targets)

        self.global_optimizer.zero_grad()
        global_loss.backward()
        self.global_optimizer.step()

        # Step 2: Update personal model with proximal regularization
        personal_outputs = self.local_model(inputs)
        personal_loss = nn.functional.cross_entropy(personal_outputs, targets)

        # Add proximal regularization term
        proximal_reg = 0.0
        for personal_param, global_param in zip(
            self.local_model.parameters(), self.global_model.parameters()
        ):
            proximal_reg += torch.sum((personal_param - global_param.detach()) ** 2)

        total_loss = personal_loss + 0.5 * self.ditto_lambda * proximal_reg

        return total_loss, batch_size

    def round_start(self, round_idx: int) -> None:
        """
        Synchronize the global model at the start of each round.
        """
        self.global_model = self.comm.broadcast(self.global_model, src=0)

    def round_end(self, round_idx: int) -> None:
        """
        Aggregate global models across clients. Personal models remain local.
        """
        # Aggregate sample counts
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

        # Calculate data proportion for weighted aggregation
        data_proportion = self.round_total_samples / total_samples
        print(
            f"Ditto Round {round_idx}: Processed {self.round_total_samples}/{total_samples} samples (weight: {data_proportion:.4f})"
        )

        # Scale global model parameters by data proportion
        utils.scale_params(self.global_model, data_proportion)

        # Aggregate global models (personal models remain local)
        self.global_model = self.comm.aggregate(
            self.global_model,
            communicate_params=True,
            compute_mean=False,
        )
