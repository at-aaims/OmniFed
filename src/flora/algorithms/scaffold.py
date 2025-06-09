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

import torch

from src.flora.communicator import Communicator
from src.flora.helper.node_config import NodeConfig
from src.flora.helper.training_params import ScaffoldTrainingParameters


class Scaffold:
    """Implementation of Stochastic Controlled Averaging algorithm or SCAFFOLD"""

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: Communicator,
        total_clients: int,
        train_params: ScaffoldTrainingParameters,
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
        self.local_step = 0
        self.training_samples = 0

        dev_id = NodeConfig().get_gpus() % self.total_clients
        self.device = torch.device(
            "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.server_control_variates = {
            name: torch.zeros_like(param.data)
            for name, param in self.model.named_parameters()
        }
        self.client_control_variates = {
            name: torch.zeros_like(param.data)
            for name, param in self.model.named_parameters()
        }
        self.old_client_control_variates = copy.deepcopy(self.client_control_variates)
        self.model_delta = {
            name: torch.zeros_like(param.data)
            for name, param in self.model.named_parameters()
        }
        self.control_variate_delta = {
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
            pred = self.model(inputs)
            loss = self.loss(pred, labels)
            self.training_samples += inputs.size(0)
            loss.backward()
            # update gradients to account for client drift
            for name, param in self.model.named_parameters():
                param.grad.add_(
                    self.server_control_variates[name]
                    - self.client_control_variates[name]
                )

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.local_step += 1
            lr = self.optimizer.param_groups[0]["lr"]

            # update client-local control variate
            for (name1, param1), (name2, param2) in zip(
                self.global_model.named_parameters(), self.model.named_parameters()
            ):
                assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
                # update old state of client control-variate
                self.old_client_control_variates[name1].copy_(
                    self.client_control_variates[name1]
                )
                self.client_control_variates[name1].copy_(
                    self.client_control_variates[name1]
                    - self.server_control_variates[name1]
                    + (param2 - param1) / (self.comm_freq * lr)
                )

            if self.local_step % self.comm_freq == 0:
                # compute client specific model-delta and control variate delta
                for (name1, param1), (name2, param2) in zip(
                    self.global_model.named_parameters(), self.model.named_parameters()
                ):
                    assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
                    self.model_delta[name1].copy_(param2.data - param1.data)
                    self.control_variate_delta[name1].copy_(
                        self.client_control_variates[name1]
                        - self.old_client_control_variates[name1]
                    )

                # average model-deltas and control-variate deltas across clients
                avg_model_delta = self.communicator.aggregate(
                    msg=self.model_delta, compute_mean=True
                )
                avg_control_variate_delta = self.communicator.aggregate(
                    msg=self.control_variate_delta, compute_mean=True
                )

                # update the global model and server control variates
                for name, param in self.global_model.named_parameters():
                    param.add_(lr * avg_model_delta[name])
                    self.server_control_variates[name].add_(
                        avg_control_variate_delta[name]
                    )

                del avg_model_delta, avg_control_variate_delta

    def train(self):
        self.model = self.broadcast_model(model=self.model)
        if self.epochs is not None and isinstance(self.epochs, int) and self.epochs > 0:
            for epoch in range(self.epochs):
                self.train_loop()
        else:
            while True:
                self.train_loop()
