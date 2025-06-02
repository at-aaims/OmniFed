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

from src.flora.algorithms import BaseClient, BaseServer
from src.flora.communicator import Communicator
from src.flora.helper.node_config import NodeConfig
from src.flora.helper.training_params import TrainingParameters


class FederatedAveragingServer(BaseServer):
    """implementation of Synchronous Federated Averaging"""

    def __init__(
        self,
        model: torch.nn.Module,
        data: torch.utils.data.DataLoader,
        communicator: Communicator,
        id: int,
        total_clients: int,
    ):
        super().__init__(model, data, communicator, id, total_clients)

    def broadcast_model(self):
        # model broadcasted from central server with id 0
        self.model = self.communicator.broadcast(msg=self.model, id=self.id)

    def aggregate_updates(self):
        self.model = self.communicator.aggregate(
            msg=self.model, communicate_params=True
        )


class FederatedAveragingClient(BaseClient):
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: Communicator,
        id: int,
        total_clients: int,
        train_params: TrainingParameters,
    ):
        """
        :param optimizer: optimizer used for training
        :param comm_freq: communication frequency w.r.t. training steps/iterations in fedavg when updates are aggregated
        """
        super().__init__(model, train_data, communicator, id, total_clients)
        self.train_params = train_params
        self.optimizer = self.train_params.optimizer
        self.comm_freq = self.train_params.comm_freq
        self.loss = self.train_params.loss
        self.epochs = self.train_params.epochs
        self.local_step = 0
        self.training_samples = 0
        dev_id = NodeConfig().get_gpus() % total_clients
        self.device = torch.device(
            "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def broadcast_model(self):
        # model broadcasted from central server with id 0
        self.model = self.communicator.broadcast(msg=self.model, id=self.id)

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
                local_samples = self.training_samples
                # collect total samples processed across all clients
                total_samples = self.communicator.aggregate(
                    torch.Tensor([self.training_samples])
                )
                weight_scaling = local_samples / total_samples.item()
                for _, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue
                    param.data *= weight_scaling

                self.model = self.communicator.aggregate(self.model)
                self.training_samples = 0

    def train_model(self):
        if self.epochs is not None and isinstance(self.epochs, int) and self.epochs > 0:
            for epoch in range(self.epochs):
                self.train_loop()
        else:
            while True:
                self.train_loop()
