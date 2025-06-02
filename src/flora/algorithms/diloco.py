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
import copy

from src.flora.algorithms import BaseServer, BaseClient
from src.flora.communicator import Communicator
from src.flora.helper.training_params import TrainingParameters
from src.flora.helper.node_config import NodeConfig


class DiLocoServer(BaseServer):
    def __init__(self, model: torch.nn.Module, data: torch.utils.data.DataLoader, communicator: Communicator,
                 id: int, total_clients: int, train_params: TrainingParameters):
        super().__init__(model, data, communicator, id, total_clients)
        self.train_params = train_params
        dev_id = NodeConfig().get_gpus() % total_clients
        self.device = torch.device("cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.diff_params = copy.deepcopy(self.model)
        self.global_model, self.diff_params = self.global_model.to(self.device), self.diff_params.to(self.device)
        self.outer_lr = self.train_params.outer_lr
        self.outer_momentum = self.train_params.outer_momentum
        self.velocity = {name: torch.zeros_like(param.data) for name, param in self.model.named_parameters()}

    def initialize_model(self):
        # model broadcasted from central server with id 0
        self.model = self.communicator.broadcast(msg=self.model, id=self.id)

    def aggregate_updates(self):
        averaged_model = self.communicator.aggregate(msg=self.model, communicate_params=True, compute_mean=True)
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(self.global_model.parameters(), averaged_model.parameters()):
                assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
                target_param = dict(self.diff_params.named_parameters())[name1]
                target_param.copy_((param1 / self.total_clients) - param2)

        del averaged_model
        # outer optimization
        self.global_model = self._outer_step()

    def _zero_velocity(self):
        for v in self.velocity.values():
            v.zero_()

    def _outer_step(self):
        with torch.no_grad():
            for (name, param), (_, param_delta) in zip(self.global_model.named_parameters(), self.diff_params.named_parameters()):
                v = self.velocity[name]
                # Momentum update rule
                v.mul_(self.outer_momentum).add_(param_delta.data, alpha=self.outer_lr)
                param.data.add_(v)  # Update the model param

        return self.global_model


class DiLocoClient(BaseClient):
    def __init__(self, model: torch.nn.Module, train_data: torch.utils.data.DataLoader, communicator: Communicator,
                 id: int, total_clients: int, train_params: TrainingParameters):
        super().__init__(model, train_data, communicator, id, total_clients)
        self.train_params = train_params
        # AdamW optimizer for LLM training in DiLoCo
        self.optimizer = self.train_params.optimizer
        self.comm_freq = self.train_params.comm_freq
        self.loss = self.train_params.loss
        self.epochs = self.train_params.epochs
        self.local_step = 0
        self.training_samples = 0
        dev_id = NodeConfig().get_gpus() % total_clients
        self.device = torch.device("cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def initialize_model(self):
        # model broadcasted from central server with id 0
        self.model = self.communicator.broadcast(msg=self.model, id=self.id)

    # Implement aggregate updates here!

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


    def train_model(self):
        if self.epochs is not None and isinstance(self.epochs, int) and self.epochs > 0:
            for epoch in range(self.epochs):
                self.train_loop()
        else:
            while True:
                self.train_loop()
