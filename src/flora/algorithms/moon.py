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
from src.flora.helper.training_params import MOONTrainingParameters


class MoonWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.proj_head = torch.nn.Identity()

    def forward(self, input):
        features = self.base_model.features(input)
        logits = self.base_model.classifier(features)
        representation = self.proj_head(features)

        return logits, representation


class Moon:
    """Implementation of Model-Contrastive Federated Learning or MOON"""

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: Communicator,
        total_clients: int,
        train_params: MOONTrainingParameters,
    ):
        """
        :param model: model to train
        :param data: data to train
        :param communicator: communicator object
        :param total_clients: total number of clients / world size
        :param train_params: training hyperparameters
        """
        self.model = MoonWrapper(model)
        self.train_data = train_data
        self.communicator = communicator
        self.total_clients = total_clients
        self.train_params = train_params
        self.optimizer = self.train_params.get_optimizer()
        self.comm_freq = self.train_params.get_comm_freq()
        self.loss = self.train_params.get_loss()
        self.epochs = self.train_params.get_epochs()
        self.num_prev_models = self.train_params.get_num_prev_models()
        self.temperature = self.train_params.get_temperature()
        self.mu = self.train_params.get_mu()
        # history of previous models tracked for contrastive loss calculation
        self.prev_models = []
        self.local_step = 0
        self.training_samples = 0
        dev_id = NodeConfig().get_gpus() % self.total_clients
        self.device = torch.device(
            "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.global_model = self.global_model.to(self.device)

    def broadcast_model(self, model):
        # broadcast model from central server with id 0
        model = self.communicator.broadcast(msg=model, id=0)
        return model

    def train_loop(self):
        for inputs, labels in self.train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # local model prediction and representation
            pred, local_repr = self.model(inputs)
            self.training_samples += inputs.size(0)
            with torch.no_grad():
                _, global_repr = self.global_model(inputs)
                if len(self.prev_models) > 0:
                    negative_reprs = [
                        prev_model(inputs)[1] for prev_model in self.prev_models
                    ]

            loss = self.loss(pred, labels)
            if len(negative_reprs) > 0:
                local_repr = torch.nn.functional.normalize(local_repr, dim=1)
                global_repr = torch.nn.functional.normalize(global_repr, dim=1)
                negative_reprs = [
                    torch.nn.functional.normalize(repr, dim=1)
                    for repr in negative_reprs
                ]
                pos_sim = torch.exp(
                    torch.sum(local_repr * global_repr, dim=1) / self.temperature
                )
                neg_sim = torch.stack(
                    [
                        torch.exp(torch.sum(local_repr * neg, dim=1) / self.temperature)
                        for neg in negative_reprs
                    ],
                    dim=1,
                ).sum(dim=1)

                contrastive_loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
                loss += self.mu * contrastive_loss

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
                for _, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue
                    param.data *= weight_scaling

                self.global_model = self.communicator.aggregate(
                    msg=self.model, communicate_params=True, compute_mean=False
                )
                self.model.load_state_dict(self.global_model.state_dict())
                self.training_samples = 0

                model_copy = copy.deepcopy(self.global_model)
                model_copy.eval()
                if len(self.prev_models) == self.num_prev_models:
                    self.prev_models.pop()
                self.prev_models.insert(0, model_copy)

    def train(self):
        self.model.train()
        self.global_model.eval()
        self.model = self.broadcast_model(model=self.model)
        if self.epochs is not None and isinstance(self.epochs, int) and self.epochs > 0:
            for epoch in range(self.epochs):
                self.train_loop()
        else:
            while True:
                self.train_loop()
