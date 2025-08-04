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
import logging
from time import perf_counter_ns

import torch

from src.flora.communicator import Communicator
from src.flora.helper.node_config import NodeConfig
from src.flora.helper.training_stats import topK_accuracy
from src.flora.helper.training_params import MOONTrainingParameters
from src.flora.helper.training_stats import (
    AverageMeter,
    train_img_accuracy,
    test_img_accuracy,
)

nanosec_to_millisec = 1e6

# class MoonWrapper(torch.nn.Module):
#     def __init__(self, base_model):
#         super().__init__()
#         self.base_model = base_model
#         self.proj_head = torch.nn.Identity()
#
#     def forward(self, input):
#         features = self.base_model.features(input)
#         logits = self.base_model.classifier(features)
#         representation = self.proj_head(features)
#
#         return logits, representation

# ResNet18
class MoonWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.proj_head = torch.nn.Identity()

    def extract_features(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, input):
        features = self.extract_features(input)
        logits = self.base_model.fc(features)
        representation = self.proj_head(features)
        return logits, representation

# VGG11
# class MoonWrapper(torch.nn.Module):
#     def __init__(self, base_model: torch.nn.Module, num_classes: int = 100, projection_dim: int = 128):
#         super().__init__()
#
#         self.features = base_model.features
#         self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = torch.nn.Flatten()
#
#         # Rebuild classifier for CIFAR-100
#         self.classifier = torch.nn.Sequential(
#             torch.nn.Linear(512, 4096),
#             torch.nn.ReLU(True),
#             torch.nn.Dropout(),
#             torch.nn.Linear(4096, 4096),
#             torch.nn.ReLU(True),
#             torch.nn.Dropout(),
#             torch.nn.Linear(4096, num_classes)
#         )
#
#         # Projection head for contrastive representation
#         self.proj_head = torch.nn.Linear(512, projection_dim)
#
#     def extract_features(self, x):
#         x = self.features(x)     # -> [B, 512, 1, 1] for CIFAR-100 (32x32)
#         x = self.avgpool(x)
#         x = self.flatten(x)      # -> [B, 512]
#         return x
#
#     def forward(self, x):
#         features = self.extract_features(x)           # -> [B, 512]
#         logits = self.classifier(features)            # -> [B, 100]
#         representation = self.proj_head(features)     # -> [B, projection_dim]
#         return logits, representation


# for AlexNet
# class MoonWrapper(torch.nn.Module):
#     def __init__(
#         self,
#         base_model: torch.nn.Module,
#         num_classes: int = 102,
#         projection_dim: int = 128,
#     ):
#         super().__init__()
#
#         self.features = base_model.features  # Conv layers
#         self.avgpool = base_model.avgpool  # AdaptiveAvgPool2d((6, 6)) by default
#         self.flatten = torch.nn.Flatten()
#
#         # Modify classifier for Caltech-101 (102 classes)
#         self.classifier = torch.nn.Sequential(
#             torch.nn.Dropout(),
#             torch.nn.Linear(256 * 6 * 6, 4096),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Dropout(),
#             torch.nn.Linear(4096, 4096),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Linear(4096, num_classes),
#         )
#
#         # Projection head for contrastive representation (input = 256 * 6 * 6)
#         self.proj_head = torch.nn.Linear(256 * 6 * 6, projection_dim)
#
#     def extract_features(self, x):
#         x = self.features(x)  # [B, 256, 6, 6] for 224×224 inputs
#         x = self.avgpool(x)  # Still [B, 256, 6, 6]
#         x = self.flatten(x)  # [B, 9216]
#         return x
#
#     def forward(self, x):
#         features = self.extract_features(x)  # [B, 9216]
#         logits = self.classifier(features)  # [B, 102]
#         representation = self.proj_head(features)  # [B, projection_dim]
#         return logits, representation


# MobileNet-V3
# class MoonWrapper(torch.nn.Module):
#     def __init__(self, base_model: torch.nn.Module, projection_dim: int = 128):
#         super().__init__()
#
#         self.features = base_model.features  # feature extractor
#         self.avgpool = base_model.avgpool  # AdaptiveAvgPool2d((1, 1))
#         self.classifier = base_model.classifier  # already modified externally
#
#         # Flatten to match output after avgpool
#         self.flatten = torch.nn.Flatten()
#
#         # Infer feature dimension (e.g., 960) from classifier[0]
#         feature_dim = base_model.classifier[0].in_features
#
#         # Projection head for MOON
#         self.proj_head = torch.nn.Linear(feature_dim, projection_dim)
#
#     def extract_features(self, x):
#         x = self.features(x)  # [B, 960, H, W]
#         x = self.avgpool(x)  # [B, 960, 1, 1]
#         x = self.flatten(x)  # [B, 960]
#         return x
#
#     def forward(self, x):
#         features = self.extract_features(x)  # [B, 960]
#         logits = self.classifier(features)  # [B, 257]
#         representation = self.proj_head(features)  # [B, projection_dim]
#         return logits, representation


class Moon:
    """Implementation of Model-Contrastive Federated Learning or MOON"""

    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        test_data: torch.utils.data.DataLoader,
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

        self.model = MoonWrapper(base_model=model)
        # July 18, 2025 for VGG
        # model.classifier[6] = torch.nn.Linear(4096, 10)
        # July 19 2025 for AlexNet
        # self.model = MoonWrapper(base_model=model, num_classes=102)
        # July 20, 2025 MobileNet-v3
        # self.model = MoonWrapper(base_model=model, projection_dim=128)
        # self.model = torchvision.models.resnet18(pretrained=True).l
        self.train_data = train_data
        self.test_data = test_data
        self.communicator = communicator
        self.total_clients = total_clients
        self.train_params = train_params
        self.optimizer = self.train_params.get_optimizer()
        self.comm_freq = self.train_params.get_comm_freq()
        self.loss = self.train_params.get_loss()
        self.lr_scheduler = self.train_params.get_lr_scheduler()
        self.epochs = self.train_params.get_epochs()
        self.num_prev_models = self.train_params.get_num_prev_models()
        self.temperature = self.train_params.get_temperature()
        self.mu = self.train_params.get_mu()
        # history of previous models tracked for contrastive loss calculation
        self.prev_models = []
        self.local_step = 0
        self.training_samples = 0
        # dev_id = NodeConfig().get_gpus() % self.total_clients
        # self.device = torch.device(
        #     "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        # )
        # self.device = torch.device("cuda:" + str(client_id)) if torch.cuda.is_available() else torch.device("cpu")
        dev_id = client_id % 4
        self.device = (
            torch.device("cuda:" + str(dev_id))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model = self.model.to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.global_model = self.global_model.to(self.device)
        self.negative_reprs = []
        self.train_loss = AverageMeter()
        self.top1_acc, self.top5_acc, self.top10_acc = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )

    def broadcast_model(self, model):
        # broadcast model from central server with id 0
        model = self.communicator.broadcast(msg=model, id=0)
        return model

    def train_loop(self, epoch):
        epoch_strt = perf_counter_ns()
        for inputs, labels in self.train_data:
            itr_strt = perf_counter_ns()
            init_time = perf_counter_ns()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # local model prediction and representation
            pred, local_repr = self.model(inputs)
            self.training_samples += inputs.size(0)
            with torch.no_grad():
                _, global_repr = self.global_model(inputs)
                if len(self.prev_models) > 0:
                    self.negative_reprs = [
                        prev_model(inputs)[1] for prev_model in self.prev_models
                    ]

            loss = self.loss(pred, labels)
            if len(self.negative_reprs) > 0:
                local_repr = torch.nn.functional.normalize(local_repr, dim=1)
                global_repr = torch.nn.functional.normalize(global_repr, dim=1)
                self.negative_reprs = [
                    torch.nn.functional.normalize(repr, dim=1)
                    for repr in self.negative_reprs
                ]
                pos_sim = torch.exp(
                    torch.sum(local_repr * global_repr, dim=1) / self.temperature
                )
                neg_sim = torch.stack(
                    [
                        torch.exp(torch.sum(local_repr * neg, dim=1) / self.temperature)
                        for neg in self.negative_reprs
                    ],
                    dim=1,
                ).sum(dim=1)

                contrastive_loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
                loss += self.mu * contrastive_loss.mean()

            loss.backward()
            compute_time = (perf_counter_ns() - init_time) / nanosec_to_millisec
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.local_step += 1
            sync_time = None
            if self.local_step % self.comm_freq == 0:
                init_time = perf_counter_ns()
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
                sync_time = (perf_counter_ns() - init_time) / nanosec_to_millisec
                self.model.load_state_dict(self.global_model.state_dict())
                self.training_samples = 0

                model_copy = copy.deepcopy(self.global_model)
                model_copy.eval()
                if len(self.prev_models) == self.num_prev_models:
                    self.prev_models.pop()
                self.prev_models.insert(0, model_copy)

            itr_time = (perf_counter_ns() - itr_strt) / nanosec_to_millisec
            logging.info(
                f"training_metrics local_step: {self.local_step} epoch {epoch} compute_time {compute_time} ms "
                f"sync_time: {sync_time} ms itr_time: {itr_time} ms"
            )

        epoch_time = (perf_counter_ns() - epoch_strt) / nanosec_to_millisec
        logging.info(f"epoch completion time for epoch {epoch} is {epoch_time} ms")

        train_img_accuracy(
            epoch=epoch,
            iteration=self.local_step,
            input=inputs,
            label=labels,
            output=pred,
            loss=loss,
            train_loss=self.train_loss,
            top1acc=self.top1_acc,
            top5acc=self.top5_acc,
            top10acc=self.top10_acc,
        )
        self.moon_test_img_accuracy(
            epoch=epoch,
            device=self.device,
            model=self.model,
            test_loader=self.test_data,
            loss_fn=self.loss,
            iteration=self.local_step,
        )

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def train(self):
        self.model.train()
        self.global_model.eval()
        self.model = self.broadcast_model(model=self.model)
        if self.epochs is not None and isinstance(self.epochs, int) and self.epochs > 0:
            for epoch in range(self.epochs):
                self.train_loop(epoch=epoch)
        else:
            i = 0
            while True:
                self.train_loop(epoch=i)
                i += 1

    def moon_test_img_accuracy(
        self, epoch, device, model, test_loader, loss_fn, iteration
    ):
        model.eval()
        with torch.no_grad():
            test_loss, top1acc, top5acc, top10acc = (
                AverageMeter(),
                AverageMeter(),
                AverageMeter(),
                AverageMeter(),
            )
            for input, label in test_loader:
                input, label = input.to(device), label.to(device)
                output, _ = model(input)
                loss = loss_fn(output, label)
                topKaccuracy = topK_accuracy(
                    output=output, target=label, topk=(1, 5, 10)
                )
                top1acc.update(topKaccuracy[0], input.size(0))
                top5acc.update(topKaccuracy[1], input.size(0))
                top10acc.update(topKaccuracy[2], input.size(0))
                test_loss.update(loss.item(), input.size(0))

            logging.info(
                f"Logging test_metrics iteration {iteration} epoch {epoch} test_loss {test_loss.avg} top1_acc "
                f"{top1acc.avg.cpu().numpy().item()} top5_acc {top5acc.avg.cpu().numpy().item()} top10_acc "
                f"{top10acc.avg.cpu().numpy().item()}"
            )
            model.train()
