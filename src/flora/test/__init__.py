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

# module for running preliminary stuff

import torch
import torchvision

import src.flora.helper as helper


def get_model(model_name, determinism, args, device=torch.device("cpu")):
    if model_name == "resnet18":
        model_obj = ResNet18Object(
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            seed=args.seed,
            gamma=args.gamma,
            determinism=determinism,
            device=device,
        )
        return model_obj

    return None


class ResNet18Object(object):
    def __init__(
        self,
        lr,
        momentum,
        weight_decay,
        seed,
        gamma,
        determinism,
        device=torch.device("cpu"),
        use_lars=False,
    ):
        helper.set_seed(seed, determinism)
        self.lr = lr
        self.momentum = momentum
        self.weightdecay = weight_decay
        self.gamma = gamma
        self.loss = torch.nn.CrossEntropyLoss()
        self.model = torchvision.models.resnet18(progress=True, pretrained=False)
        self.optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weightdecay,
        )
        milestones = [100, 150, 200]
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optim, milestones=milestones, gamma=self.gamma, last_epoch=-1
        )

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.optim

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        return self.lr_scheduler
