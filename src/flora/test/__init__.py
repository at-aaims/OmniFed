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


def get_model(model_name, determinism, args):
    if model_name == "resnet18":
        model_obj = ResNet18Object(
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            seed=args.seed,
            gamma=args.gamma,
            determinism=determinism,
        )
        return model_obj
    elif model_name == "alexnet":
        model_obj = AlexNetObject(
            lr=args.lr,
            momentum=args.momentum,
            weightdecay=args.weight_decay,
            seed=args.seed,
            determinism=determinism,
            gamma=args.gamma,
        )
        return model_obj
    elif model_name == "vgg11":
        model_obj = VGG11Object(
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            seed=args.seed,
            determinism=determinism,
            gamma=args.gamma,
        )
        return model_obj
    elif model_name == "mobilenetv3":
        model_obj = MobileNetV3Object(
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            seed=args.seed,
            determinism=determinism,
            gamma=args.gamma,
            num_classes=args.mobv3_num_classes,
            lr_step_size=args.mobv3_lr_step_size,
        )
        return model_obj

    return None


class ResNet18Object(object):
    def __init__(self, lr, momentum, weight_decay, seed, gamma, determinism):
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

    def get_lr(self):
        return self.lr

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.optim

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        return self.lr_scheduler


class AlexNetObject(object):
    def __init__(self, lr, gamma, seed, momentum, weightdecay, determinism):
        helper.set_seed(seed, determinism)
        self.lr = lr
        self.gamma = gamma
        self.momentum = momentum
        self.weightdecay = weightdecay
        self.loss = torch.nn.CrossEntropyLoss()
        self.model = torchvision.models.alexnet(progress=True, pretrained=False)
        self.opt = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weightdecay,
        )

        milestones = [25, 50, 75]
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.opt, milestones=milestones, gamma=self.gamma, last_epoch=-1
        )

    def get_lr(self):
        return self.lr

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.opt

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        return self.lr_scheduler


class VGG11Object(object):
    def __init__(self, lr, momentum, seed, weight_decay, gamma, determinism):
        helper.set_seed(seed, determinism)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.loss = torch.nn.CrossEntropyLoss()
        self.model = torchvision.models.vgg11(pretrained=False, progress=True)
        self.optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optim, milestones=[50, 75, 100], gamma=self.gamma, last_epoch=-1
        )

    def get_lr(self):
        return self.lr

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.optim

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        return self.lr_scheduler


class MobileNetV3Object(object):
    def __init__(
        self,
        lr,
        num_classes,
        seed,
        momentum,
        weight_decay,
        gamma,
        lr_step_size,
        determinism,
    ):
        helper.set_seed(seed, determinism)
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.model = torchvision.models.mobilenet_v3_large(
            progress=True, pretrained=False
        )
        # self.model.classifier[3] = torch.nn.Linear(
        #     self.model.classifier[3].in_features, num_classes
        # )
        # July 20, 2025
        self.model.classifier[3] = torch.nn.Linear(
            self.model.classifier[3].in_features, 1000
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=self.lr_step_size, gamma=self.gamma
        )

    def get_lr(self):
        return self.lr

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.optim

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        return self.lr_scheduler
