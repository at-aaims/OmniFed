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

import logging
from time import perf_counter_ns
import torch

from src.flora.compression.lora_approximation import PowerSGDCompression
from src.flora.communicator.torch_mpi import TorchMPICommunicator
from src.flora.helper.training_params import FedAvgTrainingParameters
from src.flora.helper.training_stats import (
    AverageMeter,
    train_img_accuracy,
    test_img_accuracy,
)

nanosec_to_millisec = 1e6


class PowerSGDCompressTrain:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: torch.utils.data.DataLoader,
            test_data: torch.utils.data.DataLoader,
            communicator: TorchMPICommunicator,
            client_id: int,
            total_clients: int,
            train_params: FedAvgTrainingParameters,
            compression: PowerSGDCompression,
    ):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.communicator = communicator
        self.client_id = client_id
        self.total_clients = total_clients
        self.train_params = train_params
        self.compression = compression
        self.optimizer = self.train_params.get_optimizer()
        self.comm_freq = self.train_params.get_comm_freq()
        self.loss = self.train_params.get_loss()
        self.lr_scheduler = self.train_params.get_lr_scheduler()
        self.epochs = self.train_params.get_epochs()
        self.local_step = 0
        self.training_samples = 0

        dev_id = self.client_id % 4
        self.device = torch.device(
            "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
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
            pred = self.model(inputs)
            loss = self.loss(pred, labels)
            self.training_samples += inputs.size(0)
            loss.backward()
            compute_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

            for name, param in self.model.named_parameters():
                # Store original gradient for error feedback
                original_grad = param.grad.clone()
                param_P, param_Q, param_og_shape = self.compression.compress(tensor=param, param_name=name)

                param_P = self.communicator.aggregate(msg=param_P,
                                                      communicate_params=False,
                                                      compute_mean=True)
                param_Q = self.communicator.aggregate(msg=param_Q,
                                                      communicate_params=False,
                                                      compute_mean=True)
                self.compression._update_Q(param_Q=param_Q, param_name=name)

                decompressed_grad = self.compression.decompress(P=param_P,
                                                                Q=param_Q,
                                                                original_shape=param_og_shape)
                self.compression.update_error_feedback(original_grad=original_grad,
                                                       compressed_grad=decompressed_grad,
                                                       param_name=name)

