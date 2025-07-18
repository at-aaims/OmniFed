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

from src.flora.compression import Compression
from src.flora.communicator.torch_mpi import TorchMPICommunicator
from src.flora.helper.training_params import FedAvgTrainingParameters
from src.flora.privacy.homomorphic_encryption import HomomorphicEncryption
from src.flora.helper.training_stats import (
    AverageMeter,
    train_img_accuracy,
    test_img_accuracy,
)

nanosec_to_millisec = 1e6

class HomomorphicEncryptionBSP:
    def __init__(
            self,
            client_id: int,
            model: torch.nn.Module,
            train_data: torch.utils.data.DataLoader,
            test_data: torch.utils.data.DataLoader,
            communicator: TorchMPICommunicator,
            total_clients: int,
            train_params: FedAvgTrainingParameters
    ):
        self.model = model
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
        self.local_step = 0
        self.training_samples = 0
        dev_id = client_id % 4
        self.device = torch.device("cuda:" + str(dev_id)) if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)
        self.train_loss = AverageMeter()
        self.top1_acc, self.top5_acc, self.top10_acc = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        self.he_object = HomomorphicEncryption()
        self.encrypt_grads = True

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
            init_time = perf_counter_ns()
            with torch.no_grad():
                encrypted_updates = self.he_object.encrypt(model=self.model, encrypt_grads=self.encrypt_grads)

            he_encryption_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

            init_time = perf_counter_ns()
            encrypted_updates = self.communicator.encrypted_aggregation(encrypted_dict=encrypted_updates,
                                                                        compute_mean=True)
            encrypted_sync_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

            for (name1, param1), (key, avg_value) in zip(self.model.named_parameters(), encrypted_updates.items()):
                if self.encrypt_grads:
                    param1.grad = avg_value
                else:
                    param1.data = avg_value

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.local_step += 1
            itr_time = (perf_counter_ns() - itr_strt) / nanosec_to_millisec
            logging.info(
                f"training_metrics local_step: {self.local_step} epoch {epoch} compute_time {compute_time} ms "
                f"he_encryption_time: {he_encryption_time} ms encrypted_sync_time: {encrypted_sync_time} ms "
                f"itr_time: {itr_time} ms"
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
        test_img_accuracy(
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
        print("going to broadcast model across clients...")
        self.model = self.broadcast_model(model=self.model)
        if self.epochs is not None and isinstance(self.epochs, int) and self.epochs > 0:
            for epoch in range(self.epochs):
                print("going to start epoch {}/{}".format(epoch, self.epochs))
                self.train_loop(epoch=epoch)
        else:
            i = 0
            while True:
                self.train_loop(epoch=i)
                i += 1
