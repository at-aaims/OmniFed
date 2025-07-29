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

from src.flora.compression.quantization import QSGDQuantCompression
from src.flora.communicator.torch_mpi import TorchMPICommunicator
from src.flora.helper.training_params import FedAvgTrainingParameters
from src.flora.helper.training_stats import (
    AverageMeter,
    train_img_accuracy,
    test_img_accuracy,
)

nanosec_to_millisec = 1e6


class QSGDCompressTraining:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: torch.utils.data.DataLoader,
            test_data: torch.utils.data.DataLoader,
            communicator: TorchMPICommunicator,
            client_id: int,
            total_clients: int,
            train_params: FedAvgTrainingParameters,
            compression: QSGDQuantCompression,
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

            compress_init = perf_counter_ns()
            compress_time, compress_sync_time = None, None

            if self.compression.__class__.__name__ == "QSGDQuantCompression":
                gradients = [p.grad for p in self.model.parameters() if p.requires_grad]

                # Quantize gradients locally
                quantized_data = self.compression.compress(gradients=gradients)
                del gradients
                compress_time = (perf_counter_ns() - compress_init) / nanosec_to_millisec

                sync_init = perf_counter_ns()
                # Method 1: Proper QSGD aggregation (recommended)
                self._aggregate_qsgd_method1(quantized_data)

                # Alternative Method 2: If you want to average quantization parameters
                # self._aggregate_qsgd_method2(quantized_data)

                # compress_sync_time = (perf_counter_ns() - compress_init - compress_time * nanosec_to_millisec) / nanosec_to_millisec
                compress_sync_time = (perf_counter_ns() - sync_init) / nanosec_to_millisec
                del quantized_data

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.local_step += 1
            itr_time = (perf_counter_ns() - itr_strt) / nanosec_to_millisec
            logging.info(
                f"training_metrics local_step: {self.local_step} epoch {epoch} compute_time {compute_time} ms "
                f"compress_time: {compress_time} ms compress_sync_time: {compress_sync_time} ms "
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

    def _aggregate_qsgd_method1(self, quantized_data):
        """
        Proper QSGD aggregation: dequantize locally, aggregate, then update gradients
        This is the mathematically correct approach for QSGD.
        """
        for (i, param) in enumerate(self.model.parameters()):
            if param.grad is not None and quantized_data['tensors'][i] is not None:
                # Dequantize locally first
                q_tensor = quantized_data['tensors'][i].float()
                scale = quantized_data['scales'][i]
                zero_point = quantized_data['zero_points'][i]

                # Dequantize: scale * (quantized - zero_point)
                dequantized = scale * (q_tensor - zero_point)
                dequantized = dequantized.reshape(param.grad.shape)

                # Now aggregate the dequantized gradients
                param.grad.data = dequantized
                param.grad = self.communicator.aggregate(
                    msg=param.grad,
                    communicate_params=False,
                    compute_mean=True
                )

    def _aggregate_qsgd_method2(self, quantized_data):
        """
        Alternative method: Aggregate quantized values and parameters separately
        This requires careful handling of quantization parameters.
        """
        for (i, param) in enumerate(self.model.parameters()):
            if param.grad is not None and quantized_data['tensors'][i] is not None:
                q_tensor = quantized_data['tensors'][i]
                scale = quantized_data['scales'][i]
                zero_point = quantized_data['zero_points'][i]

                # Use float32 to avoid overflow issues
                q_tensor_float = q_tensor.float()

                # Create tensors for aggregation
                scale_tensor = torch.tensor(scale, device=self.device, dtype=torch.float32)
                zero_point_tensor = torch.tensor(zero_point, device=self.device, dtype=torch.float32)

                # Aggregate quantized values (sum, not average yet)
                self.communicator.quantized_aggregate(
                    msg=q_tensor_float,
                    communicate_params=False,
                    compute_mean=False
                )

                # Aggregate scales and zero points (sum, not average yet)
                self.communicator.quantized_aggregate(
                    msg=scale_tensor,
                    communicate_params=False,
                    compute_mean=False
                )
                self.communicator.quantized_aggregate(
                    msg=zero_point_tensor,
                    communicate_params=False,
                    compute_mean=False
                )

                # Average the aggregated values
                q_tensor_avg = q_tensor_float / self.total_clients
                scale_avg = scale_tensor / self.total_clients
                zero_point_avg = zero_point_tensor / self.total_clients

                # Dequantize using averaged parameters
                dequantized = scale_avg * (q_tensor_avg - zero_point_avg)
                param.grad = dequantized.reshape(param.grad.shape)

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