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

# from src.flora.compression.quantization_noEF import QSGDCompression
from src.flora.compression.quantization import QSGDCompressionDebug
from src.flora.communicator.torch_mpi import TorchMPICommunicator
from src.flora.helper.node_config import NodeConfig
from src.flora.helper.training_params import FedAvgTrainingParameters
from src.flora.helper.training_stats import (
    AverageMeter,
    train_img_accuracy,
    test_img_accuracy,
)

nanosec_to_millisec = 1e6


class QuantizedBSPTraining:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        test_data: torch.utils.data.DataLoader,
        communicator: TorchMPICommunicator,
        total_clients: int,
        train_params: FedAvgTrainingParameters,
        # compression: QSGDCompression,
        compression: QSGDCompressionDebug,
    ):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.communicator = communicator
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

        dev_id = NodeConfig().get_gpus() % self.total_clients
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

            # loss scaling in AMP training
            if self.compression.__class__.__name__ == "AMPCompression":
                loss = self.compression.loss_scaling(loss=loss)

            loss.backward()
            compute_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

            quantized_grads = {}
            compression_time = 0.0
            compress_sync_time = 0.0
            # if self.compression.__class__.__name__ == "QSGDCompression":
            #     # perform gradient quantization here
            #     with torch.no_grad():
            #         for name, param in self.model.named_parameters():
            #             init_time = perf_counter_ns()
            #             quant_min, quant_max = self.compression.get_compress_minmax(
            #                 tensor=param.grad
            #             )
            #             compression_time += (
            #                 perf_counter_ns() - init_time
            #             ) / nanosec_to_millisec
            #             sync_init = perf_counter_ns()
            #             quant_min, quant_max = self.communicator.quantized_minmaxval(
            #                 min_val=quant_min, max_val=quant_max
            #             )
            #             compress_sync_time += (
            #                 perf_counter_ns() - sync_init
            #             ) / nanosec_to_millisec
            #             init_time = perf_counter_ns()
            #             quantized_grads[name], _ = self.compression.compress(
            #                 tensor=param.grad, name=name, min_val=quant_min, max_val=quant_max
            #             )
            #             compression_time += (
            #                 perf_counter_ns() - init_time
            #             ) / nanosec_to_millisec
            #
            #     sync_init = perf_counter_ns()
            #     quantized_grads = self.communicator.quantized_aggregate(
            #         quantized_dict=quantized_grads, compute_mean=True
            #     )
            #     compress_sync_time += (
            #         perf_counter_ns() - sync_init
            #     ) / nanosec_to_millisec
            #
            #     init_time = perf_counter_ns()
            #     for (name, param), (q_name, quantized_grad) in zip(
            #         self.model.named_parameters(), quantized_grads.items()
            #     ):
            #         assert name == q_name, f"Parameter mismatch: {name} vs {q_name}"
            #
            #         ctx = quant_min, quant_max, quantized_grad.shape
            #         param.grad = self.compression.decompress(tensor=quantized_grad, ctx=ctx)
            #
            #     compression_time += (
            #         perf_counter_ns() - init_time
            #     ) / nanosec_to_millisec

            if self.compression.__class__.__name__ == "QSGDCompression":
                # Perform layer-wise gradient quantization
                with torch.no_grad():
                    quantized_grads = {}
                    param_contexts = {}

                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            init_time = perf_counter_ns()

                            # Get min/max for THIS specific parameter tensor
                            quant_min, quant_max = self.compression.get_compress_minmax(param.grad)

                            compression_time += (perf_counter_ns() - init_time) / nanosec_to_millisec

                            # OPTION 1: Per-parameter quantization (recommended)
                            # Compress with parameter-specific min/max
                            quantized_grads[name], param_contexts[name] = self.compression.compress(
                                tensor=param.grad, name=name, min_val=quant_min, max_val=quant_max
                            )

                            # OPTION 2: If you must use global min/max, collect all first:
                            # Store min/max for later global synchronization
                            # local_mins[name] = quant_min
                            # local_maxs[name] = quant_max

                    # Aggregate quantized gradients
                    sync_init = perf_counter_ns()

                    # For per-parameter approach
                    quantized_grads = self.communicator.quantized_aggregate(
                        quantized_dict=quantized_grads, compute_mean=True
                    )

                    compress_sync_time += (perf_counter_ns() - sync_init) / nanosec_to_millisec

                    # Decompress gradients
                    init_time = perf_counter_ns()
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and name in quantized_grads:
                            # Use the stored context for this parameter
                            param.grad = self.compression.decompress(
                                tensor=quantized_grads[name],
                                ctx=param_contexts[name]
                            )

                    compression_time += (perf_counter_ns() - init_time) / nanosec_to_millisec

            elif self.compression.__class__.__name__ == "AMPCompression":
                with torch.no_grad():
                    init_time = perf_counter_ns()
                    for name, param in self.model.named_parameters():
                        quantized_grads[name], _ = self.compression.compress(
                            tensor=param.grad, name=name)

                    compression_time += (
                        perf_counter_ns() - init_time
                    ) / nanosec_to_millisec

                init_time = perf_counter_ns()
                quantized_grads = self.communicator.quantized_aggregate(
                    quantized_dict=quantized_grads, compute_mean=True
                )
                compress_sync_time += (
                    perf_counter_ns() - init_time
                ) / nanosec_to_millisec

                init_time = perf_counter_ns()
                for (name, param), (q_name, quantized_grad) in zip(
                    self.model.named_parameters(), quantized_grads.items()
                ):
                    assert name == q_name, f"Parameter mismatch: {name} vs {q_name}"
                    param.grad = self.compression.decompress(tensor=quantized_grad, ctx=quantized_grad.shape)

                compression_time += (
                    perf_counter_ns() - init_time
                ) / nanosec_to_millisec

                self.model = self.compression.gradient_unscaling(model=self.model)
                logging.info(f"model did gradient unscaling!!!")

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.local_step += 1
            itr_time = (perf_counter_ns() - itr_strt) / nanosec_to_millisec
            logging.info(
                f"training_metrics local_step: {self.local_step} epoch {epoch} compute_time {compute_time} ms "
                f"compress_time: {compression_time} ms compress_sync_time: {compress_sync_time} ms "
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
