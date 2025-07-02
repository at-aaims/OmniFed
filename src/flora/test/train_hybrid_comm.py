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
import socket
from time import perf_counter_ns

import torch

from src.flora.test import get_model
from src.flora.communicator import torch_mpi
from src.flora.communicator import grpc_communicator as rpc_comm
from src.flora.datasets.image_classification import cifar

nanosec_to_millisec = 1e6

class HybridTrainer(object):
    def __init__(self, args):
        self.args = args
        self.train_bsz = args.bsz
        self.test_bsz = args.test_bsz
        self.logdir = args.dir
        self.model_name = args.model
        self.global_rank = args.global_rank
        self.local_rank = args.local_rank
        self.backend = args.backend
        self.epochs = args.epochs
        self.comm_freq = args.comm_freq

        self.rpc_server_addr = '127.0.0.1'
        self.rpc_server_port = 50051
        self.mpi1_addr = '127.0.0.1'
        self.mpi1_port = 28250
        self.mpi2_addr = '127.0.0.1'
        self.mpi2_port = 28290

        if torch.cuda.is_available():
            self.device = torch.device('cuda:' + str(self.global_rank))
        else:
            self.device = torch.device('cpu')

        logging.basicConfig(
            filename=self.logdir
            + "/g"
            + str(self.global_rank)
            + "/"
            + self.model_name
            + "-"
            + str(self.global_rank)
            + ".log",
            level=logging.INFO,
        )
        self.dataset_name = args.dataset
        self.determinism = False

        self.model_obj = get_model(
            self.model_name, determinism=self.determinism, args=args
        )
        if self.model_obj is None:
            raise RuntimeError(f"invalid model or model_name {self.model_name}")

        logging.info("initialized model object...")
        self.model = self.model_obj.get_model()
        self.loss_fn = self.model_obj.get_loss()
        self.optimizer = self.model_obj.get_optim()
        self.lr_scheduler = self.model_obj.get_lrscheduler()
        self.local_step = 0
        self.training_samples = 0
        # gRPC consists of 1 server and 2 clients, so total_clients=3
        if self.global_rank == 0:
            self.mpi_comm = None
            self.rpc_comm = rpc_comm.GrpcCommunicator(model=self.model, id=self.global_rank,
                                                      total_clients=3,
                                                      master_addr=self.rpc_server_addr,
                                                      master_port=self.rpc_server_port,
                                                      accumulate_updates=True
                                                      )
            logging.info("started gRPC server on global_rank 0...")

        elif self.global_rank == 1:
            self.mpi_comm = torch_mpi.TorchMPICommunicator(id=self.local_rank,
                                                           total_clients=3,
                                                           backend=self.backend,
                                                           master_addr=self.mpi1_addr,
                                                           master_port=self.mpi1_port
                                                           )
            logging.info("started MPI process on global_rank 1...")
            self.rpc_comm = rpc_comm.GrpcCommunicator(model=self.model, id=self.global_rank,
                                                      total_clients=3,
                                                      master_addr=self.rpc_server_addr,
                                                      master_port=self.rpc_server_port,
                                                      accumulate_updates=True
                                                      )
            logging.info("started gRPC client on global_rank 1...")
        elif self.global_rank == 2:
            self.mpi_comm = torch_mpi.TorchMPICommunicator(id=self.local_rank,
                                                           total_clients=3,
                                                           backend=self.backend,
                                                           master_addr=self.mpi2_addr,
                                                           master_port=self.mpi2_port
                                                           )
            logging.info("started MPI process on global_rank 2...")
            self.rpc_comm = rpc_comm.GrpcCommunicator(model=self.model, id=self.global_rank,
                                                      total_clients=3,
                                                      master_addr=self.rpc_server_addr,
                                                      master_port=self.rpc_server_port,
                                                      accumulate_updates=True
                                                      )
            logging.info("started gRPC client on global_rank 2...")
        elif self.global_rank == 3 or self.global_rank == 4:
            self.mpi_comm = torch_mpi.TorchMPICommunicator(id=self.local_rank,
                                                           total_clients=3,
                                                           backend=self.backend,
                                                           master_addr=self.mpi1_addr,
                                                           master_port=self.mpi1_port
                                                           )
            logging.info("started MPI process on global_rank 3/4...")
            self.rpc_comm = None
        elif self.global_rank == 5 or self.global_rank == 6:
            self.mpi_comm = torch_mpi.TorchMPICommunicator(id=self.local_rank,
                                                           total_clients=3,
                                                           backend=self.backend,
                                                           master_addr=self.mpi2_addr,
                                                           master_port=self.mpi2_port
                                                           )
            logging.info("started MPI process on global_rank 5/6...")
            self.rpc_comm = None

        logging.info("initialized communicator object...")

        self.train_dataloader, self.test_dataloader = cifar.cifar10Data(
            client_id=self.global_rank,
            total_clients=6,
            datadir=args.dir,
            train_bsz=self.train_bsz,
            test_bsz=self.test_bsz,
            partition_dataset=False,
        )
        logging.info("initialized dataloader object...")
        args.hostname = socket.gethostname()
        args.optimizer = self.optimizer.__class__.__name__
        logging.info(f"training/job specific parameters: {args}")

    def start_training(self):
        for epoch in range(self.epochs):
            for inputs, labels in self.train_dataloader:
                init_time = perf_counter_ns()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                pred = self.model(inputs)
                loss = self.loss_fn(pred, labels)
                self.training_samples += inputs.size(0)
                loss.backward()
                compute_time = (perf_counter_ns() - init_time) / nanosec_to_millisec
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.local_step += 1
                mpi_sync_time = None
                if self.global_rank != 0:
                    init_time = perf_counter_ns()
                    self.model = self.mpi_comm.aggregate(msg=self.model,
                                                         communicate_params=True,
                                                         compute_mean=False
                                                         )
                    mpi_sync_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

                rpc_sync_time = None
                if self.local_step % self.comm_freq == 0:
                    init_time = perf_counter_ns()
                    print("going to initial rpc call now...")
                    if self.global_rank == 0 or self.global_rank == 1 or self.global_rank == 2:
                        self.model = self.rpc_comm.aggregate(msg=self.model,
                                                             communicate_params=True,
                                                             compute_mean=False,
                                                             batch_samples=self.training_samples
                                                             )
                        self.training_samples = 0
                        rpc_sync_time = (perf_counter_ns() - init_time) / nanosec_to_millisec

                    # perform an MPI broadcast here next to distributed cross-facility aggregated model
                    # corresponds to processes with global_rank 1 and 2
                    if self.local_rank == 0:
                        self.model = self.mpi_comm.broadcast(msg=self.model, id=self.local_rank)

                logging.info(
                    f"training_metrics local_step: {self.local_step} compute_time {compute_time} ms "
                    f"mpi_sync_time: {mpi_sync_time} ms rpc_sync_time: {rpc_sync_time} ms"
                )