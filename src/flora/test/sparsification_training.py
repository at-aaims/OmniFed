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

import torch

from src.flora.communicator import torch_mpi
from src.flora.datasets.image_classification import cifar, caltech
from src.flora.test import get_model
from src.flora.helper import training_params
from src.flora.helper.node_config import NodeConfig
from src.flora.algorithms.sparse_bsp import SparseBSPTraining
from src.flora.compression import sparsification


class SparseCompressionTrainer(object):
    def __init__(self, args):
        self.args = args
        self.train_bsz = args.bsz
        self.test_bsz = args.test_bsz
        self.logdir = args.dir
        self.model_name = args.model
        self.rank = args.rank
        self.world_size = args.world_size
        self.backend = args.backend
        self.epochs = args.epochs

        dev_id = NodeConfig().get_gpus() % self.world_size
        device = torch.device(
            "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        )
        self.compression_type = args.compression_type
        self.compress_ratio = args.compress_ratio
        if self.compression_type == "topK":
            self.compression = sparsification.TopKCompression(
                device=device, compress_ratio=self.compress_ratio
            )
        elif self.compression_type == "dgc":
            self.compression = sparsification.DGCCompression(
                device=device, compress_ratio=self.compress_ratio
            )
        elif self.compression_type == "redsync":
            self.compression = sparsification.RedsyncCompression(
                device=device, compress_ratio=self.compress_ratio
            )
        elif self.compression_type == "sidco":
            self.compression = sparsification.SIDCoCompression(
                num_stages=3, device=device, compress_ratio=self.compress_ratio
            )
        elif self.compression_type == "randomK":
            self.compression = sparsification.RandomKCompression(
                device=device, compress_ratio=self.compress_ratio
            )

        logging.basicConfig(
            filename=self.logdir
            + "/g"
            + str(self.rank)
            + "/"
            + self.model_name
            + "-"
            + str(self.rank)
            + ".log",
            level=logging.INFO,
        )
        self.dataset_name = args.dataset
        # hard-coded for now
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
        if self.lr_scheduler is None:
            self.lr = self.model_obj.get_lr()
        else:
            self.lr = self.lr_scheduler.get_last_lr()[0]

        self.communicator = torch_mpi.TorchMPICommunicator(
            id=self.rank,
            total_clients=self.world_size,
            backend=self.backend,
            master_addr=args.master_addr,
            master_port=args.master_port,
        )
        logging.info("initialized communicator object...")

        if self.dataset_name == "cifar10":
            self.train_dataloader, self.test_dataloader = cifar.cifar10Data(
                client_id=self.rank,
                total_clients=self.world_size,
                datadir=args.dir,
                train_bsz=self.train_bsz,
                test_bsz=self.test_bsz,
                partition_dataset=False,
            )
        elif self.dataset_name == "caltech101":
            self.train_dataloader, self.test_dataloader = caltech.caltech101Data(
                client_id=self.rank,
                total_clients=self.world_size,
                datadir=args.dir,
                train_bsz=self.train_bsz,
                test_bsz=self.test_bsz,
                partition_dataset=False,
            )
        elif self.dataset_name == "cifar100":
            self.train_dataloader, self.test_dataloader = cifar.cifar100Data(
                client_id=self.rank,
                total_clients=self.world_size,
                datadir=args.dir,
                train_bsz=self.train_bsz,
                test_bsz=self.test_bsz,
                partition_dataset=False,
            )
        elif self.dataset_name == "caltech256":
            self.train_dataloader, self.test_dataloader = caltech.caltech256Data(
                client_id=self.rank,
                total_clients=self.world_size,
                datadir=args.dir,
                train_bsz=self.train_bsz,
                test_bsz=self.test_bsz,
                partition_dataset=False,
            )

        logging.info("initialized dataloader object...")
        self.fedavg_params = training_params.FedAvgTrainingParameters(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            epochs=self.epochs,
            comm_freq=1,
            lr_scheduler=self.lr_scheduler,
        )
        self.trainer = SparseBSPTraining(
            model=self.model,
            train_data=self.train_dataloader,
            test_data=self.test_dataloader,
            communicator=self.communicator,
            total_clients=self.world_size,
            train_params=self.fedavg_params,
            compression=self.compression,
        )
        logging.info("initialized trainer object...")

        args.hostname = socket.gethostname()
        args.optimizer = self.optimizer.__class__.__name__
        args.running_job = "CompressionTrainer"
        logging.info(f"training/job specific parameters: {args}")

    def start(self):
        print("going to start training")
        self.trainer.train()
