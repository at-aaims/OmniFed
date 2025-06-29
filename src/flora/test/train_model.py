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

from src.flora.communicator import torch_mpi, torch_rpc
from src.flora.datasets.image_classification import cifar
from src.flora.test import get_model
from src.flora.helper import training_params
from src.flora.algorithms import fedavg


class ModelTrainer(object):
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
        self.comm_freq = args.comm_freq

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

        if args.communicator == "Collective":
            self.communicator = torch_mpi.TorchMPICommunicator(
                id=self.rank,
                total_clients=self.world_size,
                backend=self.backend,
                master_addr=args.master_addr,
                master_port=args.master_port,
            )
        elif args.communicator == "RPC":
            self.communicator = torch_rpc.TorchRpcCommunicator(
                id=self.rank,
                model=self.model,
                total_clients=self.world_size,
                master_addr=args.master_addr,
                master_port=args.master_port,
            )
        logging.info("initialized communicator object...")

        self.train_dataloader, self.test_dataloader = cifar.cifar10Data(
            client_id=self.rank,
            total_clients=self.world_size,
            datadir=args.dir,
            train_bsz=self.train_bsz,
            test_bsz=self.test_bsz,
            partition_dataset=False,
        )
        logging.info("initialized dataloader object...")

        # Now specify Federated Averaging algorithm here...
        self.trainable_params = training_params.FedAvgTrainingParameters(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            epochs=self.epochs,
            comm_freq=self.comm_freq,
        )
        logging.info("initialized trainable_params object...")

        self.fedavg_trainer = fedavg.FederatedAveraging(
            model=self.model,
            train_data=self.train_dataloader,
            test_data=self.test_dataloader,
            communicator=self.communicator,
            total_clients=self.world_size,
            train_params=self.trainable_params,
        )
        logging.info("initialized fedavg_trainer object...")

        args.hostname = socket.gethostname()
        args.optimizer = self.optimizer.__class__.__name__
        logging.info(f"training/job specific parameters: {args}")

    def start(self):
        # self.fedavg_trainer should have overwritable functions train_eval(..) and test_eval(..) to measure training
        # and test performance respectively
        print("going to start training")
        self.fedavg_trainer.train()
