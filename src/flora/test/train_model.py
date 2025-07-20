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

# from src.flora.flora_rpc import grpc_communicator
from src.flora.communicator import grpc_communicator
from src.flora.datasets.image_classification import cifar, caltech
from src.flora.test import get_model
from src.flora.helper import training_params
from src.flora.algorithms import SimpleFedPerHead, fedper, feddyn
from src.flora.algorithms import (
    fedavg,
    fedprox,
    fedmom,
    fednova,
    scaffold,
    diloco,
    moon,
    ditto,
    fedbn,
)


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
        self.algo = args.algo
        self.fedprox_mu = args.fedprox_mu
        self.fedmom_momentum = args.fedmom_momentum
        self.fednova_weight_decay = args.fednova_weight_decay
        self.diloco_outer_lr = args.diloco_outer_lr
        self.diloco_outer_momentum = args.diloco_outer_momentum
        self.moon_num_prev_models = args.moon_num_prev_models
        self.moon_temperature = args.moon_temperature
        self.moon_mu = args.moon_mu
        self.feddyn_regularizer_alpha = args.feddyn_regularizer_alpha

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

        self.ditto_regularizer = args.ditto_regularizer
        self.ditto_global_loss = torch.nn.CrossEntropyLoss()
        self.global_model = get_model(
            self.model_name, determinism=self.determinism, args=args
        )
        self.ditto_global_optimizer = torch.optim.SGD(
            self.global_model.get_model().parameters(),
            lr=self.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        if args.communicator == "Collective":
            self.communicator = torch_mpi.TorchMPICommunicator(
                id=self.rank,
                total_clients=self.world_size,
                backend=self.backend,
                master_addr=args.master_addr,
                master_port=args.master_port,
            )
        elif args.communicator == "RPC":
            self.communicator = grpc_communicator.GrpcCommunicator(
                model=self.model,
                id=self.rank,
                total_clients=self.world_size,
                master_addr=args.master_addr,
                master_port=args.master_port,
                accumulate_updates=True,
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
        self.trainer = self.get_FL_algo(algo=self.algo)
        logging.info("initialized trainer object...")

        args.hostname = socket.gethostname()
        args.optimizer = self.optimizer.__class__.__name__
        args.running_job = "ModelTrainer"
        logging.info(f"training/job specific parameters: {args}")

    def get_FL_algo(self, algo="fedavg"):
        if algo == "fedavg":
            # specify Federated Averaging algorithm here...
            self.fedavg_params = training_params.FedAvgTrainingParameters(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                epochs=self.epochs,
                comm_freq=self.comm_freq,
                lr_scheduler=self.lr_scheduler,
            )
            logging.info("initialized trainable_params object...")
            return fedavg.FederatedAveraging(
                client_id=self.rank,
                model=self.model,
                train_data=self.train_dataloader,
                test_data=self.test_dataloader,
                communicator=self.communicator,
                total_clients=self.world_size,
                train_params=self.fedavg_params,
            )
        elif algo == "fedprox":
            self.fedprox_params = training_params.FedProxTrainingParameters(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                epochs=self.epochs,
                comm_freq=self.comm_freq,
                mu=self.fedprox_mu,
                lr_scheduler=self.lr_scheduler,
            )
            return fedprox.FedProx(
                client_id=self.rank,
                model=self.model,
                communicator=self.communicator,
                train_data=self.train_dataloader,
                test_data=self.test_dataloader,
                total_clients=self.world_size,
                train_params=self.fedprox_params,
            )
        elif algo == "fedmom":
            self.fedmom_params = training_params.FedMomTrainingParameters(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                epochs=self.epochs,
                comm_freq=self.comm_freq,
                momentum=self.fedmom_momentum,
                lr_scheduler=self.lr_scheduler,
                lr=self.lr,
            )
            return fedmom.FederatedMomentum(
                client_id=self.rank,
                model=self.model,
                communicator=self.communicator,
                train_data=self.train_dataloader,
                test_data=self.test_dataloader,
                total_clients=self.world_size,
                train_params=self.fedmom_params,
            )
        elif algo == "fednova":
            self.fednova_params = training_params.FedNovaTrainingParameters(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                epochs=self.epochs,
                comm_freq=self.comm_freq,
                lr_scheduler=self.lr_scheduler,
                weight_decay=self.fednova_weight_decay,
            )
            return fednova.FedNova(
                client_id=self.rank,
                model=self.model,
                communicator=self.communicator,
                train_data=self.train_dataloader,
                test_data=self.test_dataloader,
                total_clients=self.world_size,
                train_params=self.fednova_params,
            )
        elif algo == "scaffold":
            self.scaffold_params = training_params.ScaffoldTrainingParameters(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                epochs=self.epochs,
                comm_freq=self.comm_freq,
            )
            return scaffold.Scaffold(
                client_id=self.rank,
                model=self.model,
                communicator=self.communicator,
                train_data=self.train_dataloader,
                test_data=self.test_dataloader,
                total_clients=self.world_size,
                train_params=self.scaffold_params,
            )
        elif algo == "diloco":
            self.diloco_params = training_params.DiLocoTrainingParameters(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                epochs=self.epochs,
                comm_freq=self.comm_freq,
                outer_lr=self.diloco_outer_lr,
                outer_momentum=self.diloco_outer_momentum,
            )
            return diloco.DiLoCo(
                client_id=self.rank,
                model=self.model,
                train_data=self.train_dataloader,
                test_data=self.test_dataloader,
                communicator=self.communicator,
                total_clients=self.world_size,
                train_params=self.diloco_params,
            )
        elif algo == "moon":
            self.moon_params = training_params.MOONTrainingParameters(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                epochs=self.epochs,
                comm_freq=self.comm_freq,
                num_prev_models=self.moon_num_prev_models,
                temperature=self.moon_temperature,
                mu=self.moon_mu,
            )
            return moon.Moon(
                client_id=self.rank,
                model=self.model,
                train_data=self.train_dataloader,
                test_data=self.test_dataloader,
                communicator=self.communicator,
                total_clients=self.world_size,
                train_params=self.moon_params,
            )
        elif algo == "ditto":
            self.ditto_params = training_params.DittoTrainingParameters(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                epochs=self.epochs,
                comm_freq=self.comm_freq,
                ditto_regularizer=self.ditto_regularizer,
                global_loss=self.ditto_global_loss,
                global_optimizer=self.ditto_global_optimizer,
            )
            return ditto.Ditto(
                client_id=self.rank,
                model=self.model,
                train_data=self.train_dataloader,
                test_data=self.test_dataloader,
                communicator=self.communicator,
                total_clients=self.world_size,
                train_params=self.ditto_params,
            )
        elif algo == "fedbn":
            self.fedbn_params = training_params.FedBNTrainingParameters(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                epochs=self.epochs,
                comm_freq=self.comm_freq,
            )
            return fedbn.FederatedBatchNormalization(
                client_id=self.rank,
                model=self.model,
                train_data=self.train_dataloader,
                test_data=self.test_dataloader,
                communicator=self.communicator,
                total_clients=self.world_size,
                train_params=self.fedbn_params,
            )
        elif algo == "fedper":
            self.fedper_params = training_params.FedPerTrainingParameters(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                epochs=self.epochs,
                comm_freq=self.comm_freq,
            )
            return fedper.FedPer(
                client_id=self.rank,
                base_model=self.model,
                # for ResNet18 CIFAR10
                # personal_head=SimpleFedPerHead(input_dim=1000, num_classes=10),
                # for VGG11 CIFAR100
                # personal_head=SimpleFedPerHead(input_dim=1000, num_classes=100),
                # for AlexNet
                # personal_head=SimpleFedPerHead(input_dim=1000, num_classes=102),
                # for MobileNet-V3
                personal_head=SimpleFedPerHead(input_dim=1000, num_classes=257),
                train_data=self.train_dataloader,
                test_data=self.test_dataloader,
                communicator=self.communicator,
                total_clients=self.world_size,
                train_params=self.fedper_params,
            )
        elif algo == "feddyn":
            self.feddyn_params = training_params.FedDynTrainingParameters(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                epochs=self.epochs,
                comm_freq=self.comm_freq,
                regularizer_alpha=self.feddyn_regularizer_alpha,
            )
            return feddyn.FedDyn(
                client_id=self.rank,
                model=self.model,
                train_data=self.train_dataloader,
                test_data=self.test_dataloader,
                communicator=self.communicator,
                total_clients=self.world_size,
                train_params=self.feddyn_params,
            )

    def start(self):
        # self.fedavg_trainer should have overwritable functions train_eval(..) and test_eval(..) to measure training
        # and test performance respectively
        print("going to start training")
        self.trainer.train()
