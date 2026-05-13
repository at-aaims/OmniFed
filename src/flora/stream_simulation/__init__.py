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

import os
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import torch
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

from src.flora.datasets.image_classification.caltech import (
    caltech101Data,
    caltech256Data,
)
from src.flora.datasets.image_classification.cifar import cifar10Data, cifar100Data
from src.flora.datasets.image_classification.imagenet import imagenetData
from src.flora.datasets.image_classification.img_datasets import (
    food101Data,
    places365Data,
    emnistData,
    fmnistData,
)
# from src.flora.datasets.image_classification.medical_imaging import isicArchiveData
# from src.flora.datasets.nlp import imdbReviewsData
# from src.flora.datasets.object_detection.coco import coco2017Data
# from src.flora.datasets.object_detection.pascal_voc import pascalvocData
# from src.flora.datasets.speech_recognition import libriSpeechData, commonVoiceData

# INSTRUCTIONS:
# install kafka by downloading and unpacking kafka_2.13-3.1.0.tgz
# first download compatible jdk (tested on openjdk@11.0)
# start zookeeper (built-into kafka): bin/zookeeper-server-start.sh config/zookeeper.properties
# start kafka server: bin/kafka-server-start.sh config/server.properties

# TODO: check version compatibility between torch and torchtext for NLP dataset IMDB
# TODO: verify that topics are created if they don't already exist


class DataHandler:
    def cifar10(get_training_dataset, datadir, client_id, total_clients, is_test):
        return cifar10Data(
            get_training_dataset=get_training_dataset,
            datadir=datadir,
            client_id=client_id,
            total_clients=total_clients,
            is_test=is_test,
        )

    def cifar100(get_training_dataset, datadir, client_id, total_clients, is_test):
        return cifar100Data(
            get_training_dataset=get_training_dataset,
            datadir=datadir,
            client_id=client_id,
            total_clients=total_clients,
            is_test=is_test,
        )

    def caltech101(get_training_dataset, datadir, client_id, total_clients, is_test):
        return caltech101Data(
            get_training_dataset=get_training_dataset,
            datadir=datadir,
            client_id=client_id,
            total_clients=total_clients,
            is_test=is_test,
        )

    def caltech256(get_training_dataset, datadir, client_id, total_clients, is_test):
        return caltech256Data(
            get_training_dataset=get_training_dataset,
            datadir=datadir,
            client_id=client_id,
            total_clients=total_clients,
            is_test=is_test,
        )

    def imagenet(get_training_dataset, datadir, client_id, total_clients, is_test):
        return imagenetData(
            get_training_dataset=get_training_dataset,
            datadir=datadir,
            client_id=client_id,
            total_clients=total_clients,
            is_test=is_test,
        )

    def fmnist(get_training_dataset, datadir, client_id, total_clients, is_test):
        return fmnistData(
            get_training_dataset=get_training_dataset,
            datadir=datadir,
            client_id=client_id,
            total_clients=total_clients,
            is_test=is_test,
        )

    def places365(get_training_dataset, datadir, client_id, total_clients, is_test):
        return places365Data(
            get_training_dataset=get_training_dataset,
            datadir=datadir,
            client_id=client_id,
            total_clients=total_clients,
            is_test=is_test,
        )

    def emnist(get_training_dataset, datadir, client_id, total_clients, is_test):
        return emnistData(
            get_training_dataset=get_training_dataset,
            datadir=datadir,
            client_id=client_id,
            total_clients=total_clients,
            is_test=is_test,
        )

    def food101(get_training_dataset, datadir, client_id, total_clients, is_test):
        return food101Data(
            get_training_dataset=get_training_dataset,
            datadir=datadir,
            client_id=client_id,
            total_clients=total_clients,
            is_test=is_test,
        )


class TrainingDataset(Enum):
    CIFAR10 = DataHandler.cifar10
    CIFAR100 = DataHandler.cifar100
    CALTECH101 = DataHandler.caltech101
    CALTECH256 = DataHandler.caltech256
    IMAGENET = DataHandler.imagenet
    FOOD101 = DataHandler.food101
    PLACES365 = DataHandler.places365
    EMNIST = DataHandler.emnist
    FMNIST = DataHandler.fmnist
    # ISIC_ARCHIVE = isicArchiveData
    # IMDB = imdbReviewsData
    # COCO2017 = coco2017Data
    # PASCALVOC = pascalvocData
    # LIBRISPEECH = libriSpeechData
    # COMMONVOICE = commonVoiceData

    @staticmethod
    def execute_action(enum_dataset, **kwargs):
        dataset = enum_dataset(**kwargs)

        return dataset


class DataStreamSimulator:
    def __init__(
        self,
        dataset=None,
        dataset_type=TrainingDataset.CIFAR10,
        kafka_host="127.0.0.1",
        kafka_port=9092,
        stream_rate=32,
        datadir="~/",
        client_id=0,
        total_clients=1,
        input_shape=[1, 3, 32, 32],
        label_shape=[1],
        kafka_dir="~/kafka",
    ):
        """
        Simulates training data streaming into a kafka consumer, and currently published by the central server
        (client_id 0). default streams 32x32 RBG image and its corresponding label to a client (i.e., consumer)
        :param dataset: dataset object for the training data
        :param dataset_type: enum for type of dataset
        :param kafka_host: ip address of the kafka producer
        :param kafka_port: port of the kafka producer
        :param stream_rate: rate at which data is streamed (in samples/second). defaults to 8 samples/second
        :param datadir: directory where data to be streamed is downloaded and stored
        :param client_id: id of the current client
        :param total_clients: total number of clients/ world-size
        :param input_shape: shape of input data expected by the model. defaults to a single RGB sample of 32x32 image.
        :param label_shape: shape of the label for a given sample.
        :param kafka_dir: directory where kafka is installed. defaults to ~/kafka.
        """
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.kafka_host = kafka_host
        self.kafka_port = kafka_port
        self.stream_rate = stream_rate
        self.datadir = datadir
        self.client_id = client_id
        self.total_clients = total_clients
        self.input_shape = input_shape
        self.label_shape = label_shape
        self.kafka_dir = kafka_dir

        if self.total_clients < 2:
            raise ValueError(
                "total devices must be at least 2, for 1 client and 1 server"
            )

        if self.dataset is None and dataset_type is None:
            raise ValueError(
                "Must specify either dataset or dataset_type TrainingDataset"
            )

        if self.dataset is not None and dataset_type is not None:
            raise ValueError(
                "Specify only one: dataset or dataset_type TrainingDataset"
            )

        if self.dataset is None:
            if (
                self.dataset_type == TrainingDataset.CIFAR10
                or self.dataset_type == TrainingDataset.CIFAR100
                or self.dataset_type == TrainingDataset.CALTECH101
                or self.dataset_type == TrainingDataset.CALTECH256
                or self.dataset_type == TrainingDataset.IMAGENET
                or self.dataset_type == TrainingDataset.FOOD101
                or self.dataset_type == TrainingDataset.PLACES365
                or self.dataset_type == TrainingDataset.EMNIST
                or self.dataset_type == TrainingDataset.FMNIST
            ):
                self.dataset = TrainingDataset.execute_action(
                    enum_dataset=self.dataset_type,
                    get_training_dataset=True,
                    datadir=self.datadir,
                    client_id=self.client_id,
                    total_clients=self.total_clients,
                    is_test=False,
                )
                print(f"length of dataset: {len(self.dataset)}")

        if self.client_id == 0:
            # start zookeeper and kafka servers
            # self.start_zookeeper_kafka()
            # start producer to publish data to kafka topics
            print(f"setting up producer on client_id {self.client_id}")
            self.streamer = KafkaProducer(
                bootstrap_servers=self.kafka_host + ":" + str(self.kafka_port)
            )
            self.create_topics()
            self.produce()
        else:
            self.streamer = KafkaConsumer("client" + str(self.client_id))

    def start_zookeeper_kafka(self):
        # TODO: add Windows with .bat executables
        try:
            zookeeper_server_pth = os.path.join(
                self.kafka_dir, "bin", "zookeeper-server-start.sh"
            )
            kafka_server_pth = os.path.join(
                self.kafka_dir, "bin", "kafka-server-start.sh"
            )

            if not os.path.exists(zookeeper_server_pth) or not os.path.exists(
                kafka_server_pth
            ):
                raise FileNotFoundError(
                    f"cannot find kafka scripts at {kafka_server_pth}"
                )

            if not self.check_if_already_running(service="zookeeper"):
                zookeeper_server = [zookeeper_server_pth, "config.zookeeper.properties"]
                result = subprocess.run(
                    zookeeper_server, check=True, text=True, capture_output=True
                )
                print(result.stdout)
                wait_time = 5
                print(
                    f"starting zookeeper server...will start kafka-server in {wait_time} seconds..."
                )
                time.sleep(wait_time)

            if not self.check_if_already_running(service="kafka"):
                kafka_server = [kafka_server_pth, "config/server.properties"]
                result = subprocess.run(
                    kafka_server, check=True, text=True, capture_output=True
                )
                print(result.stdout)

        except subprocess.CalledProcessError:
            print(
                f"could not start zookeeper or kafka on server {self.kafka_host}:{self.kafka_port}"
            )

    def check_if_already_running(self, service):
        try:
            # Use pgrep on linux-like systems and tasklist on Windows
            if os.name == "posix":
                result = subprocess.run(
                    ["pgrep", "-f", service],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            else:
                # On Windows, assume the executable name is 'zookeeper-server-start.bat' or similar
                result = subprocess.run(
                    ["tasklist"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Check if any line contains 'zookeeper' in the task list
                return service in result.stdout.lower()

            # If the result return code is 0, it means the process is running
            return result.returncode == 0
        except Exception as e:
            print(f"An error occurred while checking {service} status: {e}")
            return False

    def create_topics(self):
        # TODO: check if topics already exist and handle accordingly
        try:
            consumer = KafkaConsumer(
                bootstrap_servers=self.kafka_host + ":" + str(self.kafka_port)
            )
            topics = consumer.topics()
            for client_id in range(self.total_clients):
                try:
                    topic_name = "client" + str(client_id + 1)
                    if topic_name not in topics:
                        kafka_topics_pth = os.path.join(
                            self.kafka_dir, "bin/kafka-topics.sh"
                        )
                        command = [
                            kafka_topics_pth,
                            "--create",
                            "--topic",
                            topic_name,
                            "--bootstrap-server",
                            self.kafka_host + ":" + str(self.kafka_port),
                            "--partitions",
                            "1",
                            "--replication-factor",
                            "1",
                        ]
                        print(f"command is {command}")
                        result = subprocess.run(
                            command, check=True, text=True, capture_output=True
                        )
                        print(f"topic {topic_name} created successfully.")
                        print(result.stdout)

                except subprocess.CalledProcessError as e:
                    print(f"Error creating topic {e.stderr}")
        except KafkaError as e:
            print(f"Error while checking topic existence: {e}")
        finally:
            consumer.close()

    def publish_data_to_client(self, target_client=1):
        """spawn # of threads as total clients to publish data to a queue corresponding to each client"""
        try:
            while True:
                for ix in range(len(self.dataset)):
                    inp, label = self.dataset[ix]
                    self.streamer.send(
                        topic="client" + str(target_client),
                        key=label.to_bytes(),
                        value=inp.numpy().tobytes(),
                    )

                    time.sleep(1.0 / self.stream_rate)

        except KeyboardInterrupt:
            print("Stopping streaming data")

        finally:
            self.streamer.close()

    @abstractmethod
    def produce(self):
        for client in range(self.total_clients):
            print(
                f"going to publish data to client client-{client + 1} in background thread..."
            )
            # self.publish_data_to_client(client + 1)
            thread = threading.Thread(
                target=self.publish_data_to_client, args=(client + 1,)
            )
            thread.start()

    @abstractmethod
    def consume(self):
        try:
            for data in self.streamer:
                inp = torch.from_numpy(
                    np.frombuffer(data.value, dtype=np.float32)
                ).reshape(self.input_shape)
                label = (
                    torch.from_numpy(np.frombuffer(data.key, dtype=np.int32))[0]
                    .reshape(self.label_shape)
                    .to(torch.int32)
                )

                yield inp, label

        except KeyboardInterrupt:
            print("Stopping streaming data")

        finally:
            self.streamer.close()

    def end_streaming(self):
        self.streamer.close()
        print("finished streaming")
