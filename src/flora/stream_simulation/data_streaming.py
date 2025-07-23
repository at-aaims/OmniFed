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

import time
import pickle
import threading
from kafka import KafkaProducer, KafkaConsumer

import torch

from src.flora.datasets.image_classification import cifar


class DataStreamPublisher:
    def __init__(
        self,
        dataset_type='cifar10',
        kafka_host="127.0.0.1",
        kafka_port=9092,
        stream_rate=32,
        datadir="~/",
        total_clients=1,
    ):
        self.dataset_type = dataset_type
        self.kafka_host = kafka_host
        self.kafka_port = kafka_port
        self.stream_rate = stream_rate
        self.datadir = datadir
        self.total_clients = total_clients

        if self.dataset_type is None:
            raise ValueError(
                "Must specify either dataset or dataset_type TrainingDataset"
            )

        if self.dataset_type == "cifar10":
            self.train_dataset = cifar.cifar10Data(client_id=0,
                                                   total_clients=1,
                                                   datadir=self.datadir,
                                                   is_test=False,
                                                   get_training_dataset=True)

        elif self.dataset_type == "cifar100":
            self.train_dataset = cifar.cifar100Data(client_id=0,
                                                    total_clients=1,
                                                    datadir=self.datadir,
                                                    is_test=False,
                                                    get_training_dataset=True)

        self.producer = KafkaProducer(bootstrap_servers=self.kafka_host + ":" + str(self.kafka_port),
                                      value_serializer=self.serialize_sample)
        print("Producer started...")

    def serialize_sample(self, sample):
        image, label = sample
        return pickle.dumps({
            'image': image.numpy(),
            'label': label
        })

    def stream_data(self, topic):
        try:
            while True:
                for i, sample in enumerate(self.train_dataset):
                    self.producer.send(topic=topic, value=sample)
                    time.sleep(1 / self.stream_rate)
        finally:
            self.producer.flush()

    def publish_data_to_clients(self):
        for ix in range(self.total_clients):
            client_id = "client-{}".format(ix)
            print("going to publish data to client {}".format(client_id))
            thread = threading.Thread(target=self.stream_data, args=(client_id,))
            thread.start()

    def create_topic(self, topic):
        pass


class DataStreamSubscriber:
    def __init__(self,
        kafka_host="127.0.0.1",
        kafka_port=9092,
        client_id=0,
    ):
        self.kafka_host = kafka_host
        self.kafka_port = kafka_port
        self.topic = "client-{}".format(client_id)

        self.consumer = KafkaConsumer(self.topic,
                                      bootstrap_servers=self.kafka_host + ":" + str(self.kafka_port),
                                      auto_offset_reset='earliest',
                                      enable_auto_commit=True)
        print("Consumer started...")

    def deserialize_sample(self, msg_bytes):
        data_dict = pickle.loads(msg_bytes)
        image_np = data_dict['image']
        label = data_dict['label']
        # [C, H, W]
        image_tensor = torch.tensor(image_np, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor

    def stream_data(self):
        for message in self.consumer:
            img_tensor, label_tensor = self.deserialize_sample(message.value)
            print(f"received sample label {label_tensor.item()} image tensor shape {img_tensor.shape}")
