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

import torch

from src.flora.communicator import Communicator

# TODO: implement a base class for Decentralized FL as well!

class BaseServer():
    def __init__(self, model, data, communicator, id=0, total_clients=2, **kwargs):
        """
        :param model: model to train
        :param dataset: training/validation/test dataset
        :param communicator: type of communication protocol and backend to use
        :param id: id of the current node
        :param total_clients: total number of clients/ world-size (including the server)
        :param **kwargs: variable arguments for additional utilities (e.g. compression, privacy, streaming, etc.)
        """
        self.model = model
        self.data = data
        self.communicator = communicator
        self.id = id
        self.total_clients = total_clients

    def initialize_model(self):
        raise NotImplementedError("initialize_model not implemented!")

    def receive_updates(self, src_id):
        raise NotImplementedError("receive_updates not implemented!")

    def send_updates(self, dst_id):
        raise NotImplementedError("send_updates not implemented!")

    def aggregate_updates(self):
        raise NotImplementedError("aggregate_updates not implemented!")

    def evaluate_model(self):
        raise NotImplementedError("evaluate_model not implemented!")


class BaseClient():
    def __init__(self, model, train_data, communicator, id=1, total_clients=2, **kwargs):
        """
        :param model: model to train
        :param train_data: dataset to use for training
        :param communicator: type of communication protocol and backend to use
        :param id: id of the current node
        :param total_clients: total number of clients/ world-size (including the server)
        :param **kwargs: variable arguments for additional utilities (e.g. compression, privacy, streaming, etc.)
        """
        self.model = model
        self.train_data = train_data
        self.communicator = communicator
        self.id = id
        self.total_clients = total_clients

    def initialize_model(self):
        raise NotImplementedError("initialize_model not implemented!")

    def receive_updates(self, src_id):
        raise NotImplementedError("receive_updates not implemented!")

    def send_updates(self, dst_id):
        raise NotImplementedError("send_updates not implemented!")

    def aggregate_updates(self):
        raise NotImplementedError("aggregate_updates not implemented!")

    def evaluate_model(self):
        raise NotImplementedError("evaluate_model not implemented!")

    def train_model(self):
        raise NotImplementedError("train_model not implemented!")


# class TestServer(BaseServer):
#     def __init__(self, model, dataset, communicator, client_id=0, total_clients=2, **kwargs):
#         super().__init__(model, dataset, communicator, client_id, total_clients)
#
#     def receive_updates(self, src_id):
#         print(f'model is {self.model}')
#
# if __name__ == "__main__":
#     test = TestServer('None','None', 'None')
#     test.receive_updates(0)