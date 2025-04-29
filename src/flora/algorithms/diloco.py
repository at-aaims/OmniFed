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

from src.flora.algorithms import BaseServer, BaseClient
from src.flora.communicator import Communicator


class DiLocoServer(BaseServer):
    def __init__(self, model: torch.nn.Module, data: torch.utils.data.DataLoader, communicator: Communicator,
                 id: int, total_clients: int):
        super().__init__(model, data, communicator, id, total_clients)

    def aggregate_updates(self):
        pass

    def initialize_model(self):
        # model broadcasted from central server with id 0
        self.model = self.communicator.broadcast(msg=self.model, id=self.id)


class DiLocoClient(BaseClient):