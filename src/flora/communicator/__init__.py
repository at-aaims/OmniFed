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

# contains fn calls and implementation of different communication mechanisms like MPI, gRPC and MQTT, WebSocket??
# also implements different compressors to reduce communication volume/cost

from abc import ABC, abstractmethod

class Communicator(ABC):
    def __init__(self, protocol_type):
        self.protocol_type = protocol_type

    def broadcast(self, **kwargs):
        raise NotImplemented("implement broadcast")

    def send(self, **kwargs):
        raise NotImplemented("implement send")

    def receive(self, **kwargs):
        raise NotImplemented("implement receive")

    def aggregate(self, **kwargs):
        raise NotImplemented("implement aggregate")

    def allgather(self, **kwargs):
        raise NotImplemented("implement allgather")

    def close(self, **kwargs):
        raise NotImplemented("implement close")