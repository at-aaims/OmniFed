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

import torch
import torch.distributed.rpc as rpc

from src.flora.communicator import Communicator
from src.flora.communicator.msg_queue import aggregate_updates


class RpcServer(object):
    def __init__(self, model, aggregate_sum=True, total_clients=1):
        self.model_update = [torch.zeros_like(param) for param in model.parameters()]
        self.aggregated_update = None
        self.total_clients = total_clients
        self.client_count = 0
        self.aggregate_sum = aggregate_sum

    def collect_updates(self, updates):
        self.model_update += updates
        self.client_count += 1
        if self.client_count == self.total_clients:
            self.client_count = 0
            if not self.aggregate_sum:
                self.model_update /= self.total_clients

            self.aggregated_update = self.model_update
            self.model_update = [torch.zeros_like(param) for param in self.model_update]
            return self.aggregated_update

    def server_model(self, id):
        print(f'fetching model update to server-id {id}')
        if self.aggregated_update is None:
            return self.model_update
        else:
            return self.aggregated_update


class TorchRpmCommunicator(Communicator):
    
    def __init__(self, id=0, total_clients=1, master_addr='127.0.0.1', master_port=27890):
        super().__init__(protocol_type='torch_rpc')
        self.id = id
        self.total_clients = total_clients
        self.master_addr = master_addr
        self.master_port = master_port
        self.central_server = 'central_server'
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = str(self.master_port)

        # TODO: adjust based on total available threads
        opts = rpc.TensorPipeRpcBackendOptions(num_worker_threads=4, rpc_timeout=0)
        if self.id == 0:
            rpc.init_rpc(self.central_server, rank=self.id, world_size=self.total_clients, rpc_backend_options=opts)

        else:
            rpc.init_rpc(f'worker-{self.id}', rank=self.id, world_size=self.total_clients, rpc_backend_options=opts)


    def aggregate(self, model, communicate_params=True):
        if communicate_params:
            updates = [param.data.detach() for param in model.parameters()]
        else:
            updates = [param.grad.detach() for param in model.parameters()]

        if self.id == 0:
            model = rpc.rpc_sync(self.central_server, RpcServer.server_model, args=(self.id,))

        else:
            aggregated_update = rpc.rpc_sync(self.central_server, RpcServer.collect_updates, args=(updates,))
            for param, update in zip(model.parameters(), aggregated_update):
                if communicate_params:
                    param.data.copy_(update)
                else:
                    param.grad.data.copy_(update)


        return model