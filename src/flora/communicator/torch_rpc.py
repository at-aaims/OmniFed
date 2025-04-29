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

# TODO: implement broadcast operation in torch.rpc
# TODO: implement simple aggregation on the specific data-type being called

class RpcServer(object):
    def __init__(self, model, summation=True, total_clients=1):
        """
        :param model: model to communicate
        :param summation: whether to sum the updates or average them
        :param total_clients: total number of clients/ world-size (including the server)
        """
        self.model_update = [torch.zeros_like(param) for param in model.parameters()]
        self.aggregated_update = None
        self.total_clients = total_clients
        self.client_count = 0
        self.summation = summation

    def collect_updates(self, updates):
        self.model_update += updates
        self.client_count += 1
        if self.client_count == self.total_clients:
            self.client_count = 0
            if not self.summation:
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


class TorchRpcCommunicator(Communicator):
    
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


    def aggregate(self, msg, communicate_params=True):
        if isinstance(msg, torch.nn.Module):
            # communicate either model parameters or gradients
            if communicate_params:
                updates = [param.data.detach() for param in msg.parameters()]
            else:
                updates = [param.grad.detach() for param in msg.parameters()]

            if self.id == 0:
                msg = rpc.rpc_sync(self.central_server, RpcServer.server_model, args=(self.id,))

            else:
                aggregated_update = rpc.rpc_sync(self.central_server, RpcServer.collect_updates, args=(updates,))
                for param, update in zip(msg.parameters(), aggregated_update):
                    if communicate_params:
                        param.data.copy_(update)
                    else:
                        param.grad.data.copy_(update)

            return msg
        else:
            # TODO: implement simple aggregation on the specific data-type being called
            raise TypeError("aggregate fn only supports torch.nn.Module type")