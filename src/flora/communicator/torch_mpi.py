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

import datetime
import torch

from src.flora.communicator import Communicator
import torch.distributed as dist

# TODO: not taking returned data from sent/recv fn calls...fix that!

class TorchMPICommunicator(Communicator):

    def __init__(self, id, total_clients, init_method='tcp', master_addr='127.0.0.1', master_port='27890',
                 backend='gloo', sharedfile='sharedfile'):
        """
        :param id: client or server id ranging from 0 to (total_clients - 1)
        :param total_clients: total number of clients/world-size
        :param init_method: initialization method for clients: either tcp or sharedfile
        :param master_addr: address of master node or aggregation server
        :param master_port: port to bind to master node
        :param backend: communication backend to use: either 'mpi', 'gloo' or 'nccl'
        :param sharedfile: name of the shared file used by clients
        """
        super().__init__(protocol_type='torch_mpi')
        self.world_size = total_clients
        self.backend = backend

        if init_method == 'tcp':
            timeout = datetime.timedelta(seconds=5 * 60)
            tcp_addr = 'tcp://' + str(master_addr) + ':' + str(master_port)
            dist.init_process_group(backend=self.backend, init_method=tcp_addr, rank=id,
                                    world_size=self.world_size, timeout=timeout)

        elif init_method == 'sharedfile':
            sharedfile = 'file://' + sharedfile
            dist.init_process_group(backend=self.backend, init_method=sharedfile, rank=id,
                                    world_size=self.world_size)

    def broadcast(self, msg, id=0):
        """
        :param msg: message to broadcast
        :param id: node id which initiates the broadcast
        :return: returns the broadcasted message
        """
        if isinstance(msg, torch.nn.Module):
            for _, param in msg.named_parameters():
                if not param.requires_grad: continue
                dist.broadcast(tensor=param.data, src=id)
        else:
            dist.broadcast(tensor=msg, src=id)

        return msg

    def aggregate(self, msg, communicate_params=True):
        """
        :param msg: message to aggregate
        :param communicate_params: collect model parameters if True, else aggregate model gradients
        :return: aggregated message
        """
        if isinstance(msg, torch.nn.Module):
            for _, param in msg.named_parameters():
                if not param.requires_grad: continue
                if communicate_params:
                    dist.all_reduce(tensor=param.data, op=dist.ReduceOp.SUM)
                else:
                    dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM)
        else:
            dist.all_reduce(tensor=msg, op=dist.ReduceOp.SUM)

        return msg

    def send(self, msg, id=0, communicate_params=True):
        """
        :param msg: message to send
        :param id: client or server id ranging from 0 to (total_clients - 1)
        :param communicate_params: collect model parameters if True, else aggregate model gradients
        :return: the sending message
        """
        if isinstance(msg, torch.nn.Module):
            for _, param in msg.named_parameters():
                if not param.requires_grad: continue
                if communicate_params:
                    dist.send(tensor=param.data, dst=id)
                else:
                    dist.send(tensor=param.grad, dst=id)
        else:
            dist.send(tensor=msg, dst=id)

        return msg

    def recv(self, msg, id=0, communicate_params=True):
        """
        :param msg: message to receive
        :param id: client or server id ranging from 0 to (total_clients - 1)
        :param communicate_params: collect model parameters if True, else aggregate model gradients
        :return: the receiving message
        """
        if isinstance(msg, torch.nn.Module):
            for _, param in msg.named_parameters():
                if not param.requires_grad: continue
                if communicate_params:
                    dist.recv(tensor=param.data, src=id)
                else:
                    dist.recv(tensor=param.grad, src=id)
        else:
            dist.send(tensor=msg, dst=id)

        return msg