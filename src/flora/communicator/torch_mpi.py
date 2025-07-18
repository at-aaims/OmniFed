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
from typing import Dict
import tenseal as ts

import torch
import torch.distributed as dist

from src.flora.communicator import Communicator
from src.flora.compression import layerwise_decompress

# TODO: not taking returned data from sent/recv fn calls...fix that!


class TorchMPICommunicator(Communicator):
    def __init__(
        self,
        id,
        total_clients,
        init_method="tcp",
        master_addr="127.0.0.1",
        master_port="27890",
        backend="gloo",
        sharedfile="sharedfile",
    ):
        """
        :param id: client or server id ranging from 0 to (total_clients - 1)
        :param total_clients: total number of clients/world-size
        :param init_method: initialization method for clients: either tcp or sharedfile
        :param master_addr: address of master node or aggregation server
        :param master_port: port to bind to master node
        :param backend: communication backend to use: either 'mpi', 'gloo' or 'nccl'
        :param sharedfile: name of the shared file used by clients
        """
        super().__init__(protocol_type="torch_mpi")
        self.world_size = total_clients
        self.backend = backend

        if init_method == "tcp":
            timeout = datetime.timedelta(seconds=5 * 60)
            tcp_addr = "tcp://" + str(master_addr) + ":" + str(master_port)
            dist.init_process_group(
                backend=self.backend,
                init_method=tcp_addr,
                rank=id,
                world_size=self.world_size,
                timeout=timeout,
            )

        elif init_method == "sharedfile":
            sharedfile = "file://" + sharedfile
            dist.init_process_group(
                backend=self.backend,
                init_method=sharedfile,
                rank=id,
                world_size=self.world_size,
            )

    def broadcast(self, msg, id=0):
        """
        :param msg: message to broadcast
        :param id: node id which initiates the broadcast
        :return: returns the broadcasted message
        """
        if isinstance(msg, torch.nn.Module):
            for _, param in msg.named_parameters():
                if not param.requires_grad:
                    continue
                dist.broadcast(tensor=param.data, src=id)
        else:
            dist.broadcast(tensor=msg, src=id)

        return msg

    # def he_broadcast(self, ctx_bytes, ctx_size, id=0):
    #     dist.broadcast(ctx_size, src=0)
    #     ctx_buf = torch.empty(ctx_size.item(), dtype=torch.uint8)
    #     if id == 0:
    #         ctx_buf[:] = ctx_bytes
    #
    #     dist.broadcast(ctx_buf, src=0)
    #
    #     if id != 0:
    #         serialized_ctx = bytes(ctx_buf.tolist())
    #         ctx = ts.context_from(serialized_ctx)
    #         return ctx

    def aggregate(self, msg, communicate_params=True, compute_mean=True):
        """
        :param msg: message to aggregate
        :param communicate_params: collect model parameters if True, else aggregate model gradients
        :return: aggregated message
        """
        if isinstance(msg, torch.nn.Module):
            for _, param in msg.named_parameters():
                if not param.requires_grad:
                    continue
                if communicate_params:
                    dist.all_reduce(tensor=param.data, op=dist.ReduceOp.SUM)
                    param.data /= self.world_size if compute_mean else 1
                else:
                    dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= self.world_size if compute_mean else 1
        elif isinstance(msg, Dict):
            for _, param in msg.items():
                if not param.requires_grad:
                    continue
                if communicate_params:
                    dist.all_reduce(tensor=param.data, op=dist.ReduceOp.SUM)
                    param.data /= self.world_size if compute_mean else 1
                else:
                    dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= self.world_size if compute_mean else 1
        else:
            dist.all_reduce(tensor=msg, op=dist.ReduceOp.SUM)
            msg.data /= self.world_size if compute_mean else 1

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
                if not param.requires_grad:
                    continue
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
                if not param.requires_grad:
                    continue
                if communicate_params:
                    dist.recv(tensor=param.data, src=id)
                else:
                    dist.recv(tensor=param.grad, src=id)
        else:
            dist.send(tensor=msg, dst=id)

        return msg

    # def collect(self, msg, id=None, communicate_params=True):
    #     """
    #      all-gather in decentralized MPI collectives
    #     :param msg: message to receive
    #     :param id: client_id specifying the client update comes from. redundant in MPI communication as all_gather
    #     collects by rank ids
    #     :param communicate_params: collect model parameters if True, else send model gradients
    #     :return: either nested list of layerwise model data collected from clients or a simple list of gathered data
    #     """
    #     # if msg is a model, collected_data list contains list of tuples of client_id and layerwise model updates
    #     collected_data = []
    #
    #         for _, param in msg.named_parameters():
    #             if not param.requires_grad:
    #                 continue
    #             if communicate_params:
    #                 layerwise_collection = [
    #                     torch.zeros_like(param.data) for _ in range(self.world_size)
    #                 ]
    #             else:
    #                 layerwise_collection = [
    #                     torch.zeros_like(param.grad) for _ in range(self.world_size)
    #                 ]
    #
    #             dist.all_gather(tensor_list=layerwise_collection, tensor=param.data)
    #             layerwise_collection = [
    #                 (ix, layerwise_collection[ix])
    #                 for ix in range(len(layerwise_collection))
    #             ]
    #             collected_data.append(layerwise_collection)
    #
    #     elif isinstance(msg, int) or isinstance(msg, float):
    #         collected_data = [torch.Tensor([0.0]) for _ in range(self.world_size)]
    #         dist.all_gather(tensor_list=collected_data, tensor=torch.Tensor([msg]))
    #         collected_data = [
    #             (ix, collected_data[ix]) for ix in range(len(collected_data))
    #         ]
    #
    #     return collected_data

    def sparse_aggregate(self, msg, layerwise_vals, layerwise_ixs):
        if isinstance(msg, torch.nn.Module):
            for param, update_val, update_ix in zip(
                msg.parameters(), layerwise_vals, layerwise_ixs
            ):
                tensor_sizes = [torch.LongTensor([0]) for _ in range(self.world_size)]
                tensor_size = update_val.numel()
                dist.all_gather(tensor_sizes, torch.LongTensor([tensor_size]))

                tensor_list = []
                ix_list = []
                size_list = [int(size.item()) for size in tensor_sizes]
                max_size = max(size_list)
                if max_size > 0:
                    for _ in size_list:
                        tensor_list.append(
                            torch.zeros(size=(max_size,), dtype=torch.float32)
                        )
                        ix_list.append(torch.zeros(size=(max_size,), dtype=torch.long))

                    if tensor_size != max_size:
                        g_padding = torch.zeros(
                            size=(max_size - tensor_size,), dtype=torch.float32
                        )
                        ix_padding = torch.zeros(
                            size=(max_size - tensor_size,), dtype=torch.long
                        )
                        update_val = torch.cat((update_val, g_padding), dim=0)
                        update_ix = torch.cat((update_ix, ix_padding), dim=0)

                    dist.all_gather(tensor_list, update_val)
                    dist.all_gather(ix_list, update_ix)
                    # doing gradient sparsification
                    param.grad = layerwise_decompress(
                        collected_vals=tensor_list,
                        collected_ix=ix_list,
                        tensor_shape=param.shape,
                        client_count=self.world_size,
                    )

            return msg
        else:
            raise NotImplementedError(
                "sparse_aggregate implemented to only handle compressed gradient sparsification"
            )

    def quantized_minmaxval(self, min_val, max_val):
        dist.all_reduce(tensor=min_val, op=dist.ReduceOp.MIN)
        dist.all_reduce(tensor=max_val, op=dist.ReduceOp.MAX)

        return min_val, max_val

    def quantized_aggregate(self, quantized_dict, compute_mean=True):
        quantized_aggregate = {}
        if isinstance(quantized_dict, Dict):
            for key, val in quantized_dict.items():
                dist.all_reduce(tensor=val, op=dist.ReduceOp.SUM)
                if compute_mean:
                    val /= self.world_size

                quantized_aggregate[key] = val

            return quantized_aggregate
