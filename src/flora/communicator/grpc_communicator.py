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
import grpc
from concurrent import futures
from typing import Optional, Union

import torch.nn

from src.flora.communicator import Communicator
import src.flora.communicator.grpc_communicator_pb2_grpc as flora_grpc_pb2_grpc

from src.flora.communicator.grpc_limits import GRPC_MAX_MESSAGE_BYTES
from src.flora.communicator.grpc_server import CentralServerServicer
from src.flora.communicator.grpc_client import GrpcClient
from src.flora.compression.sparsification import TopKCompression


class GrpcCommunicator(Communicator):
    def __init__(
        self,
        model: torch.nn.Module,
        id: int = 0,
        total_clients: int = 1,
        master_addr: str = "127.0.0.1",
        master_port: int = 50051,
        accumulate_updates: bool = True,
        daemon_server: bool = False,
        compressor: Optional[TopKCompression] = None,
    ):
        super().__init__(protocol_type="RPC")
        self.id = id
        # total clients excluding parameter server
        self.total_clients = total_clients - 1
        self.master_port = master_port
        self.accumulate_updates = accumulate_updates
        self.compressor = compressor
        self.server = None

        # Flora: only ``id == 0`` starts the daemon gRPC server. That is the *communicator role id*,
        # not OmniFed hybrid ``SLURM_PROCID``. Hybrid Slurm runs the server **process** on
        # ``topology.rpc.server_rank`` but must still construct the server with ``id=0`` here unless
        # this branch is refactored (see Phase B Step 8 notes in docs/archive/hybrid-engine-pipeline/HYBRID_SLURM_REFERENCE.md).
        if self.id == 0:
            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=10),
                options=[
                    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_BYTES),
                    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_BYTES),
                ],
            )

            flora_grpc_pb2_grpc.add_CentralServerServicer_to_server(
                CentralServerServicer(
                    self.total_clients,
                    model,
                    compressor=self.compressor,
                    accumulate_updates=self.accumulate_updates,
                ),
                self.server,
            )

            listen_addr = f"[::]:{self.master_port}"
            self.server.add_insecure_port(listen_addr)

            print(f"Compatible Scalable Parameter server starting on {listen_addr}")
            self.server.start()

            if daemon_server:
                return

            try:
                while True:
                    time.sleep(86400)
            except KeyboardInterrupt:
                print("Shutting down parameter server...")
                self.server.stop(0)

        else:
            client_id = "client_" + str(self.id)
            self.client = GrpcClient(
                client_id=client_id,
                master_addr=master_addr,
                master_port=self.master_port,
                compressor=self.compressor,
            )

    def grpc_shutdown(self) -> None:
        """Stop the gRPC server if this rank started one (``id == 0``)."""
        if self.server is not None:
            self.server.stop(grace=5)
            self.server = None

    def aggregate(
        self,
        msg: Union[torch.nn.Module, torch.Tensor],
        batch_samples: int,
        communicate_params: bool = True,
        compute_mean: bool = True,
    ):
        if isinstance(msg, torch.nn.Module):
            # communicate either model parameters or gradients
            if communicate_params:
                # for weighted aggregation, summed updates are divided by total_samples on the server
                updates = {
                    name: torch.mul(param.data.detach(), batch_samples)
                    for (name, param) in msg.named_parameters()
                }
            else:
                # for weighted aggregation, summed updates are divided by total_samples on the server
                updates = {
                    name: torch.mul(param.grad.detach(), batch_samples)
                    for (name, param) in msg.named_parameters()
                }

            if self.id != 0:
                self.client.send_update_to_server(
                    updates=updates, batch_samples=batch_samples
                )
                msg = self.client.get_averaged_model(
                    msg=msg, communicate_params=communicate_params
                )
                self.client.round_number += 1

        else:
            raise NotImplementedError(
                "handle other types than torch.nn.Module for aggregation in gRPC!!!"
            )

        return msg
