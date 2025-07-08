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

from concurrent import futures
from typing import Union

import grpc
import torch
from torch import nn

from . import Communicator, grpc_communicator_pb2_grpc
from .grpc_client import GrpcClient
from .grpc_server import CentralServerServicer


class GrpcCommunicator(Communicator):
    def __init__(
        self,
        local_rank: int,
        world_size: int,
        master_addr: str = "127.0.0.1",
        master_port: int = 50051,
        accumulate_updates: bool = True,
        max_workers: int = 10,
        max_send_message_length: int = 100 * 1024 * 1024,  # 100 MB
        max_receive_message_length: int = 100 * 1024 * 1024,  # 100 MB
        retry_delay: float = 2.0,  # Seconds to wait between retries
        max_retries: int = 30,  # Maximum number of retries before giving up
        **kwargs,
    ):
        print(f"{self.__class__.__name__} init...")
        # self.model: nn.Module = model

        self.local_rank: int = local_rank
        self.world_size: int = world_size

        self.master_addr: str = master_addr
        self.master_port: int = master_port

        self.accumulate_updates: bool = accumulate_updates

        # ---
        # configurable number of gRPC server threads
        self.max_workers = max_workers
        # configurable gRPC buffer sizes
        self.max_send_message_length = max_send_message_length
        self.max_receive_message_length = max_receive_message_length
        # configurable retry behavior
        self.retry_delay = retry_delay
        self.max_retries = max_retries

        # ---
        self.server = None
        self.client = None

        print(f"GrpcCommunicator initialized: {dict(locals())}")

    def setup(self, model: nn.Module):
        """Initialize the gRPC communicator."""
        print(f"{self.__class__.__name__} setup...")
        if self.local_rank == 0:
            # grpc send and receive message length & thread pool size from constructor args
            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers),
                options=[
                    ("grpc.max_send_message_length", self.max_send_message_length),
                    (
                        "grpc.max_receive_message_length",
                        self.max_receive_message_length,
                    ),
                ],
            )

            # Add the server servicer with the model
            grpc_communicator_pb2_grpc.add_CentralServerServicer_to_server(
                CentralServerServicer(
                    model,
                    num_clients=self.world_size - 1,  # Exclude server from client count
                    accumulate_updates=self.accumulate_updates,
                ),
                self.server,
            )

            self.server.add_insecure_port(f"[::]:{self.master_port}")
            print(f"gRPC parameter server binding to [::]:{self.master_port}")
            # Start server without blocking the calling thread
            self.server.start()
            print("gRPC parameter server running, background threads ready")

            # try:
            #     # while True:
            #     #     time.sleep(86400)
            #     # Block until the server is stopped or a KeyboardInterrupt occurs
            #     self.server.wait_for_termination()
            # except KeyboardInterrupt:
            #     print("Shutting down parameter server...")
            #     self.server.stop(0)

        else:
            self.client = GrpcClient(
                client_id="client_" + str(self.local_rank),
                master_addr=self.master_addr,
                master_port=self.master_port,
                max_send_message_length=self.max_send_message_length,
                max_receive_message_length=self.max_receive_message_length,
                retry_delay=self.retry_delay,
                max_retries=self.max_retries,
            )

    def broadcast(
        self,
        msg: Communicator.MsgT,
        src: int = 0,
    ) -> Communicator.MsgT:
        """
        Get the current global model without contributing an update.

        In gRPC client-server topology:
        - Server (src=0): Returns model as-is (already has global model)
        - Clients: Fetch current global model from server

        This enables initial model synchronization and round-start model distribution.
        """
        print(f"gRPC broadcast from src={src} | {type(msg)}")

        if self.local_rank == src:
            # Server node: already has the global model
            return msg
        else:
            # Client node: fetch current global model from server
            if self.client is None:
                raise RuntimeError(
                    "gRPC client is not initialized. Call setup() first."
                )

            # Save current round number and use special value for broadcast
            saved_round = self.client.round_number
            self.client.round_number = -1  # Special value indicates broadcast request

            try:
                # Get current model from server (not waiting for aggregation)
                msg = self.client.get_averaged_model(msg=msg, communicate_params=True)
                return msg
            finally:
                # Restore original round number
                self.client.round_number = saved_round

    def aggregate(
        self,
        msg: Communicator.MsgT,
        local_samples: int,
        communicate_params: bool = True,
        compute_mean: bool = True,
    ) -> Communicator.MsgT:
        """
        Aggregate model updates across all clients.

        In gRPC client-server topology:
        - Clients: Send weighted update to server, get back aggregated model
        - Server: No-op (aggregation happens in server servicer)

        Args:
            msg: Model, tensor, or dict to aggregate
            communicate_params: If True, aggregate model parameters; if False, aggregate gradients
            compute_mean: If True, compute mean (handled by server); if False, sum aggregation
            local_samples: Number of local samples for weighted aggregation (required for models)
        """
        print(
            f"gRPC aggregate | {type(msg)} communicate_params={communicate_params} compute_mean={compute_mean}"
        )

        if self.local_rank == 0:
            # Server node: aggregation handled by server servicer, return model as-is
            return msg

        # Client node: send update and get aggregated result
        if self.client is None:
            raise RuntimeError("gRPC client is not initialized. Call setup() first.")

        if isinstance(msg, torch.nn.Module):
            # client-side scaling
            # server expects weighted updates
            # server does (sum(param * samples)) / total_samples
            if communicate_params:
                updates = {
                    name: torch.mul(param.data.detach(), local_samples)
                    for (name, param) in msg.named_parameters()
                }
            else:
                updates = {
                    name: torch.mul(param.grad.detach(), local_samples)
                    for (name, param) in msg.named_parameters()
                }

            # Send update and get aggregated model
            self.client.send_update_to_server(
                updates=updates, local_samples=local_samples
            )
            msg = self.client.get_averaged_model(
                msg=msg, communicate_params=communicate_params
            )
            self.client.round_number += 1
            return msg

        raise NotImplementedError(
            f"Aggregation not implemented for type {type(msg)} in gRPC communicator. "
            f"Only torch.nn.Module is supported."
        )

    def send(
        self,
        msg: Communicator.MsgT,
        dst: int,
        communicate_params: bool = True,
    ) -> Communicator.MsgT:
        print(f"gRPC send to dst={dst} | {type(msg)}")
        raise NotImplementedError()

    def receive(
        self,
        msg: Communicator.MsgT,
        src: int,
        communicate_params: bool = True,
    ) -> Communicator.MsgT:
        print(f"gRPC receive from src={src} | {type(msg)}")
        raise NotImplementedError()

    def collect(
        self,
        msg: Union[nn.Module, torch.Tensor, float, int],
        communicate_params: bool = True,
    ) -> list:
        print(f"gRPC collect | {type(msg)}")
        raise NotImplementedError()

    def close(self):
        """
        Clean up gRPC resources.
        """
        print("Closing gRPC communicator")

        if self.server is not None:
            print("Stopping gRPC server...")
            self.server.stop(grace=5)  # 5 second grace period
            print("gRPC server stopped")

        if self.client is not None:
            print("Closing gRPC client channel...")
            self.client.channel.close()
            print("gRPC client channel closed")

        print("gRPC communicator closed")
