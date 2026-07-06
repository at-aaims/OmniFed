# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

import time
import grpc
from concurrent import futures
from typing import Optional, Union

import torch
import torch.nn

from src.omnifed.hybrid.communicator import Communicator
import src.omnifed.hybrid.communicator.global_grpc_pb2_grpc as global_grpc_pb2_grpc

from src.omnifed.hybrid.communicator.global_grpc_limits import GRPC_MAX_MESSAGE_BYTES
from src.omnifed.hybrid.communicator.global_grpc_server import CentralServerServicer
from src.omnifed.hybrid.communicator.global_grpc_client import GrpcClient
from src.omnifed.hybrid.communicator.global_grpc_compression import GlobalHybridCompressor


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
        compressor: Optional[GlobalHybridCompressor] = None,
        communicate_params: bool = True,
    ):
        super().__init__(protocol_type="RPC")
        self.id = id
        self.total_clients = total_clients - 1
        self.master_port = master_port
        self.accumulate_updates = accumulate_updates
        self.compressor = compressor
        self.communicate_params = bool(communicate_params)
        self.server = None

        # Only id == 0 starts the daemon gRPC server (communicator role id, not SLURM_PROCID).
        if self.id == 0:
            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=10),
                options=[
                    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_BYTES),
                    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_BYTES),
                ],
            )

            global_grpc_pb2_grpc.add_CentralServerServicer_to_server(
                CentralServerServicer(
                    self.total_clients,
                    model,
                    compressor=self.compressor,
                    accumulate_updates=self.accumulate_updates,
                    communicate_params=self.communicate_params,
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
            if communicate_params:
                updates = {
                    name: torch.mul(param.data.detach(), batch_samples)
                    for (name, param) in msg.named_parameters()
                }
            else:
                missing = [
                    name
                    for name, param in msg.named_parameters()
                    if param.grad is None
                ]
                if missing:
                    raise RuntimeError(
                        "gRPC gradient aggregation requires param.grad on every parameter "
                        f"before sync (missing {len(missing)} tensors, e.g. {missing[:3]}). "
                        "Use aggregate_payload=gradients only after the deferred-optimizer "
                        "training path is enabled."
                    )
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
