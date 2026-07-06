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

import grpc
from typing import Dict, Optional

import torch

import src.flora.communicator.grpc_communicator_pb2 as flora_grpc_pb2
import src.flora.communicator.grpc_communicator_pb2_grpc as flora_grpc_pb2_grpc
from src.flora.communicator.grpc_compression_utils import (
    decode_layer_tensor,
    encode_layer_state,
)
from src.flora.communicator.grpc_limits import GRPC_MAX_MESSAGE_BYTES
from src.flora.compression.sparsification import TopKCompression


class GrpcClient:
    def __init__(
        self,
        client_id: str,
        master_addr: str = "127.0.0.1",
        master_port: int = 50051,
        compressor: Optional[TopKCompression] = None,
    ):
        self.client_id = client_id
        self.compressor = compressor
        self._last_updates: Optional[Dict[str, torch.Tensor]] = None
        self.channel = grpc.insecure_channel(
            master_addr + ":" + str(master_port),
            options=[
                ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_BYTES),
                ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_BYTES),
            ],
        )
        self.stub = flora_grpc_pb2_grpc.CentralServerStub(self.channel)
        self.round_number = 0

        mode = "TopK" if compressor is not None else "dense"
        print(
            f"Client {client_id} initialized ({mode}), connecting to {master_addr}:{master_port}"
        )
        self._register_with_server()

    def _register_with_server(self):
        """Register this client with the parameter server"""
        try:
            request = flora_grpc_pb2.ClientInfo(client_id=self.client_id)
            response = self.stub.RegisterClient(request)

            if response.success:
                print(
                    f"Successfully registered with server. Total clients: {response.total_clients}"
                )
            else:
                print(f"Failed to register with server: {response.message}")

        except grpc.RpcError as e:
            print(f"Failed to connect to server: {e}")

    def _model_params_to_protobuf(self, updates: Dict):
        """Convert model parameters to protobuf format (dense or TopK)."""
        return [
            encode_layer_state(name, tensor, self.compressor)
            for name, tensor in updates.items()
        ]

    def send_update_to_server(self, updates: Dict, batch_samples: int):
        """Send model update to parameter server"""
        try:
            self._last_updates = {k: v.detach().clone() for k, v in updates.items()}
            proto_layers = self._model_params_to_protobuf(updates)

            request = flora_grpc_pb2.ModelUpdate(
                client_id=self.client_id,
                round_number=self.round_number,
                layers=proto_layers,
                number_samples=batch_samples,
            )

            response = self.stub.SendUpdate(request)

            if response.success:
                print(
                    f"Round {self.round_number}: Update sent successfully. "
                    f"Updates received: {response.updates_received}/{response.clients_registered}"
                )
                return True
            else:
                print(f"Failed to send update: {response.message}")
                return False

        except grpc.RpcError as e:
            print(f"Failed to send update to server: {e}")
            return False

    def _update_model_from_protobuf(self, communicate_params, model, proto_layers):
        """Update model parameters from protobuf format"""
        layer_by_name = {layer.layer_name: layer for layer in proto_layers}
        with torch.no_grad():
            for name, param in model.named_parameters():
                layer = layer_by_name.get(name)
                if layer is None:
                    continue
                base = param.data if communicate_params else param.grad
                decoded = decode_layer_tensor(layer, base_tensor=base)
                target = param.data if communicate_params else param.grad
                target.copy_(decoded.to(device=target.device, dtype=target.dtype))

        return model

    def get_averaged_model(self, msg: torch.nn.Module, communicate_params: bool):
        """Get averaged model from parameter server - wait indefinitely until ready"""
        print(f"Round {self.round_number}: Waiting for averaged model from server...")
        while True:
            try:
                request = flora_grpc_pb2.GetModelRequest(
                    client_id=self.client_id, round_number=self.round_number
                )
                response = self.stub.GetUpdatedModel(request)

                if response.is_ready:
                    msg = self._update_model_from_protobuf(
                        communicate_params=communicate_params,
                        model=msg,
                        proto_layers=response.layers,
                    )
                    print(
                        f"Round {self.round_number}: Received averaged model from server"
                    )
                    return msg
                else:
                    print(
                        f"Round {self.round_number}: Averaged model not ready, waiting 2 seconds..."
                    )

            except grpc.RpcError as e:
                print(f"Failed to get averaged model (will retry): {e}")
                continue
