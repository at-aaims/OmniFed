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
from typing import Dict

import grpc
import numpy as np
import torch

from . import grpc_communicator_pb2 as flora_grpc_pb2
from . import grpc_communicator_pb2_grpc as flora_grpc_pb2_grpc


class GrpcClient:
    def __init__(
        self,
        client_id: str,
        master_addr: str,
        master_port: int,
        max_send_message_length: int,
        max_receive_message_length: int,
        retry_delay: float = 5.0,
        max_retries: int = 10,
    ):
        print(f"{self.__class__.__name__} init...")

        self.client_id = client_id
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        # initialize gRPC channel with configured buffer sizes
        self.channel = grpc.insecure_channel(
            master_addr + ":" + str(master_port),
            options=[
                ("grpc.max_receive_message_length", max_receive_message_length),
                ("grpc.max_send_message_length", max_send_message_length),
            ],
        )
        self.stub = flora_grpc_pb2_grpc.CentralServerStub(self.channel)
        self.round_number = 0

        self.log(f"Initializing client, connecting to {master_addr}:{master_port}")
        self._register_with_server()

    def log(self, message: str):
        print(
            f"[{self.__class__.__name__}] [Client {self.client_id}] [Round {self.round_number}] {message}"
        )

    def _register_with_server(self):
        """Register this client with the parameter server"""
        try:
            request = flora_grpc_pb2.ClientInfo(client_id=self.client_id)
            response = self.stub.RegisterClient(request)

            if response.success:
                self.log(
                    f"Registered with server. Total clients: {response.total_clients}"
                )
            else:
                self.log(f"Registration failed: {response.message}")

        except grpc.RpcError as e:
            self.log(f"Connection to server failed: {e}")

    def _model_params_to_protobuf(self, updates: Dict):
        """Convert model parameters to protobuf format"""
        proto_layers = []
        for name, tnsr in updates.items():
            tnsr = tnsr.cpu()
            layer_proto = flora_grpc_pb2.LayerState(layer_name=name)
            layer_proto.param_update.extend(tnsr.flatten().tolist())
            layer_proto.param_shape.extend(list(tnsr.shape))

            # print(f"DEBUGGING CLIENT {self.client_id} layer_proto: {layer_proto.param_update} with shape: {layer_proto.param_shape}")
            proto_layers.append(layer_proto)

        return proto_layers

    def send_update_to_server(self, updates: Dict, local_samples: int):
        """Send model update to parameter server"""
        try:
            proto_layers = self._model_params_to_protobuf(updates)

            request = flora_grpc_pb2.ModelUpdate(
                client_id=self.client_id,
                round_number=self.round_number,
                layers=proto_layers,
                number_samples=local_samples,
            )

            response = self.stub.SendUpdate(request)

            if response.success:
                self.log(
                    f"Update sent. Updates received: {response.updates_received}/{response.clients_registered}"
                )
                return True
            else:
                self.log(f"Update failed: {response.message}")
                return False

        except grpc.RpcError as e:
            self.log(f"Update send failed: {e}")
            return False

    def _update_model_from_protobuf(self, communicate_params, model, proto_layers):
        """Update model parameters from protobuf format"""
        with torch.no_grad():
            for (name, param), layer in zip(model.named_parameters(), proto_layers):
                layer_name = layer.layer_name
                if name == layer_name:
                    if communicate_params:
                        param.data = torch.tensor(
                            np.array(layer.param_update).reshape(
                                tuple(layer.param_shape)
                            ),
                            dtype=torch.float32,
                        )
                    else:
                        param.grad = torch.tensor(
                            np.array(layer.param_update).reshape(
                                tuple(layer.param_shape)
                            ),
                            dtype=torch.float32,
                        )

        return model

    def get_global_model(self, msg: torch.nn.Module, communicate_params: bool):
        """Get model from parameter server - retry up to max_retries times, then fail."""
        self.log("Waiting for model from server...")
        attempts = 0
        while attempts < self.max_retries:
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
                    self.log(f"Model received. Layers: {len(response.layers)}")
                    return msg
                else:
                    self.log(
                        f"Model not ready, retrying in {self.retry_delay}s (attempt {attempts + 1}/{self.max_retries})..."
                    )
                    time.sleep(self.retry_delay)
                    attempts += 1
            except grpc.RpcError as e:
                self.log(f"Model fetch failed (will retry): {e}")
                time.sleep(self.retry_delay)
                attempts += 1
        raise RuntimeError(
            f"[{self.__class__.__name__}] [Client {self.client_id}] [Round {self.round_number}] Exceeded maximum retries ({self.max_retries}) waiting for model from server."
        )
