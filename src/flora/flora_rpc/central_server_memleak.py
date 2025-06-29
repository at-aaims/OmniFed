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
import threading
from typing import Dict

import torch
import numpy as np

from src.flora.flora_rpc import SimpleModel
import flora_grpc_pb2
import flora_grpc_pb2_grpc


class CentralServerServicer(flora_grpc_pb2_grpc.CentralServerServicer):
    def __init__(
            self,
            num_clients: int,
            model: torch.nn.Module,
            use_compression: bool = True,
            accumulate_updates: bool = True,
            communicate_params: bool = True,
            compute_mean: bool = True,
    ):
        self.num_clients = num_clients
        self.model = model
        self.use_compression = use_compression
        self.accumulate_updates = accumulate_updates
        self.communicate_params = communicate_params
        self.compute_mean = compute_mean

        self.registered_clients = set()
        self.current_round = 0
        self.lock = threading.Lock()

        # Track which rounds have completed averaging
        self.completed_rounds = set()
        self.round_ready_event = {}  # Dictionary to store events for each round

        # For gradient accumulation approach
        if accumulate_updates:
            self.accumulated_updates = {}
            self.update_count = 0
            self._initialize_accumulated_updates()

        print(
            f"Compatible Scalable Parameter Server initialized for {num_clients} clients"
        )
        print(
            f"Compression: {use_compression}, Updates' Accumulation: {accumulate_updates}"
        )

    def _initialize_accumulated_updates(self):
        """Initialize accumulated gradients to zero"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.accumulated_updates[name] = torch.zeros_like(param)

    def _model_updates_to_protobuf_efficient(self, model_updates: Dict):
        """Convert model parameters to protobuf format efficiently"""
        proto_layers = []

        for name, param in model_updates.items():
            # param = param.cpu().detach().numpy()
            param = param.cpu()
            layer_proto = flora_grpc_pb2.LayerState(layer_name=name)
            if self.communicate_params:
                layer_proto.param_update.extend(param.data.flatten().tolist())
                layer_proto.param_shape.extend(list(param.data.shape))
            else:
                layer_proto.param_update.extend(param.grad.flatten().tolist())
                layer_proto.param_shape.extend(list(param.grad.shape))

            proto_layers.append(layer_proto)

        return proto_layers

    def _protobuf_to_model_params_efficient(self, proto_layers):
        """Convert protobuf layers to model parameters efficiently"""
        model_params = {}
        for layer in proto_layers:
            layer_name = layer.layer_name
            model_params[layer_name] = torch.tensor(
                np.array(layer.param_update).reshape(tuple(layer.param_shape)),
                dtype=torch.float32,
            )

        return model_params

    def SendUpdate(self, request, context):
        """Receive model updates from client"""
        with self.lock:
            client_id = request.client_id
            round_number = request.round_number

            try:
                # Convert protobuf to model parameters
                update_data = self._protobuf_to_model_params_efficient(request.layers)

                if self.accumulate_updates:
                    # Updates' accumulation approach - more memory efficient
                    self._accumulate_model_updates(update_data)
                    self.update_count += 1
                    print(
                        f"Accumulated gradient from {client_id} for round {round_number}. "
                        f"Count: {self.update_count}/{self.num_clients}"
                    )

                    if self.update_count == self.num_clients:
                        print(
                            f"All gradients received for round {round_number}. Applying average..."
                        )
                        self._apply_model_updates()
                        # print(f"DEBUG:successfully applied updates for round {round_number}")
                        self._reset_update_accumulation()
                        # print(f"DEBUG:successfully reset accumulated updates for round {round_number}")

                        # Mark this round as completed
                        self.completed_rounds.add(round_number)
                        self.current_round = round_number

                        # Signal that this round is ready
                        if round_number in self.round_ready_event:
                            self.round_ready_event[round_number].set()

                        print(f"DEBUG:successfully set current round {self.current_round}")

                return flora_grpc_pb2.UpdateResponse(
                    success=True,
                    message="Update received successfully",
                    clients_registered=len(self.registered_clients),
                    updates_received=self.update_count,
                )

            except Exception as e:
                print(f"Error processing update from {client_id}: {e}")
                return flora_grpc_pb2.UpdateResponse(
                    success=False,
                    message=f"Error processing update: {str(e)}",
                    clients_registered=len(self.registered_clients),
                    updates_received=0,
                )

    def _accumulate_model_updates(self, update_data: Dict):
        """Accumulate model updates from clients for better memory efficiency"""
        print(f"DEBUG: Accumulating model updates for round {self.current_round}")
        with torch.no_grad():
            for name, update in update_data.items():
                if name in self.accumulated_updates:
                    self.accumulated_updates[name] += update

    def _apply_model_updates(self):
        """Apply model updates"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.accumulated_updates:
                    if self.compute_mean:
                        avg_update = self.accumulated_updates[name] / self.num_clients
                    else:
                        avg_update = self.accumulated_updates[name]

                    if self.communicate_params:
                        param.data = avg_update
                    else:
                        param.grad = avg_update

    def _reset_update_accumulation(self):
        """Reset model update accumulation for next round"""
        self.update_count = 0
        with torch.no_grad():
            for name in self.accumulated_updates.keys():
                self.accumulated_updates[name].zero_()

    def GetUpdatedModel(self, request, context):
        """Send current model to client using existing protobuf format"""
        client_id = request.client_id
        round_number = request.round_number

        try:
            # Check if the requested round is already completed
            with self.lock:
                if round_number in self.completed_rounds:
                    # Round is already completed, send the model immediately
                    return self._send_current_model(round_number, client_id)

                # Round is not completed yet, create an event to wait for it
                if round_number not in self.round_ready_event:
                    self.round_ready_event[round_number] = threading.Event()

                event = self.round_ready_event[round_number]

            # Wait for the round to complete (with timeout)
            print(f"Client {client_id} waiting for round {round_number} to complete...")
            if event.wait(timeout=300):  # 5 minute timeout
                with self.lock:
                    if round_number in self.completed_rounds:
                        return self._send_current_model(round_number, client_id)

            # Timeout or other issue
            print(f"Timeout waiting for round {round_number} for client {client_id}")
            return flora_grpc_pb2.ModelParameters(
                round_number=round_number, layers=[], is_ready=False
            )

        except Exception as e:
            print(f"Error sending model to {client_id}: {e}")
            return flora_grpc_pb2.ModelParameters(
                round_number=round_number, layers=[], is_ready=False
            )

    def _send_current_model(self, round_number, client_id):
        """Helper method to send the current model"""
        if self.accumulated_updates:
            model_updates = {}
            for name, param in self.model.named_parameters():
                if self.communicate_params:
                    model_updates[name] = param.data
                else:
                    model_updates[name] = param.grad

            # Convert to protobuf format
            proto_layers = self._model_updates_to_protobuf_efficient(model_updates)
            print(f"Sending current model to {client_id} for round {round_number}")

            return flora_grpc_pb2.ModelParameters(
                round_number=round_number,
                layers=proto_layers,
                is_ready=True,
            )

        return flora_grpc_pb2.ModelParameters(
            round_number=round_number, layers=[], is_ready=False
        )

    def RegisterClient(self, request, context):
        """Register a new client"""
        with self.lock:
            self.registered_clients.add(request.client_id)
            total_clients = len(self.registered_clients)

            print(
                f"Client {request.client_id} registered. Total clients: {total_clients}"
            )

            return flora_grpc_pb2.RegistrationResponse(
                success=True,
                message=f"Client {request.client_id} registered successfully",
                total_clients=total_clients,
            )


def start_server(model, port=50051, num_clients=3, accumulate_updates=True):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
        ],
    )

    flora_grpc_pb2_grpc.add_CentralServerServicer_to_server(
        CentralServerServicer(
            num_clients, model, accumulate_updates=accumulate_updates
        ),
        server,
    )

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    print(f"Compatible Scalable Parameter server starting on {listen_addr}")
    server.start()

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        print("Shutting down parameter server...")
        server.stop(0)


if __name__ == "__main__":
    start_server(
        model=SimpleModel(), port=50051, num_clients=3, accumulate_updates=True
    )