# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

import grpc
from concurrent import futures
import threading
from typing import Dict, Optional

import torch

import src.omnifed.hybrid.communicator.global_grpc_pb2 as global_grpc_pb2
import src.omnifed.hybrid.communicator.global_grpc_pb2_grpc as global_grpc_pb2_grpc
from src.omnifed.hybrid.communicator.global_grpc_compression import (
    decode_updates_dict,
    encode_layer_state,
)
from src.omnifed.hybrid.communicator.global_grpc_limits import GRPC_MAX_MESSAGE_BYTES
from src.omnifed.hybrid.compression.topk import TopKCompression


class CentralServerServicer(global_grpc_pb2_grpc.CentralServerServicer):
    def __init__(
        self,
        num_clients: int,
        model: torch.nn.Module,
        compressor: Optional[TopKCompression] = None,
        accumulate_updates: bool = True,
        communicate_params: bool = True,
        compute_mean: bool = True,
    ):
        self.num_clients = num_clients
        self.model = model
        self.compressor = compressor
        self.accumulate_updates = accumulate_updates
        self.communicate_params = communicate_params
        self.compute_mean = compute_mean

        self.registered_clients = set()
        self.current_round = -1
        self.lock = threading.Lock()

        self.round_complete_event = threading.Event()
        self.round_in_progress = -1

        if accumulate_updates:
            self.accumulated_updates = {}
            self.update_count = 0
            self.total_samples = 0
            self._initialize_accumulated_updates()

        print(
            f"Compatible Scalable Parameter Server initialized for {num_clients} clients"
        )
        print(
            f"Compression: {compressor is not None}, Updates' Accumulation: {accumulate_updates}"
        )

    def _initialize_accumulated_updates(self):
        """Initialize accumulated gradients to zero"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.accumulated_updates[name] = torch.zeros_like(param)

    def _model_updates_to_protobuf_efficient(self, model_updates: Dict):
        """Convert model tensors to protobuf (dense or TopK)."""
        return [
            encode_layer_state(name, tensor, self.compressor)
            for name, tensor in model_updates.items()
        ]

    def _protobuf_to_model_params_efficient(self, proto_layers):
        """Convert protobuf layers to dense model tensors."""
        return decode_updates_dict(proto_layers)

    def SendUpdate(self, request, context):
        """Receive model updates from client"""
        with self.lock:
            client_id = request.client_id
            round_number = request.round_number

            try:
                if round_number > self.round_in_progress:
                    print(f"Starting new round {round_number}")
                    self.round_in_progress = round_number
                    self.round_complete_event.clear()
                    self.update_count = 0
                    self.total_samples = 0
                    self._initialize_accumulated_updates()

                if round_number != self.round_in_progress:
                    print(
                        f"Ignoring update from {client_id} for round {round_number}, current round is {self.round_in_progress}"
                    )
                    return global_grpc_pb2.UpdateResponse(
                        success=False,
                        message=f"Round {round_number} is not the current round ({self.round_in_progress})",
                        clients_registered=len(self.registered_clients),
                        updates_received=self.update_count,
                    )

                update_data = self._protobuf_to_model_params_efficient(request.layers)

                if self.accumulate_updates:
                    self._accumulate_model_updates(update_data)
                    self.update_count += 1
                    self.total_samples += request.number_samples
                    print(
                        f"Accumulated gradient from {client_id} for round {round_number}. "
                        f"Count: {self.update_count}/{self.num_clients}, "
                        f"Total samples: {self.total_samples}"
                    )

                    if self.update_count == self.num_clients:
                        print(
                            f"All gradients received for round {round_number}. Applying average..."
                            f"Total samples across all clients: {self.total_samples}"
                        )
                        self._apply_model_updates()
                        print(
                            f"DEBUG:successfully applied updates for round {round_number}"
                        )

                        self.current_round = round_number
                        self.round_complete_event.set()

                        print(
                            f"DEBUG:successfully completed round {self.current_round}"
                        )

                return global_grpc_pb2.UpdateResponse(
                    success=True,
                    message="Update received successfully",
                    clients_registered=len(self.registered_clients),
                    updates_received=self.update_count,
                )

            except Exception as e:
                print(f"Error processing update from {client_id}: {e}")
                return global_grpc_pb2.UpdateResponse(
                    success=False,
                    message=f"Error processing update: {str(e)}",
                    clients_registered=len(self.registered_clients),
                    updates_received=0,
                )

    def _accumulate_model_updates(self, update_data: Dict):
        """Accumulate model updates from clients for better memory efficiency"""
        print(f"DEBUG: Accumulating model updates for round {self.round_in_progress}")
        with torch.no_grad():
            for name, update in update_data.items():
                if name in self.accumulated_updates:
                    self.accumulated_updates[name] += update

    def _apply_model_updates(self):
        """Apply model updates using total sample count for proper averaging"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.accumulated_updates:
                    if self.compute_mean:
                        avg_update = self.accumulated_updates[name] / self.total_samples
                        print(
                            f"DEBUG: Averaging {name} with total_samples={self.total_samples}"
                        )
                    else:
                        avg_update = self.accumulated_updates[name]

                    if self.communicate_params:
                        param.data = avg_update
                    else:
                        param.grad = avg_update

    def GetUpdatedModel(self, request, context):
        """Send current model to client using existing protobuf format"""
        client_id = request.client_id
        round_number = request.round_number

        try:
            print(f"Client {client_id} requesting model for round {round_number}")

            with self.lock:
                if round_number <= self.current_round:
                    print(
                        f"Round {round_number} already completed (current: {self.current_round}), sending model to {client_id}"
                    )
                    return self._send_current_model(round_number, client_id)

                if round_number != self.round_in_progress:
                    print(
                        f"Round {round_number} not in progress (current: {self.round_in_progress})"
                    )
                    return global_grpc_pb2.ModelParameters(
                        round_number=round_number, layers=[], is_ready=False
                    )

            print(f"Client {client_id} waiting for round {round_number} to complete...")
            if self.round_complete_event.wait(timeout=10):
                with self.lock:
                    if round_number <= self.current_round:
                        return self._send_current_model(round_number, client_id)

            print(f"Timeout waiting for round {round_number} for client {client_id}")
            return global_grpc_pb2.ModelParameters(
                round_number=round_number, layers=[], is_ready=False
            )

        except Exception as e:
            print(f"Error sending model to {client_id}: {e}")
            return global_grpc_pb2.ModelParameters(
                round_number=round_number, layers=[], is_ready=False
            )

    def _send_current_model(self, round_number, client_id):
        """Helper method to send the current model"""
        if self.accumulated_updates is not None:
            model_updates = {}
            for name, param in self.model.named_parameters():
                if self.communicate_params:
                    model_updates[name] = param.data
                else:
                    model_updates[name] = param.grad

            proto_layers = self._model_updates_to_protobuf_efficient(model_updates)
            print(f"Sending current model to {client_id} for round {round_number}")

            return global_grpc_pb2.ModelParameters(
                round_number=round_number,
                layers=proto_layers,
                is_ready=True,
            )

        return global_grpc_pb2.ModelParameters(
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

            return global_grpc_pb2.RegistrationResponse(
                success=True,
                message=f"Client {request.client_id} registered successfully",
                total_clients=total_clients,
            )
