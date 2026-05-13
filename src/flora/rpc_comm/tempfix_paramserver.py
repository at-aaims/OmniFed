import grpc
from concurrent import futures
import threading
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
import parameter_server_pb2
import parameter_server_pb2_grpc


class SimpleModel(nn.Module):
    """Simple neural network for demonstration"""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CompatibleScalableParameterServerServicer(
    parameter_server_pb2_grpc.ParameterServerServicer
):
    def __init__(
        self,
        num_clients: int,
        model: nn.Module,
        use_compression: bool = True,
        accumulate_gradients: bool = True,
    ):
        self.num_clients = num_clients
        self.model = model
        self.use_compression = use_compression
        self.accumulate_gradients = accumulate_gradients

        self.registered_clients = set()
        self.current_round = 0
        self.lock = threading.Lock()

        # For gradient accumulation approach
        if accumulate_gradients:
            self.accumulated_gradients = {}
            self.gradient_count = 0
            self._initialize_accumulated_gradients()
        else:
            # Traditional approach - store all client updates
            self.client_updates = {}
            self.averaged_models_history = {}

        print(
            f"Compatible Scalable Parameter Server initialized for {num_clients} clients"
        )
        print(
            f"Compression: {use_compression}, Gradient Accumulation: {accumulate_gradients}"
        )

    def _initialize_accumulated_gradients(self):
        """Initialize accumulated gradients to zero"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.accumulated_gradients[name] = torch.zeros_like(param)

    def _model_params_to_protobuf_efficient(self, model_params: Dict):
        """Convert model parameters to protobuf format efficiently"""
        proto_layers = []

        for param_name, param_tensor in model_params.items():
            # Split parameter name to get layer and type
            parts = param_name.split(".")
            if len(parts) >= 2:
                layer_name = ".".join(parts[:-1])  # Everything except last part
                param_type = parts[-1]  # weight or bias
            else:
                layer_name = param_name
                param_type = "weight"

            # Find existing layer or create new one
            layer_proto = None
            for layer in proto_layers:
                if layer.layer_name == layer_name:
                    layer_proto = layer
                    break

            if layer_proto is None:
                layer_proto = parameter_server_pb2.LayerParameters(
                    layer_name=layer_name
                )
                proto_layers.append(layer_proto)

            # Add parameter data
            param_data = param_tensor.cpu().numpy()
            if param_type == "weight":
                layer_proto.weights.extend(param_data.flatten().tolist())
                layer_proto.weight_shape.extend(list(param_data.shape))
            elif param_type == "bias":
                layer_proto.biases.extend(param_data.flatten().tolist())
                layer_proto.bias_shape.extend(list(param_data.shape))

        return proto_layers

    def _protobuf_to_model_params_efficient(self, proto_layers):
        """Convert protobuf layers to model parameters efficiently"""
        model_params = {}

        for layer in proto_layers:
            layer_name = layer.layer_name

            if layer.weights:
                weights = np.array(layer.weights).reshape(tuple(layer.weight_shape))
                model_params[f"{layer_name}.weight"] = torch.tensor(
                    weights, dtype=torch.float32
                )

            if layer.biases:
                biases = np.array(layer.biases).reshape(tuple(layer.bias_shape))
                model_params[f"{layer_name}.bias"] = torch.tensor(
                    biases, dtype=torch.float32
                )

        return model_params

    def SendUpdate(self, request, context):
        """Receive model update from client"""
        with self.lock:
            client_id = request.client_id
            round_number = request.round_number

            try:
                # Convert protobuf to model parameters
                update_data = self._protobuf_to_model_params_efficient(request.layers)

                if self.accumulate_gradients:
                    # Gradient accumulation approach - more memory efficient
                    self._accumulate_gradient_update(update_data)
                    self.gradient_count += 1

                    print(
                        f"Accumulated gradient from {client_id} for round {round_number}. "
                        f"Count: {self.gradient_count}/{self.num_clients}"
                    )

                    # Check if we have all gradients
                    if self.gradient_count == self.num_clients:
                        print(
                            f"All gradients received for round {round_number}. Applying average..."
                        )
                        self._apply_averaged_gradients()
                        self._reset_gradient_accumulation()
                        self.current_round = round_number

                else:
                    # Traditional approach - store all updates
                    if round_number not in self.client_updates:
                        self.client_updates[round_number] = {}

                    self.client_updates[round_number][client_id] = update_data
                    updates_received = len(self.client_updates[round_number])

                    print(
                        f"Received update from {client_id} for round {round_number}. "
                        f"Updates: {updates_received}/{self.num_clients}"
                    )

                    if updates_received == self.num_clients:
                        print(
                            f"All updates received for round {round_number}. Computing average..."
                        )
                        self._compute_averaged_model(round_number)
                        self.current_round = round_number
                        self._cleanup_old_rounds(round_number)

                return parameter_server_pb2.UpdateResponse(
                    success=True,
                    message="Update received successfully",
                    clients_registered=len(self.registered_clients),
                    updates_received=self.gradient_count
                    if self.accumulate_gradients
                    else len(self.client_updates.get(round_number, {})),
                )

            except Exception as e:
                print(f"Error processing update from {client_id}: {e}")
                return parameter_server_pb2.UpdateResponse(
                    success=False,
                    message=f"Error processing update: {str(e)}",
                    clients_registered=len(self.registered_clients),
                    updates_received=0,
                )

    def _accumulate_gradient_update(self, update_data: Dict):
        """Accumulate gradient updates for memory efficiency"""
        print(f"DEBUG: Accumulating gradient updates for round {self.current_round}")
        with torch.no_grad():
            for name, gradient in update_data.items():
                if name in self.accumulated_gradients:
                    self.accumulated_gradients[name] += gradient

    def _apply_averaged_gradients(self):
        """Apply averaged gradients to the model"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.accumulated_gradients:
                    # Average the accumulated gradients
                    avg_gradient = self.accumulated_gradients[name] / self.num_clients
                    # Apply the averaged gradient (simplified SGD)
                    param.data -= 0.01 * avg_gradient

    def _reset_gradient_accumulation(self):
        """Reset gradient accumulation for next round"""
        self.gradient_count = 0
        with torch.no_grad():
            for name in self.accumulated_gradients:
                self.accumulated_gradients[name].zero_()

    def GetAveragedModel(self, request, context):
        """Send current model to client using existing protobuf format"""
        with self.lock:
            client_id = request.client_id
            round_number = request.round_number

            try:
                if self.accumulate_gradients:
                    # For gradient accumulation, always send current model
                    if self.current_round >= round_number:
                        model_params = {}
                        for name, param in self.model.named_parameters():
                            model_params[name] = param.data.clone()

                        # Convert to protobuf format
                        proto_layers = self._model_params_to_protobuf_efficient(
                            model_params
                        )

                        print(
                            f"Sending current model to {client_id} for round {round_number}"
                        )

                        return parameter_server_pb2.ModelParameters(
                            round_number=round_number,
                            layers=proto_layers,
                            is_ready=True,
                        )
                else:
                    # Traditional approach
                    if round_number in self.averaged_models_history:
                        averaged_model = self.averaged_models_history[round_number]
                        proto_layers = self._model_params_to_protobuf_efficient(
                            averaged_model
                        )

                        print(
                            f"Sending averaged model to {client_id} for round {round_number}"
                        )

                        return parameter_server_pb2.ModelParameters(
                            round_number=round_number,
                            layers=proto_layers,
                            is_ready=True,
                        )

                return parameter_server_pb2.ModelParameters(
                    round_number=round_number, layers=[], is_ready=False
                )

            except Exception as e:
                print(f"Error sending model to {client_id}: {e}")
                return parameter_server_pb2.ModelParameters(
                    round_number=round_number, layers=[], is_ready=False
                )

    def _compute_averaged_model(self, round_number):
        """Compute averaged model from all client updates"""
        client_params = list(self.client_updates[round_number].values())
        averaged_params = {}

        # Get parameter names from first client
        param_names = client_params[0].keys()

        with torch.no_grad():
            for param_name in param_names:
                # Average parameters across all clients
                param_sum = sum(params[param_name] for params in client_params)
                averaged_params[param_name] = param_sum / len(client_params)

        # Store the averaged model for this round
        self.averaged_models_history[round_number] = averaged_params

        # Update the server's model
        for name, param in self.model.named_parameters():
            if name in averaged_params:
                param.data = averaged_params[name]

        print(f"Averaged model computed for round {round_number}")

    def _cleanup_old_rounds(self, current_round, keep_history_rounds=1):
        """Clean up old rounds to prevent memory leaks"""
        if not self.accumulate_gradients:
            rounds_to_remove = []

            for round_num in self.client_updates.keys():
                if round_num < current_round - keep_history_rounds:
                    rounds_to_remove.append(round_num)

            for round_num in rounds_to_remove:
                del self.client_updates[round_num]

            rounds_to_remove = []
            for round_num in self.averaged_models_history.keys():
                if round_num < current_round - keep_history_rounds:
                    rounds_to_remove.append(round_num)

            for round_num in rounds_to_remove:
                del self.averaged_models_history[round_num]

    def RegisterClient(self, request, context):
        """Register a new client"""
        with self.lock:
            self.registered_clients.add(request.client_id)
            total_clients = len(self.registered_clients)

            print(
                f"Client {request.client_id} registered. Total clients: {total_clients}"
            )

            return parameter_server_pb2.RegistrationResponse(
                success=True,
                message=f"Client {request.client_id} registered successfully",
                total_clients=total_clients,
            )


def serve(port=50051, num_clients=3, accumulate_gradients=True):
    """Start the compatible scalable parameter server"""
    # Example model - replace with your actual model
    # model = nn.Sequential(
    #     nn.Linear(784, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, 256),
    #     nn.ReLU(),
    #     nn.Linear(256, 10)
    # )

    model = SimpleModel()

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
        ],
    )

    parameter_server_pb2_grpc.add_ParameterServerServicer_to_server(
        CompatibleScalableParameterServerServicer(
            num_clients, model, accumulate_gradients=accumulate_gradients
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
    serve()
