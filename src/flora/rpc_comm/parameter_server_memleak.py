import grpc
from concurrent import futures
import threading
import time
import torch
import torch.nn as nn
import numpy as np
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


class ParameterServerServicer(parameter_server_pb2_grpc.ParameterServerServicer):
    def __init__(self, num_clients: int, model: nn.Module):
        self.num_clients = num_clients
        self.model = model
        self.registered_clients = set()
        self.client_updates = {}
        self.current_round = 0
        self.averaged_model = None
        self.lock = threading.Lock()

        print(f"Parameter server initialized for {num_clients} clients")

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

    def SendUpdate(self, request, context):
        """Receive model update from client"""
        with self.lock:
            client_id = request.client_id
            round_number = request.round_number

            # Store the update
            if round_number not in self.client_updates:
                self.client_updates[round_number] = {}

            # Convert protobuf to model parameters
            model_params = self._protobuf_to_model_params(request.layers)
            self.client_updates[round_number][client_id] = model_params

            updates_received = len(self.client_updates[round_number])

            print(
                f"Received update from {client_id} for round {round_number}. "
                f"Updates received: {updates_received}/{self.num_clients}"
            )

            # Check if we have all updates for this round
            if updates_received == self.num_clients:
                print(
                    f"All updates received for round {round_number}. Computing average..."
                )
                self._compute_averaged_model(round_number)
                self.current_round = round_number

            return parameter_server_pb2.UpdateResponse(
                success=True,
                message="Update received successfully",
                clients_registered=len(self.registered_clients),
                updates_received=updates_received,
            )

    def GetAveragedModel(self, request, context):
        """Send averaged model to client"""
        with self.lock:
            client_id = request.client_id
            round_number = request.round_number

            # Check if averaged model is ready for this round
            if (
                self.averaged_model is not None
                and round_number in self.client_updates
                and len(self.client_updates[round_number]) == self.num_clients
            ):
                # Convert model parameters to protobuf
                proto_layers = self._model_params_to_protobuf(self.averaged_model)

                print(f"Sending averaged model to {client_id} for round {round_number}")

                return parameter_server_pb2.ModelParameters(
                    round_number=round_number, layers=proto_layers, is_ready=True
                )
            else:
                return parameter_server_pb2.ModelParameters(
                    round_number=round_number, layers=[], is_ready=False
                )

    def _protobuf_to_model_params(self, proto_layers):
        """Convert protobuf layers to PyTorch model parameters"""
        model_params = {}
        for layer in proto_layers:
            layer_name = layer.layer_name

            # Reshape weights and biases
            weights = np.array(layer.weights).reshape(layer.weight_shape)
            biases = np.array(layer.biases).reshape(layer.bias_shape)

            model_params[layer_name] = {
                "weight": torch.tensor(weights, dtype=torch.float32),
                "bias": torch.tensor(biases, dtype=torch.float32),
            }

        return model_params

    def _model_params_to_protobuf(self, model_params):
        """Convert PyTorch model parameters to protobuf"""
        proto_layers = []

        for layer_name, params in model_params.items():
            weight_tensor = params["weight"]
            bias_tensor = params["bias"]

            layer_proto = parameter_server_pb2.LayerParameters(
                layer_name=layer_name,
                weights=weight_tensor.flatten().tolist(),
                biases=bias_tensor.flatten().tolist(),
                weight_shape=list(weight_tensor.shape),
                bias_shape=list(bias_tensor.shape),
            )
            proto_layers.append(layer_proto)

        return proto_layers

    def _compute_averaged_model(self, round_number):
        """Compute averaged model from all client updates"""
        client_params = list(self.client_updates[round_number].values())
        averaged_params = {}

        # Get layer names from first client
        layer_names = client_params[0].keys()

        for layer_name in layer_names:
            # Average weights
            weight_sum = sum(params[layer_name]["weight"] for params in client_params)
            weight_avg = weight_sum / len(client_params)

            # Average biases
            bias_sum = sum(params[layer_name]["bias"] for params in client_params)
            bias_avg = bias_sum / len(client_params)

            averaged_params[layer_name] = {"weight": weight_avg, "bias": bias_avg}

        self.averaged_model = averaged_params
        print(f"Averaged model computed for round {round_number}")


def serve(port=50051, num_clients=3):
    """Start the parameter server"""
    model = SimpleModel()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    parameter_server_pb2_grpc.add_ParameterServerServicer_to_server(
        ParameterServerServicer(num_clients, model), server
    )

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    print(f"Parameter server starting on {listen_addr}")
    server.start()

    try:
        while True:
            time.sleep(86400)  # Sleep for a day
    except KeyboardInterrupt:
        print("Shutting down parameter server...")
        server.stop(0)


if __name__ == "__main__":
    serve()
