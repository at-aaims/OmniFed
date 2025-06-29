import grpc
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random

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


class ParameterServerClient:
    def __init__(self, client_id: str, server_address: str = "localhost:50051"):
        self.client_id = client_id
        self.server_address = server_address
        self.channel = grpc.insecure_channel(server_address)
        self.stub = parameter_server_pb2_grpc.ParameterServerStub(self.channel)
        self.model = SimpleModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.round_number = 0

        print(f"Client {client_id} initialized, connecting to {server_address}")
        self._register_with_server()

    def _register_with_server(self):
        """Register this client with the parameter server"""
        try:
            request = parameter_server_pb2.ClientInfo(client_id=self.client_id)
            response = self.stub.RegisterClient(request)

            if response.success:
                print(
                    f"Successfully registered with server. Total clients: {response.total_clients}"
                )
            else:
                print(f"Failed to register with server: {response.message}")
        except grpc.RpcError as e:
            print(f"Failed to connect to server: {e}")

    def _model_params_to_protobuf(self, client_id):
        """Convert model parameters to protobuf format"""
        print(f"DEBUG: _model_params_to_protobuf called for client {client_id}")
        layers = []

        for name, param in self.model.named_parameters():
            print(f"DEBUG: Processing parameter {name} with shape {param.data.shape}")
            layer_name = name.split(".")[0]  # Get layer name (fc1, fc2)
            param_type = name.split(".")[1]  # Get parameter type (weight, bias)

            # Find or create layer entry
            layer_entry = None
            for layer in layers:
                if layer.layer_name == layer_name:
                    layer_entry = layer
                    break

            if layer_entry is None:
                layer_entry = parameter_server_pb2.LayerParameters(
                    layer_name=layer_name
                )
                layers.append(layer_entry)
                print(f"DEBUG: Created new layer entry for {layer_name}")

            # Add weights or biases
            if param_type == "weight":
                weights_list = param.data.flatten().tolist()
                layer_entry.weights.extend(weights_list)
                layer_entry.weight_shape.extend(list(param.data.shape))
                print(
                    f"DEBUG: Added weights for {layer_name}, count: {len(weights_list)}"
                )
            elif param_type == "bias":
                biases_list = param.data.flatten().tolist()
                layer_entry.biases.extend(biases_list)
                layer_entry.bias_shape.extend(list(param.data.shape))
                print(
                    f"DEBUG: Added biases for {layer_name}, count: {len(biases_list)}"
                )

        print(f"DEBUG: Created {len(layers)} layer entries")
        for i, layer in enumerate(layers):
            print(
                f"DEBUG: Layer {i}: {layer.layer_name}, weights: {len(layer.weights)}, biases: {len(layer.biases)}"
            )

        return layers

    def _update_model_from_protobuf(self, proto_layers, client_id):
        """Update model parameters from protobuf format"""
        print(f"DEBUG: _update_model_from_protobuf called for client {client_id}")
        print(f"DEBUG: Received {len(proto_layers)} proto_layers")

        if len(proto_layers) == 0:
            print("ERROR: No proto_layers received!")
            return

        layer_params = {}

        # Convert protobuf to dictionary
        for i, layer in enumerate(proto_layers):
            print(f"DEBUG: Processing layer {i}")
            print(f"DEBUG: Layer name: '{layer.layer_name}'")
            print(f"DEBUG: Weights count: {len(layer.weights)}")
            print(f"DEBUG: Biases count: {len(layer.biases)}")
            print(f"DEBUG: Weight shape: {list(layer.weight_shape)}")
            print(f"DEBUG: Bias shape: {list(layer.bias_shape)}")

            if len(layer.weights) == 0:
                print(f"ERROR: No weights in layer {layer.layer_name}")
                continue

            if len(layer.biases) == 0:
                print(f"ERROR: No biases in layer {layer.layer_name}")
                continue

            layer_name = layer.layer_name
            try:
                # Convert RepeatedScalarContainer to list/tuple for reshape
                weights = torch.tensor(layer.weights).reshape(tuple(layer.weight_shape))
                biases = torch.tensor(layer.biases).reshape(tuple(layer.bias_shape))
                print(f"SUCCESS: from client {client_id} weights: {weights}")
                print(f"SUCCESS: from client {client_id} biases: {biases}")

                layer_params[layer_name] = {"weight": weights, "bias": biases}
            except Exception as e:
                print(f"ERROR: Failed to process layer {layer_name}: {e}")
                continue

        print(f"DEBUG: Successfully processed {len(layer_params)} layers")

        # Update model parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                layer_name = name.split(".")[0]
                param_type = name.split(".")[1]

                if layer_name in layer_params:
                    old_param = param.data.clone()
                    param.data = layer_params[layer_name][param_type]
                    print(
                        f"DEBUG: Updated {name} from {old_param.shape} to {param.data.shape}"
                    )

    def send_update_to_server(self, client_id):
        """Send model update to parameter server"""
        print(f"DEBUG: send_update_to_server called for client {client_id}")
        try:
            proto_layers = self._model_params_to_protobuf(client_id)

            request = parameter_server_pb2.ModelUpdate(
                client_id=self.client_id,
                round_number=self.round_number,
                layers=proto_layers,
            )

            print(f"DEBUG: Sending request with {len(proto_layers)} layers")
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

    def get_averaged_model(self, client_id):
        """Get averaged model from parameter server - wait indefinitely until ready"""
        print(f"DEBUG: get_averaged_model called for client {client_id}")
        print(f"Round {self.round_number}: Waiting for averaged model from server...")

        while True:
            try:
                request = parameter_server_pb2.GetModelRequest(
                    client_id=self.client_id, round_number=self.round_number
                )

                print(f"DEBUG: Requesting model for round {self.round_number}")
                response = self.stub.GetAveragedModel(request)

                print(f"DEBUG: Received response, is_ready: {response.is_ready}")
                print(f"DEBUG: Response has {len(response.layers)} layers")
                print(
                    f"DEBUG: Response has serialized_data: {len(response.serialized_data) if response.serialized_data else 0} bytes"
                )

                if response.is_ready:
                    if response.layers:
                        print("DEBUG: Using protobuf layers format")
                        self._update_model_from_protobuf(response.layers, client_id)
                        print(
                            f"Round {self.round_number}: Received averaged model from server"
                        )
                        return True
                    elif response.serialized_data:
                        print(
                            "DEBUG: Server sent serialized_data but client doesn't support it yet"
                        )
                        return False
                    else:
                        print("ERROR: Response is ready but has no data!")
                        return False
                else:
                    # Model not ready yet, wait and try again
                    print(
                        f"Round {self.round_number}: Averaged model not ready, waiting 2 seconds..."
                    )
                    time.sleep(2)

            except grpc.RpcError as e:
                print(f"Failed to get averaged model (will retry): {e}")
                time.sleep(2)
                continue

    def generate_dummy_data(self, batch_size=32):
        """Generate dummy training data"""
        x = torch.randn(batch_size, 784)
        y = torch.randint(0, 10, (batch_size,))
        return x, y

    def train_local_epoch(self, num_batches=5):
        """Train model locally for one epoch"""
        self.model.train()
        total_loss = 0.0

        for _ in range(num_batches):
            x, y = self.generate_dummy_data()

            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(
            f"Round {self.round_number}: Local training completed. Average loss: {avg_loss:.4f}"
        )
        return avg_loss

    def federated_training_round(self, client_id):
        """Execute one round of federated training"""
        print(f"\n=== Round {self.round_number} ===")

        # Step 1: Train locally
        loss = self.train_local_epoch()

        # Step 2: Send update to server
        if not self.send_update_to_server(client_id):
            return False

        # Step 3: Get averaged model from server
        if not self.get_averaged_model(client_id):
            return False

        self.round_number += 1
        return True

    def run_federated_training(self, client_id, num_rounds=5):
        """Run complete federated training process"""
        print(f"Starting federated training for {num_rounds} rounds...")

        for round_num in range(num_rounds):
            if not self.federated_training_round(client_id):
                print(f"Training failed at round {round_num}")
                break

            # Add some random delay to simulate real-world conditions
            time.sleep(random.uniform(0.5, 2.0))

        print(f"Federated training completed for client {self.client_id}")

    def close(self):
        """Close the connection to the server"""
        self.channel.close()


def main():
    import sys

    if len(sys.argv) != 2:
        print("Usage: python scalable_client.py <client_id>")
        sys.exit(1)

    client_id = sys.argv[1]

    # Create and run client
    client = ParameterServerClient(client_id)

    try:
        # Wait a bit for other clients to connect
        print("Waiting for other clients to connect...")
        time.sleep(5)

        # Run federated training
        client.run_federated_training(num_rounds=1, client_id=client_id)

    except KeyboardInterrupt:
        print(f"\nClient {client_id} interrupted by user")
    finally:
        client.close()


if __name__ == "__main__":
    main()
