import grpc
from concurrent import futures
import threading
import time
import torch
import torch.nn as nn
import numpy as np
import io
import gzip
import pickle
from typing import Dict
import parameter_server_pb2
import parameter_server_pb2_grpc


class ScalableParameterServerServicer(
    parameter_server_pb2_grpc.ParameterServerServicer
):
    def __init__(
        self,
        num_clients: int,
        model: nn.Module,
        use_compression: bool = True,
        accumulate_gradients: bool = False,  # Changed to False for debugging
        chunk_size: int = 1024 * 1024,
    ):  # 1MB chunks
        self.num_clients = num_clients
        self.model = model
        self.use_compression = use_compression
        self.accumulate_gradients = accumulate_gradients
        self.chunk_size = chunk_size

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

        print(f"DEBUG: Scalable Parameter Server initialized for {num_clients} clients")
        print(
            f"DEBUG: Compression: {use_compression}, Gradient Accumulation: {accumulate_gradients}"
        )

    def _initialize_accumulated_gradients(self):
        """Initialize accumulated gradients to zero"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.accumulated_gradients[name] = torch.zeros_like(param)

    def RegisterClient(self, request, context):
        """Register a new client"""
        with self.lock:
            self.registered_clients.add(request.client_id)
            total_clients = len(self.registered_clients)

            print(
                f"DEBUG: Client {request.client_id} registered. Total clients: {total_clients}"
            )

            return parameter_server_pb2.RegistrationResponse(
                success=True,
                message=f"Client {request.client_id} registered successfully",
                total_clients=total_clients,
            )

    def SendUpdate(self, request, context):
        """Receive model update from client - supports both gradient and parameter updates"""
        print(
            f"DEBUG: SendUpdate called from client {request.client_id} for round {request.round_number}"
        )
        print(f"DEBUG: Request has {len(request.layers)} layers")

        with self.lock:
            client_id = request.client_id
            round_number = request.round_number

            try:
                # Debug the received layers
                for i, layer in enumerate(request.layers):
                    print(f"DEBUG: Received layer {i}: {layer.layer_name}")
                    print(
                        f"DEBUG: Layer has {len(layer.weights)} weights, {len(layer.biases)} biases"
                    )

                # Deserialize the update
                if hasattr(request, "serialized_data") and request.serialized_data:
                    print(
                        f"DEBUG: Using serialized_data ({len(request.serialized_data)} bytes)"
                    )
                    # Use efficient serialization
                    update_data = self._deserialize_model_params(
                        request.serialized_data
                    )
                else:
                    print("DEBUG: Using protobuf format")
                    # Fallback to protobuf format
                    update_data = self._protobuf_to_model_params(request.layers)

                print(f"DEBUG: Processed update_data has {len(update_data)} parameters")

                if self.accumulate_gradients:
                    print("DEBUG: Using gradient accumulation mode")
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
                    print("DEBUG: Using traditional mode (storing all updates)")
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
                print(f"ERROR: Error processing update from {client_id}: {e}")
                import traceback

                traceback.print_exc()
                return parameter_server_pb2.UpdateResponse(
                    success=False,
                    message=f"Error processing update: {str(e)}",
                    clients_registered=len(self.registered_clients),
                    updates_received=0,
                )

    def _accumulate_gradient_update(self, update_data: Dict):
        """Accumulate gradient updates for memory efficiency"""
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
                    # Apply the averaged gradient (this is simplified - in practice you'd use an optimizer)
                    param.data -= 0.01 * avg_gradient  # Simple SGD with lr=0.01

    def _reset_gradient_accumulation(self):
        """Reset gradient accumulation for next round"""
        self.gradient_count = 0
        with torch.no_grad():
            for name in self.accumulated_gradients:
                self.accumulated_gradients[name].zero_()

    def GetAveragedModel(self, request, context):
        """Send current model to client"""
        print(
            f"DEBUG: GetAveragedModel called from client {request.client_id} for round {request.round_number}"
        )

        with self.lock:
            client_id = request.client_id
            round_number = request.round_number

            try:
                if self.accumulate_gradients:
                    print("DEBUG: Using gradient accumulation mode")
                    # For gradient accumulation, always send current model
                    if self.current_round >= round_number:
                        model_params = {}
                        for name, param in self.model.named_parameters():
                            model_params[name] = param.data.clone()

                        # Use efficient serialization if supported
                        serialized_data = self._serialize_model_params(model_params)

                        print(
                            f"Sending current model to {client_id} for round {round_number}"
                        )

                        return parameter_server_pb2.ModelParameters(
                            round_number=round_number,
                            serialized_data=serialized_data,
                            is_ready=True,
                        )

                else:
                    print("DEBUG: Using traditional mode")
                    print(
                        f"DEBUG: current_round = {self.current_round}, requested round = {round_number}"
                    )
                    print(
                        f"DEBUG: averaged_models_history keys: {list(self.averaged_models_history.keys())}"
                    )

                    if round_number in self.averaged_models_history:
                        print(f"DEBUG: Found averaged model for round {round_number}")
                        averaged_model = self.averaged_models_history[round_number]

                        # Convert back to protobuf format for compatibility
                        proto_layers = self._model_params_to_protobuf(averaged_model)

                        print(
                            f"DEBUG: Sending {len(proto_layers)} layers to {client_id}"
                        )
                        for i, layer in enumerate(proto_layers):
                            print(
                                f"DEBUG: Sending layer {i}: {layer.layer_name}, weights: {len(layer.weights)}, biases: {len(layer.biases)}"
                            )

                        return parameter_server_pb2.ModelParameters(
                            round_number=round_number,
                            layers=proto_layers,
                            is_ready=True,
                        )
                    else:
                        print(
                            f"DEBUG: No averaged model found for round {round_number}"
                        )

                print("DEBUG: Returning not ready")
                return parameter_server_pb2.ModelParameters(
                    round_number=round_number, layers=[], is_ready=False
                )

            except Exception as e:
                print(f"ERROR: Error sending model to {client_id}: {e}")
                import traceback

                traceback.print_exc()
                return parameter_server_pb2.ModelParameters(
                    round_number=round_number, layers=[], is_ready=False
                )

    def _model_params_to_protobuf(self, model_params: Dict):
        """Convert model parameters to protobuf format"""
        print(
            f"DEBUG: _model_params_to_protobuf called with {len(model_params)} parameters"
        )
        layers = []

        # Group parameters by layer
        layer_dict = {}
        for param_name, param_tensor in model_params.items():
            parts = param_name.split(".")
            layer_name = parts[0]
            param_type = parts[1]

            if layer_name not in layer_dict:
                layer_dict[layer_name] = {}
            layer_dict[layer_name][param_type] = param_tensor

        # Convert to protobuf format
        for layer_name, params in layer_dict.items():
            layer_entry = parameter_server_pb2.LayerParameters(layer_name=layer_name)

            if "weight" in params:
                weights = params["weight"]
                layer_entry.weights.extend(weights.flatten().tolist())
                layer_entry.weight_shape.extend(list(weights.shape))
                print(
                    f"DEBUG: Added {len(weights.flatten())} weights for layer {layer_name}"
                )

            if "bias" in params:
                biases = params["bias"]
                layer_entry.biases.extend(biases.flatten().tolist())
                layer_entry.bias_shape.extend(list(biases.shape))
                print(
                    f"DEBUG: Added {len(biases.flatten())} biases for layer {layer_name}"
                )

            layers.append(layer_entry)

        print(f"DEBUG: Created {len(layers)} protobuf layers")
        return layers

    def _compute_averaged_model(self, round_number):
        """Compute averaged model from all client updates (traditional approach)"""
        print(f"DEBUG: _compute_averaged_model called for round {round_number}")

        client_params = list(self.client_updates[round_number].values())
        print(f"DEBUG: Averaging {len(client_params)} client updates")

        averaged_params = {}

        # Get parameter names from first client
        param_names = client_params[0].keys()
        print(f"DEBUG: Parameter names: {list(param_names)}")

        with torch.no_grad():
            for param_name in param_names:
                # Average parameters across all clients
                param_list = [params[param_name] for params in client_params]
                print(
                    f"DEBUG: Averaging parameter {param_name}, shapes: {[p.shape for p in param_list]}"
                )

                param_sum = sum(param_list)
                averaged_params[param_name] = param_sum / len(client_params)

                print(
                    f"DEBUG: Averaged {param_name}: {averaged_params[param_name].shape}"
                )

        # Store the averaged model for this round
        self.averaged_models_history[round_number] = averaged_params

        # Update the server's model
        for name, param in self.model.named_parameters():
            if name in averaged_params:
                param.data = averaged_params[name]

        print(f"DEBUG: Averaged model computed and stored for round {round_number}")

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

    def _protobuf_to_model_params(self, proto_layers):
        """Convert protobuf layers to PyTorch model parameters (fallback)"""
        print(
            f"DEBUG: _protobuf_to_model_params called with {len(proto_layers)} layers"
        )

        model_params = {}
        for i, layer in enumerate(proto_layers):
            print(f"DEBUG: Processing layer {i}: {layer.layer_name}")
            print(
                f"DEBUG: Layer has {len(layer.weights)} weights, {len(layer.biases)} biases"
            )

            layer_name = layer.layer_name

            if len(layer.weights) > 0:
                weights = np.array(layer.weights).reshape(tuple(layer.weight_shape))
                model_params[f"{layer_name}.weight"] = torch.tensor(
                    weights, dtype=torch.float32
                )
                print(f"DEBUG: Added weights for {layer_name}.weight: {weights.shape}")

            if len(layer.biases) > 0:
                biases = np.array(layer.biases).reshape(tuple(layer.bias_shape))
                model_params[f"{layer_name}.bias"] = torch.tensor(
                    biases, dtype=torch.float32
                )
                print(f"DEBUG: Added biases for {layer_name}.bias: {biases.shape}")

        print(f"DEBUG: Converted to {len(model_params)} model parameters")
        return model_params

    def _serialize_model_params(self, model_params: Dict) -> bytes:
        """Efficiently serialize model parameters"""
        # Use pickle for efficient serialization of tensors
        buffer = io.BytesIO()

        # Convert to CPU and compress if needed
        cpu_params = {}
        for name, param in model_params.items():
            if isinstance(param, torch.Tensor):
                cpu_params[name] = param.cpu()
            else:
                cpu_params[name] = param

        pickle.dump(cpu_params, buffer)
        serialized = buffer.getvalue()

        return self._compress_data(serialized)

    def _deserialize_model_params(self, data: bytes) -> Dict:
        """Efficiently deserialize model parameters"""
        decompressed = self._decompress_data(data)
        buffer = io.BytesIO(decompressed)
        return pickle.load(buffer)

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using gzip"""
        if not self.use_compression:
            return data
        return gzip.compress(data)

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data using gzip"""
        if not self.use_compression:
            return data
        return gzip.decompress(data)


def serve(port=50051, num_clients=3, use_compression=True, accumulate_gradients=False):
    """Start the scalable parameter server"""
    # Example model - replace with your actual large model
    model = nn.Sequential(
        nn.Linear(784, 128),  # Changed to match client model
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
        ],
    )

    parameter_server_pb2_grpc.add_ParameterServerServicer_to_server(
        ScalableParameterServerServicer(
            num_clients,
            model,
            use_compression=use_compression,
            accumulate_gradients=accumulate_gradients,
        ),
        server,
    )

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    print(f"DEBUG: Scalable Parameter server starting on {listen_addr}")
    server.start()

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        print("Shutting down parameter server...")
        server.stop(0)


if __name__ == "__main__":
    serve(accumulate_gradients=True)
