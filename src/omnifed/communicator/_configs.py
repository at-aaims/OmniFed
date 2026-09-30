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

from dataclasses import dataclass
from omegaconf import MISSING

from .grpc_limits import (
    GRPC_AGGREGATION_TIMEOUT_SEC,
    GRPC_CHUNK_PAYLOAD_BYTES,
    GRPC_CHUNK_THRESHOLD_BYTES,
    GRPC_CLIENT_TIMEOUT_SEC,
    GRPC_MAX_MESSAGE_BYTES,
)
from .torchdist import InitMethod


@dataclass
class BaseCommunicatorConfig:
    """Base configuration for all communicator types with shared parameters."""

    _target_: str = "src.omnifed.communicator.BaseCommunicator"

    # Core distributed parameters
    rank: int = MISSING
    world_size: int = MISSING

    # Default network configuration
    master_addr: str = "127.0.0.1"
    master_port: int = MISSING


@dataclass
class TorchDistCommunicatorConfig(BaseCommunicatorConfig):
    """Configuration for TorchDistCommunicator."""

    _target_: str = "src.omnifed.communicator.TorchDistCommunicator"

    # Initialization and backend settings
    init_method: InitMethod = InitMethod.TCP
    backend: str = "gloo"
    #backend: str = "nccl"
    sharedfile: str = "sharedfile"

    # Connection settings
    timeout: int = 60

    # Retry settings
    max_retries: int = 5

    # True: all-reduce parameters (+ buffers). False: all-reduce gradients (classic sync grad).
    communicate_params: bool = True


@dataclass
class GrpcCommunicatorConfig(BaseCommunicatorConfig):
    """Configuration for GrpcCommunicator."""

    _target_: str = "src.omnifed.communicator.GrpcCommunicator"

    # gRPC server configuration (INT32_MAX ~2 GiB; not model-specific)
    max_workers: int = 10
    max_send_message_length: int = GRPC_MAX_MESSAGE_BYTES
    max_receive_message_length: int = GRPC_MAX_MESSAGE_BYTES

    # Seconds for one aggregation (Llama-400M-scale unary). CIFAR may override shorter.
    aggregation_timeout: float = GRPC_AGGREGATION_TIMEOUT_SEC
    client_timeout: float = GRPC_CLIENT_TIMEOUT_SEC

    # Packed-size gate (not model-name). Override in tests with a tiny threshold.
    chunk_threshold_bytes: int = GRPC_CHUNK_THRESHOLD_BYTES
    chunk_payload_bytes: int = GRPC_CHUNK_PAYLOAD_BYTES

    # Retry settings
    max_retries: int = 5
    retry_delay: float = 5.0  # Seconds between retries
