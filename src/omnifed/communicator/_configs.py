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
    sharedfile: str = "sharedfile"

    # Connection settings
    timeout: int = 60

    # Retry settings
    max_retries: int = 5


@dataclass
class GrpcCommunicatorConfig(BaseCommunicatorConfig):
    """Configuration for GrpcCommunicator."""

    _target_: str = "src.omnifed.communicator.GrpcCommunicator"

    # gRPC server configuration
    max_workers: int = 10
    max_send_message_length: int = 104857600  # 100 MB
    max_receive_message_length: int = 104857600  # 100 MB

    # Timeout settings
    aggregation_timeout: float = 600.0  # Seconds for server to wait for all clients
    client_timeout: float = 60.0  # Seconds for clients to wait for aggregation result

    # Retry settings
    max_retries: int = 5
    retry_delay: float = 5.0  # Seconds between retries
