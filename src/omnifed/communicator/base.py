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

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, TypeVar

import torch
import torch.nn as nn

from ..utils import RequiredSetup

# ======================================================================================


class AggregationOp(str, Enum):
    """Distributed tensor reduction operations for federated learning aggregation."""

    SUM = "SUM"  # Sum all tensors across ranks
    MEAN = "MEAN"  # Average tensors across ranks
    MAX = "MAX"  # Element-wise maximum across ranks


class BaseCommunicator(RequiredSetup, ABC):
    """
    Abstract interface for federated learning communication backends.

    Provides protocol-agnostic operations for distributed message passing:
    broadcasting from source to all ranks, and aggregating across ranks.

    Implementations include gRPC (client-server coordination) and
    PyTorch distributed (collective operations).
    Algorithms focus on FL logic while communicators handle transport.
    """

    MsgT = TypeVar("MsgT", nn.Module, torch.Tensor, Dict[str, torch.Tensor])

    def __init__(self, rank: int, world_size: int, master_addr: str, master_port: int):
        """
        Initialize communicator with core distributed parameters.

        Args:
            rank: Unique rank ID for this node (0 for server, 1+ for clients)
            world_size: Total number of nodes in the distributed setup
            master_addr: Address of the master node (server)
            master_port: Port for communication with the master node
        """
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port

    @abstractmethod
    def broadcast(
        self,
        msg: MsgT,
        src: int = 0,
    ) -> MsgT:
        """
        Broadcast message from source rank to all other ranks.

        Args:
            msg: Model, tensor dict, or tensor to broadcast
            src: Source rank ID (default: 0)

        Returns:
            Same message type with updated values from source
        """
        pass

    @abstractmethod
    def aggregate(
        self,
        msg: MsgT,
        reduction: AggregationOp,
    ) -> MsgT:
        """
        Aggregate message across all ranks using specified reduction operation.

        Args:
            msg: Model, tensor dict, or tensor to aggregate
            reduction: SUM, MEAN, or MAX reduction operation

        Returns:
            Same message type with aggregated values
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up communication resources and connections.

        Should be called when communication is no longer needed
        to properly release network resources and process groups.
        """
        pass
