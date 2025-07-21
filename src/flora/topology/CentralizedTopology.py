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

import os
import time
from typing import Any, Dict, List

import rich.repr
import torch
from omegaconf import DictConfig

from ..Node import Node
from .BaseTopology import BaseTopology

# ======================================================================================


@rich.repr.auto
class CentralizedTopology(BaseTopology):
    """
    Template for centralized federated learning topology with aggregator-trainer architecture.

    - One aggregator node (rank 0) collects and combines model updates
    - Multiple trainer nodes (ranks 1+) perform local training
    - All communication flows through the aggregator
    """

    def __init__(
        self,
        num_clients: int,
        local_comm: DictConfig,
        init_delay: float = 1.0,
    ):
        """
        Initialize centralized topology.

        Args:
            num_clients (int): Number of client nodes (server node is added automatically)
            local_comm: Local communication configuration (required)
            init_delay (float): Simulated delay in seconds between node initializations (default: 1.0s)
        """
        super().__init__()
        self.num_clients: int = num_clients
        self.local_comm_cfg = local_comm
        self.init_delay: float = init_delay

    def create_nodes(
        self,
        algo_cfg: DictConfig,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        log_dir: str,
        node_rayopts: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> List[Node]:
        """
        Create nodes for centralized topology.

        In centralized topology:
        - Rank 0: Aggregator (no training data, coordinates aggregation)
        - Ranks 1+: Trainers (training data, perform local training)

        The global_comm_cfg parameter allows MultiGroupTopology to provide
        additional communication configuration for specific nodes.
        """
        total_nodes: int = self.num_clients + 1  # 1 server + N clients
        # Request GPU resources if available
        if torch.cuda.is_available():
            node_rayopts.setdefault("num_gpus", 1.0 / total_nodes)

        nodes: List[Node] = []

        # ----------------------------------------------------------------
        # INIT ALL NODES

        for rank in range(total_nodes):
            # Configure Ray actor options

            # Create communicator configs with injected rank and world_size
            __local_comm_cfg = DictConfig(
                {
                    **self.local_comm_cfg,
                    "rank": rank,
                    "world_size": total_nodes,
                }
            )

            # Extract global_comm_cfg from kwargs if provided (for multi-group scenarios)
            global_comm_cfg = kwargs.get("global_comm_cfg", None)

            # Generate node ID
            node_id_base = f"L{rank}"
            if global_comm_cfg is not None:
                global_rank = global_comm_cfg["rank"]
                node_id_base = f"G{global_rank}{node_id_base}"

            if rank == 0:
                node_id = f"{node_id_base}-SERVER"
                # Create node-specific data config for server (no training data)
                node_data_cfg = data_cfg.copy()
                node_data_cfg.train = None
                # Only server gets global_comm_cfg for inter-group communication
                __global_comm_cfg = global_comm_cfg
            else:
                node_id = f"{node_id_base}-Client"  # Purposefully using different casing for log readability
                node_data_cfg = data_cfg
                # Clients don't participate in inter-group communication
                __global_comm_cfg = None

            node = Node.options(**node_rayopts).remote(
                id=node_id,
                local_comm_cfg=__local_comm_cfg,
                global_comm_cfg=__global_comm_cfg,
                algo_cfg=algo_cfg,
                model_cfg=model_cfg,
                data_cfg=node_data_cfg,
                log_dir=os.path.join(log_dir, node_id),
            )

            nodes.append(node)

            # Add simulated delay between node initializations
            time.sleep(self.init_delay)

        # NOTE: node setup is handled externally by Engine
        return nodes
