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

from .BaseTopology import BaseTopology, NodeConfig

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
        **kwargs: Any,
    ):
        """
        Initialize centralized topology.

        Args:
            num_clients (int): Number of client nodes (server node is added automatically)
            local_comm: Local communication configuration (required)
        """
        super().__init__(**kwargs)
        self.num_clients: int = num_clients
        self.local_comm_cfg = local_comm

    def configure_nodes(
        self,
        algo_cfg: DictConfig,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        **kwargs: Any,
    ) -> List[NodeConfig]:
        """
        Configure nodes for centralized topology.

        In centralized topology:
        - Rank 0: Aggregator (no training data, coordinates aggregation)
        - Ranks 1+: Trainers (training data, perform local training)

        The global_comm_cfg parameter allows MultiGroupTopology to provide
        additional communication configuration for specific nodes.
        """
        world_size: int = self.num_clients + 1  # 1 server + N clients

        node_configs: List[NodeConfig] = []

        # ----------------------------------------------------------------
        # CONFIGURE ALL NODES

        for rank in range(world_size):
            # Create communicator configs with injected rank and world_size
            __local_comm_cfg = DictConfig(
                {
                    **self.local_comm_cfg,
                    "rank": rank,
                    "world_size": world_size,
                }
            )

            if rank == 0:
                node_id = "SERVER"
                # Create node-specific data config for server (no training data)
                node_data_cfg = data_cfg.copy()
                node_data_cfg.train = None
            else:
                node_id = "Client"
                node_data_cfg = data_cfg

            node_config = NodeConfig(
                id=node_id,
                algorithm_cfg=algo_cfg,
                model_cfg=model_cfg,
                datamodule_cfg=node_data_cfg,
                local_comm_cfg=__local_comm_cfg,
            )

            node_configs.append(node_config)

        return node_configs
