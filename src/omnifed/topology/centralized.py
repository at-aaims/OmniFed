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

from typing import Dict, List, Optional, cast

import rich.repr
from omegaconf import OmegaConf

from ..algorithm import BaseAlgorithmConfig
from ..communicator import BaseCommunicatorConfig
from ..data import DataModuleConfig
from ..model import ModelConfig
from ..node import NodeConfig
from ..utils import print
from .base import BaseTopology

# ======================================================================================


@rich.repr.auto
class CentralizedTopology(BaseTopology):
    """
    Classic federated learning: one server coordinating multiple clients.

    When to use: All participants can communicate directly with a central server.
    For cross-institutional setups, use HierarchicalTopology instead.

    How it works:
    - Server (rank 0) aggregates model updates, never trains locally
    - Clients (ranks 1+) train on local data, send updates to server
    - All communication stays within this single group

    Example config:
    ```yaml
    topology:
      _target_: src.omnifed.topology.CentralizedTopology
      num_clients: N
      local_comm:
        _target_: src.omnifed.communicator.TorchDistCommunicator
        backend: "gloo"
    ```

    Example with N clients (N+1 total nodes):
    - 0.0 (server): Receives updates from clients → averages → sends back.
    - 0.1, 0.2, ... 0.N (clients): Train locally → send updates → receive new model.
    """

    def __init__(
        self,
        num_clients: int,
        local_comm: BaseCommunicatorConfig,
        overrides: Optional[Dict[int, NodeConfig]] = None,
    ):
        """
        Set up server-client FL topology.

        Args:
            num_clients: How many client nodes to create (server added automatically)
            local_comm: Communication config for server-client coordination
            overrides: Custom settings per rank (0=server, 1+=clients).
                      Example: {0: {"device_hint": "cpu"}, 1: {"device_hint": "cuda:X"}}
        """
        super().__init__()
        self.num_clients: int = num_clients
        self.local_comm: BaseCommunicatorConfig = local_comm
        self.overrides: Dict[int, NodeConfig] = overrides or {}

        # ---
        print(self)

    def _setup(
        self,
        default_algorithm_cfg: BaseAlgorithmConfig,
        default_model_cfg: ModelConfig,
        default_datamodule_cfg: DataModuleConfig,
    ) -> List[NodeConfig]:
        """
        Create server and client node configurations.

        Server gets rank 0, clients get ranks 1, 2, 3, etc.
        Each node gets communication settings and can have custom overrides.

        Returns all nodes ready for the Engine to launch as Ray actors.
        """
        world_size: int = self.num_clients + 1
        node_configs: List[NodeConfig] = []

        for rank in range(world_size):
            # Create rank-specific communicator config
            local_comm_cfg: BaseCommunicatorConfig = OmegaConf.structured(
                self.local_comm
            )
            # local_comm_cfg = copy.deepcopy(self.local_comm)
            local_comm_cfg.rank = rank
            local_comm_cfg.world_size = world_size

            # Create base node configuration
            base_node_cfg = NodeConfig(
                name=f"Node0.{rank}",
                local_comm=local_comm_cfg,
                global_comm=None,
                algorithm=default_algorithm_cfg,
                model=default_model_cfg,
                datamodule=default_datamodule_cfg,
            )

            # Apply per-node overrides using OmegaConf.merge()
            node_cfg = OmegaConf.merge(
                OmegaConf.structured(base_node_cfg),
                OmegaConf.structured(self.overrides.get(rank, {})),
                # base_node_cfg,
                # self.overrides.get(rank, {}),
            )
            node_cfg = cast(NodeConfig, node_cfg)
            # node_cfg = OmegaConf.to_object(node_cfg)
            # node_cfg = cast(NodeConfig, node_cfg)
            # node_cfg = OmegaConf.to_container(node_cfg, resolve=True)

            node_configs.append(node_cfg)

        return node_configs
