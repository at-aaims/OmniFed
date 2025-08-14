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

from typing import List, cast

import rich.repr
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ..algorithm import BaseAlgorithmConfig
from ..communicator import BaseCommunicatorConfig
from ..data import DataModuleConfig
from ..model import ModelConfig
from ..node import NodeConfig
from ..utils import print
from . import BaseTopologyConfig
from .base import BaseTopology

# ======================================================================================


@rich.repr.auto
class HierarchicalTopology(BaseTopology):
    """
    Cross-institutional federated learning with hierarchical aggregation.

    When to use: Groups can't directly share data but want to coordinate globally.
    Think hospitals collaborating while keeping patient data local.

    How it works:
    - Each group runs centralized FL internally (server + clients)
    - Group servers coordinate across institutions via global communication
    - Regular clients only communicate within their local group
    - Two-level aggregation: local → global → local

    Example config:
    ```yaml
    topology:
      _target_: src.omnifed.topology.MultiGroupTopology
      groups:
        - _target_: src.omnifed.topology.CentralizedTopology
          num_clients: N
        - _target_: src.omnifed.topology.CentralizedTopology
          num_clients: M
      global_comm:
        _target_: src.omnifed.communicator.GrpcCommunicator
    ```

    Example with 2 groups, N and M clients each:
    - 0.0 (Group A server), 0.1, 0.2, ... 0.N (Group A clients)
    - 1.0 (Group B server), 1.1, 1.2, ... 1.M (Group B clients)

    Communication flow: Clients → Local server → Global aggregation → Local server → Clients
    Only 0.0, 1.0 communicate globally; others stay local.
    """

    def __init__(
        self,
        groups: List[BaseTopologyConfig],
        global_comm: BaseCommunicatorConfig,
    ):
        """
        Set up multiple FL groups with global coordination.

        Args:
            groups: Each group's topology config (usually CentralizedTopology configs)
            global_comm: How group servers communicate globally (any BaseCommunicatorConfig)
        """
        super().__init__()
        self.groups: List[BaseTopologyConfig] = groups
        self.global_comm: BaseCommunicatorConfig = global_comm

        # print(type(self.topologies_cfgs))

        if not groups:
            raise ValueError("At least one group must be specified")

        # Create the actual topology for each group (e.g., CentralizedTopology instances)
        self.topologies: List[BaseTopology] = [
            instantiate(topology_cfg, _recursive_=False) for topology_cfg in self.groups
        ]

        # ---
        print(self)

    def _setup(
        self,
        default_algorithm_cfg: BaseAlgorithmConfig,
        default_model_cfg: ModelConfig,
        default_datamodule_cfg: DataModuleConfig,
    ) -> List[NodeConfig]:
        """
        Wire up all nodes with proper naming and cross-group communication.

        Takes the nodes from each group's topology and gives them:
        1. Names like 1.2 (group 1, local rank 2)
        2. Global communication for group servers only

        Returns all nodes flattened into one list for the Engine to launch.
        """
        # Setup all child topologies first
        for topology in self.topologies:
            topology.setup(
                default_algorithm_cfg=default_algorithm_cfg,
                default_model_cfg=default_model_cfg,
                default_datamodule_cfg=default_datamodule_cfg,
            )

        world_size = len(self.topologies)
        node_configs: List[NodeConfig] = []

        for topology_idx, topology in enumerate(self.topologies):
            for node_idx, node_cfg in enumerate(topology):
                # Give each node a name showing which group and local rank
                node_cfg.name = f"Node{topology_idx}.{node_cfg.local_comm.rank}"
                # node_cfg.name = f"Node{topology_idx}.{node_cfg.local_comm['rank']}"

                # Handle global communication assignment
                if node_cfg.local_comm.rank == 0:
                    # Only group servers (local rank 0) need global communication

                    # Create global comm config with rank and world_size
                    # global_comm_cfg = copy.deepcopy(self.global_comm)
                    global_comm_cfg: BaseCommunicatorConfig = OmegaConf.structured(
                        self.global_comm
                    )
                    global_comm_cfg.rank = topology_idx
                    global_comm_cfg.world_size = world_size
                    # global_comm_cfg = OmegaConf.merge(
                    #     self.global_comm,
                    #     {
                    #         "rank": topology_idx,
                    #         "world_size": world_size,
                    #     },
                    # )

                    # Merge the global_comm config into the node_cfg using OmegaConf.merge
                    # node_cfg.global_comm = global_comm_cfg
                    merged_cfg = OmegaConf.merge(
                        node_cfg, {"global_comm": global_comm_cfg}
                    )
                    node_cfg = cast(NodeConfig, merged_cfg)

                node_configs.append(node_cfg)

        return node_configs
