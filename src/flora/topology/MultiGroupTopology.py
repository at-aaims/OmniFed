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
from typing import Dict, List

import rich.repr
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ..Node import Node
from .BaseTopology import BaseTopology

# ======================================================================================


@rich.repr.auto
class MultiGroupTopology(BaseTopology):
    """
    Multi-group federated learning topology for cross-institutional FL.

    Coordinates multiple independent federated learning groups,
    where each group runs as a standard centralized topology internally.
    Group servers can communicate across institutional boundaries.

    Each group maintains its own TorchDistributed communication for local training,
    while group servers use gRPC to coordinate global aggregation.

    The topology composes multiple CentralizedTopology instances.
    """

    def __init__(self, groups: List[DictConfig], global_comm: DictConfig):
        """
        Initialize with a list of group configurations.

        Args:
            groups: List of DictConfig objects defining each group's topology configuration.
                   Each group config should have _target_ pointing to a topology class
                   and any group-specific parameters.
            global_comm: Global communication configuration for inter-group coordination (required)

        Raises:
            ValueError: If no groups provided
        """
        super().__init__()
        self.global_comm_cfg = global_comm

        if not groups:
            raise ValueError("At least one group must be specified")

        # Instantiate all group topologies directly
        self.topologies: List[BaseTopology] = [
            instantiate(
                topology,
                _recursive_=False,
            )
            for topology in groups
        ]

    def create_nodes(
        self,
        algo_cfg: DictConfig,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        log_dir: str,
        node_rayopts: Dict[str, any] = {},
        **kwargs,
    ) -> List[Node]:
        """
        Create nodes for all groups and configure cross-group communication.

        Each group gets its own communication setup for local training.
        Group servers additionally get global communication configuration for coordinating
        with other institutions.

        Args:
            algo_cfg: Algorithm configuration
            model_cfg: Model configuration
            data_cfg: Data configuration
            log_dir: Base logging directory

        Returns:
            Flattened list of all nodes across all groups
        """

        # Prevent GPU over-allocation when multiple groups share infrastructure
        total_nodes = sum(group.num_clients + 1 for group in self.topologies)
        if torch.cuda.is_available():
            node_rayopts.setdefault("num_gpus", 1.0 / total_nodes)

        all_nodes: List[Node] = []

        for group_idx, group_topology in enumerate(self.topologies):
            # Inject group's global rank into gRPC configuration for inter-group communication
            __global_comm_cfg = DictConfig(
                {
                    **self.global_comm_cfg,
                    "rank": group_idx,  # Group index becomes global rank
                    "world_size": len(self.topologies),
                }
            )

            # Each group topology handle node creation
            group_nodes = group_topology.create_nodes(
                algo_cfg=algo_cfg,
                model_cfg=model_cfg,
                data_cfg=data_cfg,
                log_dir=os.path.join(log_dir, f"Group{group_idx}"),
                node_rayopts=node_rayopts,
                global_comm_cfg=__global_comm_cfg,  # Pass global_comm_cfg through kwargs for inter-group coordination
                **kwargs,
            )

            all_nodes.extend(group_nodes)

        return all_nodes
