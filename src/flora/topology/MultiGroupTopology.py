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

from typing import Any, List

import rich.repr
from hydra.utils import instantiate
from omegaconf import DictConfig

from .BaseTopology import BaseTopology
from ..Node import NodeSpec

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

    def __init__(
        self,
        groups: List[DictConfig],
        global_comm: DictConfig,
        **kwargs: Any,
    ):
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
        super().__init__(**kwargs)
        self.global_comm_cfg = global_comm

        if not groups:
            raise ValueError("At least one group must be specified")

        # Instantiate all group topologies directly
        self.topologies: List[BaseTopology] = [
            instantiate(
                topology,
                **kwargs,
                _recursive_=False,
            )
            for topology in groups
        ]

    def create_node_specs(self) -> List[NodeSpec]:
        """
        Configure nodes for all groups and set up cross-group communication.

        Each group topology has already been instantiated and configured through
        the constructor's instantiate() call, so their node specifications already exist.
        This method retrieves those existing specifications and injects global
        communicators for local rank 0 nodes (group servers) only.

        Group servers (local rank 0 nodes) get global communication configuration
        for coordinating across institutional boundaries.

        Returns:
            Flattened list of all node specifications across all groups with
            global communicators injected for group servers
        """

        all_node_specs: List[NodeSpec] = []

        for group_idx, group_topology in enumerate(self.topologies):
            # Inject global communicator for local rank 0 nodes only
            # Note: group_topology.__iter__() returns node_specs directly
            for node_spec in group_topology:
                # Check if this is a local rank 0 node (server/aggregator)
                if node_spec.local_comm_cfg["rank"] == 0:
                    # Inject group's global rank into gRPC configuration for inter-group communication
                    node_spec.global_comm_cfg = DictConfig(
                        {
                            **self.global_comm_cfg,
                            "rank": group_idx,  # Group index becomes global rank
                            "world_size": len(self.topologies),
                        }
                    )

                all_node_specs.append(node_spec)

        return all_node_specs
