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

from typing import List

import ray
import rich.repr
import torch
from omegaconf import DictConfig

from .. import utils
from ..Node import Node, NodeRole
from .BaseTopology import Topology

# ======================================================================================


@rich.repr.auto
class CentralizedTopology(Topology):
    """
    Template for centralized federated learning topology with aggregator-trainer architecture.

    - One aggregator node (rank 0) collects and combines model updates
    - Multiple trainer nodes (ranks 1+) perform local training
    - All communication flows through the aggregator
    """

    def __init__(self, num_nodes: int):
        """
        Initialize centralized topology.

        Args:
            self.node_cfg (DictConfig): Default configuration for nodes
        """
        super().__init__()
        self.num_nodes: int = num_nodes

    def create_nodes(
        self,
        comm_cfg: DictConfig,
        algo_cfg: DictConfig,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
    ) -> List[Node]:
        """
        Create nodes for centralized topology.

        In centralized topology:
        - Rank 0: Aggregator (no training data, coordinates aggregation)
        - Ranks 1+: Trainers (training data, perform local training)

        Returns:
            List of configured nodes
        """
        utils.log_sep("Node Creation")
        print(f"create_nodes: num_nodes={self.num_nodes}")

        nodes: List[Node] = []

        # ----------------------------------------------------------------
        # INIT ALL NODES

        node_rayopts = dict()
        if torch.cuda.is_available():
            node_rayopts.update(dict(num_gpus=1.0 / self.num_nodes))

        # ---
        for rank in range(self.num_nodes):
            if rank == 0:
                node = Node.options(**node_rayopts).remote(
                    id=f"S{rank}",
                    roles={NodeRole.AGGREGATOR},
                    comm_cfg=comm_cfg,
                    model_cfg=model_cfg,
                    algo_cfg=algo_cfg,
                    data_cfg=data_cfg,  # TODO: Remove data from aggregator nodes
                    rank=rank,
                    world_size=self.num_nodes,
                )
            else:
                node = Node.options(**node_rayopts).remote(
                    id=f"C{rank}",
                    roles={NodeRole.TRAINER},
                    comm_cfg=comm_cfg,
                    model_cfg=model_cfg,
                    algo_cfg=algo_cfg,
                    data_cfg=data_cfg,  # TODO: Only trainers should hold data
                    rank=rank,
                    world_size=self.num_nodes,
                )

            nodes.append(node)

        # ----------------------------------------------------------------
        # SETUP ALL NODES
        setup_futures = []
        for rank, node in enumerate(nodes):
            future = node.setup.remote(
                # rank=rank,
                # world_size=self.num_nodes,
            )
            setup_futures.append(future)

        # Wait for all setups to complete
        ray.get(setup_futures)
        return nodes
