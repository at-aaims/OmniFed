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
from typing import List

import rich.repr
from omegaconf import DictConfig

from .. import utils
from ..Node import Node

# ======================================================================================


@rich.repr.auto
class Topology(ABC):
    """
    Abstract network topology interface.

    Defines the structure and coordination patterns for distributed nodes.

    Each topology implementation determines how nodes are configured and how they communicate with each other.
    """

    def __init__(self):
        """
        Initialize topology.
        """
        utils.log_sep(f"{self.__class__.__name__} Init")
        self.__nodes: List[Node] = []

    def setup(
        self,
        comm_cfg: DictConfig,
        algo_cfg: DictConfig,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
    ):
        """
        Can be overridden if necessary
        """
        utils.log_sep("Topology Setup")

        self.__nodes = self.create_nodes(
            comm_cfg=comm_cfg,
            algo_cfg=algo_cfg,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
        )

        if not self.__nodes:
            raise ValueError("Topology.create_nodes() must return at least one node.")

    @abstractmethod
    def create_nodes(
        self,
        comm_cfg: DictConfig,
        algo_cfg: DictConfig,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
    ) -> List[Node]:
        """
        Create and configure nodes for this topology.

        This method is responsible for:
        1. Creating nodes with appropriate configuration
        2. Establishing communication relationships between nodes
        3. Setting up any topology-specific state

        Returns:
            List of configured nodes
        """
        pass

    def __len__(self) -> int:
        """
        Get the number of nodes in this topology.
        """
        return len(self.__nodes)

    def __getitem__(self, index: int) -> Node:
        """
        Get a node by index.
        """
        return self.__nodes[index]

    def __iter__(self):
        """
        Iterate over nodes in this topology.
        """
        return iter(self.__nodes)
