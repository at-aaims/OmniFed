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
from typing import Any, Dict, List, Optional

import rich.repr
from omegaconf import DictConfig

from .. import utils
from ..Node import NodeSpec

# ======================================================================================


@rich.repr.auto
class BaseTopology(ABC):
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

        # Lazy-initialized node specifications list
        self.__node_specs: Optional[List[NodeSpec]] = None

    @property
    def node_comm_specs(self) -> List[NodeSpec]:
        """
        Get the list of node specifications for this topology.
        """
        if self.__node_specs is None:
            self.__node_specs = self.create_node_specs()
            print(
                f"[{self.__class__.__name__}] Configured {len(self.__node_specs)} nodes"
            )

        if not self.__node_specs:
            raise ValueError(
                "Topology.configure_nodes() must return at least one node specification."
            )

        return self.__node_specs

    @abstractmethod
    def create_node_specs(self) -> List[NodeSpec]:
        """
        Configure node specifications for this topology.

        This method is responsible for:
        1. Creating node specifications with communication parameters
        2. Establishing communication relationships between nodes
        3. Setting up any topology-specific networking

        Returns:
            List of NodeSpec objects defining network structure
        """
        pass

    def __len__(self) -> int:
        """
        Get the number of node specifications in this topology.
        """
        return len(self.node_comm_specs)

    def __getitem__(self, index: int) -> NodeSpec:
        """
        Get a node specification by index.
        """
        return self.node_comm_specs[index]

    def __iter__(self):
        """
        Iterate over node specifications in this topology.
        """
        return iter(self.node_comm_specs)
