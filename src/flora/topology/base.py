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

from abc import abstractmethod
from typing import List

import rich.repr

from .. import utils
from ..algorithm import BaseAlgorithmConfig
from ..data import DataModuleConfig
from ..model import ModelConfig
from ..node import NodeConfig
from ..utils import RequiredSetup

# ======================================================================================


@rich.repr.auto
class BaseTopology(RequiredSetup):
    """
    Base class for federated learning network topologies.

    Defines how nodes are arranged and communicate in distributed FL experiments.
    Concrete implementations include CentralizedTopology and HierarchicalTopology.

    Quick decision guide:
    - Use CentralizedTopology: Single-site FL (all nodes can talk directly)
    - Use HierarchicalTopology: Multi-site FL (hospitals, institutions, etc.)
    - Extend BaseTopology: Custom communication patterns (advanced users)

    How it works:
    - Subclasses implement _setup() to define network structure
    - Engine calls setup() which creates NodeConfig objects that the Engine launches as Ray actors
    - Provides iteration interface for easy access to all nodes
    """

    def __init__(self):
        """
        Initialize topology base class.
        """
        super().__init__()
        utils.print_rule()

    @property
    def node_configs(self) -> List[NodeConfig]:
        """
        Get all node configurations for this topology.

        Requires setup() to have been called first.
        """
        # The setup result is cached by RequiredSetup mixin
        node_configs = self.setup_result

        if not node_configs:
            raise ValueError("_setup() must return at least one node configuration")

        return node_configs

    @abstractmethod
    def _setup(
        self,
        default_algorithm_cfg: BaseAlgorithmConfig,
        default_model_cfg: ModelConfig,
        default_datamodule_cfg: DataModuleConfig,
    ) -> List[NodeConfig]:
        """
        Create node configurations for this topology.

        Called by setup() method. Creates and returns node configurations
        that define the network structure.

        Subclasses must implement this to define:
        - How many nodes and their roles (server, client, etc.)
        - Communication patterns between nodes
        - Node naming conventions
        - Device assignments and resource requirements

        Args:
            default_algorithm_cfg: Default algorithm configuration for all nodes
            default_model_cfg: Default model configuration for all nodes
            default_datamodule_cfg: Default datamodule configuration for all nodes

        Returns:
            List of NodeConfig objects ready for Engine to launch as Ray actors
        """
        pass

    def __len__(self) -> int:
        """
        Get the number of nodes in this topology.
        """
        return len(self.node_configs)

    def __getitem__(self, index: int) -> NodeConfig:
        """
        Get a node configuration by index.
        """
        return self.node_configs[index]

    def __iter__(self):
        """
        Iterate over all node configurations in this topology.
        """
        return iter(self.node_configs)
