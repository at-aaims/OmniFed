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
from ..Node import NodeConfig

# ======================================================================================


@rich.repr.auto
class BaseTopology(ABC):
    """
    Abstract network topology interface.

    Defines the structure and coordination patterns for distributed nodes.

    Each topology implementation determines how nodes are configured and how they communicate with each other.
    """

    def __init__(
        self,
        algo_cfg: DictConfig,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        **kwargs: Any,
    ):
        """
        Initialize topology.
        """
        utils.log_sep(f"{self.__class__.__name__} Init")
        self.algo_cfg: DictConfig = algo_cfg
        self.model_cfg: DictConfig = model_cfg
        self.data_cfg: DictConfig = data_cfg
        self.node_kwargs: Dict[str, Any] = kwargs

        # Lazy-initialized node configurations list
        self.__node_configs: Optional[List[NodeConfig]] = None

    @property
    def node_configs(self) -> List[NodeConfig]:
        """
        Get the list of node configurations in this topology.
        """
        if self.__node_configs is None:
            self.__node_configs = self.configure_nodes(
                algo_cfg=self.algo_cfg,
                model_cfg=self.model_cfg,
                data_cfg=self.data_cfg,
                **self.node_kwargs,
            )
            print(
                f"[{self.__class__.__name__}] Configured {len(self.__node_configs)} nodes"
            )

        if not self.__node_configs:
            raise ValueError(
                "Topology.configure_nodes() must return at least one node configuration."
            )

        return self.__node_configs

    @abstractmethod
    def configure_nodes(
        self,
        algo_cfg: DictConfig,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        **kwargs: Any,
    ) -> List[NodeConfig]:
        """
        Configure node configurations for this topology.

        This method is responsible for:
        1. Creating node configurations with appropriate parameters
        2. Establishing communication relationships between nodes
        3. Setting up any topology-specific state

        Returns:
            List of node configuration dictionaries
        """
        pass

    def __len__(self) -> int:
        """
        Get the number of node configurations in this topology.
        """
        return len(self.node_configs)

    def __getitem__(self, index: int) -> NodeConfig:
        """
        Get a node configuration by index.
        """
        return self.node_configs[index]

    def __iter__(self):
        """
        Iterate over node configurations in this topology.
        """
        return iter(self.node_configs)
