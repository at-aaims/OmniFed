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
# See the License for the specific language governing conditions and
# limitations under the License.

from typing import Dict, List, Optional, cast

import rich.repr
from omegaconf import OmegaConf

from ..algorithm import BaseAlgorithmConfig
from ..communicator import BaseCommunicatorConfig
from ..data import DataModuleConfig
from ..model import ModelConfig
from ..execution.config import NodeConfig
from ..utils import print
from .base import BaseTopology

# ======================================================================================


@rich.repr.auto
class DecentralizedTopology(BaseTopology):
    """
    Single-level peer topology: every rank trains; world size equals trainer count.

    Pair with TorchDist (NCCL/Gloo all-reduce). There is no parameter-server rank.
    gRPC parameter-server jobs should keep CentralizedTopology (world = clients + 1).

    Example:
    - num_clients: 6 → ranks 0..5, all trainers, slurm.nodes=6
    """

    has_server: bool = False

    def __init__(
        self,
        num_clients: int,
        local_comm: Optional[BaseCommunicatorConfig] = None,
        overrides: Optional[Dict[int, NodeConfig]] = None,
        has_server: bool = False,
    ):
        super().__init__()
        self.num_clients: int = num_clients
        self.local_comm: Optional[BaseCommunicatorConfig] = local_comm
        self.overrides: Dict[int, NodeConfig] = overrides or {}
        self.has_server = False
        if has_server:
            raise ValueError(
                "DecentralizedTopology cannot have a server rank; every node trains"
            )
        print(self)

    def process_world_size(self) -> int:
        """Slurm / TorchDist world size: one process per trainer."""
        return self.num_clients

    def _setup(
        self,
        default_algorithm_cfg: BaseAlgorithmConfig,
        default_model_cfg: ModelConfig,
        default_datamodule_cfg: DataModuleConfig,
    ) -> List[NodeConfig]:
        if self.local_comm is None:
            raise ValueError(
                "DecentralizedTopology._setup needs local_comm "
                "(1-GPU TorchDist). Titan-only runs skip this setup."
            )
        world_size: int = self.process_world_size()
        node_configs: List[NodeConfig] = []

        for rank in range(world_size):
            local_comm_cfg: BaseCommunicatorConfig = OmegaConf.structured(
                self.local_comm
            )
            local_comm_cfg.rank = rank
            local_comm_cfg.world_size = world_size

            base_node_cfg = NodeConfig(
                name=f"Node0.{rank}",
                local_comm=local_comm_cfg,
                global_comm=None,
                algorithm=default_algorithm_cfg,
                model=default_model_cfg,
                datamodule=default_datamodule_cfg,
                has_server=False,
            )

            node_cfg = OmegaConf.merge(
                OmegaConf.structured(base_node_cfg),
                OmegaConf.structured(self.overrides.get(rank, {})),
            )
            node_cfg = cast(NodeConfig, node_cfg)
            node_configs.append(node_cfg)

        return node_configs
