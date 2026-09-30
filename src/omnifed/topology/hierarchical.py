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

from typing import Any, Dict, List, Optional, Sequence, Union, cast

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


def _trainer_count(mpi_ranks_per_facility: Union[int, Sequence[int]], num_facilities: int) -> int:
    if isinstance(mpi_ranks_per_facility, int):
        if mpi_ranks_per_facility < 1:
            raise ValueError("mpi_ranks_per_facility must be >= 1")
        return int(mpi_ranks_per_facility) * int(num_facilities)
    seq = list(mpi_ranks_per_facility)
    if len(seq) != int(num_facilities):
        raise ValueError(
            f"mpi_ranks_per_facility length {len(seq)} must equal "
            f"num_facilities={num_facilities}"
        )
    return sum(int(w) for w in seq)


@rich.repr.auto
class HierarchicalTopology(BaseTopology):
    """
    Slurm two-hop FL: facilities (inner TorchDist) plus a dedicated outer gRPC rank.

    Flat global ranks for Engine / ``SLURM_NTASKS`` (same shape as centralized:
    rank 0 is the RPC-only server when ``dedicated_rpc_server`` is true).
    Facility membership is consumed by the hierarchical runner, not by Node.local_comm.
    """

    has_server: bool = True

    def __init__(
        self,
        num_facilities: int,
        mpi_ranks_per_facility: Union[int, Sequence[int]],
        local_comm: BaseCommunicatorConfig,
        dedicated_rpc_server: bool = True,
        overrides: Optional[Dict[int, NodeConfig]] = None,
        num_clients: Optional[int] = None,
        has_server: Optional[bool] = None,
        rpc_addr: str = "127.0.0.1",
        rpc_port: int = 50051,
        facility_mpi_addr: str = "127.0.0.1",
        facility_mpi_base_port: int = 28250,
        facility_mpi_port_stride: int = 40,
        facility_name_prefix: str = "fac",
        communicators: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if not dedicated_rpc_server:
            raise NotImplementedError(
                "HierarchicalTopology currently requires dedicated_rpc_server=True "
                "(rank 0 is the outer gRPC server)."
            )
        if int(num_facilities) < 1:
            raise ValueError("num_facilities must be >= 1")
        trainers = _trainer_count(mpi_ranks_per_facility, int(num_facilities))
        if num_clients is not None and int(num_clients) != trainers:
            raise ValueError(
                f"topology.num_clients={num_clients} must equal trainers="
                f"{trainers} (num_facilities × mpi_ranks_per_facility)"
            )
        self.num_facilities = int(num_facilities)
        self.mpi_ranks_per_facility = mpi_ranks_per_facility
        self.num_clients = trainers
        self.local_comm = local_comm
        self.overrides = overrides or {}
        self.dedicated_rpc_server = True
        self.has_server = True if has_server is None else bool(has_server)
        if not self.has_server:
            raise ValueError("HierarchicalTopology with dedicated RPC requires has_server=True")
        self.rpc_addr = rpc_addr
        self.rpc_port = int(rpc_port)
        self.facility_mpi_addr = facility_mpi_addr
        self.facility_mpi_base_port = int(facility_mpi_base_port)
        self.facility_mpi_port_stride = int(facility_mpi_port_stride)
        self.facility_name_prefix = str(facility_name_prefix)
        self.communicators = communicators
        print(self)

    def process_world_size(self) -> int:
        return self.num_clients + 1

    def _setup(
        self,
        default_algorithm_cfg: BaseAlgorithmConfig,
        default_model_cfg: ModelConfig,
        default_datamodule_cfg: DataModuleConfig,
    ) -> List[NodeConfig]:
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
            )
            node_cfg = OmegaConf.merge(
                OmegaConf.structured(base_node_cfg),
                OmegaConf.structured(self.overrides.get(rank, {})),
            )
            node_configs.append(cast(NodeConfig, node_cfg))
        return node_configs


@rich.repr.auto
class HierarchicalGroupsTopology(BaseTopology):
    """
    Ray multi-group graph (unused on Slurm this cut). Kept so Ray hierarchical
    is not deleted. Slurm jobs use :class:`HierarchicalTopology`.
    """

    def __init__(
        self,
        groups: List[BaseTopologyConfig],
        global_comm: BaseCommunicatorConfig,
    ):
        super().__init__()
        self.groups: List[BaseTopologyConfig] = groups
        self.global_comm: BaseCommunicatorConfig = global_comm
        if not groups:
            raise ValueError("At least one group must be specified")
        self.topologies: List[BaseTopology] = [
            instantiate(topology_cfg, _recursive_=False) for topology_cfg in self.groups
        ]
        print(self)

    def _setup(
        self,
        default_algorithm_cfg: BaseAlgorithmConfig,
        default_model_cfg: ModelConfig,
        default_datamodule_cfg: DataModuleConfig,
    ) -> List[NodeConfig]:
        for topology in self.topologies:
            topology.setup(
                default_algorithm_cfg=default_algorithm_cfg,
                default_model_cfg=default_model_cfg,
                default_datamodule_cfg=default_datamodule_cfg,
            )
        world_size = len(self.topologies)
        node_configs: List[NodeConfig] = []
        for topology_idx, topology in enumerate(self.topologies):
            for node_cfg in topology:
                node_cfg.name = f"Node{topology_idx}.{node_cfg.local_comm.rank}"
                if node_cfg.local_comm.rank == 0:
                    global_comm_cfg: BaseCommunicatorConfig = OmegaConf.structured(
                        self.global_comm
                    )
                    global_comm_cfg.rank = topology_idx
                    global_comm_cfg.world_size = world_size
                    merged_cfg = OmegaConf.merge(
                        node_cfg, {"global_comm": global_comm_cfg}
                    )
                    node_cfg = cast(NodeConfig, merged_cfg)
                node_configs.append(node_cfg)
        return node_configs
