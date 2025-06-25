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

from enum import Enum
from typing import Any, Dict, Optional, Set

import ray
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

from .algorithms.BaseAlgorithm import Algorithm
from .communicator.BaseCommunicator import Communicator
from .dataset.DataModule import DataModule


class NodeRole(Enum):
    """
    Defines the fundamental capabilities of a Node participant in the federation.

    Each role represents a specific capability.
    Nodes can have multiple roles to express their full set of capabilities.
    """

    AGGREGATOR = "Aggregator"  # Aggregates updates from other nodes
    TRAINER = "Trainer"  # Performs local training

    # Future additions??
    # COORDINATOR = "coordinator"  # Coordinates communication/scheduling
    # RELAY = "relay"             # Forwards messages between nodes
    # VALIDATOR = "validator"      # Validates updates or models


@ray.remote
class Node:
    """
    Distributed compute node for federated learning participants.

    Responsibilities:
    - Execute local federated learning algorithm implementations
    - Manage local model state (and training data access? # TODO: local data may be unnecessary & redundant for some roles e.g. aggregator)
    - Own and manage local copy of global model model and communicator instances
    - Handle device management for hardware resources

    Integration:
    - Instantiated as Ray actors by the Engine for distributed execution
    - Configured through Hydra with algorithm and communication dependencies
    - Should enable any topology pattern through consistent and generalizable interfaces
    """

    def __init__(
        self,
        id: str,
        roles: Set[NodeRole],
        comm_cfg: DictConfig,
        algo_cfg: DictConfig,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        device: str = "auto",
        **kwargs: Any,
    ):
        print(f"{self.__class__.__name__} {id} init...")
        self.id: str = id
        self.roles: Set[NodeRole] = roles

        # ---
        self.rank: Optional[int] = rank
        self.world_size: Optional[int] = world_size

        self.device: torch.device = self.select_device(device, rank=rank)

        # ---
        # Instantiate Components
        self.comm: Communicator = instantiate(
            comm_cfg,
            rank=rank,
            world_size=world_size,
        )

        self.model: nn.Module = instantiate(model_cfg)
        self.data: DataModule = instantiate(data_cfg)

        self.algo: Algorithm = instantiate(
            algo_cfg,
            comm=self.comm,
            model=self.model,
        )

    def __repr__(self) -> str:
        role_names = [role.value for role in self.roles]
        return f"{self.id}: {role_names}"

    @staticmethod
    def select_device(device_hint: str, rank: Optional[int] = None) -> torch.device:
        """
        Setup device with local GPU detection and round-robin assignment.

        # TODO: Round-robin GPU assignment assumes all nodes are on the same machine, which may not be true for multi-node setups.
        # TODO: If some nodes lack GPUs, we may need smarter logic for heterogeneous environments.
        # TODO: Potentially move into a NodeResources mixin in the future.

        Args:
            device_hint: Device hint ("auto", "cpu", "cuda", "cuda:0", etc.)

        Returns:
            Configured torch device
        """

        if device_hint == "auto":
            local_gpu_count = torch.cuda.device_count()
            if local_gpu_count > 0:
                # Use provided rank if available; otherwise, default to 0 and warn.
                if rank is None:
                    print(
                        "WARN: No rank provided; defaulting to 0 for device assignment."
                    )
                    rank = 0

                assigned_gpu = rank % local_gpu_count
                device_str = f"cuda:{assigned_gpu}"
                print(f"Auto-detected {local_gpu_count} local GPUs, using {device_str}")
                return torch.device(device_str)

            print("No local GPUs detected, using CPU")
            return torch.device("cpu")

        print(f"Using explicit device {device_hint}")
        return torch.device(device_hint)

    def setup(self, **kwargs: Any) -> None:
        """Instantiate all dependencies with full runtime context."""
        print(f"setup: {kwargs}", flush=True)
        self.comm.setup()

        # TODO: there's probably a better place for this
        self.model.to(self.device)
        # summary(self.model, verbose=1)

    def execute_round(self, round_num: int) -> Dict[str, Any]:
        """
        Train the model for one round using the configured algorithm.

        Args:
            round_num: Current training round number

        Returns:
            Dictionary with training metrics and results
        """
        print(f"execute_round: round_num={round_num}")
        results: Dict[str, Any] = dict()

        results.update(
            self.algo.on_round_start(
                round_num,
                results,
            )
            or {}
        )

        # Basic Role-based delegation
        # TODO: Implement more complex role-based delegation logic that can generalize to any topology, algorithm, and configuration.
        # if NodeRole.TRAINER in self.roles:
        #     # Trainer nodes perform local training
        #     if self.data is None:
        #         raise ValueError(
        #             f"Expected Node {self.id} to have data for training, but no data module was provided."
        #         )

        results.update(
            self.algo.train_round(
                round_num=round_num,
                dataloader=self.data.train,
                metrics=results,
            )
            or {}
        )

        results.update(
            self.algo.on_round_end(
                round_num,
                results,
            )
            or {}
        )

        return results
