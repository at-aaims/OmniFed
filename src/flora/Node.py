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

import time
from enum import Enum
from typing import Any, Optional, Set

import ray
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

from .algorithms import utils as alg_utils
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
        # roles: Set[NodeRole],
        comm_cfg: DictConfig,
        algo_cfg: DictConfig,
        model_cfg: DictConfig,
        data_cfg: Optional[DictConfig] = None,
        max_epochs: int = 1,  # TODO: think if this is the best place for this (take into consideration the Hydra config and user experience)
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        device: str = "auto",
        **kwargs: Any,
    ):
        print(f"{self.__class__.__name__} {id} init...")
        self.id: str = id
        # self.roles: Set[NodeRole] = roles
        self.max_epochs: int = max_epochs

        # Distributed computing context
        self.rank: Optional[int] = rank
        self.world_size: Optional[int] = world_size
        self.device: torch.device = self.select_device(device, rank=rank)

        # Communication backend instantiation
        self.comm: Communicator = instantiate(
            comm_cfg,
            rank=rank,
            world_size=world_size,
        )

        # PyTorch model
        self.local_model: nn.Module = instantiate(model_cfg)

        # Data module (optional - certain roles may not hold local data)
        self.datamodule: Optional[DataModule] = None
        if data_cfg is not None:
            self.datamodule = instantiate(data_cfg)

        # Federated learning algorithm (handles computation logic)
        self.algo: Algorithm = instantiate(
            algo_cfg,
            local_model=self.local_model,
            comm=self.comm,
        )

    def __repr__(self) -> str:
        """
        String representation showing node ID and capabilities.

        Returns:
            Formatted string with node ID and role list
        """
        # role_names = [role.value for role in self.roles]
        # return f"Node {self.id}: {role_names}"
        return f"{self.id}"

    @staticmethod
    def select_device(device_hint: str, rank: Optional[int] = None) -> torch.device:
        """
        Select and configure compute device for this node.

        Supports automatic GPU detection with round-robin assignment based on rank.
        Falls back to CPU if no GPUs are available.

        # TODO: Round-robin GPU assignment assumes all nodes are on the same machine, which may not be true for multi-node setups.
        # TODO: If some nodes lack GPUs, we may need smarter logic for heterogeneous environments.
        # TODO: Potentially move into a NodeResources mixin in the future.

        Args:
            device_hint: Device specification ("auto", "cpu", "cuda", "cuda:0", etc.)
            rank: Process rank for round-robin GPU assignment (optional)

        Returns:
            Configured PyTorch device for computation
        """

        if device_hint == "auto":
            local_gpu_count = torch.cuda.device_count()
            if local_gpu_count > 0:
                # Use provided rank for round-robin GPU assignment
                if rank is None:
                    print(
                        "WARN: No rank provided for device assignment, defaulting to GPU 0"
                    )
                    rank = 0

                assigned_gpu_id = rank % local_gpu_count
                device_str = f"cuda:{assigned_gpu_id}"
                print(
                    f"Device auto-selection: {local_gpu_count} GPUs detected, assigned {device_str} (rank {rank})"
                )
                return torch.device(device_str)

            print("Device auto-selection: No GPUs detected, using CPU")
            return torch.device("cpu")

        print(f"Device explicit: Using {device_hint}")
        return torch.device(device_hint)

    def setup(self) -> None:
        """
        Initialize node dependencies and prepare for federated learning execution.

        Sets up communication backend and prepares all components for distributed training.
        Called once before federated learning begins.
        """
        print("Setup: initializing communication backend", flush=True)
        self.comm.setup()

        # summary(self.model, verbose=1)

    def execute_round(self, round_idx: int) -> dict[str, float]:
        """
        Execute federated learning round with algorithm-controlled communication.
        Node provides communication infrastructure, Algorithm controls federated lifecycle.

        Args:
            round_idx: Current training round number

        Returns:
            Dictionary with training metrics and results
        """
        print(f"Round {round_idx} START", flush=True)
        round_start_time = time.time()

        # Reset round state (simple and explicit)
        self.algo.reset_round_state()

        # Model state before any synchronization
        metrics: dict[str, float] = dict(round_idx=round_idx)
        metrics["pnorm/before_sync"] = alg_utils.get_param_norm(self.algo.local_model)
        # Round Start Logic: e.g., Synchronization
        self.algo.round_start(round_idx)
        metrics["pnorm/after_sync"] = alg_utils.get_param_norm(self.algo.local_model)
        metrics["pnorm/sync_delta"] = (
            metrics["pnorm/after_sync"] - metrics["pnorm/before_sync"]
        )
        if metrics["pnorm/sync_delta"] <= 1e-6:
            print(
                f"WARN: Model param norm did not significantly change after synchronization in round {round_idx} "
                f"(Δ={metrics['pnorm/sync_delta']:.6f})"
            )

        # 4. Local training
        if self.datamodule is not None and self.datamodule.train is not None:
            print(f"Starting local training on {len(self.datamodule.train)} batches")
            # Pre-training metrics: Model state before local training
            metrics["pnorm/before_train_round"] = alg_utils.get_param_norm(
                self.algo.local_model
            )
            # Centralized device management: ensure model is on compute device
            self.algo.local_model.to(self.device)
            # Execute local training round
            self.algo.train_round(
                self.datamodule.train,
                round_idx,
                self.max_epochs,
            )
            # Post-training metrics: Model state after local training
            metrics["pnorm/after_train_round"] = alg_utils.get_param_norm(
                self.algo.local_model
            )
            metrics["pnorm/train_round_delta"] = (
                metrics["pnorm/after_train_round"] - metrics["pnorm/before_train_round"]
            )
            if metrics["pnorm/train_round_delta"] <= 1e-6:
                print(
                    f"WARN: Model param norm did not significantly change after local training in round {round_idx} "
                    f"(Δ={metrics['pnorm/train_delta']:.6f})"
                )
        else:
            print("WARN: No local training data available, skipping local training")

        # Round End Logic: e.g. Aggregation
        metrics["pnorm/before_agg"] = alg_utils.get_param_norm(self.algo.local_model)
        self.algo.round_end(round_idx)

        metrics["pnorm/after_agg"] = alg_utils.get_param_norm(self.algo.local_model)
        metrics["pnorm/agg_delta"] = (
            metrics["pnorm/after_agg"] - metrics["pnorm/before_agg"]
        )
        if metrics["pnorm/agg_delta"] <= 1e-6:
            print(
                f"WARN: Model param norm did not significantly change after aggregation in round {round_idx} "
                f"(Δ={metrics['pnorm/agg_delta']:.6f})"
            )

        # Collect all metrics from algorithm execution
        metrics.update(self.algo.metrics.to_dict())
        metrics["time/round"] = time.time() - round_start_time

        print(
            f"Round {round_idx} END |",
            {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()},
            flush=True,
        )
        return metrics
