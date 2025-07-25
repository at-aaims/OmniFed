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
from dataclasses import dataclass
from typing import Optional

import ray
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

# from torch.utils.tensorboard import SummaryWriter
from .algorithms.BaseAlgorithm import BaseAlgorithm
from .communicator.BaseCommunicator import BaseCommunicator
from .data.DataModule import DataModule
from .mixins import SetupMixin

# class NodeRole(Enum):
#     """
#     Defines the fundamental capabilities of a Node participant in the federation.

#     Each role represents a specific capability.
#     Nodes can have multiple roles to express their full set of capabilities.
#     """

#     AGGREGATOR = "Aggregator"  # Aggregates updates from other nodes
#     TRAINER = "Trainer"  # Performs local training

#     # Future additions??
#     # COORDINATOR = "coordinator"  # Coordinates communication/scheduling
#     # RELAY = "relay"             # Forwards messages between nodes
#     # VALIDATOR = "validator"      # Validates updates or models


@dataclass
class NodeConfig:
    """Configuration for creating a Ray actor Node."""

    # Node identity
    id: str
    # roles: list[str]  # List of NodeRole names
    # ---
    # Algorithm / Model / Data
    algorithm_cfg: DictConfig
    model_cfg: DictConfig
    datamodule_cfg: DictConfig
    # ---
    # Communicators
    local_comm_cfg: DictConfig
    global_comm_cfg: Optional[DictConfig] = None  # For inter-group communication
    # ---
    # Miscellaneous
    device: str = "auto"


@ray.remote
class Node(SetupMixin):
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

    def __init__(self, cfg: NodeConfig):
        super().__init__()
        print(f"[NODE-INIT] {cfg.id}")
        self.cfg = cfg
        # self.roles: Set[NodeRole] = roles

        # Local communicator for intra-group communication
        self.local_comm: BaseCommunicator = instantiate(cfg.local_comm_cfg)

        # Global communicator for inter-group coordination (optional)
        self.global_comm: Optional[BaseCommunicator] = None
        if cfg.global_comm_cfg is not None:
            self.global_comm = instantiate(cfg.global_comm_cfg)

        # PyTorch model
        self.local_model: nn.Module = instantiate(cfg.model_cfg)

        # Data module
        self.datamodule: DataModule = instantiate(cfg.datamodule_cfg)

        # TensorBoard setup
        # self.tb_writer: SummaryWriter = SummaryWriter(log_dir=log_dir)

        # Federated learning algorithm
        self.algorithm: BaseAlgorithm = instantiate(
            cfg.algorithm_cfg,
            local_comm=self.local_comm,
            global_comm=self.global_comm,
            local_model=self.local_model,
            datamodule=self.datamodule,
            tb_writer=None,
        )

        # Extract rank information from local communicator for device selection and other uses
        self.device: torch.device = self.select_device(
            cfg.device, rank=self.local_comm.rank
        )

    def __repr__(self) -> str:
        """
        String representation showing node ID and capabilities.

        Returns:
            Formatted string with node ID and role list
        """
        # role_names = [role.value for role in self.roles]
        # return f"Node {self.id}: {role_names}"
        _time = time.strftime("%H:%M:%S", time.gmtime())
        # _time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        prefix = f"G{self.global_comm.rank}" if self.global_comm else ""
        prefix += f"L{self.local_comm.rank}"

        return f"{prefix}_{self.cfg.id} {_time}"

    def _setup(self) -> None:
        """
        Setup node with device assignment and algorithm initialization.
        """
        print(f"[NODE-SETUP] Device: {self.device}", flush=True)

        # Setup primary communicator
        self.local_comm.setup()

        # Setup global communicator if present
        if self.global_comm is not None:
            self.global_comm.setup()

        # Setup algorithm
        self.algorithm.setup(device=self.device)
        # summary(self.model, verbose=1)

    def run_experiment(self, total_rounds: int) -> dict[str, float]:
        """
        Execute complete federated learning experiment autonomously.

        Handles the full experiment lifecycle:
        - Experiment start evaluation
        - All training rounds
        - Experiment end evaluation

        Args:
            total_rounds: Total number of federated learning rounds to execute

        Returns:
            Final metrics from the completed experiment
        """
        if not self.is_ready:
            raise RuntimeError("Node not ready - call setup() first")

        print(
            f"[EXPERIMENT-START] Node starting {total_rounds} round experiment",
            flush=True,
        )

        # Experiment start evaluation
        if self.algorithm.eval_schedule.experiment_start:
            self.algorithm.run_eval_epoch(self.local_model, "global")
        # Execute all rounds
        final_metrics = {}
        for round_idx in range(total_rounds):
            round_metrics = self.algorithm.round_exec(round_idx)
            final_metrics.update(round_metrics)  # Keep accumulating metrics

        # Experiment end evaluation
        if self.algorithm.eval_schedule.experiment_end:
            self.algorithm.run_eval_epoch(self.local_model, "global")

        print(
            f"[EXPERIMENT-END] Node completed {total_rounds} round experiment",
            flush=True,
        )
        return final_metrics

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
                    f"[NODE-DEVICE] {local_gpu_count} GPUs detected | assigned {device_str} | rank {rank}"
                )
                return torch.device(device_str)

            print("[NODE-DEVICE] Device auto-selection: No GPUs detected, using CPU")
            return torch.device("cpu")

        print(f"[NODE-DEVICE] Device explicit: Using {device_hint}")
        return torch.device(device_hint)
