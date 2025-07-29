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

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import ray
import torch
from hydra.utils import instantiate
from omegaconf import MISSING
from torch import nn
from typeguard import typechecked

from .algorithms.BaseAlgorithm import BaseAlgorithm
from .algorithms.configs import AlgorithmConfig
from .communicator.BaseCommunicator import BaseCommunicator
from .communicator.configs import BaseCommunicatorConfig
from .data.configs import DataModuleConfig
from .data.DataModule import DataModule
from .mixins import SetupMixin
from .models.configs import ModelConfig


@dataclass
class RayActorConfig:
    """
    Ray actor options for Node resource allocation and scheduling.

    Contains the same options available in Ray's .options() method for controlling
    CPU/GPU assignment, memory limits, fault tolerance, and scheduling behavior.

    Most users can ignore this - defaults work for typical FL experiments.
    Useful for resource-constrained environments or when you need specific hardware.

    Example overrides in topology configs:
    ```yaml
    overrides:
      0: {ray_actor_options: {num_cpus: X.X, memory: NNNNNN}}  # Server
      1: {ray_actor_options: {num_gpus: X.X, accelerator_type: "TYPE"}}  # Client
    ```

    Reference: https://docs.ray.io/en/latest/ray-core/api/doc/ray.actor.ActorClass.options.html
    """

    # The quantity of CPU cores to reserve for the lifetime of the actor
    num_cpus: Optional[float] = None

    # The quantity of GPUs to reserve for the lifetime of the actor
    # None = automatic allocation, 0 = CPU-only, 0.5 = fractional sharing, 1+ = full GPUs
    num_gpus: Optional[float] = None

    # The quantity of various custom resources to reserve for the lifetime of the actor.
    # Dictionary mapping strings (resource names) to floats
    resources: Optional[Dict[str, float]] = None

    # Requires that the actor run on a node which meets the specified label conditions
    label_selector: Optional[Dict[str, str]] = None

    # Requires that the actor run on a node with the specified type of accelerator
    accelerator_type: Optional[str] = None

    # The heap memory request in bytes for this actor, rounded down to the nearest integer
    memory: Optional[int] = None

    # The object store memory request for actors only
    object_store_memory: Optional[int] = None

    # Maximum number of times the actor should be restarted when it dies unexpectedly.
    # 0 = no restarts (default), -1 = infinite restarts
    max_restarts: int = 0

    # How many times to retry an actor task if the task fails due to a runtime error.
    # 0 = no retries (default), -1 = retry until max_restarts limit, n > 0 = retry up to n times
    max_task_retries: int = 0

    # Max number of pending calls allowed on the actor handle. -1 = unlimited
    # max_pending_calls: int = -1

    # Max number of concurrent calls to allow for this actor (direct calls only).
    # Defaults to 1 for threaded execution, 1000 for asyncio execution
    # max_concurrency: Optional[int] = None

    # The globally unique name for the actor, retrievable via ray.get_actor(name)
    # name: Optional[str] = None

    # Override the namespace to use for the actor. Default is anonymous namespace
    namespace: Optional[str] = None

    # Actor lifetime: None (fate share with creator) or "detached" (global object)
    # lifetime: Optional[str] = None

    # Runtime environment for this actor and its children
    runtime_env: Optional[Dict[str, Any]] = None

    # Scheduling strategy: None, "DEFAULT", "SPREAD", or placement group strategies
    scheduling_strategy: Optional[str] = None

    # Extended options for Ray libraries (e.g., workflows)
    # _metadata: Optional[Dict[str, Any]] = None

    # True if task events from the actor should be reported (tracing)
    # enable_task_events: bool = True


@dataclass
class NodeConfig:
    """
    Configuration for individual federated learning nodes.

    Defines a node's identity, communication partners, and resource requirements.
    Topologies create these automatically - you typically override specific settings
    rather than creating from scratch.

    Common overrides in topology configs:
    ```yaml
    overrides:
      0: {device_hint: "cpu", ray_actor_options: {num_cpus: X.X}}  # Server settings
      1: {device_hint: "cuda:X"}  # Client gets GPU
    ```

    See conf/ directory for working topology examples.
    """

    # Unique identifier for this node within the federated learning topology.
    # Used for logging, debugging, and actor naming. Must be unique per experiment.
    name: str = MISSING

    # Local communication configuration for intra-group federated learning operations.
    # Handles model aggregation within the same communication group (e.g., local cluster).
    # Required - must specify either TorchDist (NCCL/Gloo) or GRPC communicator.
    local_comm: BaseCommunicatorConfig = MISSING

    # Global communication configuration for inter-group federated learning operations.
    # Used by hierarchical topologies where local groups communicate with global coordinators.
    # Optional - None for purely local/centralized topologies.
    global_comm: Optional[BaseCommunicatorConfig] = MISSING

    # Ray actor configuration options for distributed execution.
    # Controls resource allocation, fault tolerance, and scheduling behavior.
    # Default creates actor with Ray defaults (no special resource requirements).
    ray_actor_options: RayActorConfig = field(default_factory=RayActorConfig)

    # Device hint for this node's computation placement
    device_hint: str = "auto"

    # Experiment directory for this node's log files
    exp_dir: Optional[str] = "/tmp/flora"


@ray.remote
class Node(SetupMixin):
    """
    Distributed federated learning participant (server or client).

    Ray actor that executes FL algorithms with local data and model state.
    Each node manages its own training loop, model updates, and communication with other nodes.

    Nodes execute FL rounds autonomously once Engine calls run_experiment().

    See conf/ directory for topology examples.
    """

    def __init__(
        self,
        name: str,
        local_comm: BaseCommunicatorConfig,
        global_comm: Optional[BaseCommunicatorConfig],
        algorithm: AlgorithmConfig,
        model: ModelConfig,
        datamodule: DataModuleConfig,
        device_hint: str,
        exp_dir: str,
        **kwargs,  # Accept additional config fields (e.g., ray_actor_options)
    ):
        """
        Initialize federated learning node with configs.

        Args:
            name: Unique node identifier (e.g., "0.1" for group 0, rank 1)
            local_comm: Communication config for intra-group coordination
            global_comm: Communication config for inter-group coordination (hierarchical only)
            algorithm: FL algorithm config
            model: Neural network model config
            datamodule: Data loading and preprocessing config
            device_hint: Device placement ("auto", "cpu", "cuda:X", etc.)
            exp_dir: Base experiment directory for logs and outputs
        """
        super().__init__()
        self.name: str = name
        self.device_hint: str = device_hint
        self.log_dir: str = os.path.join(exp_dir, name)

        # Store config for model (instantiated during setup)
        self.model_cfg: ModelConfig = model

        # Instantiate components with setup phases
        self.local_comm: BaseCommunicator = instantiate(local_comm)
        self.global_comm: Optional[BaseCommunicator] = (
            instantiate(global_comm) if global_comm else None
        )
        self.algorithm: BaseAlgorithm = instantiate(algorithm, log_dir=self.log_dir)
        self.datamodule: DataModule = instantiate(datamodule)

        # Deferred instantiation
        self.__model: Optional[nn.Module] = None  # Setup-time instantiation
        self.__device: Optional[torch.device] = (
            None  # Runtime-dependent lazy initialization
        )

    def _setup(self) -> None:
        """
        Instantiate remaining components and establish connections.

        Called by Engine after all nodes are created but before experiment starts.
        Instantiates model, establishes communicator connections,
        and passes dependencies to algorithm.
        """
        self.__model = instantiate(self.model_cfg)
        if self.__model is None:
            raise RuntimeError(
                f"Failed to instantiate model from config: {self.model_cfg}"
            )

        print(f"[NODE-SETUP] Device: {self.device}", flush=True)

        # Establish communicator connections
        self.local_comm.setup()
        if self.global_comm:
            self.global_comm.setup()

        # Initialize algorithm with dependencies
        self.algorithm.setup(
            self.local_comm, self.global_comm, self.model, self.datamodule
        )

    def run_experiment(self, total_rounds: int) -> List[List[dict[str, float]]]:
        """
        Execute federated learning experiment autonomously.

        Runs the complete experiment lifecycle.
        Moves model to compute device, executes all FL rounds via the algorithm.
        Collects metrics and restores model to original device afterward.

        Args:
            total_rounds: Number of federated learning rounds to execute

        Returns:
            List of rounds, each containing epoch-level metrics dictionaries
        """
        if not self.is_ready:
            raise RuntimeError("Node not ready - call setup() first")

        print(
            f"[EXPERIMENT-START] Node starting {total_rounds} round experiment",
            flush=True,
        )

        # Device management for experiment execution
        original_device = next(self.model.parameters()).device
        self.model = self.model.to(self.device)

        all_round_metrics = []

        try:
            for round_idx in range(total_rounds):
                epoch_metrics_list = self.algorithm.round_exec(round_idx, total_rounds)
                all_round_metrics.append(epoch_metrics_list)

            print(
                f"[EXPERIMENT-END] Node completed {total_rounds} round experiment",
                flush=True,
            )

        finally:
            # Restore original device placement
            self.model = self.model.to(original_device)
            print(
                f"[EXPERIMENT-CLEANUP] Model restored to original device: {original_device}",
                flush=True,
            )

        return all_round_metrics

    def __repr__(self) -> str:
        """Node string representation with name and timestamp."""
        _time = time.strftime("%H:%M:%S", time.gmtime())
        return f"{self.name} {_time}"

    @property
    def device(self) -> torch.device:
        """Compute device with automatic GPU assignment based on rank."""
        if self.__device is None:
            self.__device = self.__resolve_device(
                self.device_hint, rank=self.local_comm.rank
            )
        return self.__device

    @property
    def model(self) -> nn.Module:
        """Neural network model, instantiated during setup."""
        if self.__model is None:
            raise RuntimeError("Model accessed before setup() - call setup() first")
        return self.__model

    @model.setter
    def model(self, value: nn.Module) -> None:
        """Update model (used during FL training)."""
        self.__model = value

    @staticmethod
    def __resolve_device(device_hint: str, rank: Optional[int] = None) -> torch.device:
        """
        Resolve device placement for this node.

        Args:
            device_hint: Device specification ("auto", "cpu", "cuda", "cuda:X", etc.)
            rank: Process rank for round-robin GPU assignment (optional)

        Returns:
            PyTorch device for computation
        """
        if device_hint != "auto":
            print(f"[NODE-DEVICE] Explicit: {device_hint}")
            return torch.device(device_hint)

        # Auto-assignment with GPU detection
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print("[NODE-DEVICE] Auto: CPU (no GPUs available)")
            return torch.device("cpu")

        # Round-robin GPU assignment
        effective_rank = rank if rank is not None else 0
        if rank is None:
            print("WARN: No rank provided, defaulting to GPU 0")

        gpu_id = effective_rank % gpu_count
        device_str = f"cuda:{gpu_id}"
        print(
            f"[NODE-DEVICE] Auto: {device_str} (rank {effective_rank}, {gpu_count} GPUs)"
        )
        return torch.device(device_str)
