from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import MISSING

from ..algorithm import BaseAlgorithmConfig
from ..communicator import BaseCommunicatorConfig
from ..data import DataModuleConfig
from ..model import ModelConfig


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

    # ---
    # The quantity of CPU cores to reserve for the lifetime of the actor
    num_cpus: Optional[float] = None

    # ---
    # The quantity of GPUs to reserve for the lifetime of the actor
    # None = automatic allocation, 0 = CPU-only, 0.5 = fractional sharing, 1+ = full GPUs
    num_gpus: Optional[float] = None

    # ---
    # The quantity of various custom resources to reserve for the lifetime of the actor.
    # Dictionary mapping strings (resource names) to floats
    resources: Optional[Dict[str, float]] = None

    # ---
    # Requires that the actor run on a node which meets the specified label conditions
    label_selector: Optional[Dict[str, str]] = None

    # ---
    # Requires that the actor run on a node with the specified type of accelerator
    accelerator_type: Optional[str] = None

    # ---
    # The heap memory request in bytes for this actor, rounded down to the nearest integer
    memory: Optional[int] = None

    # ---
    # The object store memory request for actors only
    # object_store_memory: Optional[int] = None

    # ---
    # Maximum number of times the actor should be restarted when it dies unexpectedly.
    # 0 = no restarts (default), -1 = infinite restarts
    max_restarts: int = 0

    # ---
    # How many times to retry an actor task if the task fails due to a runtime error.
    # 0 = no retries (default), -1 = retry until max_restarts limit, n > 0 = retry up to n times
    max_task_retries: int = 0

    # ---
    # Max number of pending calls allowed on the actor handle. -1 = unlimited
    # max_pending_calls: int = -1

    # ---
    # Max number of concurrent calls to allow for this actor (direct calls only).
    # Defaults to 1 for threaded execution, 1000 for asyncio execution
    # max_concurrency: Optional[int] = None

    # ---
    # The globally unique name for the actor, retrievable via ray.get_actor(name)
    # name: Optional[str] = None

    # ---
    # Override the namespace to use for the actor. Default is anonymous namespace
    namespace: Optional[str] = None

    # ---
    # Actor lifetime: None (fate share with creator) or "detached" (global object)
    # lifetime: Optional[str] = None

    # ---
    # Runtime environment for this actor and its children
    runtime_env: Optional[Dict[str, Any]] = None

    # ---
    # Scheduling strategy: None, "DEFAULT", "SPREAD", or placement group strategies
    scheduling_strategy: Optional[str] = None

    # ---
    # Extended options for Ray libraries (e.g., workflows)
    # _metadata: Optional[Dict[str, Any]] = None

    # ---
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

    # ---
    # Unique identifier for this node within the federated learning topology.
    # Used for logging, debugging, and actor naming. Must be unique per experiment.
    name: str = MISSING

    # ---
    # Local communication configuration for intra-group federated learning operations.
    # Handles model aggregation within the same communication group (e.g., local cluster).
    # Required - must specify either TorchDist (NCCL/Gloo) or GRPC communicator.
    local_comm: BaseCommunicatorConfig = MISSING

    # ---
    # Global communication configuration for inter-group federated learning operations.
    # Used by hierarchical topologies where local groups communicate with global coordinators.
    # Optional - None for purely local/centralized topologies.
    global_comm: Optional[BaseCommunicatorConfig] = MISSING

    # ---
    # Algorithm configuration for federated learning logic
    # Defines which FL algorithm this node runs (FedAvg, FedProx, etc.)
    algorithm: BaseAlgorithmConfig = MISSING

    # ---
    # Model configuration for neural network architecture
    # Defines the PyTorch model this node will train
    model: ModelConfig = MISSING

    # ---
    # DataModule configuration for data loading and preprocessing
    # Defines how this node loads and processes its training/evaluation data
    datamodule: DataModuleConfig = MISSING

    # ---
    # Ray actor configuration options for distributed execution.
    # Controls resource allocation, fault tolerance, and scheduling behavior.
    # Default creates actor with Ray defaults (no special resource requirements).
    ray_actor_options: RayActorConfig = field(default_factory=RayActorConfig)

    # ---
    # Device hint for this node's computation placement
    device_hint: str = "auto"

    # ---
    # Experiment directory for this node's log files
    # None defaults to Hydra's output directory
    log_dir_base: Optional[str] = None