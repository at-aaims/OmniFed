from enum import Enum
from typing import Any, Dict, Set

import ray
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from .algorithms.BaseAlgorithm import Algorithm
from .communicator.BaseCommunicator import Communicator


class NodeRole(Enum):
    """
    Defines the fundamental capabilities of a Node participant in the federation.

    Each role represents a specific capability.
    Nodes can have multiple roles to express their full set of capabilities.
    """

    AGGREGATOR = "aggregator"  # Aggregates updates from other nodes
    TRAINER = "trainer"  # Performs local training

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
        comm: Communicator,
        roles: Set[NodeRole],
        model: nn.Module,
        loader: DataLoader,
        algorithm: Algorithm,
    ):
        print(f"{self.__class__.__name__} {id} initializing...")
        self.id: str = id
        self.comm: Communicator = comm
        self.roles: Set[NodeRole] = roles

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: nn.Module = model
        self.loader: DataLoader = loader
        self.algorithm: Algorithm = algorithm

    def __repr__(self) -> str:
        role_names = [role.value for role in self.roles]
        return f"Node id={self.id}, roles={role_names}"

    def info(self):
        # TODO: this is not yet called anywhere, but useful info
        summary(self.model, verbose=1)

    def setup(self):
        """
        Perform any necessary setup operations.
        """
        self.comm.setup()
        # Move model to the appropriate device
        self.model.to(self.device)
        # TODO: maybe calling algorithm setup function here would be useful pattern? FOr e.g., could pass direct reference to this Node instance.

    def execute_round(self, round_num: int) -> Dict[str, Any]:
        """
        Train the model for one round using the configured algorithm.

        Args:
            round_num: Current training round number

        Returns:
            Dictionary with training metrics and results
        """
        results: Dict[str, Any] = dict()

        results.update(self.algorithm.on_round_start(round_num, results) or {})

        # TODO: Role-based delegation
        results.update(self.algorithm.on_local_round(round_num, results) or {})

        results.update(self.algorithm.on_round_end(round_num, results) or {})

        return results
