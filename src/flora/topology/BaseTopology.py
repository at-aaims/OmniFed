from abc import ABC, abstractmethod
from typing import List

import rich.repr
from omegaconf import DictConfig

from ..Node import Node


# ======================================================================================


@rich.repr.auto
class Topology(ABC):
    """
    Abstract network topology interface.

    Defines the structure and coordination patterns for distributed nodes.

    Each topology implementation determines how nodes are configured and how they communicate with each other.
    """

    def __init__(self, num_nodes: int):
        """
        Initialize topology.

        Args:
            num_nodes: Number of nodes to create
        """
        self.num_nodes = num_nodes

    @abstractmethod
    def setup_nodes(self, node_defaults: DictConfig, num_nodes: int) -> List[Node]:
        """
        Create and configure nodes for this topology.

        This method is responsible for:
        1. Creating nodes with appropriate configuration
        2. Establishing communication relationships between nodes
        3. Setting up any topology-specific state

        Args:
            node_defaults: Default configuration for nodes
            num_nodes: Number of nodes to create

        Returns:
            List of configured nodes
        """
        pass
