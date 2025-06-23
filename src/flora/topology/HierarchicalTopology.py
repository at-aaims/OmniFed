from typing import List, Dict, Any

import ray
import rich.repr
import torch
from omegaconf import DictConfig, OmegaConf

from .. import utils
from ..Node import Node, NodeRole
from .BaseTopology import Topology


# ======================================================================================


@rich.repr.auto
class HierarchicalTopology(Topology):
    """
    Hierarchical federated learning topology with multi-level aggregation.

    - Root node performs global aggregation
    - Intermediate nodes perform local aggregation and training
    - Leaf nodes perform only training
    - Updates flow up the tree from leaves to root
    """

    def __init__(
        self, node_defaults: DictConfig, num_nodes: int = 7, branching_factor: int = 2
    ):
        """
        Initialize hierarchical topology.

        Args:
            node_defaults: Default configuration for nodes
            num_nodes: Total number of nodes in the topology
            branching_factor: Number of children per non-leaf node
        """
        self.branching_factor = branching_factor
        super().__init__(node_defaults=node_defaults, num_nodes=num_nodes)

    def setup_nodes(self, node_defaults: DictConfig, num_nodes: int) -> List[Node]:
        """
        Create nodes for hierarchical topology.

        In hierarchical topology:
        - Rank 0: Root node (global aggregator, no training)
        - Intermediate ranks: Both aggregate locally and train
        - Leaf ranks: Only perform training

        Args:
            node_defaults: Default configuration for nodes
            num_nodes: Number of nodes to create

        Returns:
            List of configured nodes
        """
        utils.log_sep("Node Creation")
        print(f"Creating {num_nodes} distributed nodes")

        nodes = []

        # Calculate GPU resources
        node_options = dict()
        if torch.cuda.is_available():
            node_options.update(dict(num_gpus=1.0 / num_nodes))

        # Create node actors
        for rank in range(num_nodes):
            # Create a copy of node config for this specific node
            node_cfg = OmegaConf.create(node_defaults)
            node_cfg.comm.rank = rank
            node_cfg.comm.world_size = num_nodes

            # Determine node roles based on position in hierarchy
            if rank == 0:
                # Root node (global aggregator only)
                roles = {NodeRole.AGGREGATOR}
            elif self._is_leaf_node(rank, num_nodes):
                # Leaf node (trainer only)
                roles = {NodeRole.TRAINER}
            else:
                # Intermediate node (both trainer and aggregator)
                roles = {NodeRole.TRAINER, NodeRole.AGGREGATOR}

            # Instantiate the node
            node = Node.options(**node_options).remote(
                id=f"N{rank}",
                model_cfg=node_cfg.model,
                loader_cfg=node_cfg.loader,
                comm_cfg=node_cfg.comm,
                algorithm_cfg=node_cfg.algorithm,
                roles=roles,
            )

            nodes.append(node)

        # Configure hierarchical relationships
        self._configure_hierarchy(nodes, num_nodes)

        print(f"Configured hierarchical topology with {len(nodes)} nodes")
        return nodes

    def _configure_hierarchy(self, nodes: List[Node], num_nodes: int) -> None:
        """
        Configure hierarchical relationships between nodes.

        This sets up the parent-child relationships that define the tree structure.
        """
        # Store the tree structure for coordination
        self.parent_map = {}  # child_rank -> parent_rank
        self.children_map = {}  # parent_rank -> [child_ranks]

        # Configure parent-child relationships
        for rank in range(1, num_nodes):  # Skip root (rank 0)
            parent_rank = self._get_parent_rank(rank)

            # Update parent map
            self.parent_map[rank] = parent_rank

            # Update children map
            if parent_rank not in self.children_map:
                self.children_map[parent_rank] = []
            self.children_map[parent_rank].append(rank)

        # Print hierarchy summary
        intermediate_count = len(self.children_map) - 1  # Exclude root
        leaf_count = num_nodes - len(self.children_map)

        print(f"Root: Node 0, Branching Factor: {self.branching_factor}")
        print(f"Intermediate Nodes: {intermediate_count}, Leaf Nodes: {leaf_count}")

    def _get_parent_rank(self, rank: int) -> int:
        """Calculate the parent rank for a given node rank."""
        return (rank - 1) // self.branching_factor

    def _is_leaf_node(self, rank: int, num_nodes: int) -> bool:
        """Determine if a node is a leaf node based on its rank."""
        # A node is a leaf if it has no children
        first_child_rank = rank * self.branching_factor + 1
        return first_child_rank >= num_nodes
