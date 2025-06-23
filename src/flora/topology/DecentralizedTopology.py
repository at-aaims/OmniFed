from typing import List, Dict, Any

import math
import ray
import rich.repr
import torch
from omegaconf import DictConfig, OmegaConf

from .. import utils
from ..Node import Node, NodeRole
from .BaseTopology import Topology


# ======================================================================================


@rich.repr.auto
class DecentralizedTopology(Topology):
    """
    Decentralized federated learning topology with peer-to-peer communication.

    - All nodes are both trainers and aggregators
    - Nodes communicate directly with their peers
    - No central coordinator
    """

    def __init__(
        self, node_defaults: DictConfig, num_nodes: int = 5, connectivity: float = 0.5
    ):
        """
        Initialize decentralized topology.

        Args:
            node_defaults: Default configuration for nodes
            num_nodes: Total number of nodes in the topology
            connectivity: Proportion of peers each node connects to (0.0-1.0)
        """
        self.connectivity = min(max(connectivity, 0.0), 1.0)  # Clamp to [0,1]
        super().__init__(node_defaults=node_defaults, num_nodes=num_nodes)

    def setup_nodes(self, node_defaults: DictConfig, num_nodes: int) -> List[Node]:
        """
        Create nodes for decentralized topology.

        In decentralized topology:
        - All nodes are peers (both train and aggregate)
        - Each node connects to a subset of other nodes based on connectivity

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

            # All nodes are both trainers and aggregators in decentralized topology
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

        # Configure peer relationships
        self._configure_peer_network(num_nodes)

        print(f"Configured decentralized topology with {len(nodes)} nodes")
        return nodes

    def _configure_peer_network(self, num_nodes: int) -> None:
        """
        Configure peer relationships for decentralized topology.

        Sets up which nodes are connected to each other based on
        the connectivity parameter.
        """
        # Calculate how many peers each node should connect to
        peers_per_node = max(1, math.floor((num_nodes - 1) * self.connectivity))

        # Create adjacency list for the peer network
        self.peer_map = {}  # rank -> set of peer ranks

        for rank in range(num_nodes):
            # Determine peers for this node
            self.peer_map[rank] = set()

            # Simple strategy: connect to the next peers_per_node nodes (wrapping around)
            for i in range(1, peers_per_node + 1):
                peer_rank = (rank + i) % num_nodes
                self.peer_map[rank].add(peer_rank)
                # Make connections bidirectional
                self.peer_map.setdefault(peer_rank, set()).add(rank)

        # Print topology summary
        avg_peers = sum(len(peers) for peers in self.peer_map.values()) / num_nodes
        print(
            f"Connectivity: {self.connectivity:.2f}, Average peers per node: {avg_peers:.2f}"
        )
