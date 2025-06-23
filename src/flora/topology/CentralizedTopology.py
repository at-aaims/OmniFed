from typing import List

import rich.repr
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from .. import utils
from ..Node import Node, NodeRole
from .BaseTopology import Topology

# ======================================================================================


@rich.repr.auto
class CentralizedTopology(Topology):
    """
    Template for centralized federated learning topology with aggregator-trainer architecture.

    - One aggregator node (rank 0) collects and combines model updates
    - Multiple trainer nodes (ranks 1+) perform local training
    - All communication flows through the aggregator
    """

    def setup_nodes(self, node_defaults: DictConfig, num_nodes: int) -> List[Node]:
        """
        Create nodes for centralized topology.

        In centralized topology:
        - Rank 0: Aggregator (no training data, coordinates aggregation)
        - Ranks 1+: Trainers (training data, perform local training)

        Args:
            node_defaults: Default configuration for nodes
            num_nodes: Number of nodes to create

        Returns:
            List of configured nodes
        """
        utils.log_sep("Node Creation")
        print(f"Creating {num_nodes} distributed nodes")

        nodes: list[Node] = []

        # Calculate GPU resources
        node_options = dict()
        if torch.cuda.is_available():
            node_options.update(dict(num_gpus=1.0 / num_nodes))  # Create node actors

        for rank in range(num_nodes):
            # TODO: some of this should be streamlined a bit to look cleaner
            # Instantiate communicator from partial and inject runtime parameters
            comm_partial = instantiate(node_defaults.comm)
            comm = comm_partial(rank=rank, world_size=num_nodes)

            model = instantiate(node_defaults.model)
            loader = instantiate(node_defaults.loader)

            # Instantiate algorithm from partial and inject runtime dependencies
            algorithm_partial = instantiate(node_defaults.algorithm)
            algorithm = algorithm_partial(comm=comm, model=model, loader=loader)

            # Configure node roles based on rank
            if rank == 0:
                # Aggregator node
                roles = {NodeRole.AGGREGATOR}
            else:
                # Trainer node
                roles = {NodeRole.TRAINER}

            # Instantiate the node with pre-instantiated objects
            node = Node.options(**node_options).remote(
                id=f"N{rank}",
                comm=comm,
                roles=roles,
                model=model,
                loader=loader,
                algorithm=algorithm,
            )

            nodes.append(node)

        print(f"Configured centralized topology with {len(nodes)} nodes")
        print(f"Aggregator: Node 0, Trainers: Nodes 1-{len(nodes) - 1}")
        return nodes
