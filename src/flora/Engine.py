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
from dataclasses import asdict
from typing import Any, Dict, List

import ray
import rich.repr
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich.pretty import pprint

from . import utils
from .mixins import RequiredSetup
from .Node import Node
from .topology.BaseTopology import BaseTopology
from .utils import print
from .utils.ExperimentDisplay import ExperimentDisplay
from .utils.MetricFormatter import MetricFormatter

LOG_FLUSH_DELAY = 2.0


@rich.repr.auto
class Engine(RequiredSetup):
    """
    Central orchestrator for distributed federated learning experiments.

    When to use: This is the main entry point for running FL experiments.

    How it works:
    - Takes topology, algorithm, model, and data configurations from Hydra
    - Launches distributed Ray actors (nodes) based on topology specification
    - Coordinates experiment execution across all nodes
    - Collects and displays experiment results and metrics

    Example configs: See working examples in the `conf/` directory.
    """

    def __init__(
        self,
        flora_cfg: DictConfig,
    ):
        """
        Initialize federated learning experiment engine.

        Args:
            flora_cfg: Complete FLORA configuration including topology, algorithm,
                      model, and datamodule specifications
        """
        super().__init__()
        utils.print_rule()

        self.flora_cfg: DictConfig = flora_cfg
        self.hydra_cfg: HydraConf = HydraConfig.get()

        self.topology: BaseTopology = instantiate(
            self.flora_cfg.topology, _recursive_=False
        )
        self.global_rounds: int = flora_cfg.global_rounds

        self._formatter: MetricFormatter = MetricFormatter()
        self._display: ExperimentDisplay = ExperimentDisplay()

        self._ray_actor_refs: List[Node] = []

    def _setup(self) -> None:
        """
        Initialize Ray cluster and launch distributed nodes.

        Sets up the distributed infrastructure for FL experiment execution.
        """
        utils.print_rule()

        # Initialize Ray cluster with Hydra configuration
        ray.init(**self.flora_cfg.ray)

        # Smart GPU allocation: detect single-node vs multi-node scenarios
        ray_available_resources = ray.available_resources()
        print("ray.available_resources()")
        pprint(ray_available_resources)

        # Detect if running on single node (Ray local cluster)
        ray_nodes = ray.nodes()
        print("ray.nodes()")
        pprint(ray_nodes)

        ray_nodes_alive = [node for node in ray_nodes if node["Alive"]]
        is_single_node = len(ray_nodes_alive) == 1

        _available_gpus = ray_available_resources.get("GPU", 0)
        _total_actors = len(list(self.topology))

        # Determine GPU allocation strategy
        use_fractional_gpu = (
            is_single_node and _total_actors > _available_gpus and _available_gpus > 0
        )

        for node_config in self.topology:
            # Default to Hydra's output directory if node doesn't specify experiment directory
            node_config.exp_dir = (
                node_config.exp_dir or self.hydra_cfg.runtime.output_dir
            )

            # Configure Ray actor options with intelligent GPU allocation
            ray_actor_options = asdict(node_config.ray_actor_options)

            # If no explicit GPU configuration, auto-assign based on deployment scenario
            if ray_actor_options.get("num_gpus") is None and _available_gpus > 0:
                if use_fractional_gpu:
                    # Single-node with GPU shortage: use fractional allocation
                    ray_actor_options["num_gpus"] = _available_gpus / _total_actors
                else:
                    # Multi-node or sufficient GPUs: request 1 GPU per actor
                    ray_actor_options["num_gpus"] = 1

            node_actor = Node.options(**ray_actor_options).remote(
                **asdict(node_config),  # type: ignore - Ray's remote() typing doesn't understand dataclass unpacking
                algorithm=self.flora_cfg.algorithm,
                model=self.flora_cfg.model,
                datamodule=self.flora_cfg.datamodule,
            )
            self._ray_actor_refs.append(node_actor)

        setup_futures = [node.setup.remote() for node in self._ray_actor_refs]
        ray.get(setup_futures)

    def run_experiment(self) -> None:
        """
        Run the federated learning experiment.

        Coordinates experiment execution across all nodes and displays results.
        """
        try:
            utils.print_rule()
            print("Starting experiment...")

            experiment_start_time = time.time()

            experiment_futures = []
            for node in self._ray_actor_refs:
                future = node.run_experiment.remote(self.global_rounds)
                experiment_futures.append(future)

            print("Waiting for nodes to complete experiments...", flush=True)
            results = ray.get(experiment_futures)

            experiment_end_time = time.time()
            experiment_duration = experiment_end_time - experiment_start_time

            utils.print_rule()
            print("Finished Experiment")
            time.sleep(
                LOG_FLUSH_DELAY
            )  # Ensure async Ray logs complete before displaying results

            self._display.show_experiment_results(
                results,
                experiment_duration,
                self.global_rounds,
                len(self.topology),
            )

        finally:
            print("Shutting down...", flush=True)
            ray.shutdown()
