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
from dataclasses import asdict
from typing import Any, Dict, List

import ray
import rich.repr
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from . import utils
from .mixins import SetupMixin
from .Node import Node
from .topology.BaseTopology import BaseTopology
from .utils.ExperimentDisplay import ExperimentDisplay
from .utils.MetricFormatter import MetricFormatter

# Timing constants
LOG_FLUSH_DELAY = (
    1.0  # Prevent race condition between async log output and result display
)


@rich.repr.auto
class Engine(SetupMixin):
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
        utils.log_sep(f"{self.__class__.__name__} Init", color="blue")

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
        utils.log_sep("FLORA Engine Setup", color="blue")

        # Initialize Ray cluster with Hydra configuration
        ray.init(**self.flora_cfg.ray)

        # Smart GPU allocation: detect single-node vs multi-node scenarios
        cluster_resources = ray.available_resources()
        available_gpus = cluster_resources.get("GPU", 0)
        available_cpus = cluster_resources.get("CPU", 0)
        total_actors = len(list(self.topology))

        # Log all available cluster resources
        print(f"[CLUSTER-RESOURCES] {cluster_resources}")

        # Detect if running on single node (Ray local cluster)
        cluster_nodes = ray.nodes()
        alive_nodes = [node for node in cluster_nodes if node["Alive"]]
        is_single_node = len(alive_nodes) == 1

        # Determine GPU allocation strategy
        use_fractional_gpu = (
            is_single_node and total_actors > available_gpus and available_gpus > 0
        )

        # Log consolidated GPU allocation information
        if use_fractional_gpu:
            gpu_per_actor = available_gpus / total_actors
            strategy = f"fractional ({gpu_per_actor:.2f} GPU/actor)"
        elif available_gpus > 0:
            strategy = "full (1 GPU/actor)" + ("" if is_single_node else " multi-node")
        else:
            strategy = "CPU-only"

        print(
            f"[GPU-AUTO] {len(alive_nodes)} nodes, {available_cpus} CPUs, {available_gpus} GPUs, {total_actors} actors → {strategy}"
        )

        for node_config in self.topology:
            # Default to Hydra's output directory if node doesn't specify experiment directory
            node_config.exp_dir = (
                node_config.exp_dir or self.hydra_cfg.runtime.output_dir
            )

            # Configure Ray actor options with intelligent GPU allocation
            ray_actor_options = asdict(node_config.ray_actor_options)

            # If no explicit GPU configuration, auto-assign based on deployment scenario
            if ray_actor_options.get("num_gpus") is None and available_gpus > 0:
                if use_fractional_gpu:
                    # Single-node with GPU shortage: use fractional allocation
                    ray_actor_options["num_gpus"] = gpu_per_actor
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

    def start(self) -> None:
        """
        Run the federated learning experiment.

        Coordinates experiment execution across all nodes and displays results.
        """
        try:
            utils.log_sep("FLORA Engine Start", color="blue")

            print(
                f"# Starting {self.global_rounds} round federated learning experiment",
                flush=True,
            )
            print("# Nodes will execute autonomously", flush=True)

            experiment_start_time = time.time()

            experiment_futures = []
            for node in self._ray_actor_refs:
                future = node.run_experiment.remote(self.global_rounds)
                experiment_futures.append(future)

            print("# Waiting for nodes to complete experiments...", flush=True)
            results = ray.get(experiment_futures)

            experiment_end_time = time.time()
            experiment_duration = experiment_end_time - experiment_start_time

            utils.log_sep("FL Experiment Complete", color="blue")
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
            print("Engine shutting down...", flush=True)
            ray.shutdown()
