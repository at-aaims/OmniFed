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
from typing import List

import ray
import rich.repr
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich import box
from rich.table import Table

from . import utils
from .mixins import SetupMixin
from .Node import Node
from .topology.BaseTopology import BaseTopology
from .utils.MetricFormatter import MetricFormatter


@rich.repr.auto
class Engine(SetupMixin):
    """
    Core orchestration engine for federated learning experiments.

    - Orchestrate distributed federated learning experiments
    - Manage node lifecycle and resource allocation
    - Collect, analyze, and report metrics and statistics.

    Integration:
    - Coordinates with Topology definitions for network structure
    - Manages Node actors running Algorithm implementations
    - Instantiated through Hydra configuration
    """

    def __init__(
        self,
        flora_cfg: DictConfig,
    ):
        super().__init__()
        utils.log_sep(f"{self.__class__.__name__} Init", color="blue")

        self.flora_cfg: DictConfig = flora_cfg
        self.hydra_cfg: HydraConf = HydraConfig.get()

        self.topology: BaseTopology = instantiate(
            self.flora_cfg.topology,
            algo_cfg=self.flora_cfg.algorithm,
            model_cfg=self.flora_cfg.model,
            data_cfg=self.flora_cfg.data,
            log_dir=self.hydra_cfg.runtime.output_dir,
            _recursive_=False,
        )
        self.global_rounds: int = flora_cfg.global_rounds

        # Instantiated Ray actors populated during setup
        self._ray_actor_refs: List[Node] = []

    def _setup(self):
        utils.log_sep("FLORA Engine Setup", color="blue")

        # Initialize Ray
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=True,  # NOTE: when false, Task and Actor logs are not copied to the driver stdout.
            # logging_level=logging.INFO,  # TODO: Tie this to Hydra's logging level
            # namespace="federated_learning",
        )

        # Calculate node rayopts based on total nodes
        total_nodes = len(self.topology)
        node_rayopts = {}
        if torch.cuda.is_available():
            node_rayopts["num_gpus"] = 1.0 / total_nodes

        # Create Ray actors from configurations
        self._ray_actor_refs = []
        for node_config in self.topology:
            node_actor = Node.options(**node_rayopts).remote(node_config)
            self._ray_actor_refs.append(node_actor)
            time.sleep(
                1
            )  # NOTE: Sleep for debugging purposes for now to allow logs to flush before next node starts

        # Setup nodes
        setup_futures = [node.setup.remote() for node in self._ray_actor_refs]
        ray.get(setup_futures)

    def start(self):
        """
        NOTE: stuffed everything in here for now, plan to refactor into smaller generalized methods.
        """
        try:
            # ----------------------------------------------------------------
            utils.log_sep("FLORA Engine Start", color="blue")

            print(
                f"# Starting {self.global_rounds} round federated learning experiment",
                flush=True,
            )
            print(f"# Nodes will execute autonomously", flush=True)

            _t_experiment_start = time.time()

            # Dispatch complete experiment to all nodes
            experiment_futures = []
            for node in self._ray_actor_refs:
                future = node.run_experiment.remote(self.global_rounds)
                experiment_futures.append(future)

            # Wait for all nodes to complete their experiments
            print("# Waiting for nodes to complete experiments...", flush=True)
            results = ray.get(experiment_futures)

            _t_experiment_end = time.time()
            _experiment_duration = _t_experiment_end - _t_experiment_start

            utils.log_sep("FL Experiment Complete", color="blue")
            print(
                f"# Total experiment duration: {_experiment_duration:.2f}s", flush=True
            )
            print(f"# Completed {len(results)} node experiments", flush=True)

            # ----------------------------------------------------------------

            # Display experiment summary
            table = Table(
                title="Experiment Summary",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
            )

            table.add_column("Metric", justify="left", style="cyan")
            table.add_column("Value", justify="center", style="green")

            table.add_row("Total Rounds", str(self.global_rounds))
            table.add_row("Nodes Completed", str(len(results)))
            table.add_row("Experiment Duration", f"{_experiment_duration:.2f}s")

            # Compute aggregated metrics across all nodes
            if results and len(results) > 0:
                table.add_row("", "")  # Separator
                table.add_row("Aggregated Results", "")

                # Use MetricFormatter for intelligent formatting
                formatter = MetricFormatter()
                formatted_metrics = formatter.format_results_summary(results)

                # Display formatted metrics
                for metric, formatted_value in formatted_metrics.items():
                    table.add_row(f"  {metric}", formatted_value)

                # Show node consistency
                if len(results) > 1:
                    table.add_row("", "")
                    table.add_row(
                        "Node Consistency", f"σ metrics across {len(results)} nodes"
                    )

            utils.console.print(table)

        finally:
            print("Engine shutting down...", flush=True)
            time.sleep(3)  # Give time for final logs to flush
            ray.shutdown()
