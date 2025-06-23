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

import logging
import time
from typing import Any, Dict, List

import ray
import rich.repr
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich import box
from rich.table import Table

from . import utils
from .topology.BaseTopology import Topology


@rich.repr.auto
class Engine:
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
        topology_cfg: DictConfig,
        node_defaults: DictConfig,
        global_rounds: int,
    ):
        utils.log_sep("Engine Initialization", color="blue")
        # Initialize Ray with more verbose logging and explicit namespace
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=True,  # NOTE: when false, Task and Actor logs are not copied to the driver stdout.
            logging_level=logging.INFO,
            namespace="federated_learning",
        )

        self.topology_cfg = topology_cfg
        self.node_defaults = node_defaults
        self.global_rounds = global_rounds

    def run_experiment(self):
        """
        NOTE: stuffed everything in here for now, plan to refactor into smaller generalized methods.
        """
        try:
            # -------------------------------------------------
            utils.log_sep("FL Rounds Start", color="blue")

            topology: Topology = instantiate(self.topology_cfg)
            nodes = topology.setup_nodes(self.node_defaults, topology.num_nodes)

            setup_futures = [n.setup.remote() for n in nodes]
            ray.get(setup_futures)

            summaries = []
            for round_num in range(self.global_rounds):
                utils.log_sep(f"Round {round_num + 1}/{self.global_rounds}")
                round_start_time = time.time()

                # Execute through topology's execute_round method
                results_futures = [
                    n.execute_round.remote(int(round_num)) for n in nodes
                ]
                results = ray.get(results_futures)

                round_duration = time.time() - round_start_time

                ct_total = len(results)
                ct_success = len([r for r in results if r is not None])

                # Store metrics
                success_rate = (ct_success / ct_total) * 100
                summaries.append(
                    {
                        "round_num": round_num + 1,
                        "duration": round_duration,
                        "total_count": ct_total,
                        "success_count": ct_success,
                        "success_rate": success_rate,
                    }
                )

                # Round summary
                print(f"Round Complete | {summaries[-1]}")

            utils.log_sep("FL Rounds End", color="blue")

            # -------------------------------------------------

            table = Table(
                title="Summary",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
            )

            for key in summaries[0].keys():
                table.add_column(
                    key.replace("_", " ").title(),
                    justify="center",
                    style="cyan",
                )

            for metrics in summaries:
                table.add_row(*[str(metrics[key]) for key in metrics.keys()])

            utils.console.print(table)

        finally:
            print("Engine shutting down")
            ray.shutdown()
