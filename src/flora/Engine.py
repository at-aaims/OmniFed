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

import numpy as np
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
from .utils.MetricFormatter import MetricFormatter, OptimizationGoal


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
            print("# Nodes will execute autonomously", flush=True)

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

            time.sleep(1)  # Allow time for final logs to flush
            utils.log_sep("FL Experiment Complete", color="blue")
            print(
                f"# Total experiment duration: {_experiment_duration:.2f}s", flush=True
            )
            print(f"# Completed {len(results)} node experiments", flush=True)

            # ----------------------------------------------------------------

            # Display experiment summary
            summary_table = Table(
                title="Experiment Summary",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
            )

            summary_table.add_column("Metric", justify="left", style="cyan")
            summary_table.add_column("Value", justify="center", style="green")

            summary_table.add_row("Total Rounds", str(self.global_rounds))
            summary_table.add_row(
                "Nodes Completed", f"{len(results)}/{len(self.topology)}"
            )
            summary_table.add_row("Experiment Duration", f"{_experiment_duration:.2f}s")

            utils.console.print(summary_table)

            # Display aggregated metrics if available
            if results and len(results) > 0:
                # Extract final round metrics from each node
                final_round_results = [node_rounds[-1] for node_rounds in results]

                metrics_table = Table(
                    title=f"Final Round Aggregated Metrics ({len(results)} nodes)"
                    if len(results) > 1
                    else "Final Round Metrics",
                    box=box.ROUNDED,
                    show_header=True,
                    header_style="bold magenta",
                )

                metrics_table.add_column("Metric", justify="left", style="cyan")
                metrics_table.add_column("Mean", justify="right", style="green")
                metrics_table.add_column("Std Dev", justify="right", style="yellow")
                metrics_table.add_column("Min", justify="right", style="blue")
                metrics_table.add_column("Max", justify="right", style="blue")

                # Use MetricFormatter for intelligent formatting
                formatter = MetricFormatter()
                formatted_metrics = formatter.format_stats(final_round_results)

                # Display formatted metrics
                for metric, stats in formatted_metrics.items():
                    metrics_table.add_row(
                        metric, stats["mean"], stats["std"], stats["min"], stats["max"]
                    )

                utils.console.print(metrics_table)

                # Display round-by-round progression if multiple rounds
                if len(results[0]) > 1:  # More than one round
                    self._display_round_progression(results)

        finally:
            print("Engine shutting down...", flush=True)
            time.sleep(3)  # Give time for final logs to flush
            ray.shutdown()

    def _display_round_progression(self, results):
        """Display round-by-round progression summary with mean, std, min, max tables."""

        formatter = MetricFormatter()

        # Discover all available metrics across all rounds
        all_metrics = set()
        for node_rounds in results:
            for round_metrics in node_rounds:
                all_metrics.update(round_metrics.keys())

        all_metrics = sorted(all_metrics)
        num_rounds = len(results[0])

        if not all_metrics:
            return  # No metrics to show

        # Create four separate tables for different statistics
        stats = ["Mean", "Std Dev", "Min", "Max"]
        stat_functions = [np.mean, np.std, np.min, np.max]

        for stat_name, stat_func in zip(stats, stat_functions):
            progression_table = Table(
                title=f"Round-by-Round Progression - {stat_name}",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold blue",
            )

            # Add metric column first
            progression_table.add_column("Metric", justify="left", style="cyan")

            # Add round columns with trend indicators in between
            for round_idx in range(num_rounds):
                progression_table.add_column(
                    f"Round {round_idx + 1}", justify="right", style="green"
                )
                # Add trend column after each round (except the last)
                if round_idx < num_rounds - 1:
                    progression_table.add_column(
                        "", justify="center", style="white", width=2
                    )

            # Pre-calculate all statistics for all rounds
            metric_stats = {}
            for metric in all_metrics:
                metric_stats[metric] = []
                for round_idx in range(num_rounds):
                    round_data = [node_rounds[round_idx] for node_rounds in results]
                    values = [
                        r.get(metric) for r in round_data if r.get(metric) is not None
                    ]

                    if values:
                        stat_value = (
                            0.0
                            if stat_func == np.std and len(values) == 1
                            else stat_func(values)
                        )
                        metric_stats[metric].append(stat_value)
                    else:
                        metric_stats[metric].append(None)

            # Build table rows
            for metric in all_metrics:
                row_values = [metric]
                stats = metric_stats[metric]

                for round_idx in range(num_rounds):
                    # Add round value
                    if stats[round_idx] is not None:
                        formatted_value = formatter.format(metric, stats[round_idx])
                        row_values.append(formatted_value)
                    else:
                        row_values.append("-")

                    # Add trend indicator (except after last round)
                    if round_idx < num_rounds - 1:
                        current_val = stats[round_idx]
                        next_val = stats[round_idx + 1]

                        if current_val is not None and next_val is not None:
                            trend_symbol = self._get_trend_symbol(
                                metric, next_val, current_val, formatter
                            )
                        else:
                            trend_symbol = ""
                        row_values.append(trend_symbol)

                progression_table.add_row(*row_values)

            utils.console.print(progression_table)
            print()  # Add spacing between tables

    def _get_trend_symbol(
        self, metric: str, current: float, previous: float, formatter
    ) -> str:
        """Get colored trend symbol based on metric change."""
        if current == previous:
            return "[yellow]→[/yellow]"

        is_increasing = current > previous
        goal = formatter.optimization_goal(metric)

        # Handle neutral metrics (no trend judgment)
        if goal == OptimizationGoal.NEUTRAL:
            symbol = "↗" if is_increasing else "↘"
            return f"[dim]{symbol}[/dim]"  # Dim gray for neutral changes

        # Determine if change is good based on optimization goal
        is_good_change = (is_increasing and goal == OptimizationGoal.MAXIMIZE) or (
            not is_increasing and goal == OptimizationGoal.MINIMIZE
        )

        symbol = "↗" if is_increasing else "↘"
        color = "green" if is_good_change else "red"

        return f"[{color}]{symbol}[/{color}]"
