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
from datetime import datetime
from enum import Enum
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
from .Node import Node, NodeSpec
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

        # Create Ray actors from specifications
        self._ray_actor_refs = []
        for node_spec in self.topology:
            node_actor = Node.options(**node_rayopts).remote(
                node_spec=node_spec,
                algorithm_cfg=self.flora_cfg.algorithm,
                model_cfg=self.flora_cfg.model,
                datamodule_cfg=self.flora_cfg.datamodule,
                schedules_cfg=self.flora_cfg.schedules,
            )
            self._ray_actor_refs.append(node_actor)

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

            utils.log_sep("FL Experiment Complete", color="blue")

            # ----------------------------------------------------------------

            # Display experiment summary
            self._display_experiment_summary(results, _experiment_duration)

        finally:
            print("Engine shutting down...", flush=True)
            ray.shutdown()

    def _display_experiment_summary(self, results, duration):
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
        summary_table.add_row("Nodes Completed", f"{len(results)}/{len(self.topology)}")
        summary_table.add_row("Experiment Duration", f"{duration:.2f}s")

        utils.console.print(summary_table)
        print("\n")  # Extra spacing after summary

        # Display aggregated metrics if available
        if results and len(results) > 0:
            # Results now have structure: List[List[List[Dict]]] (nodes -> rounds -> epochs -> metrics)
            # Extract final round's epoch metrics from each node and aggregate to round-level
            final_round_epoch_lists = [node_rounds[-1] for node_rounds in results]

            # Aggregate each node's final round epoch metrics to round-level metrics
            formatter = MetricFormatter()  # Create formatter instance
            final_round_results = []
            for epoch_metrics_list in final_round_epoch_lists:
                round_metrics = formatter.aggregate_epochs_to_round(epoch_metrics_list)
                final_round_results.append(round_metrics)

            # Use MetricFormatter for intelligent formatting with structured data
            metric_stats_list = formatter.format_stats_structured(final_round_results)
            metric_groups = formatter.group_structured_metrics(metric_stats_list)

            # Create enhanced metrics table with better styling
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            metrics_table = Table(
                title=f":dart: Final Round Aggregated Metrics ({len(results)} nodes)"
                if len(results) > 1
                else ":dart: Final Round Metrics",
                box=box.HEAVY_HEAD,
                show_header=True,
                header_style="bold magenta",
                caption=f":bar_chart: {len(results)} nodes • {len(metric_stats_list)} metrics in {len(metric_groups)} groups • {timestamp}",
                caption_justify="right",
            )

            # Enhanced column headers with appropriate emojis
            metrics_table.add_column(
                ":bar_chart: Metric",
                justify="left",
                style="bold cyan",
                no_wrap=True,
            )
            metrics_table.add_column(
                ":bar_chart: Mean",
                justify="right",
                style="green",
                header_style="bold green",
            )
            metrics_table.add_column(
                ":straight_ruler: Std Dev",
                justify="right",
                style="yellow",
                header_style="bold yellow",
            )
            metrics_table.add_column(
                ":arrow_down: Min",
                justify="right",
                style="blue",
                header_style="bold blue",
            )
            metrics_table.add_column(
                ":arrow_up: Max",
                justify="right",
                style="blue",
                header_style="bold blue",
            )

            # Display metrics with clean section separators between groups
            first_group = True
            for group_name, metric_stats in metric_groups.items():
                if metric_stats:
                    # Add section separator between groups
                    if not first_group:
                        metrics_table.add_section()
                    first_group = False

                    for stats in metric_stats:
                        metrics_table.add_row(
                            stats.display_name,
                            stats.mean,
                            stats.std,
                            stats.min,
                            stats.max,
                        )

            utils.console.print(metrics_table)
            print("\n")  # Extra spacing after metrics table

            # Display round-by-round progression if multiple rounds
            if len(results[0]) > 1:  # More than one round
                self._display_round_progression(results)

    def _display_round_progression(self, results):
        """Display round-by-round progression summary with mean, std, min, max tables."""

        formatter = MetricFormatter()

        # Discover all available metrics across all rounds
        # Results structure: List[List[List[Dict]]] (nodes -> rounds -> epochs -> metrics)
        all_metrics = set()
        for node_rounds in results:
            for epoch_metrics_list in node_rounds:
                # Aggregate epoch metrics to round metrics to discover all metric names
                round_metrics = formatter.aggregate_epochs_to_round(epoch_metrics_list)
                all_metrics.update(round_metrics.keys())

        all_metrics = sorted(all_metrics)
        num_rounds = len(results[0])

        if not all_metrics:
            return  # No metrics to show

        # Group metrics by category
        metric_groups = formatter.group_metrics(all_metrics)

        # Create four separate tables for different statistics
        stats = ["Mean", "Std Dev", "Min", "Max"]
        stat_functions = [np.mean, np.std, np.min, np.max]

        for stat_name, stat_func in zip(stats, stat_functions):
            stat_emojis = {
                "Mean": ":bar_chart:",
                "Std Dev": ":straight_ruler:",
                "Min": ":arrow_down:",
                "Max": ":arrow_up:",
            }

            progression_table = Table(
                title=f"{stat_emojis[stat_name]} Round-by-Round Progression - {stat_name}",
                box=box.HEAVY_HEAD,
                show_header=True,
                header_style="bold blue",
                caption=f":clipboard: {stat_name} • {len(all_metrics)} metrics • {num_rounds} rounds",
                caption_justify="right",
            )

            # Enhanced metric column
            progression_table.add_column(
                ":bar_chart: Metric", justify="left", style="bold cyan", no_wrap=True
            )

            # Add round columns with trend indicators in between
            for round_idx in range(num_rounds):
                progression_table.add_column(
                    f":repeat: Round {round_idx + 1}",
                    justify="right",
                    style="green",
                    header_style="bold green",
                )
                # Add trend column after each round (except the last)
                if round_idx < num_rounds - 1:
                    progression_table.add_column(
                        "→", justify="center", style="dim white", width=3
                    )

            # Pre-calculate all statistics for all rounds
            metric_stats = {}
            for metric in all_metrics:
                metric_stats[metric] = []
                for round_idx in range(num_rounds):
                    # Aggregate epoch metrics to round metrics for each node in this round
                    round_data = []
                    for node_rounds in results:
                        epoch_metrics_list = node_rounds[round_idx]
                        round_metrics = formatter.aggregate_epochs_to_round(
                            epoch_metrics_list
                        )
                        round_data.append(round_metrics)

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

            # Build table rows with group separators
            first_group = True
            for group_name, metrics in metric_groups.items():
                if metrics:
                    # Add section separator between groups
                    if not first_group:
                        progression_table.add_section()
                    first_group = False

                    for metric in metrics:
                        row_values = []
                        emoji = formatter.get_emoji(metric)
                        row_values.append(f"{emoji} {metric}")
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
                                    trend_symbol = formatter.get_trend_symbol(
                                        metric, next_val, current_val
                                    )
                                else:
                                    trend_symbol = ""
                                row_values.append(trend_symbol)

                        progression_table.add_row(*row_values)

            utils.console.print(progression_table)
            print("\n")  # Extra spacing between progression tables
