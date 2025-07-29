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

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
from rich import box, print
from rich.table import Table

from .MetricFormatter import MetricFormatter

# Emoji mapping for statistics
STAT_EMOJIS = {
    "Mean": ":bar_chart:",
    "Std Dev": ":straight_ruler:",
    "Min": ":arrow_down:",
    "Max": ":arrow_up:",
}


class ExperimentDisplay:
    """
    Handles display and formatting of federated learning experiment results.

    Separates UI/display logic from experiment orchestration to improve
    maintainability, testability, and separation of concerns.
    """

    def __init__(self):
        """Initialize display with consistent metric formatter."""
        self._formatter: MetricFormatter = MetricFormatter()

    def show_experiment_results(
        self,
        results: List[List[List[Dict[str, Any]]]],
        duration: float,
        global_rounds: int,
        total_nodes: int,
    ) -> None:
        """
        Display complete experiment results including summary and metrics.

        Args:
            results: Experiment results (nodes -> rounds -> epochs -> metrics)
            duration: Total experiment duration in seconds
            global_rounds: Number of FL rounds configured
            total_nodes: Total number of nodes in topology
        """
        self._display_summary_table(results, duration, global_rounds, total_nodes)

        if not results:
            return

        self._display_final_metrics(results)

        # Show progression tables only for multi-round experiments (assumes results is non-empty)
        if len(results[0]) > 1:
            self._display_round_progression(results)

    def _display_summary_table(
        self,
        results: List[List[List[Dict[str, Any]]]],
        duration: float,
        global_rounds: int,
        total_nodes: int,
    ) -> None:
        """Display basic experiment summary table."""
        summary_table = Table(
            title="Experiment Summary",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        summary_table.add_column("Metric", justify="left", style="cyan")
        summary_table.add_column("Value", justify="center", style="green")

        summary_table.add_row("Total Rounds", str(global_rounds))
        summary_table.add_row("Nodes Completed", f"{len(results)}/{total_nodes}")
        summary_table.add_row("Experiment Duration", f"{duration:.2f}s")

        print(summary_table)
        print("\n")

    def _display_final_metrics(self, results: List[List[List[Dict[str, Any]]]]) -> None:
        """Display final round metrics in a formatted table."""
        # Extract final round metrics (last element) from each node's results
        final_round_epoch_lists = [node_rounds[-1] for node_rounds in results]

        final_round_results = []
        for epoch_metrics_list in final_round_epoch_lists:
            round_metrics = self._formatter.aggregate_epochs_to_round(
                epoch_metrics_list
            )
            final_round_results.append(round_metrics)

        metric_stats_list = self._formatter.format_stats(final_round_results)
        metric_groups = self._formatter.group_metric_stats(metric_stats_list)
        metrics_table = self._create_final_metrics_table(
            results, metric_stats_list, metric_groups
        )
        self._populate_final_metrics_table(metrics_table, metric_groups)

        print(metrics_table)
        print("\n")

    def _create_final_metrics_table(
        self,
        results: List[List[List[Dict[str, Any]]]],
        metric_stats_list: List[Any],
        metric_groups: Dict[str, List[Any]],
    ) -> Table:
        """Create the final metrics table with proper styling and columns."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Use different title based on whether metrics are aggregated across multiple nodes
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

        return metrics_table

    def _populate_final_metrics_table(
        self, metrics_table: Table, metric_groups: Dict[str, List[Any]]
    ) -> None:
        """Populate the final metrics table with data."""
        first_group = True
        for group_name, metric_stats in metric_groups.items():
            if metric_stats:
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

    def _display_round_progression(
        self, results: List[List[List[Dict[str, Any]]]]
    ) -> None:
        """
        Display round-by-round progression tables for multi-round experiments.

        Creates statistical tables (mean, std, min, max) showing how metrics
        evolve across training rounds.
        """
        all_metrics, num_rounds = self._discover_metrics_across_rounds(results)

        if not all_metrics:
            return

        metric_groups = self._formatter.group_metric_names(all_metrics)

        stats = ["Mean", "Std Dev", "Min", "Max"]
        stat_functions = [np.mean, np.std, np.min, np.max]

        for stat_name, stat_func in zip(stats, stat_functions):
            self._create_and_display_progression_table(
                stat_name, stat_func, results, all_metrics, num_rounds, metric_groups
            )

    def _discover_metrics_across_rounds(
        self, results: List[List[List[Dict[str, Any]]]]
    ) -> Tuple[List[str], int]:
        """Discover all available metrics across all rounds."""
        all_metrics = set()
        for node_rounds in results:
            for epoch_metrics_list in node_rounds:
                round_metrics = self._formatter.aggregate_epochs_to_round(
                    epoch_metrics_list
                )
                all_metrics.update(round_metrics.keys())

        # Return metrics list and round count (0 if no results to avoid index error)
        return sorted(all_metrics), len(results[0]) if results else 0

    def _create_and_display_progression_table(
        self,
        stat_name: str,
        stat_func: Any,
        results: List[List[List[Dict[str, Any]]]],
        all_metrics: List[str],
        num_rounds: int,
        metric_groups: Dict[str, List[str]],
    ) -> None:
        """Create and display a single progression table for a given statistic."""
        progression_table = self._create_progression_table_structure(
            stat_name, all_metrics, num_rounds
        )

        metric_stats = self._calculate_progression_statistics(
            all_metrics, num_rounds, results, stat_func
        )
        self._populate_progression_table(
            progression_table, metric_groups, metric_stats, num_rounds
        )

        print(progression_table)
        print("\n")

    def _create_progression_table_structure(
        self, stat_name: str, all_metrics: List[str], num_rounds: int
    ) -> Table:
        """Create the basic structure of a progression table."""
        progression_table = Table(
            title=f"{STAT_EMOJIS[stat_name]} Round-by-Round Progression - {stat_name}",
            box=box.HEAVY_HEAD,
            show_header=True,
            header_style="bold blue",
            caption=f":clipboard: {stat_name} • {len(all_metrics)} metrics • {num_rounds} rounds",
            caption_justify="right",
        )

        progression_table.add_column(
            ":bar_chart: Metric", justify="left", style="bold cyan", no_wrap=True
        )

        for round_idx in range(num_rounds):
            progression_table.add_column(
                f":repeat: Round {round_idx + 1}",
                justify="right",
                style="green",
                header_style="bold green",
            )
            if round_idx < num_rounds - 1:
                progression_table.add_column(
                    "→", justify="center", style="dim white", width=3
                )

        return progression_table

    def _calculate_progression_statistics(
        self,
        all_metrics: List[str],
        num_rounds: int,
        results: List[List[List[Dict[str, Any]]]],
        stat_func: Any,
    ) -> Dict[str, List[Any]]:
        """Calculate statistics for each metric across all rounds."""
        metric_stats = defaultdict(list)
        for metric in all_metrics:
            for round_idx in range(num_rounds):
                round_data = []
                for node_rounds in results:
                    epoch_metrics_list = node_rounds[round_idx]
                    round_metrics = self._formatter.aggregate_epochs_to_round(
                        epoch_metrics_list
                    )
                    round_data.append(round_metrics)

                values = [
                    r.get(metric) for r in round_data if r.get(metric) is not None
                ]

                if not values:
                    metric_stats[metric].append(None)
                    continue

                # Handle numpy.std edge case: avoid numpy warning when computing std of single value
                if stat_func == np.std and len(values) == 1:
                    stat_value = (
                        0.0  # Standard deviation of single value is mathematically 0
                    )
                else:
                    stat_value = stat_func(values)

                metric_stats[metric].append(stat_value)

        return dict(metric_stats)

    def _populate_progression_table(
        self,
        progression_table: Table,
        metric_groups: Dict[str, List[str]],
        metric_stats: Dict[str, List[Any]],
        num_rounds: int,
    ) -> None:
        """Populate the progression table with metric data and trend indicators."""
        first_group = True
        for group_name, metrics in metric_groups.items():
            if metrics:
                if not first_group:
                    progression_table.add_section()
                first_group = False

                for metric in metrics:
                    row_values = self._build_progression_row(
                        metric, metric_stats, num_rounds
                    )
                    progression_table.add_row(*row_values)

    def _build_progression_row(
        self, metric: str, metric_stats: Dict[str, List[Any]], num_rounds: int
    ) -> List[str]:
        """Build a single row for the progression table."""
        row_values = []
        emoji = self._formatter.get_emoji(metric)
        row_values.append(f"{emoji} {metric}")
        stats = metric_stats[metric]

        for round_idx in range(num_rounds):
            stat_value = stats[round_idx]
            if stat_value is not None:
                formatted_value = self._formatter.format(metric, stat_value)
                row_values.append(formatted_value)
            else:
                row_values.append("-")

            if round_idx < num_rounds - 1:
                current_val = stats[round_idx]
                next_val = stats[round_idx + 1]
                if current_val is not None and next_val is not None:
                    trend_symbol = self._formatter.get_trend_symbol(
                        metric, next_val, current_val
                    )
                else:
                    trend_symbol = ""
                row_values.append(trend_symbol)

        return row_values
