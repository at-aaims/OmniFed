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
from typing import Any, Dict, List

import numpy as np
from rich import box, print
from rich.table import Table

from .MetricFormatter import MetricFormatter


class ExperimentDisplay:
    """
    Handles display and formatting of federated learning experiment results.

    Separates UI/display logic from experiment orchestration to improve
    maintainability, testability, and separation of concerns.
    """

    def __init__(self):
        """Initialize display with consistent metric formatter."""
        self._formatter: MetricFormatter = MetricFormatter()
        self._metadata_keys = {"global_step", "round_idx", "epoch_idx", "batch_idx"}

    def _extract_metrics_from_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from a data row, excluding metadata fields."""
        return {
            k: v
            for k, v in row.items()
            if k not in self._metadata_keys and v is not None
        }

    def _find_max_round(self, results: List[Dict[str, Any]]) -> int:
        """Find the maximum round across all nodes and contexts."""
        max_round = 0
        for node_data in results:
            for context, rows in node_data.items():
                for row in rows:
                    max_round = max(max_round, int(row.get("round_idx", 0)))
        return max_round

    def _validate_results_format(self, results: List[Dict[str, Any]]) -> None:
        """Validate that results follow the expected MetricLogger format."""
        if not isinstance(results, list):
            raise ValueError(f"Results must be a list, got {type(results)}")

        for i, node_data in enumerate(results):
            if not isinstance(node_data, dict):
                raise ValueError(f"Node {i} data must be a dict, got {type(node_data)}")

            for context, rows in node_data.items():
                if not isinstance(rows, list):
                    raise ValueError(
                        f"Node {i} context '{context}' must be a list, got {type(rows)}"
                    )

                for j, row in enumerate(rows):
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"Node {i} context '{context}' row {j} must be a dict, got {type(row)}"
                        )

                    if "round_idx" not in row:
                        raise ValueError(
                            f"Node {i} context '{context}' row {j} missing 'round_idx' field"
                        )

    def show_experiment_results(
        self,
        results: List[Dict[str, Any]],
        duration: float,
        global_rounds: int,
        total_nodes: int,
    ) -> None:
        """
        Display complete experiment results including summary and metrics.

        Args:
            results: List of node experiment data (from MetricLogger.get_experiment_data())
                    Format: [{context: [list of rows with metadata + metrics]}, ...]
            duration: Total experiment duration in seconds
            global_rounds: Number of FL rounds configured
            total_nodes: Total number of nodes in topology
        """
        # Validate input format to prevent silent failures
        self._validate_results_format(results)

        self._display_summary_table(results, duration, global_rounds, total_nodes)

        if not results:
            return

        # New improved display structure
        self._display_epoch_progression(results)

        # Show round summaries for multi-round experiments
        if global_rounds > 1:
            self._display_round_summaries(results)

    def _display_summary_table(
        self,
        results: List[Dict[str, Any]],
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

    def _display_epoch_progression(self, results: List[Dict[str, Any]]) -> None:
        """Display epoch-by-epoch progression within rounds, showing training dynamics."""
        if not results:
            return

        # Group metrics by round and epoch
        epoch_data = self._organize_metrics_by_epoch(results)

        if not epoch_data:
            return

        # Automatically discover and organize metrics using MetricFormatter groups
        all_metrics = self._discover_all_metrics(results)
        grouped_metrics = self._formatter.group_metric_names(all_metrics)

        for round_idx in sorted(epoch_data.keys()):
            round_epochs = epoch_data[round_idx]
            if len(round_epochs) > 1:  # Only show if multiple epochs
                self._display_round_epoch_table(
                    round_idx, round_epochs, grouped_metrics
                )

    def _organize_metrics_by_epoch(
        self, results: List[Dict[str, Any]]
    ) -> Dict[int, Dict[int, Dict[str, List[float]]]]:
        """Organize metrics by round and epoch for progression analysis."""
        epoch_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for node_data in results:
            for context, rows in node_data.items():
                for row in rows:
                    round_idx = int(row.get("round_idx", 0))
                    epoch_idx = int(row.get("epoch_idx", 0))

                    metrics = self._extract_metrics_from_row(row)
                    for metric_name, value in metrics.items():
                        if value is not None:
                            epoch_data[round_idx][epoch_idx][metric_name].append(
                                float(value)
                            )

        return epoch_data

    def _discover_all_metrics(self, results: List[Dict[str, Any]]) -> List[str]:
        """Discover all unique metrics across all nodes, contexts, and rounds."""
        all_metrics = set()
        for node_data in results:
            for context, rows in node_data.items():
                for row in rows:
                    metrics = self._extract_metrics_from_row(row)
                    all_metrics.update(metrics.keys())
        return sorted(all_metrics)

    def _display_round_epoch_table(
        self, round_idx: int, round_epochs: Dict, grouped_metrics: Dict[str, List[str]]
    ) -> None:
        """Display epoch progression table for a single round."""
        table = Table(
            title=f"📈 Round {round_idx + 1} - Epoch Progression",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold blue",
        )

        table.add_column("📊 Metric", justify="left", style="cyan")

        # Add epoch columns
        epoch_indices = sorted(round_epochs.keys())
        for epoch_idx in epoch_indices:
            table.add_column(f"Epoch {epoch_idx + 1}", justify="right", style="green")

        # Add trend column
        if len(epoch_indices) > 1:
            table.add_column("📈 Trend", justify="center", style="yellow")

        # Add all metrics organized by group
        for group_name, group_metrics in grouped_metrics.items():
            for metric in group_metrics:
                if any(
                    metric in round_epochs[epoch_idx] for epoch_idx in epoch_indices
                ):
                    self._add_metric_row(table, metric, round_epochs, epoch_indices)

        print(table)
        print("\n")

    def _add_metric_row(
        self, table: Table, metric: str, round_epochs: Dict, epoch_indices: List[int]
    ) -> None:
        """Add a metric row to the epoch progression table."""
        row_data = [self._formatter.format_display_name(metric)]
        values = []

        for epoch_idx in epoch_indices:
            if metric in round_epochs[epoch_idx] and round_epochs[epoch_idx][metric]:
                value = np.mean(round_epochs[epoch_idx][metric])  # Average across nodes
                formatted_value = self._formatter.format_value(metric, value)
                row_data.append(formatted_value)
                values.append(value)
            else:
                row_data.append("-")
                values.append(None)

        # Add trend indicator if we have multiple epochs
        if len(epoch_indices) > 1 and len([v for v in values if v is not None]) > 1:
            valid_values = [v for v in values if v is not None]
            if len(valid_values) >= 2:
                trend = self._get_trend_indicator(
                    valid_values[0], valid_values[-1], metric
                )
                row_data.append(trend)

        table.add_row(*row_data)

    def _get_trend_indicator(
        self, first_value: float, last_value: float, metric: str
    ) -> str:
        """Get trend indicator using MetricFormatter's trend analysis."""
        return self._formatter.get_trend_symbol(metric, first_value, last_value)

    def _display_round_summaries(self, results: List[Dict[str, Any]]) -> None:
        """Display round boundary summaries organized by MetricFormatter groups."""
        if not results:
            return

        round_summaries = self._create_round_summaries(results)

        # Get all metrics and organize by groups
        all_metrics = self._discover_all_metrics(results)
        grouped_metrics = self._formatter.group_metric_names(all_metrics)

        # Display one summary table per group (only groups that have metrics)
        for group_name, group_metrics in grouped_metrics.items():
            if group_metrics:
                self._display_group_summary(group_name, group_metrics, round_summaries)

    def _create_round_summaries(
        self, results: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """Create round-level summaries with metric-specific aggregations."""
        round_summaries = defaultdict(lambda: defaultdict(list))

        for node_data in results:
            for context, rows in node_data.items():
                for row in rows:
                    round_idx = int(row.get("round_idx", 0))
                    epoch_idx = int(row.get("epoch_idx", 0))

                    metrics = self._extract_metrics_from_row(row)
                    for metric_name, value in metrics.items():
                        if value is not None:
                            round_summaries[round_idx][metric_name].append(
                                {
                                    "value": float(value),
                                    "epoch_idx": epoch_idx,
                                    "context": context,
                                }
                            )

        return round_summaries

    def _display_group_summary(
        self, group_name: str, group_metrics: List[str], round_summaries: Dict
    ) -> None:
        """Display summary table for a specific metric group."""
        # Map group names to display info
        group_icons = {
            "Training": "🎯",
            "Performance": "📊",
            "Round Timing": "🔄",
            "Epoch Timing": "⏱",
            "Batch Timing": "⏲",
            "Sync Timing": "🔄",
            "Dataset": "📦",
            "Other": "📊",
        }

        icon = group_icons.get(group_name, "📊")

        table = Table(
            title=f"{icon} {group_name} Summary",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold blue",
        )

        table.add_column("📊 Metric", justify="left", style="cyan")

        # Add round columns
        round_indices = sorted(round_summaries.keys())
        for round_idx in round_indices:
            table.add_column(f"Round {round_idx + 1}", justify="right", style="green")

        # Add trend column if multiple rounds
        if len(round_indices) > 1:
            table.add_column("📈 Trend", justify="center", style="yellow")

        # Add metrics for this group
        for metric in group_metrics:
            if any(metric in round_summaries[r] for r in round_indices):
                self._add_group_metric_row(
                    table, metric, round_summaries, round_indices
                )

        print(table)
        print("\n")

    def _add_group_metric_row(
        self, table: Table, metric: str, round_summaries: Dict, round_indices: List[int]
    ) -> None:
        """Add metric row using MetricFormatter aggregation strategies."""
        row_data = [self._formatter.format_display_name(metric)]
        values = []

        for round_idx in round_indices:
            if metric in round_summaries[round_idx]:
                round_data = round_summaries[round_idx][metric]

                # Use MetricFormatter's aggregation strategy for this metric
                epoch_values = [d["value"] for d in round_data]
                aggregated_value = self._formatter.aggregate_epochs_to_round(
                    metric, epoch_values
                )

                formatted_value = self._formatter.format_value(metric, aggregated_value)
                row_data.append(formatted_value)
                values.append(aggregated_value)
            else:
                row_data.append("-")
                values.append(None)

        # Add trend indicator using MetricFormatter
        if len(round_indices) > 1:
            valid_values = [v for v in values if v is not None]
            if len(valid_values) >= 2:
                trend = self._get_trend_indicator(
                    valid_values[0], valid_values[-1], metric
                )
                row_data.append(trend)

        table.add_row(*row_data)
