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

"""Unified display system for FLUX federated learning experiment results.

This module consolidates all display-related functionality into focused components:
- ExperimentDataProcessor: Data extraction and analysis for display
- SummaryTableBuilder: Experiment summary tables
- ProgressionTableBuilder: Training progression tables with headers
- StatisticsTableBuilder: Node statistics tables
- ResultsDisplayManager: Main coordinator and entry point
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace as dataclass_replace
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rich import box, print
from rich.table import Table
from typeguard import typechecked


from .metric_format import (
    DEFAULT_FORMAT_RULES,
    MetricFormatter,
    group_by_metric_rules,
)
from .rich_helpers import print_rule
from .table_style import (
    DISPLAY_PLACEHOLDER,
    COLORS,
    EMOJIS,
    VALIDATION,
)


# === Data Models ===


@dataclass
class Measurement:
    """Single measurement from a node at a specific training position."""

    global_step: int = 0
    round_idx: Optional[int] = None
    epoch_idx: Optional[int] = None
    batch_idx: Optional[int] = None
    node_id: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Measurement":
        """Create Measurement from raw dictionary."""
        # Extract known position fields
        measurement = cls(
            global_step=data.get("global_step", 0),
            round_idx=data.get("round_idx"),
            epoch_idx=data.get("epoch_idx"),
            batch_idx=data.get("batch_idx"),
            node_id=data.get("_node_id"),
        )

        # All other fields are metrics
        metrics = {}
        known_fields = {
            "global_step",
            "round_idx",
            "epoch_idx",
            "batch_idx",
            "_node_id",
        }
        for key, value in data.items():
            if key not in known_fields and value is not None:
                metrics[key] = value

        return dataclass_replace(measurement, metrics=metrics)

    @property
    def position_tuple(self) -> Tuple[Optional[int], ...]:
        """Get position as tuple for grouping."""
        return (self.round_idx, self.epoch_idx, self.batch_idx, self.global_step)


@dataclass
class ContextCoverage:
    """Coverage information for a specific context (train/eval)."""

    nodes_with_data: int
    completeness_rate: float


class ExperimentStatus(str, Enum):
    """Experiment completion status states."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    PARTIAL = "partial"


@dataclass
class MetricSummary:
    """Statistical summary of a metric across nodes."""

    name: str
    emoji: str
    group: str
    nodes_reporting: int
    total_nodes: int
    anomalies: str
    sum: str
    mean: str
    std: str
    min: str
    max: str
    median: str
    cv: str


@dataclass
class RunStatus:
    """Training run completion status."""

    nodes_done: int
    total_nodes: int
    completion_rate: float
    rounds: int
    context_coverage: Dict[str, ContextCoverage]
    icon: str
    status: ExperimentStatus
    is_complete: bool
    has_gaps: bool


@dataclass
class NodeSync:
    """Node synchronization at final step."""

    final_step: int
    synced_nodes: int
    lagging_nodes: int
    total_nodes: int
    step_distribution: Dict[int, int]
    lag_positions: Dict[int, int]
    all_synced: bool


@dataclass
class Timeline:
    """Time-series view of metrics across training steps."""

    # Core data
    metrics: List[str] = field(default_factory=list)
    column_headers: List[str] = field(default_factory=list)
    coordinate_matrix: List[List[Any]] = field(default_factory=list)
    coordinate_groups: Dict[str, Any] = field(default_factory=dict)

    # Status
    error: Optional[str] = None
    has_data: bool = False

    # Display info
    total_points: int = 0
    displayed_points: int = 0
    was_sampled: bool = False

    # Quality tracking
    quality_issues: List[str] = field(default_factory=list)
    incomplete_metrics: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TableConfiguration:
    """Standard table configuration settings."""

    # Table styling
    box_style: box.Box = box.ROUNDED
    show_header: bool = True

    # Title styling
    title_style: str = "bold bright_white"
    title_justify: str = "left"

    # Caption styling
    caption_style: str = "italic dim"
    caption_justify: str = "right"

    # Column styling
    justify_left: str = "left"
    justify_center: str = "center"
    justify_right: str = "right"
    vertical_middle: str = "middle"


@dataclass(frozen=True)
class TableConfig:
    """Configuration for Rich table creation."""

    title: str
    title_style: str = "bold cyan"
    title_justify: str = "center"
    caption: str = ""
    caption_style: str = "dim white"
    caption_justify: str = "right"
    box_style: Any = None  # Rich box style
    show_header: bool = True


@dataclass(frozen=True)
class ColumnConfig:
    """Configuration for table column."""

    header: str
    justify: str = "left"
    style: str = ""
    vertical: str = "middle"


@dataclass(frozen=True)
class FormattedStats:
    """Formatted statistics for display."""

    sum: str = DISPLAY_PLACEHOLDER
    mean: str = DISPLAY_PLACEHOLDER
    std: str = DISPLAY_PLACEHOLDER
    min: str = DISPLAY_PLACEHOLDER
    max: str = DISPLAY_PLACEHOLDER
    median: str = DISPLAY_PLACEHOLDER
    cv: str = DISPLAY_PLACEHOLDER


@dataclass(frozen=True)
class ProgressionAnalysis:
    """Result of progression analysis."""

    has_data: bool
    error: Optional[str] = None
    reason: Optional[str] = None
    levels: List[str] = field(default_factory=list)
    total_positions: int = 0
    coordinate_groups: Optional[Dict] = None
    metrics: List[str] = field(default_factory=list)
    column_headers: List[str] = field(default_factory=list)
    coordinate_matrix: List = field(default_factory=list)
    was_sampled: bool = False
    total_points: int = 0
    displayed_points: int = 0
    quality_issues: List[str] = field(default_factory=list)
    coverage: Dict[str, float] = field(default_factory=dict)
    incomplete_metrics: List[str] = field(default_factory=list)
    is_clean: bool = True


# No need for ColumnDefinitions class - columns should be created directly where used
# This removes unnecessary abstraction and indirection


# Removed CONFIG and COLUMNS - use values directly where needed to avoid indirection


def _get_level_color(level_name: str) -> str:
    """Get color for a progression level."""
    level_colors = {
        "round": COLORS.round_color,
        "epoch": COLORS.epoch_color,
        "batch": COLORS.batch_color,
        "step": COLORS.step_color,
    }
    return level_colors.get(level_name.lower(), COLORS.info)


def _format_context_name(context: str) -> str:
    """Format context name consistently across components."""
    return context.replace("_", " ").title()


def format_coordinate_label(
    level_name: str, value: int, is_boundary: bool = False
) -> str:
    """Format coordinate labels with consistent styling."""
    if not level_name.strip():
        level_name = "Unknown"

    display_value = value + 1
    color = _get_level_color(level_name)
    base_label = f"{level_name.title()} {display_value}"

    if is_boundary:
        return f"[bold underline {color}]{base_label}[/bold underline {color}]"
    else:
        return f"[{color}]{base_label}[/{color}]"


def format_metric_name(metric_name: str, emoji: str) -> str:
    """Format metric names with consistent styling."""
    name = metric_name.strip() or "Unknown Metric"
    return f"[{COLORS.metric_emoji}]{emoji}[/{COLORS.metric_emoji}] [{COLORS.metric_name}]{name}[/{COLORS.metric_name}]"


def format_group_header(group_name: str, count: int) -> str:
    """Format group headers with consistent styling."""
    name = group_name.strip() or "Unknown Group"
    safe_count = max(0, count)
    return f"[{COLORS.group_header}]{name}[/{COLORS.group_header}] [{COLORS.group_count}]({safe_count})[/{COLORS.group_count}]"


def format_coverage_display(stats: MetricSummary) -> str:
    """Format node coverage display with validation and color coding."""
    if stats.total_nodes > 0:
        percentage = (stats.nodes_reporting / stats.total_nodes) * 100
        base_display = (
            f"{stats.nodes_reporting}/{stats.total_nodes} ({percentage:.0f}%)"
        )

        # System validation: impossible node counts
        if stats.nodes_reporting > stats.total_nodes:
            return f"[bold red]{EMOJIS.system} {base_display}[/bold red]"  # SYSTEM: multiple measurements per node

        if percentage == VALIDATION.coverage_threshold_perfect * 100:
            return f"[green]{base_display}[/green]"
        elif percentage >= VALIDATION.coverage_threshold_good * 100:
            return f"[yellow]{base_display}[/yellow]"
        else:
            return f"[red]{base_display}[/red]"
    else:
        return f"[red]{EMOJIS.system} {stats.nodes_reporting}/{stats.total_nodes}[/red]"  # SYSTEM: zero total nodes


class ProgressionLevel(Enum):
    """Training progression levels (coarse to fine)."""

    ROUND = "round_idx"
    EPOCH = "epoch_idx"
    BATCH = "batch_idx"
    GLOBAL_STEP = "global_step"

    @classmethod
    def get_progression_order(cls) -> List["ProgressionLevel"]:
        """Return progression levels in coarse to fine order."""
        return [cls.ROUND, cls.EPOCH, cls.BATCH, cls.GLOBAL_STEP]

    @classmethod
    def get_field_names(cls) -> Set[str]:
        """Return all progression field names as a set."""
        return {level.value for level in cls}


class ExperimentDataProcessor:
    """Unified data processor for experiment analysis and display preparation.

    Consolidates results_processor.py + results_analyzer.py functionality.
    """

    EMPTY_SYNC = NodeSync(
        final_step=0,
        synced_nodes=0,
        lagging_nodes=0,
        total_nodes=0,
        step_distribution={},
        lag_positions={},
        all_synced=False,
    )

    def __init__(self, metadata_keys: Set[str] = None):
        """Initialize with metadata keys to exclude from metrics."""
        self._metadata_keys = metadata_keys or {
            "global_step",
            "round_idx",
            "epoch_idx",
            "batch_idx",
            "_node_id",
        }

    def is_valid_metric_key(self, key: str, value: Any) -> bool:
        """Check if key represents a valid metric for display."""
        return (
            key not in self._metadata_keys
            and not key.startswith("_")
            and value is not None
        )

    # === Basic Data Extraction ===

    def get_all_contexts(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract all metric contexts from results."""
        contexts = set()
        for node_data in results:
            contexts.update(node_data.keys())
        return sorted(list(contexts))

    def extract_context_data(
        self, results: List[Dict[str, Any]], context: str
    ) -> List[Measurement]:
        """Extract context data across nodes, preserving node identity."""
        context_data = []
        for node_id, node_data in enumerate(results):
            if context in node_data:
                for measurement_dict in node_data[context]:
                    # Create a copy with node_id included
                    measurement_with_node = measurement_dict.copy()
                    measurement_with_node["_node_id"] = node_id
                    # Convert to Measurement object
                    measurement = Measurement.from_dict(measurement_with_node)
                    context_data.append(measurement)
        return context_data

    def get_metrics_from_data(self, context_data: List[Measurement]) -> List[str]:
        """Extract metric names from data (excluding metadata)."""
        metrics = set()
        for measurement in context_data:
            for key, value in measurement.metrics.items():
                if self.is_valid_metric_key(key, value):
                    metrics.add(key)
        return sorted(list(metrics))

    def count_reporting_nodes(self, results: List[Measurement], metric: str) -> int:
        """Count unique nodes that reported this metric."""
        if not results:
            return 0

        unique_nodes = set()
        for measurement in results:
            if metric in measurement.metrics and measurement.node_id is not None:
                unique_nodes.add(measurement.node_id)

        return len(unique_nodes)

    # === Experiment Completion Analysis ===

    def analyze_experiment_completion(
        self, results: List[Dict[str, Any]], total_nodes: int, global_rounds: int
    ) -> RunStatus:
        """Analyze experiment completion and data quality."""
        if not results or total_nodes <= 0:
            return RunStatus(
                nodes_done=0,
                total_nodes=total_nodes,
                completion_rate=0.0,
                rounds=global_rounds,
                context_coverage={},
                icon=EMOJIS.error,
                status=ExperimentStatus.INCOMPLETE,
                is_complete=False,
                has_gaps=True,
            )

        nodes_done = len(results)
        completion_rate = nodes_done / total_nodes

        # Analyze context completeness
        all_contexts = {key for node in results for key in node.keys()}
        context_coverage = {}

        for context in all_contexts:
            reporting_nodes = sum(
                1 for node in results if context in node and node[context]
            )
            context_coverage[context] = ContextCoverage(
                nodes_with_data=reporting_nodes,
                completeness_rate=reporting_nodes / nodes_done if nodes_done > 0 else 0,
            )

        # Determine overall quality
        is_complete = completion_rate == 1.0 and all(
            ctx.completeness_rate == 1.0 for ctx in context_coverage.values()
        )

        has_gaps = completion_rate < VALIDATION.completion_threshold or any(
            ctx.completeness_rate < VALIDATION.completion_threshold
            for ctx in context_coverage.values()
        )

        # Set status and emoji
        if is_complete:
            icon = EMOJIS.success
            status = ExperimentStatus.COMPLETE
        elif has_gaps:
            icon = EMOJIS.error
            status = ExperimentStatus.INCOMPLETE
        else:
            icon = EMOJIS.warning
            status = ExperimentStatus.PARTIAL

        return RunStatus(
            nodes_done=nodes_done,
            total_nodes=total_nodes,
            completion_rate=completion_rate,
            rounds=global_rounds,
            context_coverage=context_coverage,
            icon=icon,
            status=status,
            is_complete=is_complete,
            has_gaps=has_gaps,
        )

    # === Final Step Data Processing ===

    def get_final_step_data(
        self, context_data: List[Measurement]
    ) -> Tuple[List[Measurement], NodeSync]:
        """Get final step data using latest available measurements with quality assessment."""
        if not context_data:
            return [], self.EMPTY_SYNC

        # Track highest step for each node
        node_final_steps = {}
        for measurement in context_data:
            if measurement.node_id is None:
                continue
            node_final_steps[measurement.node_id] = max(
                node_final_steps.get(measurement.node_id, -1), measurement.global_step
            )

        if not node_final_steps:
            return [], self.EMPTY_SYNC

        # Find maximum global step (latest available data)
        max_global_step = max(node_final_steps.values())

        # Identify nodes that reached max step vs those that stopped earlier
        synced_nodes = {
            nid for nid, step in node_final_steps.items() if step == max_global_step
        }
        lagging_nodes = {
            nid for nid, step in node_final_steps.items() if step != max_global_step
        }

        # Extract all measurements at the maximum global step
        final_measurements = [
            measurement
            for measurement in context_data
            if measurement.global_step == max_global_step
        ]

        # Build step distribution for quality assessment
        step_counts = Counter(node_final_steps.values())

        quality = NodeSync(
            final_step=max_global_step,
            synced_nodes=len(synced_nodes),
            lagging_nodes=len(lagging_nodes),
            total_nodes=len(node_final_steps),
            step_distribution=dict(step_counts),
            lag_positions={
                node_id: node_final_steps[node_id] for node_id in lagging_nodes
            },
            all_synced=not lagging_nodes,
        )

        return final_measurements, quality

    # === Progression Analysis ===

    @staticmethod
    def get_varying_progression_levels(
        context_data: List[Dict[str, Any]],
    ) -> List[ProgressionLevel]:
        """Return progression levels with variation across data."""
        if not context_data:
            return []

        varying_levels = []
        for level in ProgressionLevel.get_progression_order():
            unique_values = {
                row.get(level.value)
                for row in context_data
                if row.get(level.value) is not None  # Explicitly filter None values
            }
            if len(unique_values) > 1:
                varying_levels.append(level)

        return varying_levels

    @staticmethod
    def group_by_training_position(
        context_data: List[Dict[str, Any]],
    ) -> Dict[Tuple[Optional[int], ...], List[Measurement]]:
        """Group data by training position."""
        if not context_data:
            return {}

        grouped = defaultdict(list)
        for row in context_data:
            measurement = Measurement.from_dict(row)
            grouped[measurement.position_tuple].append(measurement)
        return dict(grouped)

    @staticmethod
    def analyze_progression_quality(
        context_data: List[Dict[str, Any]],
    ) -> ProgressionAnalysis:
        """Analyze progression data quality and completeness."""
        if not context_data:
            return ProgressionAnalysis(
                has_data=False,
                error="No context data",
                levels=[],
                total_positions=0,
            )

        levels = ExperimentDataProcessor.get_varying_progression_levels(context_data)

        if not levels:
            return ProgressionAnalysis(
                has_data=False,
                error=None,
                reason="Single checkpoint - no progression to analyze",
                levels=[],
                total_positions=1,
            )

        coordinate_groups = ExperimentDataProcessor.group_by_training_position(
            context_data
        )

        total_positions = len(coordinate_groups)
        if total_positions < 2:
            return ProgressionAnalysis(
                has_data=False,
                error="Insufficient progression points",
                levels=levels,
                total_positions=total_positions,
            )

        return ProgressionAnalysis(
            has_data=True,
            error=None,
            levels=levels,
            coordinate_groups=coordinate_groups,
            total_positions=total_positions,
        )

    @staticmethod
    def sort_positions_chronologically(
        position_tuples: List[Tuple[Optional[int], ...]],
    ) -> List[Tuple[Optional[int], ...]]:
        """Sort position tuples in chronological order."""
        return sorted(
            position_tuples,
            key=lambda x: tuple(-1 if v is None else v for v in x),
        )

    @staticmethod
    def apply_column_limits(
        sorted_positions: List[Tuple[Optional[int], ...]],
        max_epochs_per_round: Optional[int] = None,
        max_batches_per_epoch: Optional[int] = None,
    ) -> Tuple[List[Tuple[Optional[int], ...]], bool]:
        """Apply intelligent limits to prevent table overflow."""
        if not sorted_positions:
            return [], False

        # Get defaults from configuration if not provided
        if max_epochs_per_round is None:
            max_epochs_per_round = VALIDATION.max_epochs_per_round
        if max_batches_per_epoch is None:
            max_batches_per_epoch = VALIDATION.max_batches_per_epoch

        # Group by (round, epoch) pairs
        round_epoch_groups = defaultdict(list)
        for pos in sorted_positions:
            round_idx = pos[0] if pos[0] is not None else 0
            epoch_idx = pos[1] if pos[1] is not None else 0
            round_epoch_groups[(round_idx, epoch_idx)].append(pos)

        limited_positions = []
        was_limited = False

        # Group by rounds to apply epoch limits
        rounds_data = defaultdict(list)
        for (round_idx, epoch_idx), positions in round_epoch_groups.items():
            rounds_data[round_idx].append((epoch_idx, positions))

        for round_idx in sorted(rounds_data.keys()):
            epoch_data = sorted(rounds_data[round_idx])  # Sort by epoch_idx

            # Limit epochs per round
            if len(epoch_data) > max_epochs_per_round:
                epoch_data = epoch_data[:max_epochs_per_round]
                was_limited = True

            for epoch_idx, positions in epoch_data:
                # Limit batches per epoch
                if len(positions) > max_batches_per_epoch:
                    positions = positions[:max_batches_per_epoch]
                    was_limited = True

                limited_positions.extend(positions)

        return limited_positions, was_limited


# === Table Builders ===


class StatisticsTableBuilder:
    """Focused builder for node statistics tables."""

    def __init__(
        self, formatter: MetricFormatter, data_processor: ExperimentDataProcessor
    ):
        """Initialize with required dependencies."""
        self._formatter = formatter
        self._data_processor = data_processor

    def build_statistics_table(
        self,
        context: str,
        final_data: List[Measurement],
        data_quality: NodeSync,
    ) -> Table:
        """Build complete statistics table for final-state metrics."""
        if not final_data:
            # Return empty table with error message
            table = self._create_error_table(context, "No final step data available")
            return table

        # Calculate metric statistics
        metric_stats = self._calculate_all_metric_statistics(final_data, data_quality)
        if not metric_stats:
            table = self._create_error_table(context, "No metrics found")
            return table

        # Build table
        title, caption = self._build_table_metadata(context, data_quality)
        table = self._create_styled_table(title, caption)
        self._add_columns(table)
        self._add_rows(table, metric_stats)

        return table

    def _create_error_table(self, context: str, error_message: str) -> Table:
        """Create table showing error state."""
        context_name = _format_context_name(context)
        title = f"{EMOJIS.error} {context_name} - ERROR"
        caption = f"Unable to display statistics: {error_message}"

        table = self._create_styled_table(title, caption)
        table.add_column("Error", justify="left", style="red")
        table.add_row(error_message)
        return table

    def _build_table_metadata(
        self, context: str, data_quality: NodeSync
    ) -> tuple[str, str]:
        """Build table title and caption with helpful context for users."""
        context_name = _format_context_name(context)

        if data_quality.all_synced:
            title = f"{context_name} - Node Statistics"
            caption = f"Statistics across all {data_quality.total_nodes} nodes at global step {data_quality.final_step}"
        else:
            pct = (
                (data_quality.synced_nodes / data_quality.total_nodes * 100)
                if data_quality.total_nodes > 0
                else 0
            )
            title = f"{context_name} - Node Statistics ({data_quality.synced_nodes}/{data_quality.total_nodes} nodes, {pct:.0f}%)"

            # Show step distribution for debugging
            step_dist = sorted(data_quality.step_distribution.items(), reverse=True)
            dist_info = ", ".join(
                f"{count} at step {step}" for step, count in step_dist[:3]
            )
            if len(step_dist) > 3:
                dist_info += f", +{len(step_dist) - 3} more groups"
            caption = f"Statistics from {data_quality.synced_nodes} at global step {data_quality.final_step}\nStep distribution: {dist_info}"

        return title, caption

    def _create_styled_table(self, title: str, caption: str) -> Table:
        """Create styled table with consistent configuration."""
        return Table(
            title=f"{EMOJIS.node_statistics} {title}",
            title_style="bold cyan",
            title_justify="center",
            caption=caption,
            caption_style="dim white",
            caption_justify="right",
            box=box.ROUNDED,
            show_header=True,
        )

    def _add_columns(self, table: Table) -> None:
        """Add all columns for statistics table."""
        # Metric column
        table.add_column(f"{EMOJIS.metric} Metric", justify="left", style="bold cyan")

        # Statistics columns - directly defined, no abstraction needed
        table.add_column(
            f"{EMOJIS.nodes} Coverage", justify="center", style="bold white"
        )
        table.add_column(f"{EMOJIS.alert} Issues", justify="left", style="bold white")
        table.add_column(f"{EMOJIS.sum} Sum", justify="right")
        table.add_column(f"{EMOJIS.mean} Mean", justify="right")
        table.add_column(f"{EMOJIS.std} Std", justify="right")
        table.add_column(f"{EMOJIS.min} Min", justify="right")
        table.add_column(f"{EMOJIS.max} Max", justify="right")
        table.add_column(f"{EMOJIS.median} Median", justify="right")
        table.add_column(f"{EMOJIS.cv} CV", justify="right")

    def _add_rows(self, table: Table, metric_stats: List[MetricSummary]) -> None:
        """Add all rows with proper grouping."""
        grouped_stats = self._group_metric_stats(metric_stats)
        group_names = sorted(grouped_stats.keys())

        for i, group_name in enumerate(group_names):
            stats_list = grouped_stats[group_name]

            # Add section separator for groups after the first
            if i > 0:
                table.add_section()

            # Add group header if there are multiple groups
            if len(grouped_stats) > 1:
                empty_cols = [""] * 9  # 9 statistics columns
                group_header_text = format_group_header(group_name, len(stats_list))
                table.add_row(group_header_text, *empty_cols)

            # Add metric rows
            for stats in stats_list:
                table.add_row(
                    format_metric_name(stats.name, stats.emoji),
                    format_coverage_display(stats),
                    stats.anomalies,
                    stats.sum,
                    stats.mean,
                    stats.std,
                    stats.min,
                    stats.max,
                    stats.median,
                    stats.cv,
                )

    def _calculate_all_metric_statistics(
        self, final_data: List[Measurement], data_quality: NodeSync
    ) -> List[MetricSummary]:
        """Calculate statistics for all metrics using enhanced formatter."""
        if not final_data:
            return []

        # Extract all metrics
        all_metrics = set()
        for measurement in final_data:
            for key, value in measurement.metrics.items():
                if self._data_processor.is_valid_metric_key(key, value):
                    all_metrics.add(key)

        metric_stats_list = []
        total_nodes = data_quality.total_nodes

        for metric in sorted(all_metrics):
            # Extract values for this metric
            values = []
            for measurement in final_data:
                val = measurement.metrics.get(metric)
                if isinstance(val, (int, float)):
                    values.append(float(val))

            if values:
                # Count actual nodes that reported this specific metric
                metric_reporting_nodes = self._data_processor.count_reporting_nodes(
                    final_data, metric
                )

                stats = self._calculate_single_metric_stats(
                    metric, values, metric_reporting_nodes, total_nodes
                )
                metric_stats_list.append(stats)

        return metric_stats_list

    def _calculate_single_metric_stats(
        self, metric: str, values: List[float], unique_node_count: int, total_nodes: int
    ) -> MetricSummary:
        """Calculate statistics for a single metric using enhanced formatter."""
        rule = self._formatter.find_rule(metric)

        # Use enhanced MetricFormatter for statistics and validation
        stats = self._formatter.compute_statistics(values)
        anomaly_data = self._formatter.detect_outliers(values)

        # Format statistics using formatter
        formatted_stats = self._format_statistics(metric, stats, len(values) == 1)

        return MetricSummary(
            name=metric,
            emoji=rule.emoji,
            group=rule.group,
            nodes_reporting=unique_node_count,
            total_nodes=total_nodes,
            anomalies=self._formatter.format_validation_display(
                values, anomaly_data, metric
            ),
            sum=formatted_stats.sum,
            mean=formatted_stats.mean,
            std=formatted_stats.std,
            min=formatted_stats.min,
            max=formatted_stats.max,
            median=formatted_stats.median,
            cv=formatted_stats.cv,
        )

    def _calculate_cv(self, mean: float, std: float) -> str:
        """Calculate coefficient of variation."""
        if abs(mean) > VALIDATION.division_epsilon:
            cv_val = (std / abs(mean)) * 100
            return f"{cv_val:.1f}%"
        return DISPLAY_PLACEHOLDER

    def _format_statistics(
        self, metric: str, stats: Dict[str, float], is_single_value: bool
    ) -> FormattedStats:
        """Format statistics using MetricFormatter rules."""
        applicable_stats = self._formatter.get_applicable_stats(metric)

        if is_single_value:
            single_value = stats["mean"]  # For single value, mean equals the value
            return FormattedStats(
                sum=self._formatter.format(metric, single_value)
                if applicable_stats["sum"]
                else DISPLAY_PLACEHOLDER,
                mean=self._formatter.format(metric, single_value),
                std=DISPLAY_PLACEHOLDER,
                min=DISPLAY_PLACEHOLDER,
                max=DISPLAY_PLACEHOLDER,
                median=DISPLAY_PLACEHOLDER,
                cv=DISPLAY_PLACEHOLDER,
            )

        # Format all statistics
        return FormattedStats(
            sum=self._formatter.format(metric, stats["sum"])
            if applicable_stats["sum"]
            else DISPLAY_PLACEHOLDER,
            mean=self._formatter.format(metric, stats["mean"])
            if applicable_stats["mean"]
            else DISPLAY_PLACEHOLDER,
            std=self._formatter.format(metric, stats["std"])
            if applicable_stats["std"]
            else DISPLAY_PLACEHOLDER,
            min=self._formatter.format(metric, stats["min"])
            if applicable_stats["min"]
            else DISPLAY_PLACEHOLDER,
            max=self._formatter.format(metric, stats["max"])
            if applicable_stats["max"]
            else DISPLAY_PLACEHOLDER,
            median=self._formatter.format(metric, stats["median"])
            if applicable_stats["median"]
            else DISPLAY_PLACEHOLDER,
            cv=self._calculate_cv(stats["mean"], stats["std"])
            if applicable_stats["cv"]
            else DISPLAY_PLACEHOLDER,
        )

    def _group_metric_stats(
        self, metrics: List[MetricSummary]
    ) -> Dict[str, List[MetricSummary]]:
        """Group MetricSummary objects by category with proper ordering."""
        return group_by_metric_rules(metrics, self._formatter.find_rule)


class ProgressionTableBuilder:
    """Focused builder for training progression tables with coordinate headers."""

    def __init__(
        self, formatter: MetricFormatter, data_processor: ExperimentDataProcessor
    ):
        """Initialize with required dependencies."""
        self._formatter = formatter
        self._data_processor = data_processor

    def show_progression_table(
        self, context: str, context_data: List[Measurement]
    ) -> None:
        """Show progression with data validation and sampling."""
        if not context_data:
            print(f"{EMOJIS.warning}  No data available for {context} progression")
            return

        progression_analysis = self._analyze_progression_data(context, context_data)

        if progression_analysis.error:
            print(
                f"{EMOJIS.error} {context} Progression Error: {progression_analysis.error}"
            )
            return

        if not progression_analysis.has_data:
            print(f"{EMOJIS.info}  {context} - No progression data (single checkpoint)")
            return

        try:
            # Build progression table
            title, caption = self._build_progression_table_metadata(
                context, progression_analysis
            )
            table = Table(
                title=f"{EMOJIS.progression} {title}",
                title_style="bold cyan",
                title_justify="center",
                caption=caption,
                caption_style="dim white",
                caption_justify="right",
                box=box.ROUNDED,
                show_header=True,
            )
            self._add_progression_columns(table, progression_analysis)
            self._add_progression_rows(table, progression_analysis)

            print(table)
            print()
        except (ValueError, TypeError, KeyError) as e:
            print(f"{EMOJIS.error} Failed to build {context} progression table: {e}")
            # Continue execution - don't crash entire display for one table failure

    def _analyze_progression_data(
        self, context: str, context_data: List[Measurement]
    ) -> ProgressionAnalysis:
        """Analyze progression data with data quality and sampling transparency."""
        try:
            if not context_data:
                return ProgressionAnalysis(error="No context data", has_data=False)

            # Convert Measurement objects to raw dict rows for generic analysis utilities
            raw_rows: List[Dict[str, Any]] = []
            for m in context_data:
                row = {
                    "global_step": m.global_step,
                    "round_idx": m.round_idx,
                    "epoch_idx": m.epoch_idx,
                    "batch_idx": m.batch_idx,
                    "_node_id": m.node_id,
                }
                row.update(m.metrics)
                raw_rows.append(row)

            progression_quality = ExperimentDataProcessor.analyze_progression_quality(
                raw_rows
            )
            if not progression_quality.has_data:
                return progression_quality

            levels = progression_quality.levels
            coordinate_groups = progression_quality.coordinate_groups
            metrics = self._data_processor.get_metrics_from_data(context_data)

            if not metrics:
                return ProgressionAnalysis(error="No metrics found", has_data=False)

            sorted_positions = ExperimentDataProcessor.sort_positions_chronologically(
                list(coordinate_groups.keys())
            )
            coordinate_matrix, was_sampled = (
                ExperimentDataProcessor.apply_column_limits(sorted_positions)
            )

            column_headers = self._create_headers_for_positions(coordinate_matrix)

            total_points = len(coordinate_groups)
            displayed_points = len(coordinate_matrix)

            quality_issues = []
            if was_sampled:
                quality_issues.append(
                    f"Showing {displayed_points}/{total_points} checkpoints for readability"
                )

            coverage = {}
            for metric in metrics:
                available_checkpoints = 0
                for coord_tuple in coordinate_matrix:
                    coord_data = coordinate_groups.get(coord_tuple, [])
                    has_metric = any(
                        metric in measurement.metrics
                        and measurement.metrics[metric] is not None
                        for measurement in coord_data
                    )
                    if has_metric:
                        available_checkpoints += 1
                coverage[metric] = available_checkpoints / len(coordinate_matrix)

            incomplete_metrics = [
                m for m, coverage in coverage.items() if coverage < 1.0
            ]
            if incomplete_metrics:
                quality_issues.append(
                    f"{len(incomplete_metrics)} metrics have partial checkpoint data"
                )

            return ProgressionAnalysis(
                error=None,
                has_data=True,
                levels=levels,
                coordinate_groups=coordinate_groups,
                metrics=metrics,
                column_headers=column_headers,
                coordinate_matrix=coordinate_matrix,
                was_sampled=was_sampled,
                total_points=total_points,
                displayed_points=displayed_points,
                quality_issues=quality_issues,
                coverage=coverage,
                incomplete_metrics=incomplete_metrics,
                is_clean=not quality_issues,
            )

        except (KeyError, ValueError) as e:
            return ProgressionAnalysis(
                error=f"Data analysis failed: {e}", has_data=False
            )

    def _build_progression_table_metadata(
        self, context: str, analysis: ProgressionAnalysis
    ) -> Tuple[str, str]:
        """Build table title and caption for progression data."""
        context_name = _format_context_name(context)
        title = f"{context_name} - Experiment Progression"
        return title, ""

    def _add_progression_columns(
        self, table: Table, analysis: ProgressionAnalysis
    ) -> None:
        """Add progression columns."""
        # Add metric column
        table.add_column(f"{EMOJIS.metric} Metric", justify="left", style="bold cyan")

        # Add coordinate columns
        for header in analysis.column_headers:
            table.add_column(header, justify="right", style="white", vertical="middle")

    def _add_progression_rows(
        self, table: Table, analysis: ProgressionAnalysis
    ) -> None:
        """Add progression rows with metric handling."""
        metrics = analysis.metrics
        coordinate_groups = analysis.coordinate_groups
        coordinate_matrix = analysis.coordinate_matrix

        grouped_metrics = self._formatter.group_metric_names(metrics)
        group_names = list(grouped_metrics.keys())

        for i, group_name in enumerate(group_names):
            metric_list = grouped_metrics[group_name]

            # Add section separator for groups after the first
            if i > 0:
                table.add_section()

            # Add group header if there are multiple groups
            if len(grouped_metrics) > 1:
                empty_cols = [""] * len(coordinate_matrix)
                group_header_text = format_group_header(group_name, len(metric_list))
                table.add_row(group_header_text, *empty_cols)

            for metric in metric_list:
                try:
                    self._add_progression_metric_row(
                        table, metric, coordinate_groups, coordinate_matrix, analysis
                    )
                except (ValueError, KeyError, TypeError) as e:
                    print(f"{EMOJIS.warning}  Unable to display metric '{metric}': {e}")
                    error_cols = [
                        f"Unable to display: {str(e)[: VALIDATION.error_message_truncation_limit]}..."
                    ] + [DISPLAY_PLACEHOLDER] * (len(coordinate_matrix) - 1)
                    rule = self._formatter.find_rule(metric)
                    styled_metric = format_metric_name(metric, rule.emoji)
                    table.add_row(styled_metric, *error_cols)

    def _add_progression_metric_row(
        self,
        table: Table,
        metric: str,
        coordinate_groups: Dict[Tuple[Optional[int], ...], List[Dict[str, Any]]],
        coordinate_matrix: List[Tuple[Optional[int], ...]],
        analysis: Dict[str, Any],
    ) -> None:
        """Add metric row to progression table with error handling."""
        # Get metric formatting info
        rule = self._formatter.find_rule(metric)
        styled_metric = format_metric_name(metric, rule.emoji)

        # Build metric values for each coordinate point
        row_data = [styled_metric]
        values = []  # Track for percentage change calculation

        for coord_tuple in coordinate_matrix:
            try:
                coord_data = coordinate_groups.get(coord_tuple, [])

                # Extract metric values for this coordinate
                metric_values = []
                for measurement in coord_data:
                    if (
                        metric in measurement.metrics
                        and measurement.metrics[metric] is not None
                    ):
                        try:
                            value = float(measurement.metrics[metric])
                            if not (np.isnan(value) or np.isinf(value)):
                                metric_values.append(value)
                        except (ValueError, TypeError):
                            continue

                if metric_values:
                    # Use mean of values at this coordinate
                    stats = self._formatter.compute_statistics(metric_values)
                    avg_value = stats["mean"]
                    values.append(avg_value)
                    formatted_value = self._formatter.format(metric, avg_value)
                    row_data.append(formatted_value)
                else:
                    values.append(None)
                    row_data.append(DISPLAY_PLACEHOLDER)

            except (KeyError, ValueError, TypeError) as e:
                print(
                    f"{EMOJIS.warning}  Coordinate processing failed for {metric} at {coord_tuple}: {e}"
                )
                values.append(None)
                row_data.append(DISPLAY_PLACEHOLDER)

        # Add the main metric row
        table.add_row(*row_data)

        # Add percentage change row if we have progression data
        if len(coordinate_matrix) >= 2:
            self._add_percentage_change_row(table, metric, values, coordinate_matrix)

    def _add_percentage_change_row(
        self,
        table: Table,
        metric: str,
        values: List[Optional[float]],
        coordinate_matrix: List[Tuple[Optional[int], ...]],
    ) -> None:
        """Add percentage change row with error handling."""
        try:
            valid_values = [v for v in values if v is not None]
            if len(valid_values) < 2:
                return

            # Create percentage change row
            pct_row_data = [""]  # Empty first column for alignment

            for i, coord_tuple in enumerate(coordinate_matrix):
                try:
                    if i == 0:
                        pct_row_data.append(DISPLAY_PLACEHOLDER)
                        continue

                    current_value = values[i]
                    if current_value is None:
                        pct_row_data.append(DISPLAY_PLACEHOLDER)
                        continue

                    # Find previous valid value
                    prev_value = None
                    for j in range(i - 1, -1, -1):
                        if values[j] is not None:
                            prev_value = values[j]
                            break

                    if prev_value is None:
                        pct_row_data.append(DISPLAY_PLACEHOLDER)
                        continue

                    # Calculate percentage change
                    if abs(prev_value) > VALIDATION.division_epsilon:
                        pct_change = (
                            (current_value - prev_value) / abs(prev_value)
                        ) * 100

                        # Get color based on optimization goal
                        color = self._formatter.get_delta_color(
                            metric, current_value - prev_value
                        )

                        # Format percentage
                        if abs(pct_change) < VALIDATION.small_change_threshold:
                            pct_str = f"[italic {color}]~0.0%[/italic {color}]"
                        else:
                            pct_str = (
                                f"[italic {color}]{pct_change:+5.1f}%[/italic {color}]"
                            )

                        pct_row_data.append(pct_str)
                    else:
                        # Handle division by zero
                        if current_value > prev_value:
                            pct_row_data.append(
                                f"[{COLORS.positive_change}]+∞%[/{COLORS.positive_change}]"
                            )
                        elif current_value < prev_value:
                            pct_row_data.append(
                                f"[{COLORS.negative_change}]-∞%[/{COLORS.negative_change}]"
                            )
                        else:
                            pct_row_data.append(DISPLAY_PLACEHOLDER)

                except (ArithmeticError, ValueError) as e:
                    print(
                        f"{EMOJIS.warning}  Percentage calculation failed for {metric} at coordinate {i}: {e}"
                    )
                    pct_row_data.append(DISPLAY_PLACEHOLDER)

            # Add percentage change row
            table.add_row(*pct_row_data)

        except (ArithmeticError, ValueError) as e:
            print(
                f"{EMOJIS.error} Percentage change calculation failed for metric '{metric}': {e}"
            )

    def _create_headers_for_positions(
        self, position_matrix: List[Tuple[Optional[int], ...]]
    ) -> List[str]:
        """Create properly aligned table headers where each level stays in its designated row."""
        if not position_matrix:
            return []

        levels = ProgressionLevel.get_progression_order()
        level_boundaries = self._compute_level_boundaries(position_matrix, levels)
        header_matrix = self._build_header_matrix(
            position_matrix, levels, level_boundaries
        )
        self._apply_visual_continuations(header_matrix, position_matrix, levels)
        return self._assemble_column_headers(header_matrix, position_matrix, levels)

    def _compute_level_boundaries(
        self, position_matrix: List[Tuple[Optional[int], ...]], levels
    ) -> Dict[int, Dict[str, int]]:
        """Pre-compute boundary positions for visual styling."""
        level_boundaries = {}
        for level_idx, level in enumerate(levels):
            level_values = []
            for position_tuple in position_matrix:
                if (
                    level_idx < len(position_tuple)
                    and position_tuple[level_idx] is not None
                ):
                    level_values.append(position_tuple[level_idx])

            if level_values:
                unique_values = sorted(set(level_values))
                level_boundaries[level_idx] = {
                    "first": unique_values[0],
                    "last": unique_values[-1],
                }
        return level_boundaries

    def _build_header_matrix(
        self,
        position_matrix: List[Tuple[Optional[int], ...]],
        levels,
        level_boundaries: Dict[int, Dict[str, int]],
    ) -> List[List[str]]:
        """Build the header matrix with values and continuation symbols."""
        # Initialize matrix with empty strings
        header_matrix = []
        for level_idx in range(len(levels)):
            header_matrix.append([""] * len(position_matrix))

        # Fill the matrix
        for col_idx, position_tuple in enumerate(position_matrix):
            prev_position = position_matrix[col_idx - 1] if col_idx > 0 else None

            for level_idx, level in enumerate(levels):
                if level_idx < len(position_tuple):
                    value = position_tuple[level_idx]
                    if value is not None:
                        # Check if this value changed from previous column
                        value_changed = (
                            prev_position is None
                            or level_idx >= len(prev_position)
                            or prev_position[level_idx] != value
                        )

                        if value_changed:
                            # Show the actual value with boundary styling
                            is_boundary = False
                            if level_idx in level_boundaries:
                                bounds = level_boundaries[level_idx]
                                is_boundary = (
                                    value == bounds["first"] or value == bounds["last"]
                                )

                            level_name = self._normalize_level_name(level)
                            formatted_label = format_coordinate_label(
                                level_name, value, is_boundary
                            )
                            header_matrix[level_idx][col_idx] = formatted_label
                        else:
                            # Use continuation symbol with appropriate color
                            level_name = self._normalize_level_name(level)
                            color = _get_level_color(level_name)
                            header_matrix[level_idx][col_idx] = (
                                f"[{color}]...[/{color}]"
                            )

        return header_matrix

    def _apply_visual_continuations(
        self,
        header_matrix: List[List[str]],
        position_matrix: List[Tuple[Optional[int], ...]],
        levels,
    ) -> None:
        """Apply visual row spanning using special continuation characters."""
        for level_idx in range(len(levels)):
            for col_idx in range(1, len(position_matrix)):
                current_cell = header_matrix[level_idx][col_idx]
                prev_cell = header_matrix[level_idx][col_idx - 1]

                if current_cell and prev_cell and "..." in current_cell:
                    current_pos = position_matrix[col_idx]
                    prev_pos = position_matrix[col_idx - 1]

                    if (
                        level_idx < len(current_pos)
                        and level_idx < len(prev_pos)
                        and current_pos[level_idx] == prev_pos[level_idx]
                    ):
                        color = _get_level_color(
                            self._normalize_level_name(levels[level_idx])
                        )
                        header_matrix[level_idx][col_idx] = f"[{color}]━━━[/{color}]"

    def _assemble_column_headers(
        self,
        header_matrix: List[List[str]],
        position_matrix: List[Tuple[Optional[int], ...]],
        levels,
    ) -> List[str]:
        """Convert matrix to column headers by joining rows vertically."""
        headers = []
        for col_idx in range(len(position_matrix)):
            column_parts = []
            for level_idx in range(len(levels)):
                cell_value = header_matrix[level_idx][col_idx]
                if cell_value:
                    column_parts.append(cell_value)

            headers.append("\n".join(column_parts) if column_parts else "Unknown")

        return headers

    def _normalize_level_name(self, level) -> str:
        """Normalize level name for display."""
        level_name = level.value.replace("_idx", "").replace("_", " ").lower()
        return "step" if level_name == "global step" else level_name


# === Main Display Manager ===


class ResultsDisplayManager:
    """Main coordinator for FL experiment results display system."""

    def __init__(self):
        """Initialize display with focused components."""
        self._formatter = MetricFormatter(DEFAULT_FORMAT_RULES)
        self._data_processor = ExperimentDataProcessor()
        self._statistics_builder = StatisticsTableBuilder(
            self._formatter, self._data_processor
        )
        self._progression_builder = ProgressionTableBuilder(
            self._formatter, self._data_processor
        )

    def show_experiment_results(
        self,
        results: List[Dict[str, Any]],
        duration: float,
        global_rounds: int,
        total_nodes: int,
    ) -> None:
        """Display FL experiment results."""
        # Early validation returns
        if not results:
            print(f"{EMOJIS.error} No experiment results provided")
            return
        if total_nodes <= 0:
            print(f"{EMOJIS.error} Invalid configuration: total_nodes must be positive")
            return
        if duration < 0:
            print(f"{EMOJIS.error} Invalid configuration: duration cannot be negative")
            return
        if global_rounds <= 0:
            print(
                f"{EMOJIS.error} Invalid configuration: global_rounds must be positive"
            )
            return

        # Show experiment summary first
        print(f"{EMOJIS.processing} Generating experiment summary...")
        completion_analysis = self._data_processor.analyze_experiment_completion(
            results, total_nodes, global_rounds
        )
        summary_table = self._build_summary_table(completion_analysis, duration)
        print(summary_table)
        print()

        # Process each context
        contexts = self._data_processor.get_all_contexts(results)
        if not contexts:
            print(f"{EMOJIS.warning}  No experiment contexts found")
            return

        for context in contexts:
            try:
                context_data = self._data_processor.extract_context_data(
                    results, context
                )
                if context_data:
                    self._show_context_results(context, context_data, total_nodes)
                else:
                    print(f"{EMOJIS.warning}  No data available for context: {context}")
            except (KeyError, ValueError, TypeError) as e:
                print(f"{EMOJIS.error} Failed to process context '{context}': {e}")
                # Re-raise to see full traceback
                raise

    def _build_summary_table(self, analysis: RunStatus, duration: float) -> Table:
        """Build experiment summary table."""
        if analysis.is_complete:
            title = "Experiment Summary - Complete"
        elif analysis.has_gaps:
            title = "Experiment Summary - Incomplete"
        else:
            title = "Experiment Summary - Partial"

        table = Table(
            title=f"{analysis.icon} {title}",
            title_style="bold cyan",
            title_justify="center",
            box=box.ROUNDED,
            show_header=True,
        )
        table.add_column(f"{EMOJIS.metric} Metric", justify="left", style="bold cyan")
        table.add_column(f"{EMOJIS.value} Value", justify="center", style="bold green")

        # Add rows
        table.add_row(f"{EMOJIS.rounds} Total Rounds", str(analysis.rounds))
        table.add_row(
            f"{analysis.icon} Nodes Completed",
            f"{analysis.nodes_done}/{analysis.total_nodes} ({analysis.completion_rate:.0%})",
        )
        table.add_row(f"{EMOJIS.duration} Experiment Duration", f"{duration:.2f}s")

        if not analysis.is_complete and analysis.context_coverage:
            for context, ctx_analysis in analysis.context_coverage.items():
                if ctx_analysis.completeness_rate < 1.0:
                    table.add_row(
                        f"{EMOJIS.warning} {context.title()} Data",
                        f"{ctx_analysis.nodes_with_data}/{analysis.nodes_done} ({ctx_analysis.completeness_rate:.0%})",
                    )

        return table

    def _show_context_results(
        self, context: str, context_data: List[Dict[str, Any]], total_nodes: int
    ) -> None:
        """Show results for a specific context - both progression and node statistics."""
        print_rule()
        print()

        # Show node statistics
        final_data, data_quality = self._data_processor.get_final_step_data(
            context_data
        )
        if final_data:
            table = self._statistics_builder.build_statistics_table(
                context, final_data, data_quality
            )
            print(table)
            print()
        else:
            print(f"{EMOJIS.warning}  No final step data available for {context}")

        # Show progression table
        self._progression_builder.show_progression_table(context, context_data)

        print()
