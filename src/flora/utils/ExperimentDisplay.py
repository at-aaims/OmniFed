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
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rich import box, print
from rich.table import Table
from typeguard import typechecked

from .logging import print_rule
from .MetricFormatter import (
    DEFAULT_FORMAT_RULES,
    MetricDirection,
    MetricFormatRule,
    MetricFormatter,
    MetricGroup,
)
from .DisplayStyling import (
    get_table_config,
    get_column_config,
    format_coordinate_label,
    format_metric_name,
    format_group_header,
    get_colors,
    get_emojis,
)


class ProgressionLevel(Enum):
    """Training progression levels (coarse to fine)."""

    ROUND = "round_idx"  # FL rounds (coarsest)
    EPOCH = "epoch_idx"  # Training epochs within rounds
    BATCH = "batch_idx"  # Batches within epochs
    GLOBAL_STEP = "global_step"  # Global training steps (finest)

    @classmethod
    def get_progression_order(cls) -> List["ProgressionLevel"]:
        """Return progression levels in coarse to fine order."""
        return [cls.ROUND, cls.EPOCH, cls.BATCH, cls.GLOBAL_STEP]

    @classmethod
    def get_field_names(cls) -> Set[str]:
        """Return all progression field names as a set."""
        return {level.value for level in cls}


@dataclass(frozen=True)
class TrainingPosition:
    """Point in FL training progression.

    Missing levels are None for partial data.
    """

    round_idx: Optional[int] = None
    epoch_idx: Optional[int] = None
    batch_idx: Optional[int] = None
    global_step: Optional[int] = None

    @classmethod
    def from_data_row(cls, row: Dict[str, Any]) -> "TrainingPosition":
        """Create position from data row."""
        return cls(
            round_idx=row.get("round_idx"),
            epoch_idx=row.get("epoch_idx"),
            batch_idx=row.get("batch_idx"),
            global_step=row.get("global_step"),
        )

    def get_value_for_level(self, level: ProgressionLevel) -> Optional[int]:
        """Get value for progression level."""
        # Direct attribute access
        if level == ProgressionLevel.ROUND:
            return self.round_idx
        elif level == ProgressionLevel.EPOCH:
            return self.epoch_idx
        elif level == ProgressionLevel.BATCH:
            return self.batch_idx
        elif level == ProgressionLevel.GLOBAL_STEP:
            return self.global_step
        else:
            raise ValueError(f"Unknown progression level: {level}")


def get_varying_progression_levels(
    context_data: List[Dict[str, Any]],
) -> List[ProgressionLevel]:
    """Return progression levels with variation."""
    if not context_data:
        return []

    varying_levels = []
    for level in ProgressionLevel.get_progression_order():
        unique_values = {
            row.get(level.value)
            for row in context_data
            if row.get(level.value) is not None
        }
        if len(unique_values) > 1:
            varying_levels.append(level)

    return varying_levels


def group_by_training_position(
    context_data: List[Dict[str, Any]],
) -> Dict[Tuple[Optional[int], ...], List[Dict[str, Any]]]:
    """Group data by training position."""
    if not context_data:
        return {}

    grouped = defaultdict(list)
    for row in context_data:
        position = TrainingPosition.from_data_row(row)
        position_tuple = (
            position.round_idx,
            position.epoch_idx,
            position.batch_idx,
            position.global_step,
        )
        grouped[position_tuple].append(row)

    return dict(grouped)


def create_progression_headers(context_data: List[Dict[str, Any]]) -> List[str]:
    """Create aligned table headers for training progression."""
    if not context_data:
        return []

    varying_levels = get_varying_progression_levels(context_data)
    if not varying_levels:
        return ["No Progression"]

    grouped = group_by_training_position(context_data)
    sorted_positions = sorted(
        grouped.keys(), key=lambda x: tuple(-1 if v is None else v for v in x)
    )

    # Matrix where each row = level (Round, Epoch, Batch, Step)
    all_levels = ProgressionLevel.get_progression_order()
    header_matrix = []

    # Initialize matrix with empty strings
    for level in all_levels:
        header_matrix.append([""] * len(sorted_positions))

    # Fill the matrix
    for col_idx, position_tuple in enumerate(sorted_positions):
        prev_position = sorted_positions[col_idx - 1] if col_idx > 0 else None

        for level_idx, level in enumerate(all_levels):
            if level in varying_levels and level_idx < len(position_tuple):
                value = position_tuple[level_idx]
                if value is not None:
                    # Check if this value changed from previous column
                    value_changed = (
                        prev_position is None
                        or level_idx >= len(prev_position)
                        or prev_position[level_idx] != value
                    )

                    if value_changed:
                        # Show the actual value with color
                        level_name = (
                            level.value.replace("_idx", "").replace("_", " ").lower()
                        )
                        if level_name == "global step":
                            level_name = "step"
                        formatted_label = format_coordinate_label(
                            level_name, value, False
                        )
                        header_matrix[level_idx][col_idx] = formatted_label
                    else:
                        # Use continuation symbol with appropriate color
                        level_name = (
                            level.value.replace("_idx", "").replace("_", " ").lower()
                        )
                        if level_name == "global step":
                            level_name = "step"
                        # Get color for this level
                        colors = get_colors()
                        color_map = {
                            "round": colors.round_color,
                            "epoch": colors.epoch_color,
                            "batch": colors.batch_color,
                            "step": colors.step_color,
                        }
                        color = color_map.get(level_name, colors.info)
                        header_matrix[level_idx][col_idx] = f"[{color}]...[/{color}]"

    # Convert matrix to column headers by joining rows vertically
    headers = []
    for col_idx in range(len(sorted_positions)):
        # Get all non-empty row values for this column
        column_parts = []
        for level_idx in range(len(all_levels)):
            cell_value = header_matrix[level_idx][col_idx]
            if cell_value:  # Only add non-empty cells
                column_parts.append(cell_value)

        headers.append("\n".join(column_parts) if column_parts else "Unknown")

    return headers


@dataclass
class MetricStats:
    """Statistical summary of a metric across FL nodes."""

    name: str  # Metric name (e.g., "train/accuracy")
    emoji: str  # Display icon for this metric type
    group: str  # Category for table grouping
    node_count: int  # How many nodes reported this metric
    total_nodes: int  # Total nodes in experiment
    sum: str  # Formatted sum value (total across nodes)
    mean: str  # Formatted mean value
    std: str  # Formatted standard deviation
    min: str  # Formatted minimum value
    max: str  # Formatted maximum value
    median: str  # Formatted median value
    cv: str  # Formatted coefficient of variation (std/mean)

    @property
    def coverage_display(self) -> str:
        """Node coverage display (e.g., '3/5 (60%)')."""
        if self.total_nodes > 0:
            percentage = (self.node_count / self.total_nodes) * 100
            return f"{self.node_count}/{self.total_nodes} ({percentage:.0f}%)"
        else:
            return f"{self.node_count}/{self.total_nodes}"

    @property
    def display_name(self) -> str:
        """Full display name with emoji and coverage."""
        return f"{self.emoji} {self.name} {self.coverage_display}"

    @property
    def is_complete_coverage(self) -> bool:
        """True if all nodes reported this metric."""
        return self.node_count == self.total_nodes


class ExperimentDataProcessor:
    """Data processing for experiment results."""

    def __init__(self, metadata_keys: set[str]):
        self._metadata_keys = metadata_keys

    def _create_coordinate_key(self, row: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """Create coordinate key from data row.

        Returns (round_idx, epoch_idx, batch_idx, global_step).
        """
        return (
            row.get("round_idx", 0),
            row.get("epoch_idx", 0),
            row.get("batch_idx", 0),
            row.get("global_step", 0),
        )

    def get_all_contexts(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract all metric contexts from results."""
        contexts = set()
        for node_data in results:
            contexts.update(node_data.keys())
        return list(contexts)

    def extract_context_data(
        self, results: List[Dict[str, Any]], context: str
    ) -> List[Dict[str, Any]]:
        """Extract context data across nodes, preserving node identity."""
        context_data = []
        for node_idx, node_data in enumerate(results):
            if context in node_data:
                # Add node_id to preserve identity
                for measurement in node_data[context]:
                    measurement_with_node = measurement.copy()
                    measurement_with_node["_node_id"] = node_idx
                    context_data.append(measurement_with_node)
        return context_data

    def get_metrics_from_data(self, context_data: List[Dict[str, Any]]) -> List[str]:
        """Extract metric names from data (excluding metadata)."""
        metrics = set()
        for row in context_data:
            for key, value in row.items():
                if (
                    key not in self._metadata_keys
                    and not key.startswith("_")
                    and value is not None
                ):
                    metrics.add(key)
        return list(metrics)

    def get_final_values_per_coordinate(
        self, context_data: List[Dict[str, Any]], preserve_coordinates: bool = False
    ) -> List[Dict[str, Any]]:
        """Get final metric values for each unique coordinate combination.

        Args:
            context_data: Raw experiment data for a specific context
            preserve_coordinates: If True, include coordinate fields in output
        """
        # Group by coordinate combination using centralized key creation
        coordinate_groups = defaultdict(list)

        for row in context_data:
            coord_key = self._create_coordinate_key(row)
            coordinate_groups[coord_key].append(row)

        # For each coordinate, take the final (aggregated) values
        final_data = []
        for coord_key, coord_rows in coordinate_groups.items():
            if coord_rows:
                # Use the last row for this coordinate (typically the aggregated result)
                final_row = coord_rows[-1]

                # Extract metric values and optionally coordinates
                metric_row = {}
                for key, value in final_row.items():
                    # Include coordinates if requested
                    if preserve_coordinates and key in {
                        "round_idx",
                        "epoch_idx",
                        "batch_idx",
                        "global_step",
                    }:
                        metric_row[key] = value
                    # Include non-metadata, non-private fields
                    elif (
                        key not in self._metadata_keys
                        and not key.startswith("_")
                        and value is not None
                        and key
                        not in {
                            "round_idx",
                            "epoch_idx",
                            "batch_idx",
                            "global_step",
                        }  # Avoid duplication
                    ):
                        metric_row[key] = value

                if metric_row:  # Only add if there are actual metrics
                    final_data.append(metric_row)

        return final_data


class ExperimentResultsDisplay:
    """FL experiment results display system."""

    # Display configuration constants
    SIGNIFICANCE_THRESHOLD = 15.0  # Percentage change threshold for bold formatting
    SMALL_CHANGE_THRESHOLD = 0.1  # Threshold for ~0.0% display
    DIVISION_EPSILON = 1e-10  # Minimum value to avoid division by zero

    def __init__(self):
        """Initialize display with formatter and data processor."""
        self._formatter = MetricFormatter(DEFAULT_FORMAT_RULES)
        self._metadata_keys = {
            "global_step",
            "round_idx",
            "epoch_idx",
            "batch_idx",
            "_node_id",
        }
        self._data_processor = ExperimentDataProcessor(self._metadata_keys)

    def _validate_inputs(self, data: Any, data_name: str) -> bool:
        """Central input validation with transparent error reporting."""
        if data is None:
            print(f"❌ No {data_name} provided")
            return False
        if isinstance(data, list) and len(data) == 0:
            print(f"⚠️  No {data_name} available")
            return False
        return True

    @typechecked
    def show_experiment_results(
        self,
        results: List[Dict[str, Any]],
        duration: float,
        global_rounds: int,
        total_nodes: int,
    ) -> None:
        """Display FL experiment results."""
        # Validate inputs
        if not self._validate_inputs(results, "experiment results"):
            return
        if total_nodes <= 0:
            print("❌ Invalid experiment configuration: total_nodes must be positive")
            return
        if duration < 0:
            print("❌ Invalid experiment configuration: duration cannot be negative")
            return
        if global_rounds <= 0:
            print("❌ Invalid experiment configuration: global_rounds must be positive")
            return

        # Display experiment summary
        self._show_experiment_summary(results, duration, global_rounds, total_nodes)

        # Process each context
        contexts = self._data_processor.get_all_contexts(results)
        if not contexts:
            print("⚠️  No experiment contexts found")
            return

        for context in self._order_contexts(contexts):
            try:
                context_data = self._data_processor.extract_context_data(
                    results, context
                )
                if context_data:
                    self._show_context_results(context, context_data, total_nodes)
                else:
                    print(f"⚠️  No data available for context: {context}")
            except Exception as e:
                # Context processing should not fail silently - this indicates a bug
                print(f"❌ Failed to process context '{context}': {e}")
                # Continue with other contexts - don't crash entire display
                continue

    def _show_experiment_summary(
        self,
        results: List[Dict[str, Any]],
        duration: float,
        global_rounds: int,
        total_nodes: int,
    ) -> None:
        """Show experiment summary with completion status."""
        if not results:
            print("❌ ERROR: No experiment results to summarize")
            return

        if total_nodes <= 0:
            print("❌ ERROR: Invalid total_nodes count for experiment summary")
            return

        # Analyze experiment completion
        completion_analysis = self._analyze_experiment_completion(
            results, total_nodes, global_rounds
        )

        # Create table with completion status
        table_title, table_caption = self._build_summary_table_metadata(
            completion_analysis
        )

        emojis = get_emojis()
        table = self._create_styled_table(
            title=table_title,
            caption=table_caption,
            title_emoji=completion_analysis["title_emoji"],
        )

        # Add columns using centralized styling
        metric_config = get_column_config("metric")
        value_config = get_column_config("value")

        table.add_column(
            metric_config["header"],
            justify=metric_config["justify"],
            style=metric_config["style"],
            vertical=metric_config["vertical"],
        )
        table.add_column(
            value_config["header"],
            justify=value_config["justify"],
            style=value_config["style"],
            vertical=value_config["vertical"],
        )

        # Add summary rows
        self._add_summary_rows(table, completion_analysis, duration)

        print(table)
        print()

    def _analyze_experiment_completion(
        self, results: List[Dict[str, Any]], total_nodes: int, global_rounds: int
    ) -> Dict[str, Any]:
        """Analyze experiment completion and data quality."""
        emojis = get_emojis()

        # Basic completion metrics
        completed_nodes = len(results)
        completion_rate = (completed_nodes / total_nodes) if total_nodes > 0 else 0

        # Analyze data completeness across contexts
        contexts_analysis = {}
        if results:
            all_contexts = set()
            for node_result in results:
                all_contexts.update(node_result.keys())

            # Check context completeness
            for context in all_contexts:
                nodes_with_context = sum(
                    1
                    for node_result in results
                    if context in node_result and node_result[context]
                )
                contexts_analysis[context] = {
                    "nodes_with_data": nodes_with_context,
                    "completeness_rate": nodes_with_context / completed_nodes
                    if completed_nodes > 0
                    else 0,
                }

        # Determine overall experiment quality
        is_complete = completion_rate == 1.0 and all(
            ctx["completeness_rate"] == 1.0 for ctx in contexts_analysis.values()
        )

        has_significant_missing_data = completion_rate < 0.5 or any(
            ctx["completeness_rate"] < 0.5 for ctx in contexts_analysis.values()
        )

        # Choose appropriate emoji and status
        if is_complete:
            title_emoji = emojis.success
            status = "complete"
        elif has_significant_missing_data:
            title_emoji = emojis.error
            status = "incomplete"
        else:
            title_emoji = emojis.warning
            status = "partial"

        return {
            "completed_nodes": completed_nodes,
            "total_nodes": total_nodes,
            "completion_rate": completion_rate,
            "global_rounds": global_rounds,
            "contexts_analysis": contexts_analysis,
            "title_emoji": title_emoji,
            "status": status,
            "is_complete": is_complete,
            "has_significant_missing_data": has_significant_missing_data,
        }

    def _build_summary_table_metadata(
        self, analysis: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Build table title and caption showing experiment quality."""
        completed = analysis["completed_nodes"]
        total = analysis["total_nodes"]
        rate = analysis["completion_rate"]

        # Build title based on completion quality
        if analysis["is_complete"]:
            title = f"Experiment Summary - Complete"
        elif analysis["has_significant_missing_data"]:
            title = f"Experiment Summary - Incomplete ({completed}/{total} nodes, {rate:.0%})"
        else:
            title = (
                f"Experiment Summary - Partial ({completed}/{total} nodes, {rate:.0%})"
            )

        # Build caption with context details
        if analysis["contexts_analysis"]:
            context_issues = []
            for context, ctx_analysis in analysis["contexts_analysis"].items():
                if ctx_analysis["completeness_rate"] < 1.0:
                    missing = completed - ctx_analysis["nodes_with_data"]
                    context_issues.append(f"{context}: {missing} nodes missing data")

            if context_issues:
                caption = f"Data completeness:\n{'; '.join(context_issues)}"
            else:
                caption = "All nodes completed successfully\nwith complete data"
        else:
            caption = "No experiment data available for analysis"

        return title, caption

    def _add_summary_rows(
        self, table: Table, analysis: Dict[str, Any], duration: float
    ) -> None:
        """Add summary rows with quality indicators."""
        emojis = get_emojis()

        # Rounds completed
        table.add_row(f"{emojis.rounds} Total Rounds", str(analysis["global_rounds"]))

        # Node completion with quality indicator
        completed = analysis["completed_nodes"]
        total = analysis["total_nodes"]
        rate = analysis["completion_rate"]

        completion_display = f"{completed}/{total} ({rate:.0%})"
        completion_emoji = analysis[
            "title_emoji"
        ]  # Use same emoji as title for consistency
        table.add_row(f"{completion_emoji} Nodes Completed", completion_display)

        # Duration
        table.add_row(f"{emojis.duration} Experiment Duration", f"{duration:.2f}s")

        # Add context completeness if there are issues
        if not analysis["is_complete"] and analysis["contexts_analysis"]:
            for context, ctx_analysis in analysis["contexts_analysis"].items():
                if ctx_analysis["completeness_rate"] < 1.0:
                    ctx_nodes = ctx_analysis["nodes_with_data"]
                    ctx_rate = ctx_analysis["completeness_rate"]
                    ctx_display = f"{ctx_nodes}/{completed} ({ctx_rate:.0%})"
                    table.add_row(
                        f"{emojis.warning} {context.title()} Data", ctx_display
                    )

    def _order_contexts(self, contexts: List[str]) -> List[str]:
        """Order contexts alphabetically for consistent display."""
        return sorted(contexts)

    def _create_headers_for_positions(
        self, position_matrix: List[Tuple[Optional[int], ...]]
    ) -> List[str]:
        """Create properly aligned table headers where each level stays in its designated row."""
        if not position_matrix:
            return []

        # Create a matrix where each row represents a level (Round, Epoch, Batch, Step)
        levels = ProgressionLevel.get_progression_order()
        header_matrix = []

        # Initialize matrix with empty strings
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
                            # Show the actual value with color
                            level_name = (
                                level.value.replace("_idx", "")
                                .replace("_", " ")
                                .lower()
                            )
                            if level_name == "global step":
                                level_name = "step"
                            formatted_label = format_coordinate_label(
                                level_name, value, False
                            )
                            header_matrix[level_idx][col_idx] = formatted_label
                        else:
                            # Use continuation symbol with appropriate color
                            level_name = (
                                level.value.replace("_idx", "")
                                .replace("_", " ")
                                .lower()
                            )
                            if level_name == "global step":
                                level_name = "step"
                            # Get color for this level
                            colors = get_colors()
                            color_map = {
                                "round": colors.round_color,
                                "epoch": colors.epoch_color,
                                "batch": colors.batch_color,
                                "step": colors.step_color,
                            }
                            color = color_map.get(level_name, colors.info)
                            header_matrix[level_idx][col_idx] = (
                                f"[{color}]...[/{color}]"
                            )

        # Convert matrix to column headers by joining rows vertically
        headers = []
        for col_idx in range(len(position_matrix)):
            # Get all non-empty row values for this column
            column_parts = []
            for level_idx in range(len(levels)):
                cell_value = header_matrix[level_idx][col_idx]
                if cell_value:  # Only add non-empty cells
                    column_parts.append(cell_value)

            headers.append("\n".join(column_parts) if column_parts else "Unknown")

        return headers

    def _create_styled_table(
        self,
        title: str,
        caption: str,
        title_emoji: str,
        box_style: box.Box = box.ROUNDED,
        header_style: Optional[
            str
        ] = None,  # Use None to signal using centralized styling
    ) -> Table:
        """Create a consistently styled table using centralized styling.

        Args:
            title: Table title
            caption: Table caption
            title_emoji: Emoji to prefix the title
            box_style: Rich box style for table borders (optional override)
            header_style: Style for table headers (optional override)

        Returns:
            Configured Table instance
        """
        # Get standardized configuration
        table_config = get_table_config(title, caption, title_emoji)

        # Allow optional overrides
        if box_style != box.ROUNDED:
            table_config["box"] = box_style
        if header_style is not None:
            table_config["header_style"] = header_style
        else:
            table_config["header_style"] = get_colors().header_primary

        return Table(**table_config)

    def _show_context_results(
        self, context: str, context_data: List[Dict[str, Any]], total_nodes: int
    ) -> None:
        """Show results for a specific context - both progression and node statistics."""
        print_rule()
        print()

        # Show node statistics table first
        self._show_node_statistics(context, context_data, total_nodes)

        # Show progression table second
        self._show_progression_table(context, context_data)

    def _show_progression_table(
        self, context: str, context_data: List[Dict[str, Any]]
    ) -> None:
        """Show progression with data validation and sampling."""
        if not context_data:
            print(f"⚠️  No data available for {context} progression")
            return

        # Analyze progression data
        progression_analysis = self._analyze_progression_data(context, context_data)

        if progression_analysis["error"]:
            print(f"❌ {context} Progression Error: {progression_analysis['error']}")
            return

        if not progression_analysis["has_progression"]:
            print(f"ℹ️  {context} - No progression data (single checkpoint)")
            return

        # Build table with data quality metadata
        table_title, table_caption = self._build_progression_table_metadata(
            context, progression_analysis
        )

        table = self._create_styled_table(
            title=table_title,
            caption=table_caption,
            title_emoji=get_emojis().progression,
        )

        # Add columns
        try:
            self._add_progression_columns(table, progression_analysis)
            self._add_progression_rows(table, progression_analysis)
        except Exception as e:
            # Table building failure indicates programming error - don't hide it
            print(f"❌ Failed to build {context} progression table: {e}")
            raise RuntimeError(
                f"Progression table construction failed for {context}"
            ) from e

        print(table)
        print()

    def _analyze_progression_data(
        self, context: str, context_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze progression data with data quality and sampling transparency."""
        try:
            # Use simple functions instead of over-engineered classes

            # Validate basic data structure
            if not context_data:
                return {"error": "No context data", "has_progression": False}

            # Check for progression (multiple training positions)
            active_levels = get_varying_progression_levels(context_data)
            if not active_levels:
                return {
                    "error": None,
                    "has_progression": False,
                    "reason": "Single checkpoint",
                }

            # Group data by training position and validate
            coordinate_groups = group_by_training_position(context_data)
            metrics = self._data_processor.get_metrics_from_data(context_data)

            if not metrics:
                return {"error": "No metrics found", "has_progression": False}

            if len(coordinate_groups) < 2:
                return {
                    "error": "Insufficient progression points",
                    "has_progression": False,
                }

            # Generate progression matrix with sampling transparency
            sorted_positions = sorted(
                coordinate_groups.keys(),
                key=lambda x: tuple(-1 if v is None else v for v in x),
            )
            coordinate_matrix = sorted_positions[
                :5
            ]  # Limit to 5 columns for readability
            was_sampled = len(sorted_positions) > 5

            # Create headers only for the sampled positions
            column_headers = self._create_headers_for_positions(coordinate_matrix)

            # Analyze data completeness
            total_possible_points = len(coordinate_groups)
            displayed_points = len(coordinate_matrix)

            # Check for data quality issues
            data_quality_issues = []
            if was_sampled:
                data_quality_issues.append(
                    f"Showing {displayed_points}/{total_possible_points} checkpoints for readability"
                )

            # Analyze metric availability across checkpoints
            metrics_coverage = {}
            for metric in metrics:
                available_checkpoints = 0
                for coord_tuple in coordinate_matrix:
                    coord_data = coordinate_groups.get(coord_tuple, [])
                    has_metric = any(
                        metric in row and row[metric] is not None for row in coord_data
                    )
                    if has_metric:
                        available_checkpoints += 1
                metrics_coverage[metric] = available_checkpoints / len(
                    coordinate_matrix
                )

            # Check for incomplete metrics
            incomplete_metrics = [
                m for m, coverage in metrics_coverage.items() if coverage < 1.0
            ]
            if incomplete_metrics:
                data_quality_issues.append(
                    f"{len(incomplete_metrics)} metrics have partial checkpoint data"
                )

            return {
                "error": None,
                "has_progression": True,
                "active_levels": active_levels,
                "coordinate_groups": coordinate_groups,
                "metrics": metrics,
                "column_headers": column_headers,
                "coordinate_matrix": coordinate_matrix,
                "was_sampled": was_sampled,
                "total_possible_points": total_possible_points,
                "displayed_points": displayed_points,
                "data_quality_issues": data_quality_issues,
                "metrics_coverage": metrics_coverage,
                "incomplete_metrics": incomplete_metrics,
                "data_quality_good": len(data_quality_issues) == 0,
            }

        except Exception as e:
            # Analysis failure is a programming error - don't hide it
            raise RuntimeError(
                f"Progression data analysis failed for context '{context}': {e}"
            ) from e

    def _build_progression_table_metadata(
        self, context: str, analysis: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Build transparent table title and caption for progression data."""
        context_name = context.replace("_", " ").title()

        # Build title with data display indicator
        if analysis["data_quality_good"]:
            title = f"{context_name} - Progression"
        else:
            title = f"{context_name} - Progression ℹ️"

        # Build detailed caption
        base_caption = "Metric progression across training checkpoints"

        if analysis["data_quality_issues"]:
            issues_text = "; ".join(analysis["data_quality_issues"])
            caption = f"{base_caption}.\nDisplay: {issues_text}"
        else:
            total_points = analysis["displayed_points"]
            caption = f"{base_caption}\n({total_points} checkpoints analyzed)"

        return title, caption

    def _add_progression_columns(self, table: Table, analysis: Dict[str, Any]) -> None:
        """Add progression columns with robust error handling."""
        # Add metric name column
        metric_config = get_column_config("metric")
        table.add_column(
            metric_config["header"],
            justify=metric_config["justify"],
            style=metric_config["style"],
            vertical=metric_config["vertical"],
        )

        # Add checkpoint columns
        column_headers = analysis["column_headers"]

        for header in column_headers:
            table.add_column(header, justify="right", style="white", vertical="middle")

    def _add_progression_rows(self, table: Table, analysis: Dict[str, Any]) -> None:
        """Add progression rows with robust metric handling."""
        metrics = analysis["metrics"]
        coordinate_groups = analysis["coordinate_groups"]
        coordinate_matrix = analysis["coordinate_matrix"]

        # Group metrics for display
        grouped_metrics = self._formatter.group_metric_names(metrics)
        group_names = list(grouped_metrics.keys())

        for i, group_name in enumerate(group_names):
            metric_list = grouped_metrics[group_name]

            # Add section line before each group (except the first)
            if i > 0:
                table.add_section()

            # Add group header if multiple groups
            if len(grouped_metrics) > 1:
                empty_cols = [""] * len(coordinate_matrix)
                metric_count = len(metric_list)
                group_header_text = format_group_header(group_name, metric_count)
                table.add_row(group_header_text, *empty_cols)

            # Add each metric row with robust error handling
            for metric in metric_list:
                try:
                    self._add_robust_progression_metric_row(
                        table, metric, coordinate_groups, coordinate_matrix, analysis
                    )
                except Exception as e:
                    # Metric processing failure - report clearly but continue with other metrics
                    print(f"⚠️  Unable to display metric '{metric}': {e}")
                    # Add informative row to show user what failed
                    error_cols = [f"Unable to display: {str(e)[:30]}..."] + ["-"] * (
                        len(coordinate_matrix) - 1
                    )
                    rule = self._formatter.find_rule(metric)
                    styled_metric = format_metric_name(metric, rule.emoji)
                    table.add_row(styled_metric, *error_cols)

    def _add_robust_progression_metric_row(
        self,
        table: Table,
        metric: str,
        coordinate_groups: Dict[Tuple[Optional[int], ...], List[Dict[str, Any]]],
        coordinate_matrix: List[Tuple[Optional[int], ...]],
        analysis: Dict[str, Any],
    ) -> None:
        """Add metric row to progression table with robust error handling and transparency."""
        # Get metric formatting info
        rule = self._formatter.find_rule(metric)
        styled_metric = format_metric_name(metric, rule.emoji)

        # Build metric values for each coordinate point
        row_data = [styled_metric]
        values = []  # Track for percentage change calculation

        for coord_tuple in coordinate_matrix:
            try:
                coord_data = coordinate_groups.get(coord_tuple, [])

                # Extract metric values for this coordinate with validation
                metric_values = []
                for row in coord_data:
                    if metric in row and row[metric] is not None:
                        try:
                            value = float(row[metric])
                            if not (np.isnan(value) or np.isinf(value)):
                                metric_values.append(value)
                        except (ValueError, TypeError):
                            continue  # Skip invalid values

                if metric_values:
                    # Use mean of values at this coordinate
                    avg_value = float(np.mean(metric_values))
                    values.append(avg_value)
                    formatted_value = self._formatter.format(metric, avg_value)
                    row_data.append(formatted_value)
                else:
                    values.append(None)
                    row_data.append("-")

            except Exception as e:
                # Individual coordinate processing failed - log but continue
                print(
                    f"⚠️  Coordinate processing failed for {metric} at {coord_tuple}: {e}"
                )
                values.append(None)
                row_data.append("-")

        # Add the main metric row
        table.add_row(*row_data)

        # Add percentage change row if we have progression data
        if len(coordinate_matrix) >= 2:
            self._add_robust_percentage_change_row(
                table, metric, values, coordinate_matrix
            )

    def _add_robust_percentage_change_row(
        self,
        table: Table,
        metric: str,
        values: List[Optional[float]],
        coordinate_matrix: List[Tuple[Optional[int], ...]],
    ) -> None:
        """Add percentage change row with robust error handling."""
        try:
            valid_values = [v for v in values if v is not None]
            if len(valid_values) < 2:
                return  # Need at least 2 values for comparison

            # Find first valid value as baseline
            first_valid_value = None
            for v in values:
                if v is not None:
                    first_valid_value = v
                    break

            if first_valid_value is None:
                return

            # Create percentage change row
            pct_row_data = [""]  # Empty first column for alignment

            for i, coord_tuple in enumerate(coordinate_matrix):
                try:
                    if i == 0:
                        pct_row_data.append("-")  # No comparison for first value
                        continue

                    current_value = values[i]
                    if current_value is None:
                        pct_row_data.append("-")
                        continue

                    # Calculate percentage change with division by zero protection
                    if abs(first_valid_value) > self.DIVISION_EPSILON:
                        pct_change = (
                            (current_value - first_valid_value) / abs(first_valid_value)
                        ) * 100

                        # Get color based on optimization goal
                        color = self._formatter.get_delta_color(
                            metric, current_value - first_valid_value
                        )

                        # Format percentage with significance indication
                        if abs(pct_change) < self.SMALL_CHANGE_THRESHOLD:
                            pct_str = f"[italic {color}]~0.0%[/italic {color}]"
                        else:
                            pct_str = (
                                f"[italic {color}]{pct_change:+5.1f}%[/italic {color}]"
                            )

                        pct_row_data.append(pct_str)
                    else:
                        # Handle division by zero with clear indicators
                        if current_value > first_valid_value:
                            pct_row_data.append(
                                f"[{get_colors().positive_change}]+∞%[/{get_colors().positive_change}]"
                            )
                        elif current_value < first_valid_value:
                            pct_row_data.append(
                                f"[{get_colors().negative_change}]-∞%[/{get_colors().negative_change}]"
                            )
                        else:
                            pct_row_data.append("-")

                except Exception as e:
                    # Individual percentage calculation failed - continue with other columns
                    print(
                        f"⚠️  Percentage calculation failed for {metric} at coordinate {i}: {e}"
                    )
                    pct_row_data.append("-")

            # Add percentage change row
            table.add_row(*pct_row_data)

        except Exception as e:
            # Entire percentage row construction failed - this is a programming error
            print(f"❌ Percentage change calculation failed for metric '{metric}': {e}")
            # Don't add confusing error row - let the metric row stand alone

    # Removed duplicate progression methods - using robust implementation above

    def _show_node_statistics(
        self, context: str, context_data: List[Dict[str, Any]], total_nodes: int
    ) -> None:
        """Show final-state statistics with transparent data quality indicators."""
        if not context_data:
            return

        # STEP 1: Get majority final step data with full transparency
        final_data, data_quality = self._get_majority_final_step_data(context_data)
        if not final_data:
            print(f"⚠️  No final step data available for {context}")
            return

        # STEP 2: Calculate statistics from clean final data
        metric_stats = self._calculate_final_state_stats(final_data, data_quality)
        if not metric_stats:
            return

        # STEP 3: Create table with transparent title showing data quality
        table_title = self._build_transparent_table_title(context, data_quality)
        table_caption = self._build_transparent_table_caption(data_quality)

        table = self._create_styled_table(
            title=table_title,
            caption=table_caption,
            title_emoji=get_emojis().node_statistics,
        )

        # STEP 4: Add columns (no coordinate columns - just final state statistics)
        self._add_statistics_columns(table)

        # STEP 5: Add metric rows grouped by category
        grouped_stats = self._group_metric_stats(metric_stats)
        self._add_metric_rows(table, grouped_stats)

        print(table)
        print()

    def _get_majority_final_step_data(
        self, context_data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Get final step data from majority of nodes with complete transparency.

        Returns clean final-state data and detailed quality metadata.
        No silent failures - all issues immediately visible in metadata.
        """
        if not context_data:
            return [], {
                "error": "No context data",
                "participating_nodes": 0,
                "total_nodes": 0,
            }

        # Find each node's final global step
        node_final_steps = {}
        for measurement in context_data:
            node_id = measurement.get("_node_id")
            global_step = measurement.get("global_step", 0)

            if node_id is not None:
                # Track highest step for each node
                if (
                    node_id not in node_final_steps
                    or global_step > node_final_steps[node_id]
                ):
                    node_final_steps[node_id] = global_step

        if not node_final_steps:
            return [], {
                "error": "No node IDs found",
                "participating_nodes": 0,
                "total_nodes": 0,
            }

        # Find majority final step (most common)
        from collections import Counter

        step_counts = Counter(node_final_steps.values())
        majority_step, majority_count = step_counts.most_common(1)[0]

        # Identify participating vs excluded nodes
        participating_nodes = {
            node_id
            for node_id, step in node_final_steps.items()
            if step == majority_step
        }
        excluded_nodes = {
            node_id
            for node_id, step in node_final_steps.items()
            if step != majority_step
        }

        # Extract final measurements from participating nodes only
        final_measurements = []
        for measurement in context_data:
            node_id = measurement.get("_node_id")
            global_step = measurement.get("global_step", 0)

            if node_id in participating_nodes and global_step == majority_step:
                final_measurements.append(measurement)

        # Build completely transparent quality metadata
        quality_metadata = {
            "majority_step": majority_step,
            "participating_nodes": len(participating_nodes),
            "excluded_nodes": len(excluded_nodes),
            "total_nodes": len(node_final_steps),
            "step_distribution": dict(step_counts),  # Show all step counts
            "excluded_node_steps": {
                node_id: node_final_steps[node_id] for node_id in excluded_nodes
            },
            "data_clean": len(excluded_nodes) == 0,  # Immediate boolean indicator
        }

        return final_measurements, quality_metadata

    def _build_transparent_table_title(
        self, context: str, data_quality: Dict[str, Any]
    ) -> str:
        """Build table title that immediately shows data quality status."""
        context_name = context.replace("_", " ").title()

        if "error" in data_quality:
            return f"❌ {context_name} - ERROR: {data_quality['error']}"

        step = data_quality["majority_step"]
        participating = data_quality["participating_nodes"]
        total = data_quality["total_nodes"]

        if data_quality["data_clean"]:
            # Perfect data - all nodes at same final checkpoint
            return f"📊 {context_name} - Final State (Checkpoint {step}) - All {total} nodes"
        else:
            # Imperfect data - show exactly what's excluded
            excluded = data_quality["excluded_nodes"]
            pct = (participating / total * 100) if total > 0 else 0
            return f"⚠️  {context_name} - Final State (Checkpoint {step}) - {participating}/{total} nodes ({pct:.0f}%)"

    def _build_transparent_table_caption(self, data_quality: Dict[str, Any]) -> str:
        """Build caption that shows exactly what data was included/excluded."""
        if "error" in data_quality:
            return f"Unable to process data: {data_quality['error']}"

        if data_quality["data_clean"]:
            return f"Statistics from final training state.\nAll nodes reached checkpoint {data_quality['majority_step']}."
        else:
            excluded = data_quality["excluded_nodes"]
            excluded_steps = data_quality["excluded_node_steps"]
            step_info = ", ".join(
                f"node {node_id} at checkpoint {step}"
                for node_id, step in excluded_steps.items()
            )
            return f"Statistics from most complete training state.\n{excluded} nodes at different checkpoints: {step_info}"

    def _add_statistics_columns(self, table: Table) -> None:
        """Add clean statistics columns without coordinate clutter."""
        # Metric name column
        metric_config = get_column_config("metric")
        table.add_column(
            metric_config["header"],
            justify=metric_config["justify"],
            style=metric_config["style"],
            vertical=metric_config["vertical"],
        )

        # Statistics columns in logical order
        for col_type in ["nodes", "sum", "mean", "std", "min", "max", "median", "cv"]:
            config = get_column_config(col_type)
            table.add_column(
                config["header"],
                justify=config["justify"],
                style=config["style"],
                vertical=config["vertical"],
            )

    def _add_metric_rows(
        self, table: Table, grouped_stats: Dict[str, List[MetricStats]]
    ) -> None:
        """Add metric rows grouped by category."""
        group_names = sorted(grouped_stats.keys())

        for i, group_name in enumerate(group_names):
            stats_list = grouped_stats[group_name]

            # Add section line before each group (except the first)
            if i > 0:
                table.add_section()

            # Add group header if multiple groups
            if len(grouped_stats) > 1:
                stats_count = len(stats_list)
                group_header_text = format_group_header(group_name, stats_count)
                empty_cols = [""] * 8  # 8 statistics columns
                table.add_row(group_header_text, *empty_cols)

            # Add each metric row
            for stats in stats_list:
                table.add_row(
                    format_metric_name(stats.name, stats.emoji),
                    stats.coverage_display,
                    stats.sum,
                    stats.mean,
                    stats.std,
                    stats.min,
                    stats.max,
                    stats.median,
                    stats.cv,
                )

    def _calculate_final_state_stats(
        self, final_data: List[Dict[str, Any]], data_quality: Dict[str, Any]
    ) -> List[MetricStats]:
        """Calculate statistics from clean final-state data."""
        if not final_data:
            return []

        # Get all metrics from final data
        all_metrics = set()
        for measurement in final_data:
            for key, value in measurement.items():
                if (
                    not key.startswith("_")
                    and key not in self._metadata_keys
                    and value is not None
                ):
                    all_metrics.add(key)

        metric_stats_list = []
        participating_nodes = data_quality["participating_nodes"]
        total_nodes = data_quality["total_nodes"]

        for metric in sorted(all_metrics):
            # Extract values for this metric
            values = []
            for measurement in final_data:
                if metric in measurement and isinstance(
                    measurement[metric], (int, float)
                ):
                    values.append(float(measurement[metric]))

            # Get formatting info
            rule = self._formatter.find_rule(metric)

            if values:
                # Calculate statistics using formatter rules
                stats = self._calculate_metric_statistics(final_data, metric, values)

                metric_stats = MetricStats(
                    name=metric,
                    emoji=rule.emoji,
                    group=rule.group,
                    node_count=participating_nodes,  # All participating nodes have this metric
                    total_nodes=total_nodes,
                    sum=stats["sum"],
                    mean=stats["mean"],
                    std=stats["std"],
                    min=stats["min"],
                    max=stats["max"],
                    median=stats["median"],
                    cv=stats["cv"],
                )
                metric_stats_list.append(metric_stats)

        return metric_stats_list

    # Removed old coordinate analysis - using final state approach instead

    def _calculate_experiment_stats(
        self,
        results: List[Dict[str, Any]],
        total_nodes: int,
        context_data: List[Dict[str, Any]] = None,
    ) -> List[MetricStats]:
        """Calculate statistical summary as structured MetricStats objects."""
        if not results:
            return []

        # Find all metrics present in any node (union approach)
        all_metrics = set()
        for node_results in results:
            all_metrics.update(node_results.keys())

        metric_stats_list = []

        for metric in sorted(all_metrics):
            # Extract values and metadata
            numeric_values = self._extract_numeric_values(results, metric)
            # Use context_data for node counting if available (preserves _node_id), otherwise fall back to results
            node_count = self._count_reporting_nodes(context_data or results, metric)

            # Get formatting metadata from formatter
            rule = self._formatter.find_rule(metric)
            emoji = rule.emoji
            group = rule.group

            if numeric_values:
                # Calculate statistics and format them
                stats = self._calculate_metric_statistics(
                    results, metric, numeric_values
                )
                metric_stats = MetricStats(
                    name=metric,
                    emoji=emoji,
                    group=group,
                    node_count=node_count,
                    total_nodes=total_nodes,
                    sum=stats["sum"],
                    mean=stats["mean"],
                    std=stats["std"],
                    min=stats["min"],
                    max=stats["max"],
                    median=stats["median"],
                    cv=stats["cv"],
                )
            else:
                # Handle non-numeric metrics
                all_values = [
                    str(result[metric]) for result in results if metric in result
                ]
                if all_values:
                    unique_values = list(set(all_values))
                    if len(unique_values) == 1:
                        mean_display = unique_values[0]
                    else:
                        mean_display = f"{len(unique_values)} unique values"
                else:
                    mean_display = "No data"

                metric_stats = MetricStats(
                    name=metric,
                    emoji=emoji,
                    group=group,
                    node_count=node_count,
                    total_nodes=total_nodes,
                    sum="-",
                    mean=mean_display,
                    std="-",
                    min="-",
                    max="-",
                    median="-",
                    cv="-",
                )

            metric_stats_list.append(metric_stats)

        return metric_stats_list

    def _extract_numeric_values(
        self, results: List[Dict[str, Any]], metric: str
    ) -> List[float]:
        """Extract numeric values for a metric across all results."""
        return [
            result[metric]
            for result in results
            if metric in result and isinstance(result[metric], (int, float))
        ]

    def _count_reporting_nodes(self, results: List[Dict[str, Any]], metric: str) -> int:
        """Count how many unique nodes reported this metric using preserved node identity.

        Now that we preserve node_id in extract_context_data(), we can directly count
        unique nodes instead of estimating from coordinate patterns.
        """
        if not results:
            return 0

        # Find measurements that contain this metric and extract unique node IDs
        unique_nodes = set()
        for measurement in results:
            if metric in measurement and "_node_id" in measurement:
                unique_nodes.add(measurement["_node_id"])

        return len(unique_nodes)

    def _calculate_metric_statistics(
        self, results: List[Dict[str, Any]], metric: str, values: List[float]
    ) -> Dict[str, str]:
        """Calculate formatted statistics for a list of numeric values using MetricFormatter rules."""
        # Get applicable statistics from MetricFormatter rules
        applicable_stats = self._formatter.get_applicable_stats(metric)

        if len(values) == 1:
            single_value = float(values[0])
            return {
                "sum": self._formatter.format(metric, single_value)
                if applicable_stats["sum"]
                else "-",
                "mean": self._formatter.format(metric, single_value),
                "std": "-",
                "min": "-",
                "max": "-",
                "median": "-",
                "cv": "-",
            }

        # Calculate all statistics
        sum_val = np.sum(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        median_val = np.median(values)

        # Calculate coefficient of variation (std/mean), handle division by zero
        if abs(mean_val) > 1e-10 and applicable_stats["cv"]:
            cv_val = (std_val / abs(mean_val)) * 100  # Convert to percentage
            cv_formatted = f"{cv_val:.1f}%"
        else:
            cv_formatted = "-"

        return {
            "sum": self._formatter.format(metric, float(sum_val))
            if applicable_stats["sum"]
            else "-",
            "mean": self._formatter.format(metric, float(mean_val))
            if applicable_stats["mean"]
            else "-",
            "std": self._formatter.format(metric, float(std_val))
            if applicable_stats["std"]
            else "-",
            "min": self._formatter.format(metric, float(min_val))
            if applicable_stats["min"]
            else "-",
            "max": self._formatter.format(metric, float(max_val))
            if applicable_stats["max"]
            else "-",
            "median": self._formatter.format(metric, float(median_val))
            if applicable_stats["median"]
            else "-",
            "cv": cv_formatted,
        }

    # Removed unused separator logic - cleaner display without separators

    def _group_metric_stats(
        self, metrics: List[MetricStats]
    ) -> Dict[str, List[MetricStats]]:
        """Group MetricStats objects by their group category with proper ordering."""
        groups = defaultdict(list)
        group_orders = {}  # Track the order for each group

        for metric_stats in metrics:
            # MetricGroup is defined as str, Enum so .value is always available
            group_name = metric_stats.group.value
            groups[group_name].append(metric_stats)

            # Store the order for this group (get it from the formatter rule)
            if group_name not in group_orders:
                rule = self._formatter.find_rule(metric_stats.name)
                group_orders[group_name] = rule.group_order

        # Sort metrics within each group by name
        for group in groups:
            groups[group].sort(key=lambda m: m.name)

        # Sort groups by their display order, then convert to regular dict
        sorted_groups = dict(sorted(groups.items(), key=lambda x: group_orders[x[0]]))
        return sorted_groups

    def _format_with_highlighting(
        self, metric: str, value: float, all_values: List[Optional[float]]
    ) -> str:
        """Format value with best/worst highlighting."""
        formatted = self._formatter.format(metric, value)

        valid_values = [v for v in all_values if v is not None]
        if len(valid_values) <= 1:
            return formatted

        goal = self._formatter.find_rule(metric).optimization_goal

        if goal == MetricDirection.MINIMIZE:
            best_value = min(valid_values)
            worst_value = max(valid_values)
        elif goal == MetricDirection.MAXIMIZE:
            best_value = max(valid_values)
            worst_value = min(valid_values)
        else:
            return formatted  # Neutral metrics don't get highlighting

        if value == best_value and best_value != worst_value:
            return f"[bold bright_green]{formatted}[/bold bright_green]"
        elif value == worst_value and best_value != worst_value:
            return f"[bold bright_red]{formatted}[/bold bright_red]"
        else:
            return formatted
