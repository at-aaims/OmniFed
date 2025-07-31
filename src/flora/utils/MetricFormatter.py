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

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import numpy as np

from .matchers import contains, any_of


class EpochToRoundAggregation(str, Enum):
    """
    How to combine epoch-level metrics into round-level summaries.

    Used when FL algorithms train for multiple epochs per round
    and need to report a single value per metric per round.
    """

    SUM = "sum"  # Add all epoch values (total time, total samples)
    MEAN = "mean"  # Average across epochs (typical for rates)
    LAST = "last"  # Use final epoch value (final accuracy, loss)
    MAX = "max"  # Peak value across epochs (max memory usage)
    MIN = "min"  # Minimum value across epochs (best loss)
    FIRST = "first"  # Initial epoch value (baseline metrics)


class MetricDirection(str, Enum):
    """
    Whether higher or lower metric values indicate better performance.

    Used for trend analysis and color-coding metric changes.
    """

    MAXIMIZE = "maximize"  # Higher is better (accuracy, F1 score)
    MINIMIZE = "minimize"  # Lower is better (loss, error, time)
    NEUTRAL = "neutral"  # Neither higher nor lower is better (counts)


class TrendColor(str, Enum):
    """Rich console colors for trend arrows showing performance changes."""

    GOOD = "bright_green"  # Performance improved
    NEUTRAL = "white"  # No significant change or neutral metric
    BAD = "bright_red"  # Performance degraded


class MetricGroup(str, Enum):
    """Categories for grouping related metrics in display tables."""

    PERFORMANCE = "Performance"  # Accuracy, precision, F1, etc.
    TRAINING = "Training"  # Loss, gradients, learning rates
    ROUND_TIMING = "Round Timing"  # Complete FL round timing
    EPOCH_TIMING = "Epoch Timing"  # Local epoch timing
    BATCH_TIMING = "Batch Timing"  # Mini-batch processing timing
    SYNC_TIMING = "Sync Timing"  # Model synchronization timing
    DATASET = "Dataset"  # Sample counts, data statistics
    OTHER = "Other"  # Unclassified metrics


@dataclass
class MetricRule:
    """
    Formatting rule for specific metric types in FL experiments.

    Defines how to display, group, and aggregate metrics based on their names.
    Rules are matched in order, with the first matching rule being used.
    """

    matcher: Callable[[str], bool]  # Function to test metric names
    precision: int  # Decimal places for display
    units: str = ""  # Unit suffix (s, %, etc.)
    optimization_goal: MetricDirection = (
        MetricDirection.MAXIMIZE
    )  # Higher/lower is better
    format_as_integer: bool = False  # Show as integer instead of float
    emoji: str = ":bar_chart:"  # Display icon
    group: str = MetricGroup.OTHER  # Category for grouping
    description: str = ""  # Human-readable description
    epoch_agg: EpochToRoundAggregation = (
        EpochToRoundAggregation.LAST
    )  # Epoch→round aggregation


@dataclass
class MetricStats:
    """
    Statistical summary of a metric across FL nodes.

    Contains both the computed statistics (mean, std, min, max) and
    metadata for display formatting (emoji, group, coverage).
    """

    name: str  # Metric name (e.g., "train/accuracy")
    emoji: str  # Display icon for this metric type
    group: str  # Category for table grouping
    node_count: int  # How many nodes reported this metric
    total_nodes: int  # Total nodes in experiment
    mean: str  # Formatted mean value
    std: str  # Formatted standard deviation
    min: str  # Formatted minimum value
    max: str  # Formatted maximum value

    @property
    def coverage_display(self) -> str:
        """Display string showing node coverage (e.g., '(3/5)')."""
        return f"({self.node_count}/{self.total_nodes})"

    @property
    def display_name(self) -> str:
        """Full display name with emoji and coverage."""
        return f"{self.emoji} {self.name} {self.coverage_display}"

    @property
    def is_complete_coverage(self) -> bool:
        """True if all nodes reported this metric."""
        return self.node_count == self.total_nodes


class TrendThresholds:
    """
    Percentage change thresholds for categorizing metric trends.

    Used to determine which arrow symbol and color to show
    when comparing metric values across rounds.
    """

    SMALL_CHANGE = 2.0  # Below this: neutral arrow (→)
    LARGE_CHANGE = 10.0  # Above this: double arrow (⇈/⇊)


DEFAULT_FORMAT_RULES = [
    # Round-level timing metrics
    MetricRule(
        matcher=contains("time/round"),
        precision=4,
        units="s",
        optimization_goal=MetricDirection.MINIMIZE,
        emoji=":repeat:",
        group=MetricGroup.ROUND_TIMING,
        description="Federated round timing",
        epoch_agg=EpochToRoundAggregation.SUM,
    ),
    # Epoch-level timing metrics
    MetricRule(
        matcher=contains("time/epoch"),
        precision=4,
        units="s",
        optimization_goal=MetricDirection.MINIMIZE,
        emoji=":hourglass:",
        group=MetricGroup.EPOCH_TIMING,
        description="Training epoch timing",
        epoch_agg=EpochToRoundAggregation.SUM,
    ),
    # Batch-level timing metrics
    MetricRule(
        matcher=contains("time/batch"),
        precision=4,
        units="s",
        optimization_goal=MetricDirection.MINIMIZE,
        emoji=":timer_clock:",
        group=MetricGroup.BATCH_TIMING,
        description="Batch processing timing",
        epoch_agg=EpochToRoundAggregation.MEAN,
    ),
    # Synchronization timing metrics (time/sync_*)
    MetricRule(
        matcher=contains("time/sync"),
        precision=4,
        units="s",
        optimization_goal=MetricDirection.MINIMIZE,
        emoji=":arrows_counterclockwise:",
        group=MetricGroup.SYNC_TIMING,
        description="Synchronization timing",
        epoch_agg=EpochToRoundAggregation.SUM,
    ),
    # Generic timing fallback for any other time/ metrics
    MetricRule(
        matcher=contains("time/"),
        precision=4,
        units="s",
        optimization_goal=MetricDirection.MINIMIZE,
        emoji=":stopwatch:",
        group=MetricGroup.OTHER,
        description="Generic timing",
        epoch_agg=EpochToRoundAggregation.SUM,
    ),
    # Loss metrics (lower is better)
    MetricRule(
        matcher=contains("loss"),
        precision=4,
        units="",
        optimization_goal=MetricDirection.MINIMIZE,
        emoji=":chart_decreasing:",
        group=MetricGroup.TRAINING,
        description="Loss metrics",
        epoch_agg=EpochToRoundAggregation.LAST,
    ),
    # Error metrics (lower is better)
    MetricRule(
        matcher=any_of("error", "mse", "mae", "rmse"),
        precision=4,
        units="",
        optimization_goal=MetricDirection.MINIMIZE,
        emoji=":x:",
        group=MetricGroup.TRAINING,
        description="Error metrics",
        epoch_agg=EpochToRoundAggregation.LAST,
    ),
    # Accuracy and performance metrics (higher is better)
    MetricRule(
        matcher=any_of("accuracy", "precision", "recall", "f1"),
        precision=5,
        units="",
        optimization_goal=MetricDirection.MAXIMIZE,
        emoji=":dart:",
        group=MetricGroup.PERFORMANCE,
        description="Performance metrics",
        epoch_agg=EpochToRoundAggregation.LAST,
    ),
    # Count metrics (samples, batches, etc.) - neutral (higher neither good nor bad)
    MetricRule(
        matcher=any_of("samples", "batches", "count", "num_"),
        precision=1,
        units="",
        optimization_goal=MetricDirection.NEUTRAL,
        format_as_integer=True,
        emoji=":package:",
        group=MetricGroup.DATASET,
        description="Count metrics",
        epoch_agg=EpochToRoundAggregation.SUM,
    ),
    # Gradient-related metrics (context dependent, but generally lower is better)
    MetricRule(
        matcher=contains("grad"),
        precision=6,
        units="",
        optimization_goal=MetricDirection.MINIMIZE,
        emoji=":chart_increasing:",
        group=MetricGroup.TRAINING,
        description="Gradient metrics",
        epoch_agg=EpochToRoundAggregation.LAST,
    ),
]


class MetricFormatter:
    """
    Formats and aggregates FL metrics for display.

    Handles metric classification, epoch-to-round aggregation, and statistical
    summaries across nodes.
    Provides consistent formatting rules and trend indicators.
    """

    _AGGREGATION_FUNCTIONS = {
        EpochToRoundAggregation.SUM: sum,
        EpochToRoundAggregation.MEAN: lambda values: sum(values) / len(values),
        EpochToRoundAggregation.LAST: lambda values: values[-1],
        EpochToRoundAggregation.FIRST: lambda values: values[0],
        EpochToRoundAggregation.MAX: max,
        EpochToRoundAggregation.MIN: min,
    }

    def __init__(self, rules: Optional[List[MetricRule]] = None):
        """
        Args:
            rules: Custom formatting rules
        """
        self.rules = rules or DEFAULT_FORMAT_RULES.copy()
        self._rule_cache: Dict[str, MetricRule] = {}

        # Default rule for unmatched metrics
        self._default_rule = MetricRule(
            matcher=lambda name: True,
            precision=3,
            optimization_goal=MetricDirection.NEUTRAL,
            emoji=":question:",
            group=MetricGroup.OTHER,
            description="Default formatting",
        )

    def _find_rule(self, metric_name: str) -> MetricRule:
        """Find the formatting rule that matches the given metric name with caching."""
        if metric_name in self._rule_cache:
            return self._rule_cache[metric_name]

        for rule in self.rules:
            if rule.matcher(metric_name):
                self._rule_cache[metric_name] = rule
                return rule

        self._rule_cache[metric_name] = self._default_rule
        return self._default_rule

    def format(self, metric_name: str, value: float) -> str:
        """Format a single metric value using the appropriate rule."""
        rule = self._find_rule(metric_name)

        if rule.format_as_integer:
            return f"{int(value):,}{rule.units}"
        else:
            return f"{value:.{rule.precision}f}{rule.units}"

    def get_emoji(self, metric_name: str) -> str:
        """Get emoji icon for the given metric based on formatting rules."""
        return self._find_rule(metric_name).emoji

    def get_group(self, metric_name: str) -> str:
        """Get category group for the given metric based on formatting rules."""
        return self._find_rule(metric_name).group

    def get_aggregation_strategy(self, metric_name: str) -> EpochToRoundAggregation:
        """Get aggregation strategy for converting epoch-level to round-level metrics."""
        return self._find_rule(metric_name).epoch_agg

    def optimization_goal(self, metric_name: str) -> MetricDirection:
        """Get optimization goal for metric (MAXIMIZE, MINIMIZE, or NEUTRAL)."""
        return self._find_rule(metric_name).optimization_goal

    def group_metric_names(self, metric_names: List[str]) -> Dict[str, List[str]]:
        """Group metrics by their category based on formatting rules."""
        groups = defaultdict(list)
        for metric_name in metric_names:
            group = self.get_group(metric_name)
            groups[group].append(metric_name)

        # Sort metrics within each group and convert to regular dict
        for group in groups:
            groups[group].sort()

        return dict(groups)

    def _calculate_metric_statistics(
        self, values: List[float], rule: MetricRule
    ) -> Dict[str, str]:
        """Calculate formatted statistics for a list of numeric values."""
        if len(values) == 1:
            return {
                "mean": f"{values[0]:.{rule.precision}f}{rule.units}",
                "std": "-",
                "min": "-",
                "max": "-",
            }

        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        return {
            "mean": f"{mean_val:.{rule.precision}f}{rule.units}",
            "std": f"{std_val:.{rule.precision}f}{rule.units}",
            "min": f"{min_val:.{rule.precision}f}{rule.units}",
            "max": f"{max_val:.{rule.precision}f}{rule.units}",
        }

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
        """Count how many nodes reported this metric."""
        return sum(1 for result in results if metric in result)

    def aggregate_epochs_to_round(
        self, epoch_metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate epoch-level metrics to round-level metrics using defined strategies."""
        if not epoch_metrics_list:
            return {}

        # Find all metrics across all epochs
        all_metrics = set()
        for epoch_metrics in epoch_metrics_list:
            all_metrics.update(epoch_metrics.keys())

        round_metrics = {}
        for metric_name in all_metrics:
            # Extract non-None values for this metric across all epochs
            values = [
                epoch_metrics[metric_name]
                for epoch_metrics in epoch_metrics_list
                if metric_name in epoch_metrics
                and epoch_metrics[metric_name] is not None
            ]

            if not values:
                continue

            # Get aggregation strategy and apply function
            strategy = self.get_aggregation_strategy(metric_name)
            aggregation_func = self._AGGREGATION_FUNCTIONS.get(
                strategy, lambda values: values[-1]
            )
            round_metrics[metric_name] = aggregation_func(values)

        return round_metrics

    def format_stats(self, results: List[Dict[str, Any]]) -> List[MetricStats]:
        """Format statistical summary as structured MetricStats objects."""
        if not results:
            return []

        # Find all metrics present in any node (union approach)
        all_metrics = set()
        for node_results in results:
            all_metrics.update(node_results.keys())

        metric_stats_list = []
        total_nodes = len(results)

        for metric in sorted(all_metrics):
            # Extract values and metadata using helper methods
            numeric_values = self._extract_numeric_values(results, metric)
            node_count = self._count_reporting_nodes(results, metric)

            # Get formatting metadata
            rule = self._find_rule(metric)

            if numeric_values:
                # Calculate statistics using extracted method
                stats = self._calculate_metric_statistics(numeric_values, rule)
                metric_stats = MetricStats(
                    name=metric,
                    emoji=rule.emoji,
                    group=rule.group,
                    node_count=node_count,
                    total_nodes=total_nodes,
                    mean=stats["mean"],
                    std=stats["std"],
                    min=stats["min"],
                    max=stats["max"],
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
                    emoji=rule.emoji,
                    group=rule.group,
                    node_count=node_count,
                    total_nodes=total_nodes,
                    mean=mean_display,
                    std="-",
                    min="-",
                    max="-",
                )

            metric_stats_list.append(metric_stats)

        return metric_stats_list

    def group_metric_stats(
        self, metrics: List[MetricStats]
    ) -> Dict[str, List[MetricStats]]:
        """Group MetricStats objects by their group category."""
        groups = defaultdict(list)
        for metric_stats in metrics:
            groups[metric_stats.group].append(metric_stats)

        # Sort metrics within each group by name and convert to regular dict
        for group in groups:
            groups[group].sort(key=lambda m: m.name)

        return dict(groups)

    def get_trend_symbol(
        self, metric_name: str, new_value: float, old_value: float
    ) -> str:
        """Get colored trend symbol based on metric change magnitude and direction."""
        if new_value == old_value:
            return f"[bold {TrendColor.NEUTRAL}]→[/bold {TrendColor.NEUTRAL}]"

        # Calculate percentage change magnitude
        pct_change = (
            abs((new_value - old_value) / old_value * 100)
            if old_value != 0
            else abs(new_value - old_value) * 100
        )

        # Select symbol based on change magnitude
        symbol = self._get_trend_symbol_for_magnitude(pct_change, new_value > old_value)

        # Select color based on performance impact
        color = self._get_trend_color(metric_name, pct_change, new_value > old_value)

        return f"[bold {color}]{symbol}[/bold {color}]"

    def _get_trend_symbol_for_magnitude(
        self, pct_change: float, is_increase: bool
    ) -> str:
        """Get trend symbol based on percentage change magnitude."""
        if pct_change < TrendThresholds.SMALL_CHANGE:
            return "→"  # Small change
        elif pct_change < TrendThresholds.LARGE_CHANGE:
            return "↑" if is_increase else "↓"  # Medium change
        else:
            return "⇈" if is_increase else "⇊"  # Large change

    def _get_trend_color(
        self, metric_name: str, pct_change: float, is_increase: bool
    ) -> str:
        """Get trend color based on optimization goal and change direction."""
        goal = self.optimization_goal(metric_name)

        # Neutral metrics or small changes - no performance judgment
        if goal == MetricDirection.NEUTRAL or pct_change < TrendThresholds.SMALL_CHANGE:
            return TrendColor.NEUTRAL

        # Good change: increase for MAXIMIZE metrics, decrease for MINIMIZE metrics
        is_good_change = is_increase == (goal == MetricDirection.MAXIMIZE)
        return TrendColor.GOOD if is_good_change else TrendColor.BAD
