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
import numpy as np

from .matchers import contains, any_of


class AggregationStrategy(str, Enum):
    """Strategy for aggregating epoch-level metrics to round-level metrics."""

    SUM = "sum"
    MEAN = "mean"
    LAST = "last"
    MAX = "max"
    MIN = "min"
    FIRST = "first"


class OptimizationGoal(str, Enum):
    """How metric values should be interpreted for trend analysis."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    NEUTRAL = "neutral"


class TrendColor(str, Enum):
    """Colors for trend indicators based on performance impact."""

    GOOD = "bright_green"
    NEUTRAL = "white"
    BAD = "bright_red"


class MetricGroup(str, Enum):
    """Standard metric group categories for organizing display tables."""

    PERFORMANCE = "Performance"
    TRAINING = "Training"
    ROUND_TIMING = "Round Timing"
    EPOCH_TIMING = "Epoch Timing"
    BATCH_TIMING = "Batch Timing"
    SYNC_TIMING = "Sync Timing"
    DATASET = "Dataset"
    OTHER = "Other"


@dataclass
class MetricFormatRule:
    """Rule-based formatting system for federated learning metrics."""

    matcher: Callable[[str], bool]
    precision: int
    units: str = ""
    optimization_goal: OptimizationGoal = OptimizationGoal.MAXIMIZE
    format_as_integer: bool = False
    emoji: str = ":bar_chart:"
    group: str = MetricGroup.OTHER
    description: str = ""
    aggregation_strategy: AggregationStrategy = AggregationStrategy.LAST


@dataclass
class MetricStats:
    """Structured metric statistics with metadata for display and analysis."""

    name: str
    emoji: str
    group: str
    node_count: int
    total_nodes: int
    mean: str
    std: str
    min: str
    max: str

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


@dataclass
class DisplayConfiguration:
    """Configuration for display formatting and styling."""
    
    show_node_coverage: bool = True
    use_emoji_indicators: bool = True
    group_metrics: bool = True
    decimal_precision: int = 4
    
    @classmethod
    def create_default(cls) -> 'DisplayConfiguration':
        """Create default display configuration."""
        return cls()


class TrendThresholds:
    """Threshold values for trend analysis."""
    SMALL_CHANGE = 2.0
    LARGE_CHANGE = 10.0


DEFAULT_FORMAT_RULES = [
    # Round-level timing metrics (actual: time/round)
    MetricFormatRule(
        matcher=contains("time/round"),
        precision=4,
        units="s",
        optimization_goal=OptimizationGoal.MINIMIZE,
        emoji=":repeat:",
        group=MetricGroup.ROUND_TIMING,
        description="Federated round timing",
        aggregation_strategy=AggregationStrategy.SUM,
    ),
    # Epoch-level timing metrics (actual: time/epoch_train, time/epoch_eval)
    MetricFormatRule(
        matcher=contains("time/epoch"),
        precision=4,
        units="s",
        optimization_goal=OptimizationGoal.MINIMIZE,
        emoji=":hourglass:",
        group=MetricGroup.EPOCH_TIMING,
        description="Training epoch timing",
        aggregation_strategy=AggregationStrategy.SUM,
    ),
    # Batch-level timing metrics (actual: time/batch_train, time/batch_eval, time/batch_data_*, time/batch_compute_*)
    MetricFormatRule(
        matcher=contains("time/batch"),
        precision=4,
        units="s",
        optimization_goal=OptimizationGoal.MINIMIZE,
        emoji=":timer_clock:",
        group=MetricGroup.BATCH_TIMING,
        description="Batch processing timing",
        aggregation_strategy=AggregationStrategy.MEAN,
    ),
    # Synchronization timing metrics (time/sync_*)
    MetricFormatRule(
        matcher=contains("time/sync"),
        precision=4,
        units="s",
        optimization_goal=OptimizationGoal.MINIMIZE,
        emoji=":arrows_counterclockwise:",
        group=MetricGroup.SYNC_TIMING,
        description="Synchronization timing",
        aggregation_strategy=AggregationStrategy.SUM,
    ),
    # Generic timing fallback for any other time/ metrics
    MetricFormatRule(
        matcher=contains("time/"),
        precision=4,
        units="s",
        optimization_goal=OptimizationGoal.MINIMIZE,
        emoji=":stopwatch:",
        group=MetricGroup.OTHER,
        description="Generic timing",
        aggregation_strategy=AggregationStrategy.SUM,
    ),
    # Loss metrics (lower is better)
    MetricFormatRule(
        matcher=contains("loss"),
        precision=4,
        units="",
        optimization_goal=OptimizationGoal.MINIMIZE,
        emoji=":chart_decreasing:",
        group=MetricGroup.TRAINING,
        description="Loss metrics",
        aggregation_strategy=AggregationStrategy.LAST,
    ),
    # Error metrics (lower is better)
    MetricFormatRule(
        matcher=any_of("error", "mse", "mae", "rmse"),
        precision=4,
        units="",
        optimization_goal=OptimizationGoal.MINIMIZE,
        emoji=":x:",
        group=MetricGroup.TRAINING,
        description="Error metrics",
        aggregation_strategy=AggregationStrategy.LAST,
    ),
    # Accuracy and performance metrics (higher is better)
    MetricFormatRule(
        matcher=any_of("accuracy", "precision", "recall", "f1"),
        precision=5,
        units="",
        optimization_goal=OptimizationGoal.MAXIMIZE,
        emoji=":dart:",
        group=MetricGroup.PERFORMANCE,
        description="Performance metrics",
        aggregation_strategy=AggregationStrategy.LAST,
    ),
    # Count metrics (samples, batches, etc.) - neutral (higher neither good nor bad)
    MetricFormatRule(
        matcher=any_of("samples", "batches", "count", "num_"),
        precision=1,
        units="",
        optimization_goal=OptimizationGoal.NEUTRAL,
        format_as_integer=True,
        emoji=":package:",
        group=MetricGroup.DATASET,
        description="Count metrics",
        aggregation_strategy=AggregationStrategy.SUM,
    ),
    # Gradient-related metrics (context dependent, but generally lower is better)
    MetricFormatRule(
        matcher=contains("grad"),
        precision=6,
        units="",
        optimization_goal=OptimizationGoal.MINIMIZE,
        emoji=":chart_increasing:",
        group=MetricGroup.TRAINING,
        description="Gradient metrics",
        aggregation_strategy=AggregationStrategy.LAST,
    ),
]


class MetricFormatter:
    """
    Rule-based metric formatting, aggregation, and display utilities.

    Used by Engine for formatting final results without metric collection.
    """

    # Class-level aggregation functions to avoid recreating on every call
    _AGGREGATION_FUNCTIONS = {
        AggregationStrategy.SUM: sum,
        AggregationStrategy.MEAN: lambda values: sum(values) / len(values),
        AggregationStrategy.LAST: lambda values: values[-1],
        AggregationStrategy.FIRST: lambda values: values[0],
        AggregationStrategy.MAX: max,
        AggregationStrategy.MIN: min,
    }

    def __init__(self, rules: Optional[List[MetricFormatRule]] = None):
        self.rules = rules or DEFAULT_FORMAT_RULES.copy()
        self._rule_cache: Dict[str, MetricFormatRule] = {}

        # Default rule for unmatched metrics
        self._default_rule = MetricFormatRule(
            matcher=lambda name: True,
            precision=3,
            optimization_goal=OptimizationGoal.NEUTRAL,
            emoji=":question:",
            group=MetricGroup.OTHER,
            description="Default formatting",
        )

    def _find_rule(self, metric_name: str) -> MetricFormatRule:
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

    def get_aggregation_strategy(self, metric_name: str) -> AggregationStrategy:
        """Get aggregation strategy for converting epoch-level to round-level metrics."""
        return self._find_rule(metric_name).aggregation_strategy

    def optimization_goal(self, metric_name: str) -> OptimizationGoal:
        """Get optimization goal for metric (MAXIMIZE, MINIMIZE, or NEUTRAL)."""
        return self._find_rule(metric_name).optimization_goal

    def group_metrics(self, metric_names: List[str]) -> Dict[str, List[str]]:
        """Group metrics by their category based on formatting rules."""
        groups = {}
        for metric_name in metric_names:
            group = self.get_group(metric_name)
            if group not in groups:
                groups[group] = []
            groups[group].append(metric_name)

        # Sort metrics within each group
        for group in groups:
            groups[group].sort()

        return groups

    def _calculate_metric_statistics(self, values: List[float], rule: MetricFormatRule) -> Dict[str, str]:
        """Calculate formatted statistics for a list of numeric values."""
        if len(values) == 1:
            return {
                "mean": f"{values[0]:.{rule.precision}f}{rule.units}",
                "std": "-",
                "min": "-", 
                "max": "-"
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

    def _extract_numeric_values(self, results: List[Dict[str, Any]], metric: str) -> List[float]:
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


    def format_stats_structured(self, results: List[Dict[str, Any]]) -> List[MetricStats]:
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
            emoji = rule.emoji
            group = rule.group

            if numeric_values:
                # Calculate statistics using extracted method
                stats = self._calculate_metric_statistics(numeric_values, rule)
                metric_stats = MetricStats(
                    name=metric,
                    emoji=emoji,
                    group=group,
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
                    str(result[metric]) 
                    for result in results 
                    if metric in result
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
                    mean=mean_display,
                    std="-",
                    min="-",
                    max="-",
                )

            metric_stats_list.append(metric_stats)

        return metric_stats_list

    def group_structured_metrics(self, metrics: List[MetricStats]) -> Dict[str, List[MetricStats]]:
        """Group MetricStats objects by their group category."""
        groups = {}
        for metric_stats in metrics:
            group = metric_stats.group
            if group not in groups:
                groups[group] = []
            groups[group].append(metric_stats)

        # Sort metrics within each group by name
        for group in groups:
            groups[group].sort(key=lambda m: m.name)

        return groups

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

        # Determine magnitude and select symbol
        if pct_change < TrendThresholds.SMALL_CHANGE:
            symbol = "→"  # Small change
        elif pct_change < TrendThresholds.LARGE_CHANGE:
            symbol = "↑" if new_value > old_value else "↓"  # Medium change
        else:
            symbol = "⇈" if new_value > old_value else "⇊"  # Large change

        # Determine color based on performance impact
        goal = self.optimization_goal(metric_name)

        # Neutral metrics - no performance judgment
        if goal == OptimizationGoal.NEUTRAL or pct_change < TrendThresholds.SMALL_CHANGE:
            color = TrendColor.NEUTRAL
        else:
            # Good change: increase for MAXIMIZE metrics, decrease for MINIMIZE metrics
            is_good_change = (new_value > old_value) == (
                goal == OptimizationGoal.MAXIMIZE
            )
            color = TrendColor.GOOD if is_good_change else TrendColor.BAD

        return f"[bold {color}]{symbol}[/bold {color}]"
