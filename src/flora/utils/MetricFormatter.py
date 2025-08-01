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

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import numpy as np
import warnings
import re
from typeguard import typechecked


class MetricDirection(str, Enum):
    """Optimization direction for metrics (higher/lower is better)."""

    MAXIMIZE = "maximize"  # Higher is better (accuracy, F1 score)
    MINIMIZE = "minimize"  # Lower is better (loss, error, time)
    NEUTRAL = "neutral"  # Neither higher nor lower is better (counts)


class MetricGroup(str, Enum):
    """Metric categories for table organization."""

    LOSS_METRICS = "Loss & Error"  # Loss, error metrics (minimize)
    PERFORMANCE_METRICS = "Performance"  # Accuracy, precision, recall, F1 (maximize)
    GRADIENT_METRICS = "Gradients"  # Gradient norms, magnitudes
    TIMING_METRICS = "Timing"  # All timing metrics
    DATASET_METRICS = "Dataset"  # Sample counts, data statistics
    REPORTING_METADATA = "Reporting Metadata"  # Training coordinates and indices
    OTHER = "Other"  # Unclassified metrics


@dataclass
class MetricFormatRule:
    """Formatting rules for FL metrics.

    Rules are matched by regex pattern in order.
    """

    regex: str  # Regex pattern to match metric names
    precision: int = 3  # Decimal places for display
    units: str = ""  # Unit suffix (s, %, etc.)
    optimization_goal: MetricDirection = (
        MetricDirection.NEUTRAL
    )  # Higher/lower is better
    format_as_integer: bool = False  # Show as integer instead of float
    emoji: str = ""  # Display icon (empty for unknown metrics)
    group: str = MetricGroup.OTHER  # Category for grouping
    group_order: int = 50  # Display order (lower numbers first, higher numbers last)
    description: str = "Default formatting"  # Human-readable description

    # Which statistics to show for this metric type
    show_sum: bool = False  # Total across nodes
    show_mean: bool = True  # Average
    show_std: bool = True  # Standard deviation
    show_min: bool = True  # Minimum
    show_max: bool = True  # Maximum
    show_median: bool = True  # Median
    show_cv: bool = True  # Coefficient of variation


DEFAULT_FORMAT_RULES = [
    # All timing metrics - matches any metric containing 'time'
    MetricFormatRule(
        regex=r"time",
        precision=3,
        units="s",
        optimization_goal=MetricDirection.MINIMIZE,
        emoji=":stopwatch:",
        group=MetricGroup.TIMING_METRICS,
        group_order=40,
        description="Timing metrics",
        show_sum=False,
    ),
    # Loss and error metrics (lower is better)
    MetricFormatRule(
        regex=r"(loss|error|mse|mae|rmse)",
        precision=4,
        units="",
        optimization_goal=MetricDirection.MINIMIZE,
        emoji=":chart_decreasing:",
        group=MetricGroup.LOSS_METRICS,
        group_order=10,
        description="Loss and error metrics",
        show_sum=False,
    ),
    # Accuracy and performance metrics (higher is better)
    MetricFormatRule(
        regex=r"(accuracy|precision|recall|f1)",
        precision=4,
        units="",
        optimization_goal=MetricDirection.MAXIMIZE,
        emoji=":dart:",
        group=MetricGroup.PERFORMANCE_METRICS,
        group_order=20,
        description="Performance metrics",
        show_sum=False,
    ),
    # Count metrics
    MetricFormatRule(
        regex=r"(count|num_)",
        precision=0,
        units="",
        optimization_goal=MetricDirection.NEUTRAL,
        format_as_integer=True,
        emoji=":package:",
        group=MetricGroup.DATASET_METRICS,
        group_order=5,
        description="Count metrics",
        show_sum=True,  # Sum makes sense for counts
    ),
    # Gradient metrics (gradient norms, magnitudes, etc.)
    MetricFormatRule(
        regex=r"grad",
        precision=4,
        units="",
        optimization_goal=MetricDirection.NEUTRAL,
        emoji=":bar_chart:",
        group=MetricGroup.GRADIENT_METRICS,
        group_order=30,
        description="Gradient metrics",
        show_sum=False,
    ),
    # Training coordinate indices (batch_idx, epoch_idx, global_step, round_idx)
    MetricFormatRule(
        regex=r"(batch_idx|epoch_idx|global_step|round_idx)",
        precision=0,
        units="",
        optimization_goal=MetricDirection.NEUTRAL,
        format_as_integer=True,
        emoji=":spiral_calendar:",
        group=MetricGroup.REPORTING_METADATA,
        group_order=90,  # Show last
        description="Training progression coordinates",
        show_sum=False,  # Averaging makes more sense than summing
        show_mean=True,
        show_std=True,
        show_min=True,
        show_max=True,
        show_median=True,
        show_cv=True,
    ),
]


class MetricFormatter:
    """FL metrics formatter with validation and caching."""

    def __init__(self, rules: List[MetricFormatRule] = None):
        """Initialize formatter.

        Args:
            rules: Custom formatting rules. Uses defaults if None.
        """
        # Set and validate rules
        if rules is None:
            rules = DEFAULT_FORMAT_RULES

        if not isinstance(rules, list):
            raise ValueError(f"Rules must be a list, got {type(rules)}")

        if not rules:
            raise ValueError("Rules list cannot be empty")

        # Check each rule is valid
        validated_rules = []
        for i, rule in enumerate(rules):
            if not isinstance(rule, MetricFormatRule):
                raise ValueError(f"Rule {i} is not a MetricFormatRule: {type(rule)}")
            if not isinstance(rule.regex, str) or not rule.regex.strip():
                raise ValueError(f"Rule {i} has invalid regex: {rule.regex}")
            validated_rules.append(rule)

        self.rules = validated_rules
        self._rule_cache = {}  # Cache for metric name -> rule lookups
        self._validation_stats = {"cache_hits": 0, "cache_misses": 0, "fallbacks": 0}

    @typechecked
    def find_rule(self, metric_name: str) -> MetricFormatRule:
        """Find matching formatting rule for metric."""
        # Handle empty metric names
        if not metric_name.strip():
            metric_name = "empty_metric"
            self._validation_stats["fallbacks"] += 1

        # Check cache first
        if metric_name in self._rule_cache:
            self._validation_stats["cache_hits"] += 1
            return self._rule_cache[metric_name]

        self._validation_stats["cache_misses"] += 1

        # Find first matching rule
        try:
            for rule in self.rules:
                try:
                    if re.search(rule.regex, metric_name, re.IGNORECASE):
                        self._rule_cache[metric_name] = rule
                        return rule
                except re.error as e:
                    warnings.warn(f"Invalid regex in rule for {rule.regex}: {e}")
                    continue
        except Exception as e:
            warnings.warn(f"Error searching rules for metric '{metric_name}': {e}")

        # Default rule for unmatched metrics
        default_rule = MetricFormatRule(
            regex=r".*",
            group_order=80,
            description=f"Auto-generated default for '{metric_name}'",
        )
        self._rule_cache[metric_name] = default_rule
        self._validation_stats["fallbacks"] += 1
        return default_rule

    @typechecked
    def format(self, metric_name: str, value: float) -> str:
        """Format metric value according to rules."""
        # Handle special numeric values
        if np.isnan(value):
            return "NaN"
        if np.isinf(value):
            return "+∞" if value > 0 else "-∞"

        try:
            rule = self.find_rule(metric_name)

            # Apply formatting rule
            if rule.format_as_integer:
                try:
                    formatted_value = f"{int(round(value)):,}{rule.units}"
                except (ValueError, OverflowError):
                    formatted_value = f"{value:.0f}{rule.units}"
            else:
                precision = max(0, min(10, rule.precision))
                formatted_value = f"{value:.{precision}f}{rule.units}"

            return formatted_value

        except Exception as e:
            raise ValueError(
                f"Failed to format metric '{metric_name}' with value {value}: {e}"
            ) from e

    @typechecked
    def group_metric_names(self, metric_names: List[str]) -> Dict[str, List[str]]:
        """Group metrics by category."""
        # Handle empty input
        if not metric_names:
            return {}

        # Clean up metric names
        validated_names = []
        for i, name in enumerate(metric_names):
            if not name.strip():
                name = f"empty_metric_{i}"
                self._validation_stats["fallbacks"] += 1
            validated_names.append(name.strip())

        try:
            groups = defaultdict(list)
            group_orders = {}

            for metric_name in validated_names:
                rule = self.find_rule(metric_name)
                # Use enum string value
                group = rule.group.value
                groups[group].append(metric_name)
                # Track group display order
                if group not in group_orders:
                    group_orders[group] = rule.group_order

            # Sort metrics within each group
            for group in groups:
                groups[group].sort()

            # Sort by display order
            sorted_groups = dict(
                sorted(groups.items(), key=lambda x: group_orders[x[0]])
            )
            return sorted_groups

        except Exception as e:
            raise RuntimeError(f"Failed to group metrics: {e}") from e

    @typechecked
    def get_applicable_stats(self, metric_name: str) -> Dict[str, bool]:
        """Get which statistics to show for a metric.

        Args:
            metric_name: Metric name

        Returns:
            Dict of stat name -> show flag
        """
        # Handle empty metric name
        if not metric_name.strip():
            metric_name = "empty_metric"
            self._validation_stats["fallbacks"] += 1

        try:
            rule = self.find_rule(metric_name)
            return {
                "sum": rule.show_sum,
                "mean": rule.show_mean,
                "std": rule.show_std,
                "min": rule.show_min,
                "max": rule.show_max,
                "median": rule.show_median,
                "cv": rule.show_cv,
            }
        except Exception as e:
            raise RuntimeError(
                f"Failed to get applicable stats for metric '{metric_name}': {e}"
            ) from e

    @typechecked
    def get_delta_color(
        self, metric_name: str, delta: float, threshold: float = 1e-4
    ) -> str:
        """Get color for metric change.

        Args:
            metric_name: Metric name
            delta: Change value (new - old)
            threshold: Minimum significant change

        Returns:
            Rich color string
        """
        # Handle invalid values
        if np.isnan(delta) or np.isinf(delta):
            return "dim white"

        if threshold <= 0:
            threshold = 1e-4

        try:
            # No significant change
            if abs(delta) < threshold:
                return "dim white"

            # Color based on whether change is good/bad
            rule = self.find_rule(metric_name)
            goal = rule.optimization_goal

            if goal == MetricDirection.NEUTRAL:
                return "dim white"

            is_good_change = (delta < 0 and goal == MetricDirection.MINIMIZE) or (
                delta > 0 and goal == MetricDirection.MAXIMIZE
            )
            return "bright_green" if is_good_change else "bright_red"

        except Exception as e:
            raise RuntimeError(
                f"Failed to get delta color for metric '{metric_name}': {e}"
            ) from e
