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

import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, DefaultDict, Tuple

import numpy as np
from typeguard import typechecked

from .table_style import DISPLAY_PLACEHOLDER, EMOJIS

OUTLIER_DEVIATION_THRESHOLD = 3.0  # Standard deviation threshold for outliers
EXTREME_OUTLIER_THRESHOLD = 4  # Order of magnitude for extreme outliers


def group_by_metric_rules(
    items: List[Any], rule_finder: Callable[[str], Any]
) -> Dict[str, List[Any]]:
    """Shared utility for grouping items by metric rules."""
    groups: DefaultDict[str, List[Any]] = defaultdict(list)
    group_orders: Dict[str, int] = {}

    for item in items:
        # Handle both string names and objects with name attribute
        try:
            name = item.name  # type: ignore[attr-defined]
        except AttributeError:
            name = str(item)

        rule = rule_finder(name)
        group = rule.group.value
        groups[group].append(item)
        if group not in group_orders:
            group_orders[group] = getattr(rule, "group_order", 100)

    # Sort items within each group
    for group, group_items in groups.items():
        try:
            group_items.sort(key=lambda x: x.name)  # type: ignore[attr-defined]
        except Exception:
            try:
                group_items.sort()
            except Exception:
                pass

    # Sort groups by display order
    return dict(sorted(groups.items(), key=lambda x: group_orders.get(x[0], 100)))


class MetricDirection(str, Enum):
    """Optimization direction for metrics (higher/lower is better)."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    NEUTRAL = "neutral"


class MetricGroup(str, Enum):
    """Metric categories for table organization."""

    LOSS_METRICS = "Loss & Error"
    PERFORMANCE_METRICS = "Performance"
    GRADIENT_METRICS = "Gradients"
    TIMING_METRICS = "Timing"
    DATASET_METRICS = "Dataset"
    REPORTING_METADATA = "Reporting Metadata"
    OTHER = "Other"


class ValidationErrorType(str, Enum):
    """Metric validation error types."""

    NAN = "NaN"  # Not a number values
    INFINITE = "infinite"  # Infinite values
    NEGATIVE_TIME = "negative-time"  # Time cannot be negative
    NEGATIVE_VALUE = "negative-value"  # Value cannot be negative
    IMPOSSIBLE_ORDER = "impossible-order"  # min > max contradiction


class ValidationRule(str, Enum):
    """Validation rule identifiers."""

    NO_NEGATIVE = "no_negative"  # Reject negative values
    NO_INFINITE = "no_infinite"  # Reject infinite values


@dataclass
class MetricFormatRule:
    """Complete metric rules for FL metrics: formatting, validation, statistics.

    Rules are matched by regex pattern in order.
    Each rule defines ALL aspects of how a metric type should be handled.
    """

    regex: str
    precision: int = 3
    units: str = ""
    optimization_goal: MetricDirection = MetricDirection.NEUTRAL
    format_as_integer: bool = False
    emoji: str = ""
    group: str = MetricGroup.OTHER
    description: str = "Default formatting"
    # NEW: explicit display ordering within grouped tables (lower = earlier)
    group_order: int = 50

    # Statistical computation rules
    show_sum: bool = False
    show_mean: bool = True
    show_std: bool = True
    show_min: bool = True
    show_max: bool = True
    show_median: bool = True
    show_cv: bool = True

    # Validation rules
    validation_rules: List[ValidationRule] = field(default_factory=list)

    # Display formatting rules
    display_config: Dict[str, Any] = field(default_factory=dict)


DEFAULT_FORMAT_RULES = [
    MetricFormatRule(
        regex=r"time",
        precision=4,
        units="s",
        optimization_goal=MetricDirection.MINIMIZE,
        emoji="⏱️",
        group=MetricGroup.TIMING_METRICS,
        group_order=40,
        description="Timing metrics",
        show_sum=False,
        validation_rules=[
            ValidationRule.NO_NEGATIVE,
            ValidationRule.NO_INFINITE,
        ],  # Time cannot be negative or infinite
    ),
    MetricFormatRule(
        regex=r"(loss|error|mse|mae|rmse)",
        precision=4,
        units="",
        optimization_goal=MetricDirection.MINIMIZE,
        emoji="📉",
        group=MetricGroup.LOSS_METRICS,
        group_order=10,
        description="Loss and error metrics",
        show_sum=False,
    ),
    MetricFormatRule(
        regex=r"(accuracy|precision|recall|f1)",
        precision=4,
        units="",
        optimization_goal=MetricDirection.MAXIMIZE,
        emoji="🎯",
        group=MetricGroup.PERFORMANCE_METRICS,
        group_order=20,
        description="Performance metrics",
        show_sum=False,
    ),
    MetricFormatRule(
        regex=r"(count|num_)",
        precision=0,
        units="",
        optimization_goal=MetricDirection.NEUTRAL,
        format_as_integer=True,
        emoji="📦",
        group=MetricGroup.DATASET_METRICS,
        group_order=5,
        description="Count metrics",
        show_sum=True,  # Sum makes sense for counts
    ),
    MetricFormatRule(
        regex=r"grad",
        precision=4,
        units="",
        optimization_goal=MetricDirection.NEUTRAL,
        emoji="📊",
        group=MetricGroup.GRADIENT_METRICS,
        group_order=30,
        description="Gradient metrics",
        show_sum=False,
    ),
    MetricFormatRule(
        regex=r"(batch_idx|epoch_idx|global_step|round_idx)",
        precision=0,
        units="",
        optimization_goal=MetricDirection.NEUTRAL,
        format_as_integer=True,
        emoji="🗓️",
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


@typechecked
class MetricFormatter:
    """FL metrics formatter with validation and caching."""

    def __init__(self, rules: List[MetricFormatRule] = DEFAULT_FORMAT_RULES):
        """Initialize formatter.

        Args:
            rules: Custom formatting rules. Uses defaults if None.
        """
        if not rules:
            raise ValueError("Rules list cannot be empty")

        for i, rule in enumerate(rules):
            if not rule.regex.strip():
                raise ValueError(f"Rule {i} has invalid regex: {rule.regex}")

        self.rules: List[MetricFormatRule] = rules
        self._rule_cache: Dict[str, MetricFormatRule] = {}

    def find_rule(self, metric_name: str) -> MetricFormatRule:
        """Find matching formatting rule for metric."""
        if not metric_name.strip():
            metric_name = "empty_metric"

        if metric_name in self._rule_cache:
            return self._rule_cache[metric_name]

        for rule in self.rules:
            try:
                if re.search(rule.regex, metric_name, re.IGNORECASE):
                    self._rule_cache[metric_name] = rule
                    return rule
            except re.error as e:
                warnings.warn(f"Invalid regex in rule for {rule.regex}: {e}")
                continue

        default_rule = MetricFormatRule(
            regex=r".*",
            group_order=80,
            description=f"Auto-generated default for '{metric_name}'",
        )
        self._rule_cache[metric_name] = default_rule
        return default_rule

    def format(self, metric_name: str, value: float) -> str:
        """Format metric value according to rules."""
        if np.isnan(value):
            return "NaN"
        if np.isinf(value):
            return "+∞" if value > 0 else "-∞"

        try:
            rule = self.find_rule(metric_name)

            if rule.format_as_integer:
                try:
                    formatted_value = f"{int(round(value)):,}{rule.units}"
                except (ValueError, OverflowError):
                    formatted_value = f"{value:.0f}{rule.units}"
            else:
                precision = max(0, min(10, rule.precision))
                formatted_value = f"{value:.{precision}f}{rule.units}"

            return formatted_value

        except (TypeError, AttributeError) as e:
            warnings.warn(
                f"Failed to format metric '{metric_name}' with value {value}: {e}"
            )
            return f"{value}"  # Return basic string representation as fallback

    def group_metric_names(self, metric_names: List[str]) -> Dict[str, List[str]]:
        """Group metrics by category."""
        if not metric_names:
            return {}
        validated_names: List[str] = []
        for i, name in enumerate(metric_names):
            if not name or not name.strip():
                name = f"empty_metric_{i}"
            validated_names.append(name.strip())
        return group_by_metric_rules(validated_names, self.find_rule)

    def get_applicable_stats(self, metric_name: str) -> Dict[str, bool]:
        """Get which statistics to show for a metric.

        Args:
            metric_name: Metric name

        Returns:
            Dict of stat name -> show flag
        """
        if not metric_name.strip():
            metric_name = "empty_metric"

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
        if np.isnan(delta) or np.isinf(delta):
            return "dim white"

        if threshold <= 0:
            threshold = 1e-4

        try:
            if abs(delta) < threshold:
                return "dim white"

            rule = self.find_rule(metric_name)
            goal = rule.optimization_goal

            if goal == MetricDirection.NEUTRAL:
                return "dim white"

            is_good_change = (delta < 0 and goal == MetricDirection.MINIMIZE) or (
                delta > 0 and goal == MetricDirection.MAXIMIZE
            )
            return "bright_green" if is_good_change else "bright_red"

        except AttributeError as e:
            warnings.warn(f"Failed to get delta color for metric '{metric_name}': {e}")
            return "dim white"  # Return neutral color as fallback

    def compute_statistics(self, values: List[float]) -> Dict[str, float]:
        """Compute basic statistics for values.

        Args:
            values: List of numeric values

        Returns:
            Dictionary with computed statistics
        """
        if not values:
            return {
                "sum": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
            }

        return {
            "sum": float(np.sum(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }

    def detect_outliers(self, values: List[float]) -> Dict[str, Any]:
        """Detect statistical outliers using order-of-magnitude analysis.

        Args:
            values: List of numeric values to analyze

        Returns:
            Dictionary with outlier information: count, percentage, details
        """
        if len(values) < 2:
            return {
                "count": 0,
                "percentage": 0.0,
                "details": DISPLAY_PLACEHOLDER,
                "outlier_indices": [],
            }

        # Convert to log scale for order-of-magnitude analysis
        log_values: List[Tuple[int, float]] = []
        for i, v in enumerate(values):
            if v > 0:
                log_values.append((i, math.log10(v)))
            elif v == 0:
                log_values.append((i, -10.0))  # Assign very low log value for zeros
            else:
                log_values.append((i, math.log10(abs(v)) if v != 0 else -10.0))

        if len(log_values) < 2:
            return {
                "count": 0,
                "percentage": 0.0,
                "details": DISPLAY_PLACEHOLDER,
                "outlier_indices": [],
            }

        # Calculate log scale statistics
        log_vals_only: List[float] = [lv[1] for lv in log_values]
        log_median: float = float(np.median(log_vals_only))

        # Define outlier threshold (3+ orders of magnitude difference from median)
        outlier_threshold = OUTLIER_DEVIATION_THRESHOLD
        magnitude_threshold = EXTREME_OUTLIER_THRESHOLD

        outliers: List[Tuple[int, float]] = []
        for idx, log_val in log_values:
            deviation: float = abs(log_val - log_median)
            if deviation > outlier_threshold:
                outliers.append((idx, deviation))

        # Create details string
        if not outliers:
            details = DISPLAY_PLACEHOLDER
        elif len(outliers) == 1:
            deviation = outliers[0][1]
            if deviation >= magnitude_threshold:
                details = f"{magnitude_threshold}+ord-mag"
            elif deviation >= 3:
                details = "3ord-mag"
            else:
                details = "extreme"
        else:
            max_deviation = max(o[1] for o in outliers)
            if max_deviation >= magnitude_threshold:
                details = f"{magnitude_threshold}+ord-mag"
            elif max_deviation >= 3:
                details = "3ord-mag"
            else:
                details = "extreme"

        outlier_count = len(outliers)
        percentage = (outlier_count / len(values)) * 100 if values else 0.0

        return {
            "count": outlier_count,
            "percentage": percentage,
            "details": details,
            "outlier_indices": [o[0] for o in outliers],
        }

    def validate_metric_values(
        self, values: List[float], metric_name: str
    ) -> List[ValidationErrorType]:
        """Validate metric values based on metric-specific rules.

        Args:
            values: List of numeric values to validate
            metric_name: Name of the metric being validated

        Returns:
            List of validation errors found
        """
        errors: List[ValidationErrorType] = []

        # Universal validation: NaN and infinite values
        for val in values:
            if val != val:  # NaN check
                errors.append(ValidationErrorType.NAN)
                break
            elif abs(val) == float("inf"):
                errors.append(ValidationErrorType.INFINITE)
                break

        # Metric-specific validation based on rules
        rule = self.find_rule(metric_name)
        if ValidationRule.NO_NEGATIVE in rule.validation_rules:
            for val in values:
                if val < 0:
                    if "time" in metric_name.lower():
                        errors.append(ValidationErrorType.NEGATIVE_TIME)
                    else:
                        errors.append(ValidationErrorType.NEGATIVE_VALUE)
                    break

        # Mathematical consistency validation
        if len(values) >= 2:
            min_val = min(values)
            max_val = max(values)
            if min_val > max_val:
                errors.append(ValidationErrorType.IMPOSSIBLE_ORDER)

        return errors

    def format_validation_display(
        self, values: List[float], anomaly_data: Dict[str, Any], metric_name: str
    ) -> str:
        """Format combined validation and statistical anomaly display.

        Args:
            values: Raw metric values
            anomaly_data: Statistical anomaly information
            metric_name: Name of the metric

        Returns:
            Formatted display string with appropriate emojis and colors
        """
        errors = self.validate_metric_values(values, metric_name)
        anomaly_count = anomaly_data["count"]

        if errors:
            error_summary = ",".join(errors[:2])
            if len(errors) > 2:
                error_summary += f"+{len(errors) - 2}"

            # Simple classification: impossible vs implementation errors
            if any(
                e
                in [
                    ValidationErrorType.NEGATIVE_TIME,
                    ValidationErrorType.IMPOSSIBLE_ORDER,
                ]
                for e in errors
            ):
                return f"[bold red]{EMOJIS.blocked} {len(errors)} ({error_summary})[/bold red]"
            else:
                return f"[bold red]{EMOJIS.forbidden} {len(errors)} ({error_summary})[/bold red]"
        elif anomaly_count > 0:
            return f"[orange3]{EMOJIS.investigate} {anomaly_count} ({anomaly_data['details']})[/orange3]"
        else:
            return DISPLAY_PLACEHOLDER
