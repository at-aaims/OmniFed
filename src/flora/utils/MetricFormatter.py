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

from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Union, Optional
from enum import Enum
import numpy as np
from .matchers import contains, any_of

# ======================================================================================
# CONSTANTS AND ENUMS
# ======================================================================================


class MetricGroup(str, Enum):
    """Enum defining standard metric group categories for organizing display tables."""

    PERFORMANCE = "Performance"  # Accuracy, precision, recall, F1, etc.
    TRAINING = "Training"  # Loss, error, gradient metrics
    ROUND_TIMING = "Round Timing"  # Federated round duration (time/round)
    EPOCH_TIMING = "Epoch Timing"  # Local training epoch times (time/epoch_*)
    BATCH_TIMING = "Batch Timing"  # Batch processing times (time/batch_*)
    SYNC_TIMING = "Sync Timing"  # Synchronization operations (time/sync_*)
    DATASET = "Dataset"  # Sample counts, batch sizes, data metrics
    OTHER = "Other"  # Default/miscellaneous metrics


class TrendMagnitude(str, Enum):
    """Enum defining trend change magnitude categories based on percentage change."""

    SMALL = "small"  # < 2% change
    MEDIUM = "medium"  # 2-10% change
    LARGE = "large"  # > 10% change


class TrendColor(str, Enum):
    """Enum defining colors for trend indicators based on performance impact."""

    GOOD = "bright_green"  # Beneficial changes (clear, vibrant green)
    NEUTRAL = "white"  # Small/neutral changes (clear white)
    BAD = "bright_red"  # Detrimental changes (clear, vibrant red)


class OptimizationGoal(str, Enum):
    """
    Enum defining how metric values should be interpreted for trend analysis.

    Used by MetricFormatRule to determine trend direction and color coding.

    Values:
        MAXIMIZE: Higher values indicate better performance (accuracy, F1, precision, recall)
        MINIMIZE: Lower values indicate better performance (loss, error, training time, latency)
        NEUTRAL: No performance judgment - changes are neutral (counts, samples, node IDs)

    Usage:
        - Controls trend arrow colors (green for good changes, red for bad)
        - Determines optimization direction in federated learning metrics
        - Can be extended with additional goals like TARGET_RANGE for future use
    """

    MAXIMIZE = "maximize"  # Higher values are better (accuracy, F1, etc.)
    MINIMIZE = "minimize"  # Lower values are better (loss, error, time, etc.)
    NEUTRAL = "neutral"  # No trend judgment (counts, IDs, etc.)


@dataclass
class MetricFormatRule:
    """
    Rule-based formatting system for federated learning metrics.

    This class defines how metrics should be formatted, styled, and interpreted based on name patterns.
    Used by MetricFormatter to provide consistent, intelligent formatting across the FLORA framework.

    Purpose:
        - Format metric values with appropriate precision and units
        - Determine optimization goals for trend analysis and color coding
        - Provide visual indicators (emojis) for different metric types
        - Support both floating-point and integer formatting

    Usage:
        Rules are matched against metric names using the 'matcher' function. The first matching
        rule determines how the metric is formatted and interpreted.

        Example:
            rule = MetricFormatRule(
                matcher=contains("loss"),
                precision=4,
                optimization_goal=OptimizationGoal.MINIMIZE,
                emoji="📉"
            )

    Extensions:
        - Add custom matchers for domain-specific metrics
        - Extend with additional formatting options (scientific notation, percentages)
        - Add conditional formatting based on metric values
        - Support for locale-specific number formatting

    Attributes:
        matcher: Function that returns True if rule applies to metric name
        precision: Number of decimal places to show
        units: Unit suffix to append (e.g., 's' for seconds)
        optimization_goal: How to interpret metric changes for trend analysis
        format_as_integer: Whether to format values as integers (for counts, etc.)
        emoji: Emoji icon representing this metric type
        group: Category group for organizing metrics in tables
        description: Human-readable description of what this rule matches
    """

    matcher: Callable[[str], bool]
    precision: int
    units: str = ""
    optimization_goal: OptimizationGoal = OptimizationGoal.MAXIMIZE
    format_as_integer: bool = False
    emoji: str = ":bar_chart:"
    group: str = MetricGroup.OTHER
    description: str = ""


# ======================================================================================
# DEFAULT FORMATTING RULES
# ======================================================================================

# Trend magnitude thresholds (percentage changes)
TREND_THRESHOLD_SMALL = 2.0  # Below this is considered small change
TREND_THRESHOLD_LARGE = 10.0  # Above this is considered large change

# Unicode symbols for trend visualization (using regular arrows with double arrows for large changes)
TREND_ARROWS = {
    TrendMagnitude.SMALL: "→",  # Regular horizontal arrow for small changes
    TrendMagnitude.MEDIUM: {"up": "↑", "down": "↓"},  # Single arrows for medium changes
    TrendMagnitude.LARGE: {
        "up": "⇈",
        "down": "⇊",
    },  # Double arrows for large/significant changes
}

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
    ),
    # Batch-level timing metrics (actual: time/batch_train, time/batch_eval, time/batch_data_*, time/batch_compute_*)
    MetricFormatRule(
        matcher=contains("time/batch"),
        precision=4,
        units="s",
        optimization_goal=OptimizationGoal.MINIMIZE,
        emoji=":clock:",
        group=MetricGroup.BATCH_TIMING,
        description="Batch processing timing",
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
    ),
]


# ======================================================================================
# METRIC FORMATTER CLASS
# ======================================================================================


class MetricFormatter:
    """
    Intelligent rule-based formatter and trend analyzer for federated learning metrics.

    This class provides comprehensive formatting, styling, and trend analysis for metrics
    in the FLORA federated learning framework. It automatically detects metric types and
    applies appropriate formatting rules for consistent display across tables and reports.

    Purpose:
        - Format individual metric values with proper precision and units
        - Generate statistical summaries (mean, std, min, max) across nodes
        - Analyze trends between metric values with colored arrows and magnitude detection
        - Provide emoji icons for enhanced visual identification
        - Support both real-time display and final reporting

    Core Features:
        1. **Rule-based formatting**: Automatically matches metrics to formatting rules
        2. **Statistical aggregation**: Calculates stats across federated nodes
        3. **Trend analysis**: Colored arrows with magnitude-based symbols and performance-aware coloring
        4. **Visual enhancement**: Provides emojis and styling for better UX
        5. **Type flexibility**: Handles integers, floats, and specialized formats

    Usage Examples:
        # Basic formatting
        formatter = MetricFormatter()
        formatted = formatter.format("loss/train", 0.1234)  # "📉 0.1234"

        # Statistical summary across nodes
        results = [{"loss": 0.1}, {"loss": 0.2}, {"loss": 0.15}]
        stats = formatter.format_stats(results)
        # Returns: {"loss": {"mean": "0.1500", "std": "0.0500", ...}}

        # Trend analysis with visual indicators
        trend_symbol = formatter.get_trend_symbol("accuracy", 0.95, 0.90)  # "[bold green]↑[/bold green]"
        emoji = formatter.get_emoji("time/communication")  # ":stopwatch:"

    Extensibility:
        - Add custom rules for domain-specific metrics
        - Extend optimization goals (e.g., TARGET_RANGE)
        - Support additional statistical measures
        - Integrate with external visualization libraries
        - Add conditional formatting based on threshold values

    Integration:
        Used throughout FLORA's Engine.py for:
        - Final round metrics tables
        - Round-by-round progression displays
        - Real-time training dashboards
        - Experiment result exports
    """

    def __init__(self, rules: List[MetricFormatRule] = None):
        """
        Initialize formatter with optional custom rules.

        Args:
            rules: List of formatting rules. Uses DEFAULT_FORMAT_RULES if None.
        """
        self.rules = rules or DEFAULT_FORMAT_RULES.copy()
        self._default_rule = MetricFormatRule(
            matcher=lambda x: True,
            precision=5,
            units="",
            optimization_goal=OptimizationGoal.MAXIMIZE,
            emoji=":bar_chart:",
            group=MetricGroup.OTHER,
            description="Default",
        )

    def format_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
        """
        Format statistical summary across multiple result sets.

        Args:
            results: List of result dictionaries from different nodes

        Returns:
            Dictionary mapping metric names to dictionaries with 'mean', 'std', 'min', 'max' keys
        """
        if not results:
            return {}

        # Find metrics common to all nodes
        common_metrics = set(results[0].keys())
        for node_results in results[1:]:
            common_metrics &= set(node_results.keys())

        formatted_metrics = {}

        for metric in sorted(common_metrics):
            # Handle numeric metrics with statistical formatting
            numeric_values = [
                result[metric]
                for result in results
                if isinstance(result[metric], (int, float))
            ]

            if numeric_values:
                # Find first matching format rule
                matched_rule = self._default_rule
                for rule in self.rules:
                    if rule.matcher(metric):
                        matched_rule = rule
                        break

                mean_val = np.mean(numeric_values)
                std_val = np.std(numeric_values) if len(numeric_values) > 1 else 0
                min_val = np.min(numeric_values)
                max_val = np.max(numeric_values)

                # Format with appropriate precision and units
                precision = matched_rule.precision
                units = matched_rule.units

                if len(numeric_values) == 1:
                    # Single node - show value in mean column, others empty
                    formatted_metrics[metric] = {
                        "mean": f"{mean_val:.{precision}f}{units}",
                        "std": "-",
                        "min": "-",
                        "max": "-",
                    }
                else:
                    # Multiple nodes - show all statistics properly
                    formatted_metrics[metric] = {
                        "mean": f"{mean_val:.{precision}f}{units}",
                        "std": f"{std_val:.{precision}f}{units}",
                        "min": f"{min_val:.{precision}f}{units}",
                        "max": f"{max_val:.{precision}f}{units}",
                    }
            else:
                # Handle non-numeric metrics
                all_values = [str(result[metric]) for result in results]
                unique_values = list(set(all_values))
                if len(unique_values) == 1:
                    formatted_metrics[metric] = {
                        "mean": unique_values[0],
                        "std": "-",
                        "min": "-",
                        "max": "-",
                    }
                else:
                    formatted_metrics[metric] = {
                        "mean": f"{len(unique_values)} unique values",
                        "std": "-",
                        "min": "-",
                        "max": "-",
                    }

        return formatted_metrics

    def format(self, metric_name: str, value: float) -> str:
        """Format a single metric value using the appropriate rule."""
        matched_rule = self._find_rule(metric_name)

        # Format with appropriate precision and units
        if matched_rule.format_as_integer:
            return f"{int(value):,}{matched_rule.units}"
        else:
            return f"{value:.{matched_rule.precision}f}{matched_rule.units}"

    def get_rule_property(self, metric_name: str, property_name: str):
        """Generic method to get any property from the matching rule."""
        matched_rule = self._find_rule(metric_name)
        return getattr(matched_rule, property_name)

    def optimization_goal(self, metric_name: str) -> OptimizationGoal:
        """Get optimization goal for metric (MAXIMIZE, MINIMIZE, or NEUTRAL)."""
        return self.get_rule_property(metric_name, "optimization_goal")

    def get_emoji(self, metric_name: str) -> str:
        """Get emoji icon for the given metric based on formatting rules."""
        return self.get_rule_property(metric_name, "emoji")

    def get_group(self, metric_name: str) -> str:
        """Get category group for the given metric based on formatting rules."""
        return self.get_rule_property(metric_name, "group")

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

    def get_trend_symbol(
        self, metric_name: str, current: float, previous: float
    ) -> str:
        """Get colored trend symbol based on metric change magnitude and direction."""
        # No change - neutral
        if current == previous:
            return f"[bold {TrendColor.NEUTRAL}]→[/bold {TrendColor.NEUTRAL}]"

        # Calculate percentage change magnitude
        pct_change = (
            abs((current - previous) / previous * 100)
            if previous != 0
            else abs(current - previous) * 100
        )

        # Determine magnitude and select symbol
        if pct_change < TREND_THRESHOLD_SMALL:
            symbol = "→"  # Small change
        elif pct_change < TREND_THRESHOLD_LARGE:
            symbol = "↑" if current > previous else "↓"  # Medium change
        else:
            symbol = "⇈" if current > previous else "⇊"  # Large change

        # Determine color based on performance impact
        goal = self.optimization_goal(metric_name)

        # Neutral metrics - no performance judgment
        if goal == OptimizationGoal.NEUTRAL or pct_change < TREND_THRESHOLD_SMALL:
            color = TrendColor.NEUTRAL
        else:
            # Good change: increase for MAXIMIZE metrics, decrease for MINIMIZE metrics
            is_good_change = (current > previous) == (goal == OptimizationGoal.MAXIMIZE)
            color = TrendColor.GOOD if is_good_change else TrendColor.BAD

        return f"[bold {color}]{symbol}[/bold {color}]"

    def _find_rule(self, metric_name: str) -> MetricFormatRule:
        """Find the formatting rule that matches the given metric name."""
        for rule in self.rules:
            if rule.matcher(metric_name):
                return rule
        return self._default_rule
