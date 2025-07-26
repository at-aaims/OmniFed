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


class OptimizationGoal(str, Enum):
    """Enum defining how metric values should be interpreted for trend analysis."""

    MAXIMIZE = "maximize"  # Higher values are better (accuracy, F1, etc.)
    MINIMIZE = "minimize"  # Lower values are better (loss, error, time, etc.)
    NEUTRAL = "neutral"  # No trend judgment (counts, IDs, etc.)


@dataclass
class MetricFormatRule:
    """
    Rule for formatting metrics based on name patterns.

    Attributes:
        matcher: Function that returns True if rule applies to metric name
        precision: Number of decimal places to show
        units: Unit suffix to append (e.g., 's' for seconds)
        optimization_goal: How to interpret metric changes for trend analysis
        format_as_integer: Whether to format values as integers (for counts, etc.)
        description: Human-readable description of what this rule matches
    """

    matcher: Callable[[str], bool]
    precision: int
    units: str = ""
    optimization_goal: OptimizationGoal = OptimizationGoal.MAXIMIZE
    format_as_integer: bool = False
    description: str = ""


# ======================================================================================
# DEFAULT FORMATTING RULES
# ======================================================================================

DEFAULT_FORMAT_RULES = [
    # Time metrics with various naming patterns (lower is better)
    MetricFormatRule(
        matcher=any_of("time", "duration", "latency", "communication"),
        precision=4,
        units="s",
        optimization_goal=OptimizationGoal.MINIMIZE,
        description="Time metrics",
    ),
    # Loss metrics (lower is better)
    MetricFormatRule(
        matcher=contains("loss"),
        precision=4,
        units="",
        optimization_goal=OptimizationGoal.MINIMIZE,
        description="Loss metrics",
    ),
    # Error metrics (lower is better)
    MetricFormatRule(
        matcher=any_of("error", "mse", "mae", "rmse"),
        precision=4,
        units="",
        optimization_goal=OptimizationGoal.MINIMIZE,
        description="Error metrics",
    ),
    # Accuracy and performance metrics (higher is better)
    MetricFormatRule(
        matcher=any_of("accuracy", "precision", "recall", "f1"),
        precision=5,
        units="",
        optimization_goal=OptimizationGoal.MAXIMIZE,
        description="Performance metrics",
    ),
    # Count metrics (samples, batches, etc.) - neutral (higher neither good nor bad)
    MetricFormatRule(
        matcher=any_of("samples", "batches", "count", "num_"),
        precision=1,
        units="",
        optimization_goal=OptimizationGoal.NEUTRAL,
        format_as_integer=True,
        description="Count metrics",
    ),
    # Gradient-related metrics (context dependent, but generally lower is better)
    MetricFormatRule(
        matcher=contains("grad"),
        precision=6,
        units="",
        optimization_goal=OptimizationGoal.MINIMIZE,
        description="Gradient metrics",
    ),
]


# ======================================================================================
# METRIC FORMATTER CLASS
# ======================================================================================


class MetricFormatter:
    """
    Formats metrics for display with intelligent type-based formatting.

    Uses a rule-based system to apply appropriate precision, units, and
    aggregation strategies based on metric names and types.
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

    def optimization_goal(self, metric_name: str) -> OptimizationGoal:
        """Get optimization goal for metric (MAXIMIZE, MINIMIZE, or NEUTRAL)."""
        matched_rule = self._find_rule(metric_name)
        return matched_rule.optimization_goal

    def is_higher_better(self, metric_name: str) -> bool:
        """Check if higher values are better for this metric. Returns False for MINIMIZE/NEUTRAL."""
        return self.optimization_goal(metric_name) == OptimizationGoal.MAXIMIZE

    def is_lower_better(self, metric_name: str) -> bool:
        """Check if lower values are better for this metric. Returns False for MAXIMIZE/NEUTRAL."""
        return self.optimization_goal(metric_name) == OptimizationGoal.MINIMIZE

    def _find_rule(self, metric_name: str) -> MetricFormatRule:
        """Find the formatting rule that matches the given metric name."""
        for rule in self.rules:
            if rule.matcher(metric_name):
                return rule
        return self._default_rule
