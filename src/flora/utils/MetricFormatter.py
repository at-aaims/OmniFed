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
from typing import Callable, List, Dict, Any, Union
import numpy as np
from .matchers import contains, any_of

# ======================================================================================


@dataclass
class MetricFormatRule:
    """
    Rule for formatting metrics based on name patterns.

    Attributes:
        matcher: Function that returns True if rule applies to metric name
        precision: Number of decimal places to show
        show_total: Whether to show total sum for count-like metrics
        units: Unit suffix to append (e.g., 's' for seconds)
        description: Human-readable description of what this rule matches
    """

    matcher: Callable[[str], bool]
    precision: int
    show_total: bool = False
    units: str = ""
    description: str = ""


# ======================================================================================
# DEFAULT FORMATTING RULES
# ======================================================================================

DEFAULT_FORMAT_RULES = [
    # Time metrics with various naming patterns
    MetricFormatRule(
        matcher=any_of("time", "duration", "latency"),
        precision=4,
        show_total=False,
        units="s",
        description="Time metrics",
    ),
    # Loss metrics
    MetricFormatRule(
        matcher=contains("loss"),
        precision=4,
        show_total=False,
        units="",
        description="Loss metrics",
    ),
    # Accuracy and performance metrics
    MetricFormatRule(
        matcher=any_of("accuracy", "precision", "recall", "f1"),
        precision=5,
        show_total=False,
        units="",
        description="Performance metrics",
    ),
    # Count metrics (samples, batches, etc.) - show totals
    MetricFormatRule(
        matcher=any_of("samples", "batches", "count", "num_"),
        precision=1,
        show_total=True,
        units="",
        description="Count metrics",
    ),
    # Gradient-related metrics (for debugging)
    MetricFormatRule(
        matcher=contains("grad"),
        precision=6,
        show_total=False,
        units="",
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
            show_total=False,
            units="",
            description="Default",
        )

    def add_rule(self, rule: MetricFormatRule) -> None:
        """Add a new formatting rule (inserted at beginning for priority)."""
        self.rules.insert(0, rule)

    def format_metric(
        self, metric_name: str, numeric_values: List[Union[int, float]]
    ) -> str:
        """
        Format a metric with multiple values using statistical aggregation.

        Args:
            metric_name: Name of the metric
            numeric_values: List of numeric values from different nodes

        Returns:
            Formatted string representation of the metric
        """
        if not numeric_values:
            return "N/A"

        # Find first matching format rule
        matched_rule = self._default_rule
        for rule in self.rules:
            if rule.matcher(metric_name):
                matched_rule = rule
                break

        return self._format_with_rule(numeric_values, matched_rule)

    def _format_with_rule(
        self, numeric_values: List[Union[int, float]], rule: MetricFormatRule
    ) -> str:
        """Format numeric values according to a specific rule."""
        mean_val = np.mean(numeric_values)
        std_val = np.std(numeric_values) if len(numeric_values) > 1 else 0
        single_node = len(numeric_values) == 1

        # Choose formatting strategy
        if rule.show_total and not single_node:
            value_str = f"Σ{sum(numeric_values):,} (μ{mean_val:.{rule.precision}f} ±{std_val:.{rule.precision}f})"
        elif single_node:
            value_str = f"{mean_val:.{rule.precision}f}"
        else:
            value_str = f"{mean_val:.{rule.precision}f} ±{std_val:.{rule.precision}f}"

        return f"{value_str}{rule.units}"

    def format_results_summary(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Format a complete results summary from multiple nodes.

        Args:
            results: List of result dictionaries from different nodes

        Returns:
            Dictionary mapping metric names to formatted display strings
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
                formatted_metrics[metric] = self.format_metric(metric, numeric_values)
            else:
                # Handle non-numeric metrics
                all_values = [str(result[metric]) for result in results]
                unique_values = list(set(all_values))
                if len(unique_values) == 1:
                    formatted_metrics[metric] = unique_values[0]
                else:
                    formatted_metrics[metric] = f"{len(unique_values)} unique values"

        return formatted_metrics

    def get_rule_info(self) -> List[Dict[str, str]]:
        """Get information about all active formatting rules."""
        return [
            {
                "description": rule.description,
                "precision": str(rule.precision),
                "show_total": str(rule.show_total),
                "units": rule.units or "none",
            }
            for rule in self.rules
        ]
