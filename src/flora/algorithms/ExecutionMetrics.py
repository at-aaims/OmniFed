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

import os
import csv
from typing import Dict, List, Optional
from pathlib import Path
from enum import Enum

from torch.utils.tensorboard import SummaryWriter

from ..utils.MetricFormatter import MetricFormatter


class AccumulationMode(str, Enum):
    """
    How individual metric values accumulate within a training phase.

    Controls how metric values combine as training progresses within
    epochs (batch-to-batch accumulation).
    """

    AVG = "average"  # Running average (weighted mean)
    SUM = "sum"  # Running sum (accumulative)


class SimpleMetric:
    """
    Lightweight metric accumulator for FL training statistics.

    Supports both running averages (weighted by sample count) and
    accumulative sums for different metric types.
    """

    def __init__(self, reduction: AccumulationMode):
        """
        Initialize metric accumulator.

        Args:
            reduction: How to accumulate values (AVG or SUM)
        """
        self.reduction = reduction
        self.reset()

    def reset(self) -> None:
        """Reset accumulator state for new collection period."""
        self.total = 0.0
        self.count = 0
        self.weight_sum = 0.0

    def update(self, value: float, weight: int = 1) -> None:
        """
        Add a new metric value to accumulator.

        Args:
            value: Metric value to accumulate
            weight: Sample weight (for weighted averages)
        """
        match self.reduction:
            case AccumulationMode.SUM:
                self.total += value
                self.count += 1
            case AccumulationMode.AVG:
                self.total += value * weight
                self.weight_sum += weight
                self.count += 1
            case _:
                raise ValueError(f"Unknown reduction type: {self.reduction}")

    def compute(self) -> float:
        """
        Calculate final accumulated metric value.

        Returns:
            Final metric value based on reduction strategy
        """
        if self.count == 0:
            return 0.0

        match self.reduction:
            case AccumulationMode.SUM:
                return self.total
            case AccumulationMode.AVG:
                return self.total / self.weight_sum if self.weight_sum > 0 else 0.0
            case _:
                raise ValueError(f"Unknown reduction type: {self.reduction}")


class ExecutionMetrics:
    """
    Integrated metrics collection and logging for FL algorithm training.

    **Architectural Responsibility:**
    This class handles algorithm-specific metrics instrumentation during local
    training phases. It provides unified collection, accumulation, and logging
    capabilities specifically for FL algorithm implementations.

    **Used by:** BaseAlgorithm and its subclasses during training execution
    **Not used by:** Node (infrastructure), Engine (global results), or other components

    **Key capabilities:**
    - Accumulates training metrics with different reduction strategies
    - Logs to TensorBoard with FL-aware round/epoch step calculation
    - Exports to CSV for additional analysis
    - Provides consistent metric formatting
    """

    def __init__(self, log_dir: str):
        """
        Initialize metrics collection and logging infrastructure.

        Args:
            log_dir: Directory for TensorBoard logs and CSV exports
        """
        # Metric collection state
        self._metrics: Dict[str, SimpleMetric] = {}
        self.num_samples_trained: int = 0

        # Logging infrastructure
        os.makedirs(log_dir, exist_ok=True)
        self._log_dir = log_dir
        self._tensorboard_writer = SummaryWriter(log_dir=log_dir)

        # Formatting for consistent display
        self._formatter = MetricFormatter()

    def log_metric(
        self, name: str, value: float, reduction: AccumulationMode, weight: int = 1
    ) -> None:
        """
        Record a metric value with specified aggregation strategy.

        Creates new metric accumulator if needed, then updates with value.

        Args:
            name: Metric identifier (e.g., "train/loss", "eval/accuracy")
            value: Metric value to record
            reduction: How to accumulate values (AVG or SUM)
            weight: Sample weight for weighted averages
        """
        if name not in self._metrics:
            self._metrics[name] = SimpleMetric(reduction)
        self._metrics[name].update(value, weight)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Calculate final values for all accumulated metrics.

        Returns:
            Dictionary mapping metric names to computed values
        """
        return {name: metric.compute() for name, metric in self._metrics.items()}

    def log_to_tensorboard(
        self, round_idx: int, epoch_idx: int, max_epochs_per_round: int
    ) -> None:
        """
        Write current metrics to TensorBoard with FL-aware step calculation.

        Uses deterministic step calculation for consistent visualization
        across FL rounds and epochs. Each algorithm execution logs to its
        own TensorBoard timeline.

        Args:
            round_idx: Current FL round number (0-based)
            epoch_idx: Current epoch within round (0-based)
            max_epochs_per_round: Maximum epochs per round for step calculation
        """
        metrics = self.compute_metrics()
        if metrics:
            global_step = round_idx * max_epochs_per_round + epoch_idx
            self._tensorboard_writer.add_scalars("epoch_metrics", metrics, global_step)
        self._tensorboard_writer.flush()

    def export_to_csv(
        self, round_idx: int, epoch_idx: int, filepath: Optional[str] = None
    ) -> None:
        """
        Export current metrics to CSV file for additional analysis.

        Args:
            round_idx: Current FL round number
            epoch_idx: Current epoch number
            filepath: Custom CSV path (default: auto-generated in log_dir)
        """
        if filepath is None:
            filepath = os.path.join(self._log_dir, "metrics.csv")

        metrics = self.compute_metrics()
        if not metrics:
            return

        # Create CSV with headers if file doesn't exist
        file_exists = os.path.exists(filepath)
        with open(filepath, "a", newline="") as csvfile:
            fieldnames = ["round", "epoch"] + list(metrics.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            row = {"round": round_idx, "epoch": epoch_idx}
            row.update(metrics)
            writer.writerow(row)

    def format_metric(self, metric_name: str, value: float) -> str:
        """
        Format metric value using consistent formatting rules.

        Args:
            metric_name: Name of metric for formatting rules
            value: Numeric value to format

        Returns:
            Formatted string representation
        """
        return self._formatter.format(metric_name, value)

    def reset_metrics(self) -> None:
        """Clear all accumulated metrics for new training phase."""
        self._metrics.clear()
        self.num_samples_trained = 0

    def close(self) -> None:
        """Clean up logging resources."""
        if self._tensorboard_writer:
            self._tensorboard_writer.close()
