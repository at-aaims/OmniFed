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
from typing import List, Dict, Optional
from pathlib import Path
from enum import Enum

from torch.utils.tensorboard import SummaryWriter

from .MetricFormatter import MetricFormatter


class MetricReduction(Enum):
    """Type of metric aggregation during collection."""

    AVG = "average"  # Running average (weighted mean)
    SUM = "sum"  # Running sum (accumulative)


class SimpleMetric:
    """Lightweight metric accumulator replacing torchmetrics for better control."""

    def __init__(self, reduction: MetricReduction):
        self.reduction = reduction
        self.reset()

    def reset(self) -> None:
        """Reset the metric state."""
        self.total = 0.0
        self.count = 0
        self.weight_sum = 0.0

    def update(self, value: float, weight: int = 1) -> None:
        """Update the metric with a new value."""
        if self.reduction == MetricReduction.SUM:
            self.total += value
            self.count += 1
        elif self.reduction == MetricReduction.AVG:
            self.total += value * weight
            self.weight_sum += weight
            self.count += 1

    def compute(self) -> float:
        """Compute the final metric value."""
        if self.count == 0:
            return 0.0

        if self.reduction == MetricReduction.SUM:
            return self.total
        elif self.reduction == MetricReduction.AVG:
            return self.total / self.weight_sum if self.weight_sum > 0 else 0.0

        return 0.0


class ExecutionSummary:
    """
    Metrics collection, TensorBoard logging, and computation system for FLORA.

    Handles metric collection and provides basic formatting through composition.
    Designed for use in Node/BaseAlgorithm where metrics lifecycle is needed.
    """

    def __init__(self, log_dir: str):
        """
        Initialize the metrics collection manager.

        Args:
            log_dir: Directory for logging (TensorBoard, CSVs, etc.).
        """
        # Internal formatter for basic formatting needs
        self._formatter = MetricFormatter()

        # Metric collection state
        self._metrics: Dict[str, SimpleMetric] = {}
        self.num_samples_trained: int = 0

        # TensorBoard integration
        os.makedirs(log_dir, exist_ok=True)
        self._tensorboard_writer = SummaryWriter(log_dir=log_dir)

    def log_metric(
        self, name: str, value: float, reduction: MetricReduction, weight: int = 1
    ) -> None:
        """Log a metric with specified aggregation type."""
        if name not in self._metrics:
            self._metrics[name] = SimpleMetric(reduction)
        self._metrics[name].update(value, weight)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute and return all metrics as a flat dictionary."""
        return {name: metric.compute() for name, metric in self._metrics.items()}

    def log_epoch_to_tensorboard(
        self, round_idx: int, epoch_idx: int, max_epochs_per_round: int
    ) -> None:
        """
        Log current computed metrics to TensorBoard with deterministic step calculation.

        Args:
            round_idx: Current round number (0-based)
            epoch_idx: Current epoch within round (0-based)
            max_epochs_per_round: Maximum epochs per round (for deterministic step calculation)
        """
        metrics = self.compute_metrics()
        if metrics:
            global_step = round_idx * max_epochs_per_round + epoch_idx
            self._tensorboard_writer.add_scalars("epoch_metrics", metrics, global_step)
        self._tensorboard_writer.flush()

    def format(self, metric_name: str, value: float) -> str:
        """Format a single metric value using internal formatter."""
        return self._formatter.format(metric_name, value)

    def reset_metrics(self) -> None:
        """Reset all metrics for new collection period."""
        self._metrics.clear()
        self.num_samples_trained = 0
