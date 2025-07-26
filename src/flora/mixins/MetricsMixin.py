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

from abc import ABC
from collections import defaultdict
from enum import Enum
from typing import Dict

from torchmetrics import MeanMetric, SumMetric


class MetricReduction(Enum):
    """Type of metric aggregation."""

    AVG = "average"
    SUM = "sum"


class MetricsMixin(ABC):
    """
    Mixin providing modular metrics collection and tracking for federated learning nodes.

    Features:
        - Supports both averaged (mean) and cumulative (sum) metrics.
        - Lazy initialization of metric containers for efficiency.
        - Simple logging and retrieval interface.

    Usage:
        - log_metric(name, value, MetricType.AVG, weight=1): Log a value to an averaged metric.
        - log_metric(name, value, MetricType.SUM): Log a value to a cumulative metric.
        - compute_metrics(): Returns a flat dict of all computed metrics.
        - reset_metrics(): Clears all tracked metrics and counters.

    Attributes:
        num_samples_trained (int): Tracks the number of samples processed (user-managed).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__mean_metrics: Dict[str, MeanMetric] | None = None
        self.__sum_metrics: Dict[str, SumMetric] | None = None
        self.num_samples_trained = 0

    @property
    def mean_metrics(self) -> Dict[str, MeanMetric]:
        if self.__mean_metrics is None:
            self.__mean_metrics = defaultdict(lambda: MeanMetric(sync_on_compute=False))
        return self.__mean_metrics

    @property
    def sum_metrics(self) -> Dict[str, SumMetric]:
        if self.__sum_metrics is None:
            self.__sum_metrics = defaultdict(lambda: SumMetric(sync_on_compute=False))
        return self.__sum_metrics

    def log_metric(
        self, name: str, value: float, reduction: MetricReduction, weight: int = 1
    ) -> None:
        """Log a metric with specified aggregation type."""
        if reduction == MetricReduction.AVG:
            self.mean_metrics[name].update(value, weight)
        elif reduction == MetricReduction.SUM:
            self.sum_metrics[name].update(value)
        else:
            raise ValueError(f"Unknown metric type: {reduction}")

    def compute_metrics(self) -> Dict[str, float]:
        """Compute and return all metrics as a flat dictionary."""
        metrics = {}
        if self.__mean_metrics is not None:
            metrics.update(
                {k: v.compute().item() for k, v in self.__mean_metrics.items()}
            )
        if self.__sum_metrics is not None:
            metrics.update(
                {k: v.compute().item() for k, v in self.__sum_metrics.items()}
            )
        return metrics

    def reset_metrics(self) -> None:
        """Reset all metrics for new collection period."""
        self.__mean_metrics = None
        self.__sum_metrics = None
        self.num_samples_trained = 0
