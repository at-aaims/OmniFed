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


class MetricType(Enum):
    """Type of metric aggregation."""

    AVG = "average"
    SUM = "sum"


class MetricsMixin(ABC):
    """
    Mixin providing comprehensive metrics collection and tracking capabilities.

    Usage:
    - log_metric(name, value, MetricType.AVG) for averaged metrics
    - log_metric(name, value, MetricType.SUM) for cumulative metrics
    - get_metrics() to retrieve all metrics
    - reset_metrics() to reset all metrics
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mean_metrics: Dict[str, MeanMetric] = defaultdict(
            lambda: MeanMetric(sync_on_compute=False)
        )
        self._sum_metrics: Dict[str, SumMetric] = defaultdict(
            lambda: SumMetric(sync_on_compute=False)
        )
        self.num_samples_trained = 0

    def log_metric(
        self, name: str, value: float, metric_type: MetricType, weight: int = 1
    ) -> None:
        """Log a metric with specified aggregation type."""
        if metric_type == MetricType.AVG:
            self._mean_metrics[name].update(value, weight)
        elif metric_type == MetricType.SUM:
            self._sum_metrics[name].update(value)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

    def get_metrics(self) -> Dict[str, float]:
        """Compute and return all metrics as a flat dictionary."""
        return {
            **{k: v.compute().item() for k, v in self._mean_metrics.items()},
            **{k: v.compute().item() for k, v in self._sum_metrics.items()},
        }

    def reset_metrics(self) -> None:
        """Reset all metrics for new collection period."""
        for metric in self._mean_metrics.values():
            metric.reset()
        for metric in self._sum_metrics.values():
            metric.reset()
