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

from collections import defaultdict
from typing import Dict

from torchmetrics import MeanMetric, SumMetric


# ======================================================================================


class RoundMetrics:
    """
    Unified metrics tracking manager for training rounds.
    """

    def __init__(self):
        self._mean_metrics: Dict[str, MeanMetric] = defaultdict(
            lambda: MeanMetric(sync_on_compute=False)  # Default CPU placement
        )
        self._sum_metrics: Dict[str, SumMetric] = defaultdict(
            lambda: SumMetric(sync_on_compute=False)  # Default CPU placement
        )

    def update_mean(self, name: str, value: float, weight: int = 1) -> None:
        """Update a mean-tracked metric with automatic tensor conversion."""
        self._mean_metrics[name].update(value, weight)

    def update_sum(self, name: str, value: float) -> None:
        """Update a sum-tracked metric with automatic tensor conversion."""
        self._sum_metrics[name].update(value)

    def compute_all(self) -> Dict[str, float]:
        """Compute and return all metrics as a flat dictionary."""
        return {
            **{k: v.compute().item() for k, v in self._mean_metrics.items()},
            **{k: v.compute().item() for k, v in self._sum_metrics.items()},
        }

    def reset_all(self) -> None:
        """Reset all metrics for a new round."""
        for metric in self._mean_metrics.values():
            metric.reset()
        for metric in self._sum_metrics.values():
            metric.reset()
