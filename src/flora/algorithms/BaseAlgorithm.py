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
from typing import Any, Dict

import rich.repr
from torch import nn

# ======================================================================================


@rich.repr.auto
class Algorithm(ABC):
    """
    Abstract base class for federated learning algorithms.

    This class defines a modular interface for federated learning algorithms, allowing
    flexible implementation of various FL paradigms.

    Responsibilities:
    - Define algorithm-specific training strategy
    - Provide lifecycle hooks for algorithm customization
    - Remain topology-agnostic through flexible interfaces

    Integration:
    - Instantiated by Node actors
    - Node calls methods at appropriate points in the FL lifecycle
    - Node provides access to resources

    Design Philosophy:
    - Separation of concerns: Node manages resources, Algorithm provides strategy
    - Flexibility: Implementations can override only what they need
    - Extensibility: Designed to support diverse FL paradigms and topologies
    """

    def on_round_start(
        self,
        round_num: int,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform operations before local computation in each round.

        Args:
            round_num (int): The current round number.
            metrics (Dict[str, Any]): Dictionary to store metrics and statistics.
        Returns:
            Dictionary with metrics and statistics
        """
        return metrics

    # @abstractmethod
    def on_local_round(
        self,
        round_num: int,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform local training for the current round.

        TODO: Updates to this to enable the override-only-what-is-needed paradigm.

        Args:
            round_num (int): The current round number.
            metrics (Dict[str, Any]): Dictionary to store metrics and statistics.
        """
        return metrics

    # @abstractmethod
    def train_epoch(
        self,
        round_num: int,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform local training for a single epoch.

        NOTE: Currently not used.
        TODO: Update this to enable the override-only-what-is-needed paradigm.

        Args:
            round_num (int): The current round number.
            metrics (Dict[str, Any]): Dictionary to store metrics and statistics.
        """
        return metrics

    # @abstractmethod
    def train_step(
        self,
        round_num: int,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform local training for a single step.

        NOTE: Currently not used.
        TODO: Update this to enable the override-only-what-is-needed paradigm.

        Args:
            round_num (int): The current round number.
            metrics (Dict[str, Any]): Dictionary to store metrics and statistics.
        """
        return metrics

    def on_round_end(
        self,
        round_num: int,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform operations after local computation in each round.

        Args:
            round_num (int): The current round number.
            metrics (Dict[str, Any]): Dictionary to store metrics and statistics.
        Returns:
            Dictionary with metrics and statistics
        """
        return metrics

    # -----------------------------------------------------------------

    def get_param_norm(self, model: nn.Module) -> float:
        """
        Calculate the L2 norm of the model parameters.

        Returns:
            The L2 norm of the model parameters.
        """
        param_norm = 0.0
        for p in model.parameters():
            if p.requires_grad:
                param_norm += p.data.norm(2).item() ** 2
        param_norm = param_norm**0.5
        return param_norm

    def get_grad_norm(self, model: nn.Module) -> float:
        """
        Calculate the L2 norm of the gradients of the model parameters.

        Returns:
            The L2 norm of the gradients of the model parameters.
        """
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm**0.5
        return grad_norm
