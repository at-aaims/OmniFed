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
from typing import List, Optional, Union

from omegaconf import MISSING


@dataclass
class TriggerConfig:
    """
    Structured config for execution condition triggers.

    Defines when FL operations should execute based on call frequency
    or specific timing conditions.
    """

    _target_: str = "src.flora.algorithms.ExecutionSchedules.Trigger"

    every: Optional[int] = None  # Execute every N calls
    at: Optional[List[int]] = None  # Execute at specific call numbers


@dataclass
class AggregationTriggersConfig:
    """
    Structured config for FL model aggregation scheduling.

    Controls when nodes combine their local models with others
    during different phases of federated learning.
    """

    _target_: str = "src.flora.algorithms.ExecutionSchedules.AggregationTriggers"

    round_end: Union[TriggerConfig, bool, None] = True
    epoch_end: Union[TriggerConfig, bool, None] = True
    batch_end: Union[TriggerConfig, bool, None] = True


@dataclass
class EvaluationTriggersConfig:
    """
    Structured config for FL model evaluation scheduling.

    Controls when nodes evaluate their models on test/validation data
    during different phases of federated learning.
    """

    _target_: str = "src.flora.algorithms.ExecutionSchedules.EvaluationTriggers"

    experiment_start: Union[TriggerConfig, bool, None] = True
    experiment_end: Union[TriggerConfig, bool, None] = False
    pre_aggregation: Union[TriggerConfig, bool, None] = False
    post_aggregation: Union[TriggerConfig, bool, None] = False


@dataclass
class ExecutionSchedulesConfig:
    """
    Structured config for coordinated FL scheduling.

    Combines aggregation and evaluation schedules into unified configuration.
    """

    _target_: str = "src.flora.algorithms.ExecutionSchedules.ExecutionSchedules"

    aggregation: Optional[AggregationTriggersConfig] = None
    evaluation: Optional[EvaluationTriggersConfig] = None


@dataclass
class ExecutionMetricsConfig:
    """
    Structured config for algorithm metrics collection and logging.

    Controls how algorithms collect, accumulate, and log training metrics
    during federated learning execution.
    """

    _target_: str = "src.flora.algorithms.ExecutionMetrics.ExecutionMetrics"

    log_dir: str = MISSING  # Directory for TensorBoard and CSV output


@dataclass
class AlgorithmConfig:
    """Base configuration for all federated learning algorithms."""

    # Algorithm-specific training hyperparameters (required)
    local_lr: float = MISSING
    max_epochs_per_round: int = MISSING
    schedules: ExecutionSchedulesConfig = MISSING
    log_dir: str = MISSING  # Directory for metrics logging and TensorBoard output


@dataclass
class FedAvgConfig(AlgorithmConfig):
    """Configuration for FedAvg algorithm."""

    _target_: str = "src.flora.algorithms.fedavg.FedAvg"


@dataclass
class FedProxConfig(AlgorithmConfig):
    """Configuration for FedProx algorithm."""

    _target_: str = "src.flora.algorithms.fedprox.FedProx"

    # FedProx-specific parameters
    mu: float = 0.01  # Proximal term coefficient


@dataclass
class ScaffoldConfig(AlgorithmConfig):
    """Configuration for Scaffold algorithm."""

    _target_: str = "src.flora.algorithms.scaffold.Scaffold"


@dataclass
class FedNovaConfig(AlgorithmConfig):
    """Configuration for FedNova algorithm."""

    _target_: str = "src.flora.algorithms.fednova.FedNova"

    # FedNova-specific parameters
    weight_decay: float = 0.0  # Weight decay parameter


@dataclass
class FedBNConfig(AlgorithmConfig):
    """Configuration for FedBN algorithm."""

    _target_: str = "src.flora.algorithms.fedbn.FedBN"


@dataclass
class MoonConfig(AlgorithmConfig):
    """Configuration for Moon algorithm."""

    _target_: str = "src.flora.algorithms.moon.MOON"

    # Moon-specific parameters
    mu: float = 1.0  # Contrastive loss weight
    temperature: float = 0.5  # Temperature for contrastive learning
    num_prev_models: int = 1  # Number of previous models to store


@dataclass
class FedPerConfig(AlgorithmConfig):
    """Configuration for FedPer algorithm."""

    _target_: str = "src.flora.algorithms.fedper.FedPer"

    # FedPer-specific parameters
    personal_layers: Optional[List[str]] = None  # Layer names for personalization


@dataclass
class FedMomConfig(AlgorithmConfig):
    """Configuration for FedMom algorithm."""

    _target_: str = "src.flora.algorithms.fedmom.FedMom"

    # FedMom-specific parameters
    momentum: float = 0.9  # Momentum parameter


@dataclass
class FedDynConfig(AlgorithmConfig):
    """Configuration for FedDyn algorithm."""

    _target_: str = "src.flora.algorithms.feddyn.FedDyn"

    # FedDyn-specific parameters
    alpha: float = 0.1  # Regularization parameter


@dataclass
class DittoConfig(AlgorithmConfig):
    """Configuration for Ditto algorithm."""

    _target_: str = "src.flora.algorithms.ditto.Ditto"

    # Ditto-specific parameters
    global_lr: float = 0.01  # Global learning rate
    ditto_lambda: float = 0.1  # Regularization parameter


@dataclass
class DilocoConfig(AlgorithmConfig):
    """Configuration for Diloco algorithm."""

    _target_: str = "src.flora.algorithms.diloco.DiLoCo"

    # Diloco-specific parameters
    outer_lr: float = 0.7  # Outer learning rate
    outer_momentum: float = 0.9  # Outer momentum
