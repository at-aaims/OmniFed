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

import logging
import time
from abc import ABC, abstractmethod

from typing import Any, Optional

import rich.repr
import torch
from torch import nn
from torch.utils.data import DataLoader
from ..communicator.BaseCommunicator import Communicator
from ..helper.RoundMetrics import RoundMetrics
from . import utils

# ======================================================================================


@rich.repr.auto
class Algorithm(ABC):
    """
    Base class for federated learning algorithms with hooks-based architecture.

    - Override only what you need: sensible defaults with selective customization
    - Automatic infrastructure: metrics, timing, device handling, sample counting
    """

    def __init__(self, local_model: nn.Module, comm: Communicator, max_epochs: int):
        """
        Initialize the Algorithm instance.

        Args:
            local_model: The neural network model for this algorithm instance
            comm: The communicator for federated operations
        """
        # Core federated learning components
        self.local_model: nn.Module = local_model
        self.comm: Communicator = comm
        self.max_epochs: int = max_epochs

        # Round state - properly initialized
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._metrics: Optional[RoundMetrics] = None
        self._local_samples: int = 0
        # Round, epoch, and batch indices
        self._round_idx: int = 0
        self._epoch_idx: int = 0
        self._batch_idx: int = 0

    # =============================================================================
    # PROPERTIES
    # =============================================================================

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Current optimizer - created lazily when first accessed."""
        if self._optimizer is None:
            self._optimizer = self.configure_optimizer()
        return self._optimizer

    @property
    def metrics(self) -> RoundMetrics:
        """Current round metrics - created on first access."""
        if self._metrics is None:
            self._metrics = RoundMetrics()
        return self._metrics

    @property
    def round_idx(self) -> int:
        """Current federated round index with validation."""
        return self._round_idx

    @round_idx.setter
    def round_idx(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"round_idx must be non-negative, got {value}")
        self._round_idx = value

    @property
    def epoch_idx(self) -> int:
        """Current local epoch index with validation."""
        return self._epoch_idx

    @epoch_idx.setter
    def epoch_idx(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"epoch_idx must be non-negative, got {value}")
        self._epoch_idx = value

    @property
    def batch_idx(self) -> int:
        """Current local batch index with validation."""
        return self._batch_idx

    @batch_idx.setter
    def batch_idx(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"batch_idx must be non-negative, got {value}")
        self._batch_idx = value

    @property
    def local_samples(self) -> int:
        """Local samples processed by this node in current round - this node's contribution."""
        return self._local_samples

    @local_samples.setter
    def local_samples(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"local_samples must be non-negative, got {value}")
        self._local_samples = value

    def round_setup(self, round_idx: int) -> None:
        """Reset state for a new round - called by framework before round_start."""
        self._optimizer = None
        self._metrics = None
        self.round_idx = round_idx
        self.epoch_idx = 0
        self.batch_idx = 0
        self.local_samples = 0

    # =============================================================================

    @abstractmethod
    def configure_optimizer(self) -> torch.optim.Optimizer:
        """
        Configure optimizer for the given model.
        """
        pass

    # =============================================================================
    # TRAINING LOGIC
    # =============================================================================

    def train_round(
        self,
        dataloader: DataLoader[Any],
        round_idx: int,
    ):
        """
        Execute federated round computation across multiple epochs.

        DEFAULT: Sequential epoch processing with automatic timing and metrics.

        Override Use-Cases:
        - Early stopping: break epoch loop based on validation metrics
        - Dynamic epochs: adjust epoch count based on convergence
        - Learning rate schedules: step schedulers between epochs
        - Cross-epoch state: maintain state across epochs (e.g., momentum buffers)
        """
        self.local_model.train()

        for epoch_idx in range(self.max_epochs):
            # Update current epoch index
            self.epoch_idx = epoch_idx
            print(
                f"TRAIN_ROUND {round_idx + 1} | EPOCH {epoch_idx + 1} START",
                flush=True,
            )
            # Epoch timing
            epoch_start_time = time.time()
            self.epoch_start(epoch_idx)

            # Epoch computation
            self.train_epoch(dataloader, epoch_idx)

            # Overridable epoch end hook
            self.epoch_end(epoch_idx)
            epoch_time = time.time() - epoch_start_time

            self.metrics.update_mean("time/epoch", epoch_time)

            print(
                f"TRAIN_ROUND {round_idx + 1} | EPOCH {epoch_idx + 1} END |",
                {
                    k: round(v, 2) if isinstance(v, float) else v
                    for k, v in self.metrics.to_dict().items()
                },
                flush=True,
            )

    def train_epoch(self, dataloader: DataLoader[Any], epoch_idx: int) -> None:
        """
        Execute computation for all batches in a single epoch.

        DEFAULT: Sequential batch processing with automatic device transfer and timing.

        Override Use-Cases:
        - Gradient accumulation: accumulate gradients over multiple batches before optimizer step
        - Custom sampling: dynamic batch selection or weighted sampling
        - Multi-dataloader: alternate between multiple data sources
        - Batch preprocessing: custom transformations before device transfer
        """
        data_iter = iter(dataloader)

        for batch_idx in range(len(dataloader)):
            # Update current batch index
            self.batch_idx = batch_idx
            # Batch timing
            batch_start_time = time.time()
            # Overridable batch start hook
            self.batch_start(batch_idx)

            # ---
            # Sample batch and transfer to device
            data_start_time = time.time()
            batch = self.transfer_batch_to_device(
                next(data_iter),
                next(self.local_model.parameters()).device,
            )
            data_time = time.time() - data_start_time

            # ---
            # Batch computation
            compute_start_time = time.time()
            self.train_batch(batch, batch_idx)
            compute_time = time.time() - compute_start_time

            # ---
            # Overridable batch end hook
            self.batch_end(batch_idx)
            batch_time = time.time() - batch_start_time

            # ---
            self.metrics.update_mean("time/step_data", data_time)
            self.metrics.update_mean("time/step_compute", compute_time)
            self.metrics.update_mean("time/step", batch_time)

    def train_batch(self, batch: Any, batch_idx: int) -> None:
        """
        Execute computation for a single batch.

        Override Use-Cases:
        - Multiple optimizer steps: perform several gradient steps per batch
        - Custom backward passes: manual gradient computation or accumulation
        - Mixed precision: automatic or manual scaling for float16 training
        - Gradient clipping: apply gradient norm clipping before optimizer step
        """
        # Forward pass hook (implemented by subclasses)
        loss, batch_size = self.train_step(batch, batch_idx)
        # Automatic sample tracking
        self.local_samples = self.local_samples + batch_size
        self.metrics.update_sum("train/num_samples", batch_size)
        self.metrics.update_sum("train/num_batches", 1)

        # Automatic loss tracking
        self.metrics.update_mean("train/loss", loss.detach().item(), batch_size)

        self.optimizer.zero_grad()
        # Backward pass hook
        self.backward_pass(loss, batch_idx)

        # Automatic gradient tracking
        self.metrics.update_mean(
            "train/grad_norm", utils.get_grad_norm(self.local_model)
        )

        # Optimizer step hook
        self.optimizer_step(batch_idx)

    # =============================================================================

    @abstractmethod
    def train_step(self, batch: Any, batch_idx: int) -> tuple[torch.Tensor, int]:
        """
        Compute loss for a single batch (REQUIRED override).

        Args:
            batch: Single batch from DataLoader (already on device)
            batch_idx: Current batch index within epoch

        Returns:
            loss: Scalar tensor for backward pass
            batch_size: Number of samples in batch (for metrics)
        """
        pass

    def backward_pass(self, loss: torch.Tensor, batch_idx: int) -> None:
        """
        Compute gradients from loss tensor.

        DEFAULT: Standard loss.backward() for automatic differentiation.

        Override Use-Cases:
        - Manual gradients: torch.autograd.grad() for specific parameters
        - Gradient penalty: add regularization terms during backward pass
        - Mixed precision: scale loss before backward pass
        """
        loss.backward()

    def optimizer_step(self, batch_idx: int) -> None:
        """
        Apply parameter updates using computed gradients.

        DEFAULT: Standard optimizer.step() with current gradients.

        Override Use-Cases:
        - Conditional updates: skip updates based on gradient norms or loss values
        - Per-parameter learning rates: apply different step sizes to different layers
        - Momentum modifications: adjust momentum based on training progress
        """
        self.optimizer.step()

    # =============================================================================
    # LIFECYCLE HOOKS
    # =============================================================================

    def round_start(self, round_idx: int) -> None:
        """
        Called at the start of each federated round before local training.

        Use for federated communication and model synchronization.

        Override Use-Cases:
        - Receive global model: broadcast from server to all participants
        - Initialize round state: reset algorithm-specific variables or buffers
        - Sync control variables: exchange auxiliary parameters (SCAFFOLD control variates)
        """
        pass

    def round_end(self, round_idx: int) -> None:
        """
        Called at the end of each federated round after local training.

        Use for federated aggregation and model updates.

        Override Use-Cases:
        - Send local updates: transmit trained model or gradients to aggregator
        - Aggregate models: combine updates from all participants (if aggregator role)
        - Apply server optimization: use server-side optimizers on aggregated updates
        """
        pass

    def epoch_start(self, epoch_idx: int) -> None:
        """
        Called at the start of each local training epoch.

        Use for per-epoch setup and local state management.

        Override Use-Cases:
        - Learning rate scheduling: step LR scheduler based on epoch progress
        - Epoch state reset: clear per-epoch counters or loss accumulators
        - Dynamic configuration: adjust dropout rates or data augmentation per epoch
        """
        pass

    def epoch_end(self, epoch_idx: int) -> None:
        """
        Called at the end of each local training epoch.

        Use for per-epoch finalization and local validation.

        Override Use-Cases:
        - Local validation: evaluate model on local validation set
        - Epoch metrics: compute and log per-epoch training statistics
        - Early stopping: check local convergence criteria
        """
        pass

    def batch_start(self, batch_idx: int) -> None:
        """
        Called before processing each batch.

        Use for per-batch setup and state preparation.

        Override Use-Cases:
        - Optimizer state: modify learning rates or momentum per batch
        - Batch tracking: initialize batch-specific counters or flags
        - Debug logging: log batch indices or data samples for debugging
        """
        pass

    def batch_end(self, batch_idx: int) -> None:
        """
        Called after processing each batch.

        Use for per-batch finalization and monitoring.

        Override Use-Cases:
        - Batch metrics: log loss values or gradient norms per batch
        - Memory cleanup: clear temporary tensors or cached computations
        - Progress tracking: update training progress indicators
        """
        pass

    # =============================================================================
    # MISC UTILITY METHODS
    # =============================================================================

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        """
        Move batch tensors to algorithm's compute device.

        DEFAULT: Handles tensor, tuple, list, dict batch formats automatically.

        Override Use-Cases:
        - Nested structures: recursive transfer for complex batch hierarchies
        - Selective transfer: move only specific tensors to GPU, keep others on CPU
        - Memory optimization: transfer tensors individually to reduce peak memory
        """
        # Single tensor
        if isinstance(batch, torch.Tensor):
            return batch.to(device)

        # Tuple/list of tensors (most common case)
        if isinstance(batch, (tuple, list)):
            transferred = []
            for item in batch:
                if isinstance(item, torch.Tensor):
                    transferred.append(item.to(device))
                else:
                    transferred.append(item)  # Keep non-tensors as-is
            return tuple(transferred) if isinstance(batch, tuple) else transferred

        # Dictionary with tensor values
        if isinstance(batch, dict):
            transferred = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    transferred[key] = value.to(device)
                else:
                    transferred[key] = value  # Keep non-tensors as-is
            return transferred

        # Unsupported type - warn user
        logging.warning(
            f"Batch type '{type(batch).__name__}' not handled by transfer_batch_to_device(). "
            "Override this method for custom batch formats."
        )
        return batch
