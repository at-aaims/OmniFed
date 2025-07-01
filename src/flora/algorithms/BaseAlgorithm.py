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

    def __init__(self, local_model: nn.Module, comm: Communicator):
        """
        Initialize the Algorithm instance.

        Args:
            model: The neural network model for this algorithm instance
            communicator: The communicator for federated operations
        """
        # Core federated learning components
        self.local_model: nn.Module = local_model
        self.comm: Communicator = comm

        # Active execution context
        self._round_optimizer: Optional[torch.optim.Optimizer] = None
        self._round_metrics: Optional[RoundMetrics] = None
        self._round_total_samples: int = 0

    # =============================================================================
    # PROPERTIES
    # =============================================================================

    @property
    def round_optimizer(self) -> torch.optim.Optimizer:
        """Current optimizer (only available during computation)."""
        if self._round_optimizer is None:
            raise RuntimeError("optimizer only available during active computation")
        return self._round_optimizer

    @property
    def round_metrics(self) -> RoundMetrics:
        """Current metrics manager (only available during computation)."""
        if self._round_metrics is None:
            raise RuntimeError("metrics only available during active computation")
        return self._round_metrics

    @property
    def round_total_samples(self) -> int:
        """Total samples processed in current round."""
        if self._round_total_samples < 0:
            raise ValueError("round_total_samples is negative")
        return self._round_total_samples

    # =============================================================================
    # =============================================================================

    @abstractmethod
    def configure_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Configure optimizer for the given model.
        """
        pass

    def round_init(self) -> None:
        """
        Initialize algorithm state for a new federated round.

        Sets up execution context.
        """
        self._round_optimizer = self.configure_optimizer(self.local_model)
        self._round_total_samples = 0
        self._round_metrics = RoundMetrics()

    # =============================================================================
    # TRAINING LOGIC
    # =============================================================================

    def train_round(
        self,
        dataloader: DataLoader[Any],
        round_idx: int,
        max_epochs: int,
        device: torch.device,
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
        device_prev = next(self.local_model.parameters()).device

        self.local_model.train()
        self.local_model.to(device)

        for epoch_idx in range(max_epochs):
            # Epoch timing
            epoch_start_time = time.time()
            self.epoch_start(epoch_idx)

            # Epoch computation
            self.train_epoch(dataloader, epoch_idx)

            # Overridable epoch end hook
            self.epoch_end(epoch_idx)
            epoch_time = time.time() - epoch_start_time

            self.round_metrics.update_mean("time/epoch", epoch_time)

            print(
                f"Round {round_idx + 1} | Epoch {epoch_idx + 1}/{max_epochs} | {self.round_metrics.compute_all()}"
            )

        self.local_model.to(device_prev)

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
            self.round_metrics.update_mean("time/step_data", data_time)
            self.round_metrics.update_mean("time/step_compute", compute_time)
            self.round_metrics.update_mean("time/step", batch_time)

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
        self._round_total_samples += batch_size
        self.round_metrics.update_sum("data/samples", batch_size)
        self.round_metrics.update_sum("data/batches", 1)

        # Automatic loss tracking
        self.round_metrics.update_mean("loss/compute", loss.detach().item(), batch_size)

        self.round_optimizer.zero_grad()
        # Backward pass hook
        self.backward_pass(loss, batch_idx)

        # Automatic gradient tracking
        self.round_metrics.update_mean(
            "compute/grad_norm", utils.get_grad_norm(self.local_model)
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
        self.round_optimizer.step()

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

    def calculate_batch_size(self, batch: Any) -> int:
        """
        Extract number of samples from batch for metrics tracking.
        Handles common PyTorch batch formats automatically.

        Override for custom batch structures.
        """
        if isinstance(batch, torch.Tensor):
            return batch.size(0)

        if isinstance(batch, (tuple, list)) and len(batch) > 0:
            # Try first element
            first = batch[0]
            if isinstance(first, torch.Tensor):
                return first.size(0)

        if hasattr(batch, "__len__"):
            return len(batch)

        raise ValueError(f"Cannot estimate batch size for type {type(batch)}")

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
