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
import math
import time
import warnings
from abc import abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import rich.repr
import torch
from torch import nn
from typeguard import typechecked

from ..communicator.BaseCommunicator import AggregationOp, BaseCommunicator
from ..data.DataModule import DataModule
from ..mixins import SetupMixin
from ..mixins.LifecycleHooksMixin import LifecycleHooksMixin
from . import utils
from .ExecutionMetrics import AccumulationMode, ExecutionMetrics
from .ExecutionSchedules import ExecutionSchedules

# ======================================================================================


@rich.repr.auto
class BaseAlgorithm(SetupMixin, LifecycleHooksMixin):
    """
    Base class for federated learning algorithms with hooks-based architecture.

    **Required implementations:**
    - `_configure_local_optimizer()`: Return optimizer (e.g., SGD, Adam)
    - `_compute_loss()`: Forward pass, return (loss, batch_size)
    - `_aggregate()`: Federated aggregation using self.local_comm

    **Optional lifecycle hooks:**
    - `_setup()`, `_round_start()`, `_round_end()` for round lifecycle
    - `_train_epoch_start()`, `_train_epoch_end()` for training epoch lifecycle
    - `_train_batch_start()`, `_train_batch_end()` for training batch lifecycle
    - `_eval_epoch_start()`, `_eval_epoch_end()` for evaluation epoch lifecycle
    - `_eval_batch_start()`, `_eval_batch_end()` for evaluation batch lifecycle
    - `_backward_pass()`, `_optimizer_step()` for training customization
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Apply decorators to required abstract methods if they exist
        try:
            cls._aggregate = cls._log_param_changes(cls._aggregate)
        except AttributeError:
            pass  # Method not implemented yet, will be caught by abstract method validation

    @typechecked
    def __init__(
        self,
        # Training hyperparameters (algorithm-specific)
        local_lr: float,
        max_epochs_per_round: int,
        schedules: ExecutionSchedules,
        log_dir: str,
    ):
        """
        Initialize the BaseAlgorithm instance.

        Args:
            local_lr (float): Learning rate for local training.
            max_epochs_per_round (int): Maximum number of epochs per round.
            schedules (ExecutionSchedules): Execution schedules for timing control.
            log_dir (str): Directory for metrics logging and TensorBoard output.
        """
        # Initialize parent mixins
        super().__init__()

        # Validate algorithm-specific parameters
        if local_lr <= 0:
            raise ValueError(f"local_lr must be positive, got {local_lr}")
        if max_epochs_per_round <= 0:
            raise ValueError(
                f"max_epochs_per_round must be positive, got {max_epochs_per_round}"
            )

        # Store algorithm-specific parameters
        self.local_lr: float = local_lr
        self.max_epochs_per_round: int = max_epochs_per_round
        # Store execution schedules
        self.schedules: ExecutionSchedules = schedules
        # Initialize metrics collection and logging infrastructure
        self.metrics: ExecutionMetrics = ExecutionMetrics(log_dir=log_dir)

        # Node context dependencies (injected via _setup())
        self.__local_comm: Optional[BaseCommunicator] = None
        self.__global_comm: Optional[BaseCommunicator] = None
        self.__local_model: Optional[nn.Module] = None
        self.__datamodule: Optional[DataModule] = None

        # Training state indices
        self.__round_idx: int = 0
        self.__epoch_idx: int = 0
        self.__batch_idx: int = 0

        # Training components (initialized per round)
        self.__local_optimizer: Optional[torch.optim.Optimizer] = None

        # Distributed training parameters (discovered during setup)
        self._local_iters_per_epoch: Optional[int] = None
        self._global_max_iters_per_epoch: Optional[int] = None
        self._global_max_epochs_per_round: Optional[int] = None

    # =============================================================================
    # PROPERTIES
    # =============================================================================

    @property
    def local_comm(self) -> BaseCommunicator:
        """Local communication interface for FL operations."""
        if self.__local_comm is None:
            raise RuntimeError(
                "local_comm accessed before setup() - call setup() first"
            )
        return self.__local_comm

    @property
    def global_comm(self) -> Optional[BaseCommunicator]:
        """Global communication interface for hierarchical FL operations."""
        return self.__global_comm

    @property
    def local_model(self) -> nn.Module:
        """Current ML model being trained."""
        if self.__local_model is None:
            raise RuntimeError("model accessed before setup() - call setup() first")
        return self.__local_model

    @local_model.setter
    def local_model(self, value: nn.Module) -> None:
        """Update the ML model (used during FL training phases)."""
        self.__local_model = value

    @property
    def datamodule(self) -> DataModule:
        """Data loading interface providing train/eval dataloaders."""
        if self.__datamodule is None:
            raise RuntimeError(
                "datamodule accessed before setup() - call setup() first"
            )
        return self.__datamodule

    @property
    def local_optimizer(self) -> torch.optim.Optimizer:
        """Current optimizer for local training.

        Created during round initialization in __reset_round_state().
        """
        if self.__local_optimizer is None:
            raise RuntimeError("local_optimizer accessed before round initialization")
        return self.__local_optimizer

    @local_optimizer.setter
    def local_optimizer(self, value: torch.optim.Optimizer) -> None:
        self.__local_optimizer = value

    @property
    def round_idx(self) -> int:
        """Current federated learning round index."""
        return self.__round_idx

    @round_idx.setter
    def round_idx(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"round_idx must be non-negative, got {value}")
        self.__round_idx = value

    @property
    def epoch_idx(self) -> int:
        """Current local training epoch index within the current round."""
        return self.__epoch_idx

    @epoch_idx.setter
    def epoch_idx(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"epoch_idx must be non-negative, got {value}")
        self.__epoch_idx = value

    @property
    def batch_idx(self) -> int:
        """Current batch index within the current epoch."""
        return self.__batch_idx

    @batch_idx.setter
    def batch_idx(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"batch_idx must be non-negative, got {value}")
        self.__batch_idx = value

    @property
    def local_iters_per_epoch(self) -> int:
        """Number of local iterations per epoch for this node.

        Used with global_max_iters_per_epoch for synchronized training loops
        across nodes with different data sizes.
        """
        if self._local_iters_per_epoch is None:
            raise RuntimeError(
                "local_iters_per_epoch accessed before distributed training initialization"
            )
        return self._local_iters_per_epoch

    @property
    def global_max_iters_per_epoch(self) -> int:
        """Global maximum iterations per epoch across all nodes.

        Used by training loops to synchronize all nodes for the same number of
        iterations, preventing nodes with less data from finishing early.
        """
        if self._global_max_iters_per_epoch is None:
            raise RuntimeError(
                "global_max_iters_per_epoch accessed before distributed training initialization"
            )
        return self._global_max_iters_per_epoch

    @property
    def global_max_epochs_per_round(self) -> int:
        """Global maximum epochs per round across all nodes.

        Used by training loops to synchronize all nodes for the same number of
        epochs, maintaining consistency even if nodes have different max_epochs settings.
        """
        if self._global_max_epochs_per_round is None:
            raise RuntimeError(
                "global_max_epochs_per_round accessed before distributed training initialization"
            )
        return self._global_max_epochs_per_round

    @property
    def progress_context(self) -> str:
        """
        Current progress context string with maximum detail.

        Used for consistent logging across training, evaluation, and aggregation phases.
        Always provides full context (Round + Epoch + Batch) for maximum visibility
        into where we are in the federated learning lifecycle.
        """
        return f"ROUND {self.round_idx + 1} EPOCH {self.epoch_idx + 1} BATCH {self.batch_idx + 1}"

    # =============================================================================
    # SETUP
    # =============================================================================

    def _setup(
        self,
        local_comm: BaseCommunicator,
        global_comm: Optional[BaseCommunicator],
        model: nn.Module,
        datamodule: DataModule,
    ) -> None:
        """
        Setup algorithm with injected dependencies.

        Args:
            local_comm: Local communication interface for intra-group operations
            global_comm: Optional global communication interface for hierarchical FL
            model: ML model being trained
            datamodule: Data loading interface providing train/eval dataloaders

        Override for algorithm-specific initialization logic.
        ALWAYS call super()._setup() first when overriding.
        """
        # Store injected dependencies
        self.__local_comm = local_comm
        self.__global_comm = global_comm
        self.__local_model = model
        self.__datamodule = datamodule

        # Standard federated learning setup: broadcast initial model from server
        _t_sync_start = time.time()
        self.local_model = self.local_comm.broadcast(self.local_model)
        _t_sync_end = time.time()
        self.metrics.log_metric(
            "time/sync_broadcast_init",
            _t_sync_end - _t_sync_start,
            AccumulationMode.AVG,
        )

    # =============================================================================
    # MINIMAL OVERRIDES
    # =============================================================================

    @abstractmethod
    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        Configure optimizer for local training (REQUIRED override).

        Create and return the optimizer that will be used for local model updates.
        This is called once per round when the optimizer is first accessed.

        Args:
            local_lr: Learning rate for local training

        Returns:
            local_optimizer: Configured optimizer instance
        """
        pass

    @abstractmethod
    def _compute_loss(self, batch: Any) -> tuple[torch.Tensor, int]:
        """
        Compute loss for a single batch (REQUIRED override).

        Perform forward pass and compute loss tensor. This method only handles
        the forward pass - backward pass, optimizer step, and metrics are
        handled automatically by the base class.

        Args:
            batch: Single batch from DataLoader (already on device)

        Returns:
            loss: Scalar tensor for backward pass
            batch_size: Number of samples in batch (for metrics)
        """
        pass

    @abstractmethod
    def _aggregate(self) -> nn.Module:
        """
        Perform local aggregation within the group (REQUIRED override).

        This method is called after local training when aggregation should occur
        (controlled by schedules.aggregation triggers). Use self.local_comm for communication.

        Common patterns:
        - FedAvg: return self.local_comm.aggregate(self.model, AllReduceOp.MEAN)
        - Custom: Scale local weights → aggregate → apply optimizations → return result

        Returns:
            local_model: Aggregated model after local group aggregation
        """
        pass

    # =============================================================================
    # =============================================================================

    def __synchronize(self) -> None:
        """
        Internal method: Coordinate federated model aggregation and evaluation.

        Orchestrates the complete synchronization process including local aggregation,
        inter-group coordination, and pre/post-aggregation evaluation phases.

        Called at different granularities based on schedules.aggregation configuration:
        - round_end: After complete training rounds (most common)
        - epoch_end: After local training epochs
        - batch_end: After individual training batches

        Four-phase execution:
        0. Pre-aggregation evaluation (local model)
        1. Intra-group aggregation: All nodes use local_comm.aggregate()
        2. Inter-group coordination: Group servers aggregate via global_comm
        3. Conditional broadcast: Only when inter-group coordination occurred
        4. Post-aggregation evaluation (global model)
        """
        # Pre-aggregation evaluation (local model)
        if self.schedules.evaluation.pre_aggregation():
            self.run_eval_epoch(self.local_model, "local")

        # Phase 1: Intra-group aggregation via all-reduce
        print(f"[LOCAL-AGG] {self.progress_context} | Start", flush=True)
        _t_sync_start = time.time()
        self.local_model = self._aggregate()
        _t_sync_end = time.time()
        self.metrics.log_metric(
            "time/sync_aggregate_local",
            _t_sync_end - _t_sync_start,
            AccumulationMode.AVG,
        )
        print(f"[LOCAL-AGG] {self.progress_context} | Complete", flush=True)

        # Phase 2: Inter-group coordination (group servers only)
        if self.global_comm is not None:
            print(f"[GLOBAL-AGG] {self.progress_context} | Start", flush=True)
            _t_sync_start = time.time()
            self.local_model = self.global_comm.aggregate(
                self.local_model, reduction=AggregationOp.MEAN
            )
            _t_sync_end = time.time()
            self.metrics.log_metric(
                "time/sync_aggregate_global",
                _t_sync_end - _t_sync_start,
                AccumulationMode.AVG,
            )
            print(f"[GLOBAL-AGG] {self.progress_context} | Complete", flush=True)

        # Phase 3: Conditional broadcast to distribute global results
        needs_final_broadcast = (
            self.local_comm.aggregate(
                torch.tensor(1.0 if self.global_comm is not None else 0.0),
                AggregationOp.MAX,
            )
            > 0
        )

        if needs_final_broadcast:
            print(f"[LOCAL-BCAST] {self.progress_context} | Start", flush=True)
            _t_sync_start = time.time()
            self.local_model = self.local_comm.broadcast(self.local_model)
            _t_sync_end = time.time()
            self.metrics.log_metric(
                "time/sync_broadcast_final",
                _t_sync_end - _t_sync_start,
                AccumulationMode.AVG,
            )
            print(f"[LOCAL-BCAST] {self.progress_context} | Complete", flush=True)
        else:
            print(f"[LOCAL-BCAST] {self.progress_context} | Skipped", flush=True)

        # Post-aggregation evaluation (global model)
        if self.schedules.evaluation.post_aggregation():
            self.run_eval_epoch(self.local_model, "global")

    def __reset_round_state(self, round_idx: int) -> None:
        """
        Internal method: Initialize state for a new federated learning round.

        Resets all round-specific state including optimizer, metrics, indices, and
        discovers distributed training parameters for synchronized execution.
        """
        # Discover distributed training parameters for synchronized loops
        self._local_iters_per_epoch = (
            len(self.datamodule.train) if self.datamodule.train is not None else 0
        )

        # Find global maximum iterations and epochs (batched for efficiency)
        local_limits = {
            "iters_per_epoch": torch.tensor(
                self._local_iters_per_epoch, dtype=torch.int
            ),
            "epochs_per_round": torch.tensor(
                self.max_epochs_per_round, dtype=torch.int
            ),
        }
        group_limits = self.local_comm.aggregate(local_limits, AggregationOp.MAX)

        self._global_max_iters_per_epoch = int(group_limits["iters_per_epoch"].item())
        self._global_max_epochs_per_round = int(group_limits["epochs_per_round"].item())

        # ---
        # Create new instances for this round
        self.__local_optimizer = self._configure_local_optimizer(self.local_lr)
        # ---
        self.round_idx = round_idx
        self.epoch_idx = 0
        self.batch_idx = 0

    def round_exec(self, round_idx: int, max_rounds: int) -> List[Dict[str, float]]:
        """
        Execute federated round computation across multiple epochs.

        Returns:
            List of epoch-level metrics dictionaries (one per epoch)

        Override for custom multi-epoch training logic.
        """

        self.__reset_round_state(round_idx)

        # Experiment start evaluation (only on first round) - before any training work
        if round_idx == 0 and self.schedules.evaluation.experiment_start():
            self.run_eval_epoch(self.local_model, "global")

        _t_start = time.time()

        print(
            f"[ROUND-START] {self.progress_context} | "
            f"global_max_epochs_per_round={self.global_max_epochs_per_round} | "
            f"global_max_iters_per_epoch={self.global_max_iters_per_epoch}",
            flush=True,
        )
        # Overridable hook for algorithm-specific logic
        self._round_start()

        # ---
        # All nodes enter synchronized epoch loop structure
        self.local_model.train()  # Future: .eval() for evaluation phases

        # Collect epoch-level metrics
        epoch_metrics_list = []

        for epoch_idx in range(self.global_max_epochs_per_round):
            # Run epoch training
            self.run_train_epoch(epoch_idx)

            # Compute epoch metrics and log to TensorBoard (after all epoch work is complete)
            epoch_metrics = self.metrics.compute_metrics()
            self.metrics.log_to_tensorboard(
                round_idx, epoch_idx, self.global_max_epochs_per_round
            )

            # For intermediate epochs: capture and reset before next epoch
            if epoch_idx < self.global_max_epochs_per_round - 1:
                epoch_metrics_list.append(epoch_metrics)
                self.metrics.reset_metrics()

            # Print epoch summary with simple formatting
            print(
                f"[EPOCH-END] {self.progress_context} |",
                {k: self.metrics.format_metric(k, v) for k, v in epoch_metrics.items()},
                flush=True,
            )

        # Check if round-level aggregation should occur
        if self.schedules.aggregation.round_end():
            self.__synchronize()

        # Overridable hook for algorithm-specific logic
        self._round_end()

        _t_end = time.time()
        round_duration = _t_end - _t_start

        # # Log round timing to the metrics system for aggregation
        # # Note: This gets logged outside the epoch loop, but round timing is a round-level metric
        # self.metrics.log_metric("time/round", round_duration, AccumulationMode.AVG)

        print(
            f"[ROUND-END] {self.progress_context} | time/round={round_duration:.4f}s",
            flush=True,
        )

        # Experiment end evaluation (only on last round)
        if round_idx == max_rounds - 1 and self.schedules.evaluation.experiment_end():
            self.run_eval_epoch(self.local_model, "global")

        # Capture final epoch with all round-end work included
        final_epoch_metrics = self.metrics.compute_metrics()
        epoch_metrics_list.append(final_epoch_metrics)

        return epoch_metrics_list

    def run_train_epoch(
        self,
        epoch_idx: int,
    ) -> None:
        """
        Internal method: Execute a single training epoch with synchronized batches.

        Runs one complete training epoch including batch processing, timing metrics,
        lifecycle hooks, epoch-level aggregation, and TensorBoard logging.
        """
        self.epoch_idx = epoch_idx

        print(
            f"[TRAIN-EPOCH-START] {self.progress_context}",
            flush=True,
        )

        _t_epoch_start = time.time()

        # Train epoch start hook
        self._train_epoch_start()

        # Initialize dataloader iterator for sequential batch processing
        dataloader_iter = iter(self.datamodule.train or [])

        # All nodes participate in synchronized batch loop
        for batch_idx in range(self.global_max_iters_per_epoch):
            # Set batch index and start batch processing
            self.batch_idx = batch_idx
            _t_batch_start = time.time()
            # Overridable hook for algorithm-specific logic
            self._train_batch_start()

            # Data preparation: fetch and transfer batch
            batch = None
            _t_batch_data_start = time.time()
            if self.epoch_idx < self.max_epochs_per_round:
                try:
                    batch = next(dataloader_iter)
                    batch = self._transfer_batch_to_device(
                        batch, next(self.local_model.parameters()).device
                    )
                except StopIteration:
                    # Node has exhausted its data - continue with None batch for synchronization
                    pass
            _t_batch_data_end = time.time()

            # Execute batch computation
            _t_batch_compute_start = time.time()
            if batch is not None:
                self.__run_train_batch(batch)
            _t_batch_compute_end = time.time()

            # Batch-level aggregation
            if self.schedules.aggregation.batch_end():
                self.__synchronize()

            # Overridable hook for algorithm-specific logic
            self._train_batch_end()

            # Update timing metrics
            _t_batch_end = time.time()

            self.metrics.log_metric(
                "time/batch_data_train",
                _t_batch_data_end - _t_batch_data_start,
                AccumulationMode.AVG,
            )
            self.metrics.log_metric(
                "time/batch_compute_train",
                _t_batch_compute_end - _t_batch_compute_start,
                AccumulationMode.AVG,
            )

            # Overall batch timing for all nodes
            self.metrics.log_metric(
                "time/batch_train", _t_batch_end - _t_batch_start, AccumulationMode.AVG
            )

        # ---
        # Epoch boundary synchronization
        print(f"[EPOCH-SYNC-START] {self.progress_context}", flush=True)
        _t_start = time.time()
        sync_signal = torch.tensor([1.0])
        total_signals = self.local_comm.aggregate(sync_signal, AggregationOp.SUM)
        sync_time = time.time() - _t_start
        self.metrics.log_metric(
            "time/sync_epoch_boundary", sync_time, AccumulationMode.AVG
        )
        print(
            f"[EPOCH-SYNC-END] {self.progress_context} | time={sync_time:.4f}s signals={int(total_signals.item())}",
            flush=True,
        )

        # Epoch-level aggregation
        if self.schedules.aggregation.epoch_end():
            self.__synchronize()

        # Train epoch end hook
        self._train_epoch_end()

        _t_epoch_end = time.time()

        # Log epoch timing (metrics will be computed later in main loop)
        self.metrics.log_metric(
            "time/epoch_train", _t_epoch_end - _t_epoch_start, AccumulationMode.AVG
        )

    def run_eval_epoch(self, model: nn.Module, eval_type: str) -> None:
        """
        Internal method: Execute a single evaluation epoch with timing and metrics.

        Runs model evaluation on the entire eval dataset using torch.no_grad()
        for memory efficiency, with comprehensive timing and lifecycle hooks.
        """
        if self.datamodule.eval is None:
            raise RuntimeError(
                f"Evaluation data not available for {self.progress_context}. "
                "Ensure datamodule.eval is properly configured or disable evaluation in the schedule."
            )

        print(
            f"[EVAL-EPOCH-START] {eval_type.upper()} | {self.progress_context}",
            flush=True,
        )
        _t_start = time.time()

        # Temporarily switch to eval mode
        was_training = model.training
        model.eval()

        # Overridable hook for algorithm-specific logic
        self._eval_epoch_start()

        # Initialize dataloader iterator for sequential batch processing
        dataloader_iter = iter(self.datamodule.eval or [])

        # Determine metric namespace based on evaluation type
        metric_namespace = f"eval_{eval_type}"

        with torch.no_grad():
            # Simple loop through eval data - no synchronization needed during eval
            for idx, batch in enumerate(dataloader_iter):
                # Start batch processing with detailed timing
                _t_batch_start = time.time()
                # Overridable hook for algorithm-specific logic
                self._eval_batch_start()

                # Data preparation: fetch and transfer batch
                _t_batch_data_start = time.time()
                batch = self._transfer_batch_to_device(
                    batch, next(model.parameters()).device
                )
                _t_batch_data_end = time.time()

                # Execute evaluation batch computation
                _t_batch_compute_start = time.time()
                self.__run_eval_batch(batch, metric_namespace)
                _t_batch_compute_end = time.time()

                # Overridable hook for algorithm-specific logic
                self._eval_batch_end()

                # Update timing metrics
                _t_batch_end = time.time()
                self.metrics.log_metric(
                    "time/batch_data_eval",
                    _t_batch_data_end - _t_batch_data_start,
                    AccumulationMode.AVG,
                )
                self.metrics.log_metric(
                    "time/batch_compute_eval",
                    _t_batch_compute_end - _t_batch_compute_start,
                    AccumulationMode.AVG,
                )
                self.metrics.log_metric(
                    "time/batch_eval",
                    _t_batch_end - _t_batch_start,
                    AccumulationMode.AVG,
                )

        # Overridable hook for algorithm-specific logic
        self._eval_epoch_end()

        _t_end = time.time()
        # Log eval timing (metrics will be computed later with training metrics)
        self.metrics.log_metric(
            "time/epoch_eval", _t_end - _t_start, AccumulationMode.AVG
        )

        # Restore original mode
        if was_training:
            model.train()

    def __run_batch(
        self, batch: Any, metric_namespace: str
    ) -> tuple[torch.Tensor, int]:
        """
        Internal method: Execute forward pass and metrics tracking for a single batch.

        This shared computation is used by both training and evaluation batch
        processing to maintain consistent metrics collection.
        """
        # Forward pass
        loss, batch_size = self._compute_loss(batch)

        # Metric tracking
        self.metrics.log_metric(
            f"loss/{metric_namespace}",
            loss.detach().item(),
            AccumulationMode.AVG,
            batch_size,
        )

        return loss, batch_size

    def __run_train_batch(self, batch: Any) -> None:
        """
        Internal method: Execute training-specific computation for a single batch.

        Handles the training-only operations including gradient computation,
        optimizer updates, and training-specific metrics collection.
        """
        loss, batch_size = self.__run_batch(batch, "train")
        self.metrics.log_metric("samples/train", batch_size, AccumulationMode.SUM)
        self.metrics.log_metric("batches/train", 1, AccumulationMode.SUM)

        # Training-only operations
        self.metrics.num_samples_trained += batch_size  # For aggregation weights only

        self.local_optimizer.zero_grad()
        self._backward_pass(loss)
        self.metrics.log_metric(
            "grad_norm/train",
            utils.get_grad_norm(self.local_model),
            AccumulationMode.AVG,
        )
        self._optimizer_step()

    def __run_eval_batch(self, batch: Any, metric_namespace: str) -> None:
        """
        Internal method: Execute evaluation-specific computation for a single batch.

        Handles evaluation-only operations with shared batch processing
        but without gradient computation or parameter updates.
        """
        loss, batch_size = self.__run_batch(batch, metric_namespace)
        self.metrics.log_metric("samples/eval", batch_size, AccumulationMode.SUM)
        self.metrics.log_metric("batches/eval", 1, AccumulationMode.SUM)

    # =============================================================================

    def _backward_pass(self, loss: torch.Tensor) -> None:
        """
        Compute gradients from loss tensor.

        DEFAULT: Standard loss.backward() for automatic differentiation.

        Override for custom gradient computation.
        """
        loss.backward()

    def _optimizer_step(self) -> None:
        """
        Apply parameter updates using computed gradients.

        DEFAULT: Standard optimizer.step() with current gradients.

        Override for custom parameter updates.
        """
        self.local_optimizer.step()

    # =============================================================================
    # MISC UTILITY METHODS
    # =============================================================================

    def _transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        """
        Move batch tensors to algorithm's compute device.


        Override this method for custom batch formats or transfer logic.
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
            f"Batch type '{type(batch).__name__}' not handled by _transfer_batch_to_device(). "
            "Override this method for custom batch formats."
        )
        return batch

    @staticmethod
    def _log_param_changes(func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to automatically track parameter changes for BaseAlgorithm methods."""

        @wraps(func)
        def wrapper(algo: "BaseAlgorithm", *args: Any, **kwargs: Any) -> Any:
            phase = func.__name__  # Automatically get the function name
            before_norm = utils.get_param_norm(algo.local_model)
            before_hash = utils.hash_model_params(algo.local_model)

            # Fatal check: model must have valid parameters before operation
            if before_norm == 0.0:
                raise RuntimeError(
                    f"Model has zero parameter norm before {phase}(). All parameters are zero."
                )
            if math.isnan(before_norm) or math.isinf(before_norm):
                raise RuntimeError(
                    f"Model has invalid parameter norm before {phase}(). Contains NaN or Inf values."
                )

            # Execute the original function
            result = func(algo, *args, **kwargs)

            after_norm = utils.get_param_norm(algo.local_model)
            after_hash = utils.hash_model_params(algo.local_model)

            delta = after_norm - before_norm
            changed = before_hash != after_hash

            print(
                f"[{phase.upper()}] local_model hash: {before_hash[:8]} → {after_hash[:8]} | "
                f"norm: {before_norm:.4f} → {after_norm:.4f} (Δ={delta:.6f}) | "
                f"{'CHANGED' if changed else 'UNCHANGED'}"
            )

            # Fatal check: operation must not break the model
            if after_norm == 0.0:
                raise RuntimeError(
                    f"Operation {phase}() zeroed all model parameters. Norm became 0.0."
                )
            if math.isnan(after_norm) or math.isinf(after_norm):
                raise RuntimeError(
                    f"Operation {phase}() caused numerical instability. Norm is now NaN or Inf."
                )

            # Non-fatal warnings for concerning patterns
            if phase == "_aggregate" and not changed:
                warnings.warn(
                    f"Aggregation operation {phase}() completed but parameters unchanged. "
                    f"No model updates occurred. "
                    f"Check if nodes have training data and aggregation weights are non-zero.",
                    UserWarning,
                )

            if after_norm > before_norm * 10:
                warnings.warn(
                    f"Parameter norm explosion in {phase}(). "
                    f"Increased {after_norm / before_norm:.1f}x from {before_norm:.4f} to {after_norm:.4f}. "
                    f"Consider reducing learning rate or adding gradient clipping.",
                    UserWarning,
                )

            if after_norm < before_norm * 0.1:
                warnings.warn(
                    f"Parameter norm vanishing in {phase}(). "
                    f"Decreased {before_norm / after_norm:.1f}x from {before_norm:.4f} to {after_norm:.4f}. "
                    f"Check for gradient vanishing, excessive regularization, or incorrect scaling.",
                    UserWarning,
                )

            return result

        return wrapper
