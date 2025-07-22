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
from enum import Enum
from functools import wraps
from typing import Any, Optional

import rich.repr
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..communicator.BaseCommunicator import BaseCommunicator, ReductionType
from ..communicator.grpc_communicator import GrpcCommunicator
from ..communicator.TorchDistCommunicator import TorchDistCommunicator
from ..data.DataModule import DataModule
from ..helper.RoundMetrics import RoundMetrics
from ..mixins import SetupMixin
from . import utils
from .utils import log_param_changes

# ======================================================================================


class AggLevel(str, Enum):
    """
    Aggregation levels granularity.

    - ROUND: Aggregate after each global round
    - EPOCH: Aggregate after each local epoch
    - ITER: Aggregate after each local batch iteration

    CONFIGURATION NOTE: Use uppercase values in config files:
    - ✓ agg_level: ROUND (correct)
    - ✗ agg_level: round (incorrect - will cause aggregation to be skipped)

    FUTURE ENHANCEMENTS:
    - TODO: Analyze algorithm incompatibilities with agg_freq > 1 (some algorithms may require specific frequencies)
    - TODO: Design declarative validation framework for algorithm subclasses to specify aggregation requirements
    """

    ROUND = "ROUND"
    EPOCH = "EPOCH"
    ITER = "ITER"  # Batch-level aggregation
    # TODO: Maybe add another level based on the number of samples processed?


@rich.repr.auto
class BaseAlgorithm(SetupMixin):
    """
    Base class for federated learning algorithms with hooks-based architecture.

    **Required implementations:**
    - `_configure_local_optimizer()`: Return optimizer (e.g., SGD, Adam)
    - `_train_step()`: Forward pass, return (loss, batch_size)
    - `_aggregate()`: Federated aggregation using self.local_comm

    **Optional lifecycle hooks:**
    - `_setup()`, `_round_start()`, `_round_end()` for round lifecycle
    - `_epoch_start()`, `_epoch_end()` for epoch lifecycle
    - `_batch_start()`, `_batch_end()` for batch lifecycle
    - `_backward_pass()`, `_optimizer_step()` for training customization

    **Available infrastructure:**
    `self.metrics`, `self.local_comm`, `self.local_model`, hyperparameters, and state tracking.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_aggregate"):
            cls._aggregate = log_param_changes(cls._aggregate)
        if hasattr(cls, "_setup"):
            cls._setup = log_param_changes(cls._setup)

    def __init__(
        self,
        # Core FL components
        local_comm: BaseCommunicator,
        global_comm: BaseCommunicator | None,
        local_model: nn.Module,
        agg_level: AggLevel,
        agg_freq: int,
        # Training hyperparameters
        local_lr: float,
        max_epochs_per_round: int,
        # Miscellaneous
        tb_writer: Optional[SummaryWriter],
        **kwargs: Any,
    ):
        """
        Initialize the Algorithm instance.

        Args:
            local_comm: Communicator for intra-group operations
            global_comm: Optional communicator for inter-group operations
            local_model: The neural network model for this algorithm instance
            agg_level: Level at which aggregation occurs (ROUND, EPOCH, or ITER)
            agg_freq: Frequency of aggregation operations
            local_lr: Learning rate for local training
            max_epochs_per_round: Maximum number of epochs per round
            tb_writer: TensorBoard writer for logging metrics
            **kwargs: Additional algorithm-specific parameters
        """
        # Core federated learning components
        self.local_comm: BaseCommunicator = local_comm
        self.global_comm: BaseCommunicator | None = global_comm
        self.local_model: nn.Module = local_model
        self.agg_level: AggLevel = agg_level
        self.agg_freq: int = agg_freq

        # Training hyperparameters
        if local_lr <= 0:
            raise ValueError(f"local_lr must be positive, got {local_lr}")
        if max_epochs_per_round <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs_per_round}")
        if agg_freq <= 0:
            raise ValueError(f"agg_freq must be positive, got {agg_freq}")

        self.local_lr: float = local_lr
        self.max_epochs_per_round: int = max_epochs_per_round

        # Infrastructure components
        # self.tb_writer: SummaryWriter = tb_writer
        # self.tb_writer: Optional[SummaryWriter] = None
        self.tb_writer = None

        # Initialize state
        self.__local_optimizer: Optional[torch.optim.Optimizer] = None
        self.__round_metrics: Optional[RoundMetrics] = None
        self.__round_idx: int = 0
        self.__epoch_idx: int = 0
        self.__batch_idx: int = 0
        self.__local_sample_count: int = 0

        # ---
        self._local_iters_per_epoch: Optional[int] = None
        self._global_max_iters_per_epoch: Optional[int] = None
        self._global_max_epochs_per_round: Optional[int] = None

    # =============================================================================
    # PROPERTIES
    # =============================================================================

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
    def round_metrics(self) -> RoundMetrics:
        """Current round metrics for tracking training statistics.

        Created during round initialization in __reset_round_state().
        """
        if self.__round_metrics is None:
            raise RuntimeError("metrics accessed before round initialization")
        return self.__round_metrics

    @round_metrics.setter
    def round_metrics(self, value: RoundMetrics) -> None:
        self.__round_metrics = value

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
    def tb_global_epoch(self) -> int:
        """Global epoch count across all rounds for TensorBoard logging."""
        return self.round_idx * self.max_epochs_per_round + self.epoch_idx

    @property
    def local_sample_count(self) -> int:
        """Local samples processed by this node in current round.

        Used for computing aggregation weights in federated learning.
        """
        return self.__local_sample_count

    @local_sample_count.setter
    def local_sample_count(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"local_sample_count must be non-negative, got {value}")
        elif value < self.__local_sample_count and value != 0:
            raise ValueError(
                f"local_sample_count can only be reset to 0 or increased (current: {self.__local_sample_count}, got {value})"
            )
        self.__local_sample_count = value

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

    # =============================================================================
    # SETUP
    # =============================================================================

    def _setup(self, device: torch.device) -> None:
        """
        Default setup for federated learning algorithms (OPTIONAL override).

        Performs standard federated learning initialization:
        - Moves model to target device
        - Broadcasts initial model from rank 0 to all participants

        Override this method if your algorithm needs custom setup behavior.
        When overriding, ALWAYS call super()._setup(device) first to ensure
        the standard model setup happens before your custom logic.

        Args:
            device: Target device for model placement

        Common override use-cases:
        - Initialize algorithm-specific state (control variates, momentum buffers)
        - Create copies of the model for reference (global model, previous models)
        - Set up auxiliary networks or transformations

        Example:
            def _setup(self, device: torch.device) -> None:
                super()._setup(device)  # Standard model setup
                self.global_model = copy.deepcopy(self.local_model)
                self.momentum_buffers = {name: torch.zeros_like(param)
                                       for name, param in self.local_model.named_parameters()}
        """
        # Move model to target device (one-time operation)
        self.local_model = self.local_model.to(device)

        # Standard federated learning setup: broadcast initial model from server
        self.local_model = self.local_comm.broadcast(self.local_model, src=0)
        # TODO: maybe require this to return the local model or perhaps a whole state object? (should we do the same to aggregate?)

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
            Configured PyTorch optimizer (e.g., SGD, Adam, AdamW)

        Example:
            return torch.optim.SGD(self.local_model.parameters(), lr=local_lr, momentum=0.9)
        """
        pass

    @abstractmethod
    def _train_step(self, batch: Any) -> tuple[torch.Tensor, int]:
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

        Example:
            inputs, targets = batch
            outputs = self.local_model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            return loss, inputs.size(0)
        """
        pass

    @abstractmethod
    def _aggregate(self) -> None:
        """
        Perform local aggregation within the group (REQUIRED override).

        This method is called after local training when aggregation should occur
        (controlled by agg_level and agg_freq). Use self.local_comm for communication.

        Common patterns:
        - FedAvg: self.local_model = self.local_comm.aggregate(self.local_model, ReductionType.MEAN)
        - Custom: Aggregate gradients, apply server-side optimization, broadcast result

        Available communication operations:
        - self.local_comm.aggregate(model, ReductionType.MEAN/SUM): All-reduce aggregation
        - self.local_comm.broadcast(model, src=0): Broadcast from specific rank
        - self.local_comm.all_gather(tensors): Gather tensors from all ranks

        Example:
            # Simple FedAvg aggregation
            self.local_model = self.local_comm.aggregate(self.local_model, ReductionType.MEAN)
        """
        pass

    def _perform_aggregation(self) -> None:
        """
        Coordinate model aggregation across local groups and (optionally) between groups.

        This gets called from three places depending on agg_level:
        - ROUND: After complete federated rounds (typical case)
        - EPOCH: After local training epochs
        - ITER: After individual training batches

        The flow works in phases:
        1. Everyone does local aggregation within their group
        2. Group servers (rank=0 nodes with global_comm) coordinate between groups
        3. All nodes participate in final broadcast to avoid race conditions

        The race condition fix in phase 3 is critical - without it, clients finish
        their local work and disconnect before servers complete inter-group coordination,
        causing broadcast timeouts. Making everyone participate prevents this.

        Note: Only group servers have global_comm set. Regular clients have global_comm=None
        and only participate in local aggregation + receiving the final result.

        Multi-group coordination requires consistent agg_freq across all groups.
        """
        # Phase 1: Local aggregation
        self._aggregate()

        # Phase 2: Inter-group coordination (only nodes with global_comm participate)
        if self.global_comm is not None:
            print(
                f"[AGG-INTER-START] Round {self.round_idx + 1} | Server initiating inter-group aggregation",
                flush=True,
            )

            # Exchange and combine models with other group servers
            self.local_model = self.global_comm.aggregate(
                self.local_model,
                reduction=ReductionType.SUM,
            )

            print(
                f"[AGG-INTER-DONE] Round {self.round_idx + 1} | Inter-group coordination complete",
                flush=True,
            )

        # Phase 3: Final broadcast - all nodes participate to prevent race conditions
        # TODO: This broadcast is unnecessary overhead for single-group topologies since
        # local aggregation already synchronized the model. Should only do this extra
        # broadcast step when we actually have multi-group coordination happening.
        node_role = "Server" if self.global_comm is not None else "Client"
        print(
            f"[AGG-FINAL-START] Round {self.round_idx + 1} | {node_role} entering final broadcast phase",
            flush=True,
        )
        self.local_model = self.local_comm.broadcast(self.local_model, src=0)
        print(
            f"[AGG-FINAL-DONE] Round {self.round_idx + 1} | {node_role} {'broadcast' if self.global_comm is not None else 'received'} final model",
            flush=True,
        )

    # =============================================================================
    # =============================================================================

    def __reset_round_state(
        self, round_idx: int, dataloader: Optional[DataLoader]
    ) -> None:
        # Discover distributed training parameters for synchronized loops
        self._local_iters_per_epoch = len(dataloader) if dataloader is not None else 0

        # Find global maximum iterations and epochs (batched for efficiency)
        discovery_tensor = torch.tensor(
            [self._local_iters_per_epoch, self.max_epochs_per_round], dtype=torch.int
        )
        global_maxes = self.local_comm.aggregate(discovery_tensor, ReductionType.MAX)

        self._global_max_iters_per_epoch = int(global_maxes[0].item())
        self._global_max_epochs_per_round = int(global_maxes[1].item())

        # ---
        # Create new instances for this round
        self.__local_optimizer = self._configure_local_optimizer(self.local_lr)
        self.__round_metrics = RoundMetrics()
        # ---
        self.round_idx = round_idx
        self.epoch_idx = 0
        self.batch_idx = 0
        # ---
        self.local_sample_count = 0

    def round_exec(self, datamodule: DataModule, round_idx: int) -> dict[str, float]:
        """
        Execute federated round computation across multiple epochs.

        DEFAULT: Sequential epoch processing with automatic timing and metrics.

        Override Use-Cases:
        - Early stopping: break epoch loop based on validation metrics
        - Dynamic epochs: adjust epoch count based on convergence
        - Learning rate schedules: step schedulers between epochs
        - Cross-epoch state: maintain state across epochs (e.g., momentum buffers)
        """
        # Future: could select datamodule.val or datamodule.test based on phase
        dataloader = datamodule.train

        _t_start = time.time()
        self.__reset_round_state(round_idx, dataloader)

        print(
            f"[START] Round {round_idx + 1} | "
            f"global_max_epochs_per_round={self.global_max_epochs_per_round} | "
            f"global_max_iters_per_epoch={self.global_max_iters_per_epoch}",
            flush=True,
        )
        # Overridable hook for algorithm-specific logic
        self._round_start()

        # ---
        # All nodes enter synchronized epoch loop structure
        self.local_model.train()  # Future: .eval() for val/test

        for epoch_idx in range(self.global_max_epochs_per_round):
            self.__run_epoch(
                epoch_idx,
                dataloader,
            )

        # ---
        # Check if aggregation should occur based on level & frequency
        if self.agg_level == AggLevel.ROUND:
            if self.round_idx % self.agg_freq == 0:
                print(
                    f"[AGG-TRIGGER] Round {self.round_idx + 1} | Level=ROUND | Freq={self.agg_freq} | Executing aggregation",
                    flush=True,
                )
                self._perform_aggregation()
            else:
                next_agg = ((self.round_idx // self.agg_freq) + 1) * self.agg_freq + 1
                print(
                    f"[AGG-SKIP] Round {self.round_idx + 1} | Level=ROUND | Freq={self.agg_freq} | Next aggregation at round {next_agg}",
                    flush=True,
                )

        # Overridable hook for algorithm-specific logic
        self._round_end()
        _t_end = time.time()
        self.round_metrics.update_mean("time/round", _t_end - _t_start)

        print(
            f"[END] Round {self.round_idx + 1} |",
            {
                k: round(v, 4)
                if isinstance(v, float) and k.startswith("time/")
                else round(v, 2)
                if isinstance(v, float)
                else v
                for k, v in self.round_metrics.to_dict().items()
            },
            flush=True,
        )

        return self.round_metrics.to_dict()

    def __run_epoch(
        self,
        epoch_idx: int,
        dataloader: Optional[DataLoader],
    ) -> None:
        """
        PRIVATE internal method to run a single epoch.
        """
        self.epoch_idx = epoch_idx

        print(
            f"[START] Round {self.round_idx + 1} Epoch {epoch_idx + 1}",
            flush=True,
        )

        _t_epoch_start = time.time()

        # Overridable hook for algorithm-specific logic
        self._epoch_start()

        # Initialize dataloader iterator for sequential batch processing
        dataloader_iter = iter(dataloader or [])

        # All nodes participate in synchronized batch loop
        for batch_idx in range(self.global_max_iters_per_epoch):
            # Set batch index and start batch processing
            self.batch_idx = batch_idx
            _t_batch_start = time.time()
            # Overridable hook for algorithm-specific logic
            self._batch_start()

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
                self._train_batch(batch)
            _t_batch_compute_end = time.time()

            # Batch-level aggregation
            if self.agg_level == AggLevel.ITER:
                if self.batch_idx % self.agg_freq == 0:
                    print(
                        f"[AGG-TRIGGER] Round {self.round_idx + 1} Epoch {self.epoch_idx + 1} Batch {self.batch_idx + 1} | Level=ITER | Freq={self.agg_freq} | Executing aggregation",
                        flush=True,
                    )
                    self._perform_aggregation()
                else:
                    next_agg = (
                        (self.batch_idx // self.agg_freq) + 1
                    ) * self.agg_freq + 1
                    print(
                        f"[AGG-SKIP] Round {self.round_idx + 1} Epoch {self.epoch_idx + 1} Batch {self.batch_idx + 1} | Level=ITER | Freq={self.agg_freq} | Next aggregation at batch {next_agg}",
                        flush=True,
                    )

            # Overridable hook for algorithm-specific logic
            self._batch_end()

            # Update timing metrics
            _t_batch_end = time.time()

            self.round_metrics.update_mean(
                "time/step_data", _t_batch_data_end - _t_batch_data_start
            )
            self.round_metrics.update_mean(
                "time/step_compute", _t_batch_compute_end - _t_batch_compute_start
            )

            # Overall batch timing for all nodes
            self.round_metrics.update_mean("time/step", _t_batch_end - _t_batch_start)

        # ---
        # Epoch boundary synchronization
        print("[EPOCH-SYNC-START]", flush=True)
        _t_start = time.time()
        sync_signal = torch.tensor([1.0])
        total_signals = self.local_comm.aggregate(sync_signal, ReductionType.SUM)
        sync_time = time.time() - _t_start
        print(
            f"[EPOCH-SYNC-END] time={sync_time:.4f}s signals={int(total_signals.item())}",
            flush=True,
        )

        # Epoch-level aggregation
        if self.agg_level == AggLevel.EPOCH:
            if self.epoch_idx % self.agg_freq == 0:
                print(
                    f"[AGG-TRIGGER] Round {self.round_idx + 1} Epoch {self.epoch_idx + 1} | Level=EPOCH | Freq={self.agg_freq} | Samples={self.local_sample_count} | Executing aggregation",
                    flush=True,
                )
                _t_start = time.time()
                self._perform_aggregation()
                agg_time = time.time() - _t_start
                print(
                    f"[AGG-COMPLETE] Round {self.round_idx + 1} Epoch {self.epoch_idx + 1} | Level=EPOCH | Duration={agg_time:.4f}s",
                    flush=True,
                )
            else:
                next_agg = ((self.epoch_idx // self.agg_freq) + 1) * self.agg_freq + 1
                print(
                    f"[AGG-SKIP] Round {self.round_idx + 1} Epoch {self.epoch_idx + 1} | Level=EPOCH | Freq={self.agg_freq} | Next aggregation at epoch {next_agg}",
                    flush=True,
                )

        # Overridable hook for algorithm-specific logic
        self._epoch_end()
        _t_epoch_end = time.time()

        self.round_metrics.update_mean("time/epoch", _t_epoch_end - _t_epoch_start)

        print(
            f"[END] Round {self.round_idx + 1} Epoch {epoch_idx + 1} |",
            {
                k: round(v, 4)
                if isinstance(v, float) and k.startswith("time/")
                else round(v, 2)
                if isinstance(v, float)
                else v
                for k, v in self.round_metrics.to_dict().items()
            },
            flush=True,
        )

        # Log epoch metrics to TensorBoard
        if self.tb_writer:
            for key, value in self.round_metrics.to_dict().items():
                self.tb_writer.add_scalar(key, value, self.tb_global_epoch)

    def _train_batch(self, batch: Any) -> None:
        """
        Execute computation for a single batch.

        Override Use-Cases:
        - Multiple optimizer steps: perform several gradient steps per batch
        - Custom backward passes: manual gradient computation or accumulation
        - Mixed precision: automatic or manual scaling for float16 training
        - Gradient clipping: apply gradient norm clipping before _optimizer_step
        """
        # Forward pass hook (implemented by subclasses)
        loss, batch_size = self._train_step(batch)
        # Automatic sample tracking
        self.local_sample_count = self.local_sample_count + batch_size
        self.round_metrics.update_sum("local/sample_count", batch_size)
        self.round_metrics.update_sum("local/batch_count", 1)

        # Automatic loss tracking
        self.round_metrics.update_mean("local/loss", loss.detach().item(), batch_size)

        self.local_optimizer.zero_grad()
        # Backward pass hook
        self._backward_pass(loss)

        # Automatic gradient tracking
        self.round_metrics.update_mean(
            "local/grad_norm", utils.get_grad_norm(self.local_model)
        )

        # Optimizer step hook
        self._optimizer_step()

    # =============================================================================

    def _backward_pass(self, loss: torch.Tensor) -> None:
        """
        Compute gradients from loss tensor.

        DEFAULT: Standard loss.backward() for automatic differentiation.

        Override Use-Cases:
        - Manual gradients: torch.autograd.grad() for specific parameters
        - Gradient penalty: add regularization terms during backward pass
        - Mixed precision: scale loss before backward pass
        """
        loss.backward()

    def _optimizer_step(self) -> None:
        """
        Apply parameter updates using computed gradients.

        DEFAULT: Standard optimizer.step() with current gradients.

        Override Use-Cases:
        - Conditional updates: skip updates based on gradient norms or loss values
        - Per-parameter learning rates: apply different step sizes to different layers
        - Momentum modifications: adjust momentum based on training progress
        """
        self.local_optimizer.step()

    # =============================================================================
    # LIFECYCLE HOOKS
    # =============================================================================

    def _round_start(self) -> None:
        """
        Algorithm-specific round start hook. Override for model sync and state reset.
        """
        pass

    def _round_end(self) -> None:
        """
        Algorithm-specific round end hook.

        Called after framework aggregation logic. Use for:
        - Custom post-training processing: model validation, state finalization
        - Algorithm-specific aggregation enhancements: custom model updates
        - Round metrics collection: algorithm-specific statistics gathering

        Example:
            def _round_end(self) -> None:
                self.compute_algorithm_metrics()
                self.save_round_checkpoint()
        """
        pass

    def _epoch_start(self) -> None:
        """
        Algorithm-specific epoch start hook.

        Called at the start of each local training epoch. Use for:
        - Learning rate scheduling: step LR scheduler based on epoch progress
        - Epoch state reset: clear per-epoch counters or loss accumulators
        - Dynamic configuration: adjust dropout rates or data augmentation per epoch

        Example:
            def _epoch_start(self) -> None:
                self.scheduler.step()
                self.epoch_loss_accumulator = 0.0
        """
        pass

    def _epoch_end(self) -> None:
        """
        Algorithm-specific epoch end hook.

        Called after framework aggregation logic. Use for:
        - Local validation: evaluate model on local validation set
        - Epoch metrics: compute and log per-epoch training statistics
        - Early stopping: check local convergence criteria

        Example:
            def _epoch_end(self) -> None:
                val_loss = self.validate_local_model()
                self.metrics.update("validation/loss", val_loss)
        """
        pass

    def _batch_start(self) -> None:
        """
        Algorithm-specific batch start hook.

        Called before processing each batch. Use for:
        - Optimizer state: modify learning rates or momentum per batch
        - Batch tracking: initialize batch-specific counters or flags
        - Debug logging: log batch indices or data samples for debugging

        Example:
            def _batch_start(self) -> None:
                if self.batch_idx % 100 == 0:
                    self.adjust_learning_rate(self.batch_idx)
        """
        pass

    def _batch_end(self) -> None:
        """
        Algorithm-specific batch end hook.

        Called after framework aggregation logic. Use for:
        - Batch metrics: log loss values or gradient norms per batch
        - Memory cleanup: clear temporary tensors or cached computations
        - Progress tracking: update training progress indicators

        Example:
            def _batch_end(self) -> None:
                if self.batch_idx % 50 == 0:
                    self.log_batch_metrics(self.batch_idx)
                torch.cuda.empty_cache()  # Memory cleanup
        """
        pass

    # =============================================================================
    # MISC UTILITY METHODS
    # =============================================================================

    def _transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
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
            f"Batch type '{type(batch).__name__}' not handled by _transfer_batch_to_device(). "
            "Override this method for custom batch formats."
        )
        return batch
