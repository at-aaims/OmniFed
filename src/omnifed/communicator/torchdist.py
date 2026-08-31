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

import datetime
import warnings
from enum import Enum

import rich.repr
import torch
import torch.distributed as dist
from torch import nn

from ..utils import print
from .base import AggregationOp, BaseCommunicator
from .compression import is_compressor, resolve_compressor
from .utils import get_msg_info

_GRPC_CFG_ALIASES = frozenset(
    {
        "client_compressor",
        "server_compressor",
        "client_timeout",
        "normalize_by_total_samples",
        "retry_delay",
    }
)


class InitMethod(str, Enum):
    """Initialization methods for PyTorch distributed process groups."""

    TCP = "tcp"  # TCP-based initialization for network communication
    FILE = "file"  # File-based initialization for shared filesystem


# ======================================================================================


@rich.repr.auto
class TorchDistCommunicator(BaseCommunicator):
    """
    Communication backend using PyTorch distributed collective operations.

    Implements broadcast and aggregation via PyTorch's efficient collective
    primitives (broadcast, all-reduce).
    Uses process groups for coordination with support for multiple backends
    (gloo, nccl) and initialization methods.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        init_method: InitMethod = InitMethod.TCP,
        backend: str = "gloo",
        sharedfile: str = "sharedfile",
        timeout: int = 600,
        max_retries: int = 5,
        communicate_params: bool = True,
        compressor=None,
        client_compressor=None,
        server_compressor=None,
        **kwargs: object,
    ) -> None:
        """
        Initialize PyTorch distributed communicator.

        Args:
            rank: Process rank in distributed group
            world_size: Total number of processes in distributed group
            master_addr: Master node address for coordination
            master_port: Master node port for coordination
            init_method: TCP or file-based process group initialization
            backend: Communication backend ('gloo' for CPU, 'nccl' for GPU)
            sharedfile: Shared file path for file-based initialization
            timeout: Process group initialization timeout (seconds)
            max_retries: Maximum initialization retry attempts
        """
        unused = {k: v for k, v in kwargs.items() if k not in _GRPC_CFG_ALIASES}
        if unused:
            warnings.warn(
                f"TorchDistCommunicator ignoring unused keyword arguments: {sorted(unused)}",
                stacklevel=2,
            )
        super().__init__(rank, world_size, master_addr, master_port)
        self.communicate_params = bool(communicate_params)
        self.compressor = resolve_compressor(
            compressor, client_compressor, server_compressor
        )
        self._aggregation_num_samples = 0
        self.logger = None
        compressor_name = (
            type(self.compressor).__name__ if self.compressor is not None else "none"
        )
        print(
            f"rank={rank}/{world_size} | backend={backend} | addr={master_addr}:{master_port} | "
            f"communicate_params={self.communicate_params} | compressor={compressor_name}"
        )

        # Core distributed parameters
        # self.rank: int = rank
        # self.world_size: int = world_size
        # self.master_addr: str = master_addr
        # self.master_port: int = master_port

        # ---
        self.init_method: str = init_method
        self.backend: str = backend
        #self.backend: str = "nccl"
        self.sharedfile: str = sharedfile
        self.timeout: datetime.timedelta = datetime.timedelta(seconds=timeout)
        self.max_retries: int = max_retries
        print(f"Inside torchdist communicator init {self.backend}")

        # Backend validation with automatic fallback
        if self.backend == "nccl" and not torch.cuda.is_available():
            print("NCCL→gloo fallback (no CUDA available)")
            self.backend = "gloo"

    def _setup(self):
        """
        Initialize PyTorch distributed process group.

        Creates the distributed process group using the specified initialization
        method and backend.

        All ranks must call this before any collective operations.
        """

        print(
            f"rank={self.rank}/{self.world_size} | {self.init_method} | backend={self.backend}"
        )

        match self.init_method:
            case InitMethod.TCP:
                addr = f"tcp://{self.master_addr}:{self.master_port}"
                dist.init_process_group(
                    backend=self.backend,
                    init_method=addr,
                    rank=self.rank,
                    world_size=self.world_size,
                    timeout=self.timeout,
                )
            case InitMethod.FILE:
                addr = f"file://{self.sharedfile}"
                dist.init_process_group(
                    backend=self.backend,
                    init_method=addr,
                    rank=self.rank,
                    world_size=self.world_size,
                    timeout=self.timeout,
                )
            case _:
                raise ValueError(
                    f"Unknown init_method: {self.init_method}. Supported: {[m.value for m in InitMethod]}"
                )
        
        print(f"[TorchDistCommunicator->_setup] init complete rank={self.rank}")
        dist.barrier()
        print(f"[TorchDistCommunicator->_setup] barrier complete rank={self.rank}")

    def set_aggregation_num_samples(self, num_samples: int) -> None:
        """API parity with GrpcCommunicator (controls when compression applies)."""
        self._aggregation_num_samples = max(int(num_samples), 0)

    def set_logger(self, logger) -> None:
        """Forward metric logger for per-iteration compress/decompress timings."""
        self.logger = logger

    def _active_compressor(self):
        # Dense when no compressor, or sample/BN phases (num_samples=0).
        # Grad (communicate_params=False) and param (True) both use compression when set.
        if self.compressor is None or self._aggregation_num_samples <= 0:
            return None
        return self.compressor

    def _aggregate_tensor(self, tensor: torch.Tensor, *, name: str, op: dist.ReduceOp) -> torch.Tensor:
        active = self._active_compressor()
        if active is None:
            dist.all_reduce(tensor, op=op)
            return tensor
        if not is_compressor(active):
            raise TypeError(
                f"TorchDist compressor must extend Compression, got {type(active)!r}"
            )
        return active.aggregate_torchdist(
            tensor,
            name=name,
            world_size=self.world_size,
            op=op,
            logger=self.logger,
        )

    @staticmethod
    def _module_aggregate_tensor(param: nn.Parameter, *, communicate_params: bool) -> torch.Tensor:
        if communicate_params:
            return param.data
        if param.grad is not None:
            return param.grad
        return torch.zeros_like(param.data)

    @staticmethod
    def _apply_module_aggregate_tensor(
        param: nn.Parameter, tensor: torch.Tensor, *, communicate_params: bool
    ) -> None:
        if communicate_params:
            param.data.copy_(tensor.to(param.device))
        else:
            param.grad = tensor.to(param.device)

    def _all_reduce_module(self, msg: nn.Module, op: dist.ReduceOp) -> None:
        with torch.no_grad():
            for pname, param in msg.named_parameters():
                if not param.requires_grad:
                    continue
                tensor = self._module_aggregate_tensor(
                    param, communicate_params=self.communicate_params
                )
                reduced = self._aggregate_tensor(tensor, name=pname, op=op)
                self._apply_module_aggregate_tensor(
                    param, reduced, communicate_params=self.communicate_params
                )
            if self.communicate_params:
                for name, buffer in msg.named_buffers():
                    if buffer is None:  # type: ignore
                        warnings.warn(f"Buffer '{name}' is None, skipping aggregation")
                        continue
                    if not buffer.dtype.is_floating_point:
                        continue
                    reduced = self._aggregate_tensor(buffer.data, name=name, op=op)
                    buffer.data.copy_(reduced.to(buffer.device))

    def broadcast(
        self,
        msg: BaseCommunicator.MsgT,
        src: int = 0,
        # TODO: Consider adding control arguments like utils.scale_params:
        # requires_grad: Optional[bool] = None,
        # include_buffers: bool = True,
        # filter_fn: Optional[Callable[[str, torch.Tensor], bool]] = None
    ) -> BaseCommunicator.MsgT:
        """
        Broadcast message from source rank to all other ranks.

        Uses PyTorch's distributed broadcast collective for efficient
        one-to-many communication within the process group.

        Args:
            msg: Model, tensor dict, or tensor to broadcast
            src: Source rank ID (default: 0)

        Returns:
            Message with broadcasted values
        """

        print(f"Inside broadcast function of torchdist rank {self.rank}")
        print(f"{get_msg_info(msg)} | src={src}")

        if isinstance(msg, nn.Module):
            # Broadcast all trainable parameters
            for _, p in msg.named_parameters():
                # Only broadcast trainable parameters (frozen params stay local)
                if p.requires_grad:
                    dist.broadcast(p.data, src=src)
            # Broadcast all buffers (batch norm stats, etc.) - only floating-point buffers
            for name, buffer in msg.named_buffers():
                if buffer is None:  # type: ignore
                    warnings.warn(f"Buffer '{name}' is None, skipping broadcast")
                    continue
                if not buffer.dtype.is_floating_point:
                    continue  # Skip integer buffers like num_batches_tracked
                dist.broadcast(buffer.data, src=src)
        elif isinstance(msg, dict):
            # Broadcast each tensor in dictionary
            for tensor in msg.values():
                dist.broadcast(tensor, src=src)
        else:
            # Broadcast single tensor
            dist.broadcast(msg, src=src)
        return msg

    def aggregate(
        self,
        msg: BaseCommunicator.MsgT,
        reduction: AggregationOp,
        # TODO: Consider adding control arguments like utils.scale_params:
        # requires_grad: Optional[bool] = None,
        # include_buffers: bool = True,
        # filter_fn: Optional[Callable[[str, torch.Tensor], bool]] = None
    ) -> BaseCommunicator.MsgT:
        """
        Aggregate message across all ranks using PyTorch all-reduce collective.

        Performs efficient element-wise reduction across all process ranks.

        Args:
            msg: Model, tensor dict, or tensor to aggregate
            reduction: SUM, MEAN, or MAX reduction operation

        Returns:
            Message with aggregated values distributed to all ranks
        """

        print(f"Inside aggregate function of torchdist rank {self.rank}")
        print(f"{get_msg_info(msg)} | reduction={reduction}")

        # Map reduction type to PyTorch operation
        reduction_ops = {
            AggregationOp.SUM: dist.ReduceOp.SUM,
            AggregationOp.MEAN: dist.ReduceOp.AVG,
            AggregationOp.MAX: dist.ReduceOp.MAX,
        }

        if reduction not in reduction_ops:
            raise ValueError(f"Unsupported reduction type: {reduction}")

        op = reduction_ops[reduction]

        print(f"[aggregate] BEFORE all_reduce rank={self.rank}")


        if isinstance(msg, nn.Module):
            self._all_reduce_module(msg, op)
        elif isinstance(msg, dict):
            for key, tensor in msg.items():
                reduced = self._aggregate_tensor(tensor, name=str(key), op=op)
                msg[key] = reduced
        else:
            msg = self._aggregate_tensor(msg, name="tensor", op=op)

        print(f"[aggregate] AFTER all_reduce rank={self.rank}")

        return msg

    def close(self):
        """
        Destroy PyTorch distributed process group and clean up resources.

        Should be called when distributed communication is no longer needed.
        All ranks must call this to properly clean up the process group.
        """
        print()
        dist.destroy_process_group()
