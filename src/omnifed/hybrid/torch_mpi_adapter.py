"""``TorchMPICommunicator`` wrapped as :class:`omnifed.communicator.BaseCommunicator`."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from src.flora.communicator.torch_mpi import TorchMPICommunicator
from src.omnifed.communicator.base import AggregationOp, BaseCommunicator

__all__ = ["TorchMPIAdapter"]


def _all_reduce_tensor(t: torch.Tensor, reduction: AggregationOp, world_size: int) -> torch.Tensor:
    if reduction == AggregationOp.SUM:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    elif reduction == AggregationOp.MAX:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    elif reduction == AggregationOp.MEAN:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= max(world_size, 1)
    else:
        raise ValueError(f"Unsupported reduction {reduction}")
    return t


class TorchMPIAdapter(BaseCommunicator):
    """
    Uses an already-initialized :class:`TorchMPICommunicator` process group
    (facility-local ranks).
    """

    def __init__(
        self,
        mpi: TorchMPICommunicator,
        *,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ) -> None:
        super().__init__(rank, world_size, master_addr, int(master_port))
        self._mpi = mpi

    def _setup(self) -> None:
        return

    def broadcast(self, msg: BaseCommunicator.MsgT, src: int = 0) -> BaseCommunicator.MsgT:
        if isinstance(msg, nn.Module):
            for _, p in msg.named_parameters():
                if p.requires_grad:
                    dist.broadcast(p.data, src=src)
            for _, buffer in msg.named_buffers():
                if buffer is None or not buffer.dtype.is_floating_point:
                    continue
                dist.broadcast(buffer.data, src=src)
        elif isinstance(msg, dict):
            for tensor in msg.values():
                dist.broadcast(tensor, src=src)
        else:
            dist.broadcast(msg, src=src)
        return msg

    def aggregate(
        self,
        msg: BaseCommunicator.MsgT,
        reduction: AggregationOp,
    ) -> BaseCommunicator.MsgT:
        if isinstance(msg, nn.Module):
            if reduction == AggregationOp.SUM:
                red_op = dist.ReduceOp.SUM
            elif reduction == AggregationOp.MAX:
                red_op = dist.ReduceOp.MAX
            elif reduction == AggregationOp.MEAN:
                red_op = getattr(dist.ReduceOp, "AVG", dist.ReduceOp.SUM)
            else:
                raise ValueError(f"Unsupported reduction {reduction}")

            manual_mean = reduction == AggregationOp.MEAN and not hasattr(
                dist.ReduceOp, "AVG"
            )

            for _, p in msg.named_parameters():
                if p.requires_grad:
                    dist.all_reduce(p.data, op=red_op)
                    if manual_mean:
                        p.data /= max(self.world_size, 1)
            for _, buffer in msg.named_buffers():
                if buffer is None or not buffer.dtype.is_floating_point:
                    continue
                dist.all_reduce(buffer.data, op=red_op)
                if manual_mean:
                    buffer.data /= max(self.world_size, 1)
        elif isinstance(msg, dict):
            for t in msg.values():
                _all_reduce_tensor(t, reduction, self.world_size)
        else:
            _all_reduce_tensor(msg, reduction, self.world_size)

        return msg

    def close(self) -> None:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
