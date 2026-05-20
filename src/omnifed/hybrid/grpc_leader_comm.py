"""Facility leader ↔ central server via Flora gRPC (global federated step)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

from torch import nn

from src.flora.communicator import grpc_communicator as rpc_comm
from src.omnifed.communicator.base import AggregationOp, BaseCommunicator

if TYPE_CHECKING:
    from src.omnifed.hybrid.comm_bridge import HybridCommBridge


__all__ = ["GrpcLeaderCommunicator"]


class GrpcLeaderCommunicator(BaseCommunicator):
    """
    ``global_comm`` for hybrid Slurm: one round trip to the parameter server per call.

    Model aggregation uses ``batch_samples`` from :class:`HybridCommBridge` set during
    facility-local sync Phase 1 (group total sample count).
    """

    def __init__(
        self,
        *,
        bridge: "HybridCommBridge",
        global_rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        rpc_total_clients: int,
    ) -> None:
        super().__init__(global_rank, world_size, master_addr, int(master_port))
        self._bridge = bridge
        self._rpc_total = int(rpc_total_clients)
        self._grpc: Optional[rpc_comm.GrpcCommunicator] = None

    def _setup(self) -> None:
        return

    def attach_model(self, model: nn.Module) -> None:
        if self._grpc is not None:
            return
        self._grpc = rpc_comm.GrpcCommunicator(
            model=model,
            id=self.rank,
            total_clients=self._rpc_total,
            master_addr=self.master_addr,
            master_port=int(self.master_port),
            accumulate_updates=True,
            daemon_server=False,
        )

    def broadcast(self, msg: BaseCommunicator.MsgT, src: int = 0) -> BaseCommunicator.MsgT:
        raise NotImplementedError("Hybrid global path uses gRPC pull from server only.")

    def aggregate(
        self,
        msg: BaseCommunicator.MsgT,
        reduction: AggregationOp,
    ) -> BaseCommunicator.MsgT:
        if self._grpc is None:
            raise RuntimeError("GrpcLeaderCommunicator.attach_model() before aggregate().")
        if not isinstance(msg, nn.Module):
            raise NotImplementedError(
                "GrpcLeaderCommunicator only aggregates nn.Module (global FedAvg step)."
            )
        if reduction not in (AggregationOp.SUM, AggregationOp.MEAN):
            raise ValueError(f"gRPC global aggregate supports SUM/MEAN, got {reduction}")

        bs = int(self._bridge.last_group_total_samples)
        if bs < 1:
            warnings.warn(
                f"Hybrid gRPC aggregate with batch_samples={bs}; "
                "group total samples missing — using 1.",
                UserWarning,
            )
            bs = max(bs, 1)

        return self._grpc.aggregate(
            msg=msg,
            batch_samples=bs,
            communicate_params=True,
            compute_mean=True,
        )

    def close(self) -> None:
        if self._grpc is not None:
            self._grpc.grpc_shutdown()
            self._grpc = None
