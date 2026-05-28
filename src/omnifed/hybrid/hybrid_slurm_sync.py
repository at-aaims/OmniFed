"""Patch FedAvg sync for hybrid Slurm (gRPC global step skips torch scalar reduce)."""

from __future__ import annotations

import types
import warnings

import torch

from src.omnifed.communicator.base import AggregationOp
from src.omnifed.hybrid.comm_bridge import HybridCommBridge
from src.omnifed.hybrid.grpc_leader_comm import GrpcLeaderCommunicator


def install_hybrid_slurm_sync(algorithm, bridge: HybridCommBridge) -> None:
    """
    Replace ``_BaseAlgorithm__sync_comm`` and optionally ``_aggregate_across_groups``
    when ``global_comm`` is :class:`GrpcLeaderCommunicator`.
    """
    algorithm._hybrid_bridge = bridge
    if getattr(algorithm, "global_comm", None) is not None and isinstance(
        algorithm.global_comm, GrpcLeaderCommunicator
    ):

        def _aggregate_across_grpc_slurm(self, comm, weight):
            del weight
            return comm.aggregate(self.local_model, AggregationOp.SUM)

        algorithm._aggregate_across_groups = types.MethodType(
            _aggregate_across_grpc_slurm, algorithm
        )

    algorithm._BaseAlgorithm__sync_comm = types.MethodType(
        _hybrid_slurm__sync_comm, algorithm
    )


def _hybrid_slurm__sync_comm(self) -> None:
    dev = next(self.local_model.parameters()).device
    with self.track_model_operation("local_agg"):
        group_total_samples = self.local_comm.aggregate(
            torch.tensor(
                [self._BaseAlgorithm__num_samples_trained],
                dtype=torch.float32,
                device=dev,
            ),
            reduction=AggregationOp.SUM,
        ).item()

        if group_total_samples == 0:
            warnings.warn(
                f"Zero samples trained across all nodes in group ({self.progress_info_str}). "
                "Check data availability or epoch scheduling. Using uniform weights for aggregation.",
                UserWarning,
            )

        within_group_weight = self._BaseAlgorithm__num_samples_trained / max(
            group_total_samples, 1
        )

        self.local_model = self._aggregate_within_group(
            self.local_comm, within_group_weight
        )

    br = getattr(self, "_hybrid_bridge", None)
    if br is not None:
        br.last_group_total_samples = int(group_total_samples)

    if self.global_comm is not None:
        with self.track_model_operation("global_agg"):
            if isinstance(self.global_comm, GrpcLeaderCommunicator):
                self.local_model = self._aggregate_across_groups(
                    self.global_comm, 0.0
                )
            else:
                global_total_samples = self.global_comm.aggregate(
                    torch.tensor([group_total_samples], dtype=torch.float32, device=dev),
                    reduction=AggregationOp.SUM,
                ).item()

                if global_total_samples == 0:
                    warnings.warn(
                        f"Zero samples trained across all groups globally ({self.progress_info_str}). "
                        "Check data availability or cross-group coordination. Using uniform weights for cross-group aggregation.",
                        UserWarning,
                    )

                across_group_weight = group_total_samples / max(global_total_samples, 1)

                self.local_model = self._aggregate_across_groups(
                    self.global_comm, across_group_weight
                )

    needs_final_bcast = (
        self.local_comm.aggregate(
            torch.tensor(1.0 if self.global_comm is not None else 0.0, device=dev),
            AggregationOp.MAX,
        )
        > 0
    )

    if needs_final_bcast:
        with self.track_model_operation("local_bcast"):
            self.local_model = self.local_comm.broadcast(self.local_model)
