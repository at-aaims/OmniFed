"""Patch FedAvg sync for hybrid Slurm (gRPC global step skips torch scalar reduce)."""

from __future__ import annotations

import types
import warnings

import torch

from src.omnifed.algorithm import utils
from src.omnifed.communicator.base import AggregationOp
from src.omnifed.hybrid.comm_bridge import HybridCommBridge
from src.omnifed.hybrid.grpc_leader_comm import GrpcLeaderCommunicator
from src.omnifed.hybrid.hybrid_grad_training import (
    apply_optimizer_grads,
    clear_model_grads,
    grad_l2_norm,
    require_model_grads,
)


def install_hybrid_slurm_sync(
    algorithm,
    bridge: HybridCommBridge,
    *,
    communicate_params: bool = True,
) -> None:
    """
    Replace ``_BaseAlgorithm__sync_comm`` and optionally ``_aggregate_across_groups``
    when ``global_comm`` is :class:`GrpcLeaderCommunicator`.

    ``communicate_params=False`` switches facility-local aggregation to sample-weighted
    gradient all-reduce (both hops use gradients) and defers ``optimizer.step`` until
    after global/local aggregation completes.
    """
    algorithm._hybrid_bridge = bridge
    algorithm._hybrid_communicate_params = bool(communicate_params)

    if not communicate_params:
        prior_round_start = algorithm._round_start

        def _round_start_grad_accum(self) -> None:
            opt = self._BaseAlgorithm__local_optimizer
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            prior_round_start()

        def _train_batch_grad_accum(self, batch):
            loss = self._compute_loss(batch)
            self._backward_pass(loss)
            return {
                "loss": loss.detach().item(),
                "grad_norm": grad_l2_norm(self.local_model),
            }

        def _optimizer_step_deferred(self) -> None:
            return

        def _aggregate_within_group_grads(self, comm, weight):
            utils.scale_grads(self.local_model, weight)
            require_model_grads(self.local_model)
            comm.aggregate(self.local_model, AggregationOp.SUM)
            return self.local_model

        algorithm._round_start = types.MethodType(_round_start_grad_accum, algorithm)
        algorithm._train_batch = types.MethodType(_train_batch_grad_accum, algorithm)
        algorithm._optimizer_step = types.MethodType(_optimizer_step_deferred, algorithm)
        algorithm._aggregate_within_group = types.MethodType(
            _aggregate_within_group_grads, algorithm
        )

    if getattr(algorithm, "global_comm", None) is not None and isinstance(
        algorithm.global_comm, GrpcLeaderCommunicator
    ):

        def _aggregate_across_grpc_slurm(self, comm, weight):
            del weight
            return comm.aggregate(self.local_model, AggregationOp.SUM)

        algorithm._aggregate_across_groups = types.MethodType(
            _aggregate_across_grpc_slurm, algorithm
        )
    elif not communicate_params:

        def _aggregate_across_groups_grads(self, comm, weight):
            utils.scale_grads(self.local_model, weight)
            require_model_grads(self.local_model)
            comm.aggregate(self.local_model, AggregationOp.SUM)
            return self.local_model

        algorithm._aggregate_across_groups = types.MethodType(
            _aggregate_across_groups_grads, algorithm
        )

    algorithm._BaseAlgorithm__sync_comm = types.MethodType(
        _hybrid_slurm__sync_comm, algorithm
    )


def _should_apply_grad_after_agg(self, *, needs_final_bcast: bool) -> bool:
    """Leaders apply after global agg; local-only groups apply when no broadcast follows."""
    if getattr(self, "_hybrid_communicate_params", True):
        return False
    if self.global_comm is not None:
        return True
    return not needs_final_bcast


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

    if _should_apply_grad_after_agg(self, needs_final_bcast=needs_final_bcast):
        with self.track_model_operation("grad_apply"):
            opt = self._BaseAlgorithm__local_optimizer
            if opt is None:
                raise RuntimeError(
                    "Hybrid grad apply: optimizer missing before optimizer.step()."
                )
            apply_optimizer_grads(self.local_model, opt)

    if needs_final_bcast:
        with self.track_model_operation("local_bcast"):
            self.local_model = self.local_comm.broadcast(self.local_model)

    if not getattr(self, "_hybrid_communicate_params", True):
        clear_model_grads(
            self.local_model,
            optimizer=self._BaseAlgorithm__local_optimizer,
        )
