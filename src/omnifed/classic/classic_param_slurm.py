# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""Classic centralized Slurm: synchronous parameter aggregation (optimizer.step before sync)."""

from __future__ import annotations

import time
import types

import torch

from src.omnifed.classic.classic_grad_slurm import _set_comm_sample_count
from src.omnifed.communicator.base import AggregationOp
from src.omnifed.utils import print


def install_classic_param_slurm_sync(algorithm, *, local_comm) -> None:
    """
    Synchronous param FedAvg: local optimizer.step, then aggregate model params.

    Per-iter CSV column mapping (shared header with grad runs):
    - ``grpc_agg_sample_s`` — sample-count SUM
    - ``grpc_agg_grad_s`` — param payload SUM (weights + BN buffers)
    - ``grpc_agg_bn_s`` / ``grad_apply_s`` — unused (empty) on param track

    Optional Top-K / QSGD: wire compressors on ``local_comm`` and call
    ``set_aggregation_num_samples`` before param aggregate (same gate as grad).
    BN buffers are still included in the payload until a dense-BN split lands.
    """
    algorithm._classic_communicate_params = True
    algorithm._classic_local_comm = local_comm

    base_round_end = getattr(algorithm, "_round_end", lambda self: None)
    base_train_batch = algorithm._train_batch

    def _round_end_with_eval(self) -> None:
        if getattr(self.datamodule, "eval", None) is not None:
            print(
                f"Round-end evaluation @ {self.progress_info_str}",
                flush=True,
            )
            self._BaseAlgorithm__eval_epoch(self.local_model)
        base_round_end()

    def _train_batch_param(self, batch):
        metrics = base_train_batch(batch)
        train_state = getattr(self, "_summary_iter_train", None)
        if train_state is not None:
            train_state.update(metrics)
        return metrics

    def _classic_param__sync_comm(self) -> None:
        dev = next(self.local_model.parameters()).device
        sync_bucket = getattr(self, "_summary_iter_sync", None)

        t0 = time.perf_counter()
        with self.track_model_operation("grpc_agg_sample"):
            _set_comm_sample_count(self.local_comm, 0)
            group_total_samples = self.local_comm.aggregate(
                torch.tensor(
                    [self._BaseAlgorithm__num_samples_trained],
                    dtype=torch.float32,
                    device=dev,
                ),
                reduction=AggregationOp.SUM,
            ).item()
        if sync_bucket is not None:
            sync_bucket["grpc_agg_sample_time"] = time.perf_counter() - t0

        within_group_weight = self._BaseAlgorithm__num_samples_trained / max(
            group_total_samples, 1
        )

        t0 = time.perf_counter()
        with self.track_model_operation("grpc_agg_param"):
            _set_comm_sample_count(
                self.local_comm, int(self._BaseAlgorithm__num_samples_trained)
            )
            self.local_model = self._aggregate_within_group(
                self.local_comm, within_group_weight
            )
        if sync_bucket is not None:
            # Reuse grad column for param payload timing (see module docstring).
            sync_bucket["grpc_agg_grad_time"] = time.perf_counter() - t0

        if sync_bucket is not None:
            local_agg_time = sum(
                float(sync_bucket.get(k, 0.0) or 0.0)
                for k in ("grpc_agg_sample_time", "grpc_agg_grad_time")
            )
            self.log_metric("local_agg_time", local_agg_time)

    algorithm._round_end = types.MethodType(_round_end_with_eval, algorithm)
    algorithm._train_batch = types.MethodType(_train_batch_param, algorithm)
    algorithm._BaseAlgorithm__sync_comm = types.MethodType(
        _classic_param__sync_comm, algorithm
    )

    print(
        "[classic] synchronous parameter aggregation (optimizer.step before sync)",
        flush=True,
    )


__all__ = ["install_classic_param_slurm_sync"]
