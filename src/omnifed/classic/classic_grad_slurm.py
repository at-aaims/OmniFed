# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""Classic centralized Slurm: synchronous gradient aggregation (deferred optimizer.step)."""

from __future__ import annotations

import time
import types

import torch

from src.omnifed.algorithm import utils
from src.omnifed.communicator.base import AggregationOp
from src.omnifed.hybrid.hybrid_grad_training import (
    apply_optimizer_grads,
    clear_model_grads,
    grad_l2_norm,
    normalize_accumulated_grads,
    require_model_grads,
)
from src.omnifed.utils import print


def _ensure_model_grad_tensors(model) -> None:
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad and param.grad is None:
                param.grad = torch.zeros_like(param.data)


def install_classic_grad_slurm_sync(algorithm, *, local_comm) -> None:
    """
    Synchronous gradient FedAvg: backward locally, aggregate grads, then optimizer.step().

    Use with ``algorithm/schedules/aggregation: batch_end`` for per-minibatch sync.
    """
    algorithm._classic_communicate_params = False
    algorithm._classic_local_comm = local_comm

    base_round_start = algorithm._round_start
    base_round_end = getattr(algorithm, "_round_end", lambda self: None)

    def _round_start_grad(self) -> None:
        opt = self._BaseAlgorithm__local_optimizer
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        self._classic_batches_since_sync = 0
        base_round_start()

    def _round_end_with_eval(self) -> None:
        if getattr(self.datamodule, "eval", None) is not None:
            print(
                f"Round-end evaluation @ {self.progress_info_str}",
                flush=True,
            )
            self._BaseAlgorithm__eval_epoch(self.local_model)
        base_round_end()

    def _train_batch_grad(self, batch):
        loss = self._compute_loss(batch)
        self._backward_pass(loss)
        self._classic_batches_since_sync = (
            int(getattr(self, "_classic_batches_since_sync", 0)) + 1
        )
        metrics = {
            "loss": loss.detach().item(),
            "grad_norm": grad_l2_norm(self.local_model),
        }
        train_state = getattr(self, "_summary_iter_train", None)
        if train_state is not None:
            train_state.update(metrics)
        return metrics

    def _aggregate_within_group_sync_grads(self, comm, weight):
        nb = int(getattr(self, "_classic_batches_since_sync", 0))
        if nb < 1:
            _ensure_model_grad_tensors(self.local_model)
        else:
            normalize_accumulated_grads(self.local_model, nb)
            utils.scale_grads(self.local_model, weight)
        require_model_grads(self.local_model)
        _set_comm_sample_count(comm, int(self._BaseAlgorithm__num_samples_trained))
        comm.aggregate(self.local_model, AggregationOp.SUM)
        return self.local_model

    def _optimizer_step_deferred(self) -> None:
        return

    def _classic_grad__sync_comm(self) -> None:
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
        with self.track_model_operation("grpc_agg_grad"):
            self.local_model = self._aggregate_within_group(
                self.local_comm, within_group_weight
            )
        if sync_bucket is not None:
            sync_bucket["grpc_agg_grad_time"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        with self.track_model_operation("grad_apply"):
            opt = self._BaseAlgorithm__local_optimizer
            if opt is None:
                raise RuntimeError(
                    "Classic grad apply: optimizer missing before optimizer.step()."
                )
            apply_optimizer_grads(self.local_model, opt)
        if sync_bucket is not None:
            sync_bucket["grad_apply_time"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        with self.track_model_operation("grpc_agg_bn"):
            _aggregate_buffers_sample_weighted(
                self.local_model,
                self.local_comm,
                within_group_weight,
            )
        if sync_bucket is not None:
            sync_bucket["grpc_agg_bn_time"] = time.perf_counter() - t0

        # Backward-compatible aggregate for per-round summary (sum of gRPC sync ops).
        if sync_bucket is not None:
            local_agg_time = sum(
                float(sync_bucket.get(k, 0.0) or 0.0)
                for k in (
                    "grpc_agg_sample_time",
                    "grpc_agg_grad_time",
                    "grpc_agg_bn_time",
                )
            )
            self.log_metric("local_agg_time", local_agg_time)

        clear_model_grads(
            self.local_model,
            optimizer=self._BaseAlgorithm__local_optimizer,
        )
        self._classic_batches_since_sync = 0

    algorithm._round_start = types.MethodType(_round_start_grad, algorithm)
    algorithm._round_end = types.MethodType(_round_end_with_eval, algorithm)
    algorithm._train_batch = types.MethodType(_train_batch_grad, algorithm)
    algorithm._optimizer_step = types.MethodType(_optimizer_step_deferred, algorithm)
    algorithm._aggregate_within_group = types.MethodType(
        _aggregate_within_group_sync_grads, algorithm
    )
    algorithm._BaseAlgorithm__sync_comm = types.MethodType(
        _classic_grad__sync_comm, algorithm
    )

    print(
        "[classic] synchronous gradient aggregation (deferred optimizer.step)",
        flush=True,
    )


def _set_comm_sample_count(comm, num_samples: int) -> None:
    setter = getattr(comm, "set_aggregation_num_samples", None)
    if setter is not None:
        setter(max(int(num_samples), 0))


def _aggregate_buffers_sample_weighted(model, comm, weight: float) -> None:
    buf_dict: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, buf in model.named_buffers():
            if buf is None or not buf.dtype.is_floating_point:
                continue
            buf_dict[name] = buf.data.detach().clone().mul(weight)
    if not buf_dict:
        return
    _set_comm_sample_count(comm, 0)
    agg = comm.aggregate(buf_dict, AggregationOp.SUM)
    with torch.no_grad():
        for name, buf in model.named_buffers():
            if name in agg:
                buf.data.copy_(agg[name].to(buf.device))


__all__ = ["install_classic_grad_slurm_sync"]
