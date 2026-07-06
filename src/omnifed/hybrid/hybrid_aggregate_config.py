# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""Hybrid Slurm aggregate payload: sample-weighted parameters vs gradients (both hops)."""

from __future__ import annotations

from typing import Literal, Union

from omegaconf import DictConfig, OmegaConf

AggregatePayload = Literal["params", "gradients"]

AGGREGATE_PAYLOAD_PARAMS: AggregatePayload = "params"
AGGREGATE_PAYLOAD_GRADIENTS: AggregatePayload = "gradients"
_VALID_PAYLOADS = frozenset({AGGREGATE_PAYLOAD_PARAMS, AGGREGATE_PAYLOAD_GRADIENTS})


def normalize_aggregate_payload(raw: object) -> AggregatePayload:
    """Parse Hydra value; default ``params`` (current FedAvg weight averaging)."""
    if raw is None:
        return AGGREGATE_PAYLOAD_PARAMS
    key = str(raw).strip().lower()
    if key in ("param", "params", "parameters", "weights", "weight"):
        return AGGREGATE_PAYLOAD_PARAMS
    if key in ("grad", "grads", "gradient", "gradients"):
        return AGGREGATE_PAYLOAD_GRADIENTS
    raise ValueError(
        f"engine.hybrid.aggregate_payload must be 'params' or 'gradients', got {raw!r}"
    )


def hybrid_aggregate_payload_from_cfg(cfg: Union[DictConfig, object]) -> AggregatePayload:
    raw = OmegaConf.select(cfg, "engine.hybrid.aggregate_payload", default="params")
    return normalize_aggregate_payload(raw)


def hybrid_communicate_params_from_cfg(cfg: Union[DictConfig, object]) -> bool:
    """``True`` → aggregate/broadcast model parameters; ``False`` → gradients."""
    return hybrid_aggregate_payload_from_cfg(cfg) == AGGREGATE_PAYLOAD_PARAMS


__all__ = [
    "AGGREGATE_PAYLOAD_GRADIENTS",
    "AGGREGATE_PAYLOAD_PARAMS",
    "AggregatePayload",
    "hybrid_aggregate_payload_from_cfg",
    "hybrid_communicate_params_from_cfg",
    "normalize_aggregate_payload",
]
