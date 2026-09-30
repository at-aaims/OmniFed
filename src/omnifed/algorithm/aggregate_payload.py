# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""Shared params vs gradients payload flag (1-GPU and hierarchical)."""

from __future__ import annotations

from typing import Literal, Union

from omegaconf import DictConfig, OmegaConf

AggregatePayload = Literal["params", "gradients"]

AGGREGATE_PAYLOAD_PARAMS: AggregatePayload = "params"
AGGREGATE_PAYLOAD_GRADIENTS: AggregatePayload = "gradients"
_VALID_PAYLOADS = frozenset({AGGREGATE_PAYLOAD_PARAMS, AGGREGATE_PAYLOAD_GRADIENTS})


def normalize_aggregate_payload(raw: object) -> AggregatePayload:
    """Parse Hydra value; default ``params`` (FedAvg weight averaging)."""
    if raw is None:
        return AGGREGATE_PAYLOAD_PARAMS
    key = str(raw).strip().lower()
    if key in ("param", "params", "parameters", "weights", "weight"):
        return AGGREGATE_PAYLOAD_PARAMS
    if key in ("grad", "grads", "gradient", "gradients"):
        return AGGREGATE_PAYLOAD_GRADIENTS
    raise ValueError(
        f"aggregate_payload must be 'params' or 'gradients', got {raw!r}"
    )


def aggregate_payload_from_cfg(cfg: Union[DictConfig, object]) -> AggregatePayload:
    """Prefer ``algorithm.aggregate_payload``, then ``engine.hierarchical``."""
    algo_raw = OmegaConf.select(cfg, "algorithm.aggregate_payload", default=None)
    if algo_raw is not None:
        return normalize_aggregate_payload(algo_raw)
    raw = OmegaConf.select(
        cfg, "engine.hierarchical.aggregate_payload", default=None
    )
    if raw is not None:
        return normalize_aggregate_payload(raw)
    return AGGREGATE_PAYLOAD_PARAMS


def communicate_params_from_cfg(cfg: Union[DictConfig, object]) -> bool:
    return aggregate_payload_from_cfg(cfg) != "gradients"


def format_aggregate_payload_policy(cfg: Union[DictConfig, object]) -> str:
    return f"aggregate_payload={aggregate_payload_from_cfg(cfg)!r}"


__all__ = [
    "AGGREGATE_PAYLOAD_GRADIENTS",
    "AGGREGATE_PAYLOAD_PARAMS",
    "AggregatePayload",
    "aggregate_payload_from_cfg",
    "communicate_params_from_cfg",
    "format_aggregate_payload_policy",
    "normalize_aggregate_payload",
]
