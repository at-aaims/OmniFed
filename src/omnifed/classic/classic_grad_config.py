# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""Classic centralized Slurm: gradient vs parameter aggregation."""

from __future__ import annotations

from typing import Union

from omegaconf import DictConfig, OmegaConf

from src.omnifed.hybrid.hybrid_aggregate_config import normalize_aggregate_payload

_CLASSIC_AGGREGATE_KEY = "engine.classic.aggregate_payload"


def classic_aggregate_payload_from_cfg(cfg: Union[DictConfig, object]):
    raw = OmegaConf.select(cfg, _CLASSIC_AGGREGATE_KEY, default="params")
    return normalize_aggregate_payload(raw)


def classic_communicate_params_from_cfg(cfg: Union[DictConfig, object]) -> bool:
    return classic_aggregate_payload_from_cfg(cfg) != "gradients"


def format_classic_grad_policy(cfg: Union[DictConfig, object]) -> str:
    return f"aggregate_payload={classic_aggregate_payload_from_cfg(cfg)!r}"


__all__ = [
    "classic_aggregate_payload_from_cfg",
    "classic_communicate_params_from_cfg",
    "format_classic_grad_policy",
]
