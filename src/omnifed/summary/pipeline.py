"""Resolve training pipeline mode for summary generation."""

from __future__ import annotations

from typing import Any

from src.omnifed.engine_communication import communication_mode

_SUMMARY_MODES = ("classic", "hybrid")


def summary_mode_from_cfg(cfg: Any) -> str:
    """Return ``classic`` or ``hybrid`` from ``engine.communication_mode``."""
    mode = communication_mode(cfg)
    if mode not in _SUMMARY_MODES:
        raise ValueError(f"unsupported summary pipeline mode: {mode!r}")
    return mode
