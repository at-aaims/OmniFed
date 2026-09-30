"""Resolve training pipeline mode for summary generation."""

from __future__ import annotations

from typing import Any

from src.omnifed.engine_communication import is_hierarchical_cfg

_SUMMARY_MODES = ("single_level", "hierarchical")


def summary_mode_from_cfg(cfg: Any) -> str:
    """Return ``hierarchical`` or ``single_level`` from ``topology._target_``."""
    mode = "hierarchical" if is_hierarchical_cfg(cfg) else "single_level"
    if mode not in _SUMMARY_MODES:
        raise ValueError(f"unsupported summary pipeline mode: {mode!r}")
    return mode
