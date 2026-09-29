from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf


SUPPORTED_EXECUTION_MODES = frozenset({"ray", "slurm"})


def validate_execution_mode(execution_mode: str) -> str:
    """Return normalized ``engine.mode``. Ray vs Slurm only."""
    mode = str(execution_mode).lower()
    if mode not in SUPPORTED_EXECUTION_MODES:
        supported = ", ".join(sorted(SUPPORTED_EXECUTION_MODES))
        raise ValueError(
            f"engine.mode must be one of: {supported}, got {execution_mode!r}"
        )
    return mode


def uses_torchtitan(cfg: Any) -> bool:
    """True when this run's **client** is Titan (``torchtitan.module`` is set).

    Not a scenario flag. Topology still chooses the hop (server or not).
    """
    module = OmegaConf.select(cfg, "torchtitan.module", default=None)
    if module is None:
        return False
    return str(module).strip() not in ("", "null", "None")
