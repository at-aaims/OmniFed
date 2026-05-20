# Phase B Step 6: shared parsing for engine.communication_mode (classic vs hybrid Slurm).
from __future__ import annotations

from typing import Any, Optional

from omegaconf import OmegaConf

__all__ = [
    "communication_mode",
    "hybrid_topology_config_for_slurm",
    "hybrid_slurm_world_size_from_conf_name",
    "resolve_slurm_ntasks",
]


def communication_mode(cfg: Any) -> str:
    raw = OmegaConf.select(cfg, "engine.communication_mode", default="classic")
    s = str(raw).lower()
    if s not in ("classic", "hybrid"):
        raise ValueError(
            f"engine.communication_mode must be 'classic' or 'hybrid', got {raw!r}"
        )
    return s


def hybrid_topology_config_for_slurm(cfg: Any) -> Optional[str]:
    name = OmegaConf.select(cfg, "engine.hybrid.topology_config", default=None)
    if name is None or str(name).strip() == "":
        return None
    return str(name)


def hybrid_slurm_world_size_from_conf_name(topology_config: str) -> int:
    from src.omnifed.hybrid.hydra_loader import (
        hybrid_slurm_world_size_from_topology_yaml,
    )

    return hybrid_slurm_world_size_from_topology_yaml(topology_config)


def resolve_slurm_ntasks(cfg: Any, topology_node_count: int) -> int:
    """
    Return Slurm ``--ntasks`` for the frozen :mod:`slurm_worker` world.

    * **classic** — one task per OmniFed topology node (``len(topology)``).
    * **hybrid** — one task per global hybrid rank (``topology.world_size`` in ``conf_hybrid``);
      must match ``topology_node_count`` to avoid mis-sized allocations.
    """
    mode = communication_mode(cfg)
    if mode == "classic":
        return topology_node_count

    htc = hybrid_topology_config_for_slurm(cfg)
    if not htc:
        raise ValueError(
            "engine.communication_mode=hybrid requires engine.hybrid.topology_config "
            "(Hybrids under conf_hybrid/topology/, e.g. built_symmetric_2x3.yaml)."
        )
    hybrid_ws = hybrid_slurm_world_size_from_conf_name(htc)
    if topology_node_count != hybrid_ws:
        raise ValueError(
            f"Hybrid layout {htc!r} has world_size={hybrid_ws} but len(topology)={topology_node_count}. "
            "Align centralized (or other) node count with hybrid ranks, e.g. num_clients=world_size-1 "
            "when using one aggregator node plus gRPC-only rank 0 in the hybrid map (Phase B step 7 "
            "will reconcile FL roles)."
        )
    return hybrid_ws
