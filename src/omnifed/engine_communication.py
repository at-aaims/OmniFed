# Dispatch and Slurm world-size helpers. Topology kind (not communication_mode)
# decides centralized vs decentralized vs hierarchical.
from __future__ import annotations

from typing import Any, Optional

from omegaconf import MISSING, OmegaConf

__all__ = [
    "is_hierarchical_cfg",
    "topology_target_name",
    "hierarchical_topology_config_for_slurm",
    "hierarchical_slurm_world_size_from_conf_name",
    "hierarchical_world_size_from_cfg",
    "validate_hierarchical_slurm_topology_alignment",
    "resolve_slurm_ntasks",
]


def topology_target_name(cfg: Any) -> str:
    raw = OmegaConf.select(cfg, "topology._target_", default="")
    if raw is None:
        return ""
    return str(raw).rsplit(".", 1)[-1]


def is_hierarchical_cfg(cfg: Any) -> bool:
    """True when yaml ``topology._target_`` is Slurm :class:`HierarchicalTopology`."""
    return topology_target_name(cfg) == "HierarchicalTopology"


def hierarchical_topology_config_for_slurm(cfg: Any) -> Optional[str]:
    """Optional YAML name under ``conf_hybrid/topology/`` (legacy preset path)."""
    name = OmegaConf.select(cfg, "engine.hierarchical.topology_config", default=None)
    if name is None or str(name).strip() == "":
        return None
    return str(name)


def hierarchical_slurm_world_size_from_conf_name(topology_config: str) -> int:
    from src.omnifed.hierarchical.hydra_loader import (
        hierarchical_slurm_world_size_from_topology_yaml,
    )

    return hierarchical_slurm_world_size_from_topology_yaml(topology_config)


def _topology_num_clients_resolved(cfg: Any) -> Optional[int]:
    raw = OmegaConf.select(cfg, "topology.num_clients", default=None)
    if raw is None or raw is MISSING:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def hierarchical_world_size_from_cfg(cfg: Any) -> int:
    """Facility graph on ``topology`` or legacy ``engine.hierarchical.topology_config``."""
    from src.omnifed.hierarchical.hydra_loader import (
        engine_has_facility_topology,
        hierarchical_slurm_world_size_from_engine_layout,
    )

    if engine_has_facility_topology(cfg):
        return hierarchical_slurm_world_size_from_engine_layout(cfg)

    htc = hierarchical_topology_config_for_slurm(cfg)
    if not htc:
        raise ValueError(
            "Hierarchical Slurm requires topology.num_facilities and "
            "topology.mpi_ranks_per_facility (or engine.hierarchical.topology_config)."
        )
    return hierarchical_slurm_world_size_from_conf_name(htc)


def validate_hierarchical_slurm_topology_alignment(
    cfg: Any,
    *,
    topology_node_count: int,
    slurm_ntasks: Optional[int] = None,
) -> int:
    hierarchical_ws = hierarchical_world_size_from_cfg(cfg)

    if topology_node_count != hierarchical_ws:
        raise ValueError(
            "Hierarchical Slurm requires len(topology) == facility world_size: "
            f"facility graph => world_size={hierarchical_ws}, but "
            f"len(topology)={topology_node_count}."
        )

    nc = _topology_num_clients_resolved(cfg)
    if nc is not None:
        expected_nodes = nc + 1
        if expected_nodes != hierarchical_ws:
            raise ValueError(
                "Hierarchical Slurm requires topology.num_clients + 1 == world_size: "
                f"topology.num_clients={nc} implies {expected_nodes} logical nodes but "
                f"world_size={hierarchical_ws}."
            )
        if topology_node_count != expected_nodes:
            raise ValueError(
                "Inconsistent OmniFed topology: topology.num_clients=%s implies "
                "len(topology)=%s but len(topology)=%s."
                % (nc, expected_nodes, topology_node_count)
            )

    if slurm_ntasks is not None and int(slurm_ntasks) != hierarchical_ws:
        raise ValueError(
            "Hierarchical Slurm requires SLURM_NTASKS == world_size: "
            f"SLURM_NTASKS={slurm_ntasks}, world_size={hierarchical_ws}."
        )

    return hierarchical_ws


def resolve_slurm_ntasks(cfg: Any, topology_node_count: int) -> int:
    """
    Return Slurm ``--ntasks`` for the frozen worker world.

    * **centralized / decentralized** — ``len(topology)``.
    * **hierarchical** — facility graph world_size (must match ``len(topology)``).
    """
    if is_hierarchical_cfg(cfg):
        return validate_hierarchical_slurm_topology_alignment(
            cfg, topology_node_count=topology_node_count, slurm_ntasks=None
        )
    return topology_node_count
