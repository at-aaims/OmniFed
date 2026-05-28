# Phase B Step 6: shared parsing for engine.communication_mode (classic vs hybrid Slurm).
# Phase B roadmap: single hybrid world_size resolution + centralized validation helpers.
from __future__ import annotations

from typing import Any, Optional

from omegaconf import MISSING, OmegaConf

__all__ = [
    "communication_mode",
    "hybrid_topology_config_for_slurm",
    "hybrid_slurm_world_size_from_conf_name",
    "hybrid_world_size_from_cfg",
    "validate_hybrid_slurm_topology_alignment",
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
    """YAML name under ``conf_hybrid/topology/`` — optional when ``engine.hybrid.layout`` is set."""
    name = OmegaConf.select(cfg, "engine.hybrid.topology_config", default=None)
    if name is None or str(name).strip() == "":
        return None
    return str(name)


def hybrid_slurm_world_size_from_conf_name(topology_config: str) -> int:
    from src.omnifed.hybrid.hydra_loader import (
        hybrid_slurm_world_size_from_topology_yaml,
    )

    return hybrid_slurm_world_size_from_topology_yaml(topology_config)


def _topology_num_clients_resolved(cfg: Any) -> Optional[int]:
    """
    Resolved ``topology.num_clients`` when present (OmniFed centralized slot count minus server).
    Returns ``None`` if the knob is absent or structural ``MISSING``.
    """
    raw = OmegaConf.select(cfg, "topology.num_clients", default=None)
    if raw is None or raw is MISSING:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _validate_engine_hybrid_layout_topology_preset_alignment(cfg: Any) -> None:
    """
    If both ``engine.hybrid.layout`` and ``engine.hybrid.topology_config`` are set,
    they must imply the **same** ``world_size``. Runtime loading uses ``layout`` only when
    both are present — duplicate conflicting sizes are rejected here.
    """
    from src.omnifed.hybrid.hydra_loader import (
        engine_has_runtime_hybrid_layout,
        hybrid_slurm_world_size_from_engine_layout,
    )

    if not engine_has_runtime_hybrid_layout(cfg):
        return
    htc = hybrid_topology_config_for_slurm(cfg)
    if not htc:
        return
    ws_layout = hybrid_slurm_world_size_from_engine_layout(cfg)
    ws_yaml = hybrid_slurm_world_size_from_conf_name(htc)
    if ws_layout != ws_yaml:
        raise ValueError(
            "engine.hybrid.layout implies "
            f"world_size={ws_layout}, but engine.hybrid.topology_config={htc!r} "
            f"implies world_size={ws_yaml}. At runtime layout wins; remove "
            "`topology_config` or change one side so both agree."
        )


def hybrid_world_size_from_cfg(cfg: Any) -> int:
    """
    Single authoritative hybrid global rank count from ``cfg`` (**layout or YAML preset only**).

    Does **not** read ``topology.num_clients``. Call
    :func:`resolve_slurm_ntasks` / :func:`validate_hybrid_slurm_topology_alignment`
    for full Slurm × centralized topology checks.
    """
    from src.omnifed.hybrid.hydra_loader import (
        engine_has_runtime_hybrid_layout,
        hybrid_slurm_world_size_from_engine_layout,
    )

    if engine_has_runtime_hybrid_layout(cfg):
        return hybrid_slurm_world_size_from_engine_layout(cfg)

    htc = hybrid_topology_config_for_slurm(cfg)
    if not htc:
        raise ValueError(
            "engine.communication_mode=hybrid requires engine.hybrid.layout "
            "(with num_facilities, mpi_ranks_per_facility, ...) or "
            "engine.hybrid.topology_config (YAML under conf_hybrid/topology/, "
            "e.g. built_symmetric_2x3.yaml)."
        )
    return hybrid_slurm_world_size_from_conf_name(htc)


def validate_hybrid_slurm_topology_alignment(
    cfg: Any,
    *,
    topology_node_count: int,
    slurm_ntasks: Optional[int] = None,
) -> int:
    """
    **Roadmap Phase B:** Fail fast when hybrid ``world_size`` disagrees with
    centralized OmniFed node count / ``topology.num_clients`` or (optionally)
    ``SLURM_NTASKS`` inside an allocation.

    Returns the resolved hybrid ``world_size``.
    """
    _validate_engine_hybrid_layout_topology_preset_alignment(cfg)

    hybrid_ws = hybrid_world_size_from_cfg(cfg)

    if topology_node_count != hybrid_ws:
        lbl = (
            "engine.hybrid.layout"
            if _engine_has_runtime_hybrid_layout_for_msg(cfg)
            else repr(hybrid_topology_config_for_slurm(cfg))
        )
        raise ValueError(
            "Hybrid Slurm requires len(topology) == hybrid world_size: "
            f"hybrid {lbl} => world_size={hybrid_ws}, but len(topology)={topology_node_count}. "
            "With CentralizedTopology + dedicated RPC hybrid rank use "
            f"topology.num_clients={hybrid_ws - 1} (one logical node per hybrid global rank)."
        )

    nc = _topology_num_clients_resolved(cfg)
    if nc is not None:
        expected_nodes = nc + 1
        if expected_nodes != hybrid_ws:
            raise ValueError(
                "Hybrid Slurm requires topology.num_clients + 1 == hybrid world_size: "
                f"topology.num_clients={nc} implies {expected_nodes} logical nodes but "
                f"hybrid world_size={hybrid_ws}. Either set topology.num_clients to "
                f"{hybrid_ws - 1} or adjust engine.hybrid.layout / topology_config."
            )
        if topology_node_count != expected_nodes:
            raise ValueError(
                "Inconsistent OmniFed topology: topology.num_clients=%s implies len(topology)=%s "
                "but len(topology)=%s."
                % (nc, expected_nodes, topology_node_count)
            )

    if slurm_ntasks is not None and int(slurm_ntasks) != hybrid_ws:
        raise ValueError(
            "Hybrid Slurm requires SLURM_NTASKS == hybrid world_size: "
            f"SLURM_NTASKS={slurm_ntasks}, hybrid world_size={hybrid_ws}. "
            "Re-submit with Slurm --ntasks matching hybrid layout "
            "(Engine resolves this automatically when configs are aligned)."
        )

    return hybrid_ws


def _engine_has_runtime_hybrid_layout_for_msg(cfg: Any) -> bool:
    from src.omnifed.hybrid.hydra_loader import engine_has_runtime_hybrid_layout

    return engine_has_runtime_hybrid_layout(cfg)


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

    return validate_hybrid_slurm_topology_alignment(
        cfg, topology_node_count=topology_node_count, slurm_ntasks=None
    )
