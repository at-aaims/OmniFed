"""Hydra compose for ``conf_hybrid`` (no torch dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from src.omnifed.hierarchical.topology_builder import (
    DEFAULT_HIERARCHICAL_COMMUNICATORS,
    build_hierarchical_topology,
)

__all__ = [
    "load_hierarchical_cfg",
    "load_hierarchical_cfg_for_engine",
    "hierarchical_slurm_world_size_from_topology_yaml",
    "engine_has_facility_topology",
    "hierarchical_slurm_world_size_from_engine_layout",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _topology_name_from_arg(config_arg: str) -> str:
    name = Path(config_arg).name
    return name[:-5] if name.endswith(".yaml") else name


_FACILITY_BUILDER_KEYS = (
    "num_facilities",
    "mpi_ranks_per_facility",
    "dedicated_rpc_server",
    "rpc_addr",
    "rpc_port",
    "facility_mpi_addr",
    "facility_mpi_base_port",
    "facility_mpi_port_stride",
    "facility_name_prefix",
    "communicators",
)


def _facility_mapping(engine_cfg_root: Any) -> dict | None:
    """Facility knobs live on ``topology`` (not a separate layout group)."""
    node = OmegaConf.select(engine_cfg_root, "topology", default=None)
    if node is None:
        return None
    try:
        d = OmegaConf.to_container(node, resolve=True)
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    if "num_facilities" in d and "mpi_ranks_per_facility" in d:
        return d
    return None


def engine_has_facility_topology(engine_cfg_root: Any) -> bool:
    """True when ``topology`` has facility fields for :func:`build_hierarchical_topology`."""
    return _facility_mapping(engine_cfg_root) is not None


def _layout_kwargs_and_communicators(layout_node: Any) -> tuple[dict, dict | None]:
    if isinstance(layout_node, dict):
        raw = dict(layout_node)
    else:
        raw = OmegaConf.to_container(layout_node, resolve=True)
        if not isinstance(raw, dict):
            raise TypeError(f"facility topology must be a mapping, got {type(raw)}")
    kwargs = {k: raw[k] for k in _FACILITY_BUILDER_KEYS if k in raw and raw[k] is not None}
    comm = kwargs.pop("communicators", None)
    if comm is not None and not isinstance(comm, dict):
        raise TypeError("topology communicators override must be a mapping")
    return kwargs, comm


def _ensure_topology_declares_communicators(topo_oc: Any) -> None:
    """Static YAML (no layout merge) gets default ``communicators`` for PR clarity."""
    if OmegaConf.select(topo_oc, "communicators", default=None) is not None:
        return
    with open_dict(topo_oc):
        topo_oc.communicators = OmegaConf.create(dict(DEFAULT_HIERARCHICAL_COMMUNICATORS))


def hierarchical_slurm_world_size_from_engine_layout(engine_cfg_root: Any) -> int:
    """``world_size`` from ``topology`` facility fields (resolved via builder)."""
    layout = _facility_mapping(engine_cfg_root)
    if layout is None:
        raise ValueError("topology is missing num_facilities / mpi_ranks_per_facility")
    kwargs, comm = _layout_kwargs_and_communicators(layout)
    return int(build_hierarchical_topology(**kwargs, communicators=comm)["world_size"])


def hierarchical_slurm_world_size_from_topology_yaml(topology_config: str) -> int:
    """
    Return ``topology.world_size`` for a name under ``conf_hybrid/topology/``.

    Does not use Hydra (``GlobalHydra`` may already be initialized by the main app).
    """
    name = _topology_name_from_arg(topology_config)
    topo_path = _repo_root() / "conf_hybrid" / "topology" / f"{name}.yaml"
    if not topo_path.is_file():
        raise FileNotFoundError(
            f"Hybrid topology YAML not found: {topo_path} (from {topology_config!r})"
        )
    raw = OmegaConf.load(topo_path)

    ws = OmegaConf.select(raw, "topology.world_size", default=None)
    if ws is not None:
        return int(ws)

    layout = OmegaConf.select(raw, "topology.layout", default=None)
    if layout is not None:
        kwargs, comm = _layout_kwargs_and_communicators(layout)
        return int(build_hierarchical_topology(**kwargs, communicators=comm)["world_size"])

    if "layout" in raw:
        kwargs, comm = _layout_kwargs_and_communicators(raw.layout)
        return int(build_hierarchical_topology(**kwargs, communicators=comm)["world_size"])

    raise ValueError(
        f"{topo_path} has no topology.world_size and no layout block; "
        "cannot derive Slurm world size."
    )


def _maybe_merge_layout_into_topology(cfg) -> None:
    """
    If ``cfg.topology.layout`` is set, replace it with the output of
    :func:`build_hierarchical_topology` (Phase A: generated ranks / facilities).
    """
    if "topology" not in cfg or "layout" not in cfg.topology:
        return

    layout = cfg.topology.layout
    kwargs, comm = _layout_kwargs_and_communicators(layout)
    topo_dict = build_hierarchical_topology(**kwargs, communicators=comm)
    built = OmegaConf.create(topo_dict)

    with open_dict(cfg.topology):
        del cfg.topology.layout
        for key in built:
            cfg.topology[key] = built[key]


def load_hierarchical_cfg(config_arg: str):
    conf_dir = _repo_root() / "conf_hybrid"
    topology_name = _topology_name_from_arg(config_arg)

    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="base", overrides=[f"topology={topology_name}"])

    _maybe_merge_layout_into_topology(cfg)

    with open_dict(cfg):
        nested_training = None
        if "topology" in cfg and "training" in cfg.topology:
            nested_training = cfg.topology.training

        if "topology" in cfg and "topology" in cfg.topology:
            cfg.topology = cfg.topology.topology

        if "training" not in cfg and nested_training is not None:
            cfg.training = nested_training

        # Layout-based topology keeps ``training`` alongside ``world_size`` until now.
        if nested_training is not None and "training" in cfg.topology:
            with open_dict(cfg.topology):
                del cfg.topology.training

    _ensure_topology_declares_communicators(cfg.topology)
    return cfg


def load_hierarchical_cfg_for_engine(engine_cfg_root: Any) -> OmegaConf:
    """
    Build the OmegaConf blob used by :func:`run_hierarchical_training`.

    Facility knobs come from ``topology``. Optional
    ``engine.hierarchical.topology_config`` still loads a ``conf_hybrid`` preset.
    """
    if engine_has_facility_topology(engine_cfg_root):
        layout = _facility_mapping(engine_cfg_root)
        kwargs, comm = _layout_kwargs_and_communicators(layout)
        topo_dict = build_hierarchical_topology(**kwargs, communicators=comm)
        out = OmegaConf.create({"topology": topo_dict})
        train_node = OmegaConf.select(engine_cfg_root, "engine.hierarchical.training", default=None)
        if train_node is not None:
            train_dict = OmegaConf.to_container(train_node, resolve=True)
            if not isinstance(train_dict, dict):
                raise TypeError("engine.hierarchical.training must be a mapping when provided")
            with open_dict(out):
                out.training = OmegaConf.create(train_dict)
        return out

    tc = OmegaConf.select(engine_cfg_root, "engine.hierarchical.topology_config", default=None)
    if tc is None or str(tc).strip() == "":
        raise ValueError(
            "Hierarchical Slurm requires topology.num_facilities and "
            "topology.mpi_ranks_per_facility, or engine.hierarchical.topology_config."
        )
    return load_hierarchical_cfg(str(tc))
