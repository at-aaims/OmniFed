"""Hydra compose for ``conf_hybrid`` (no torch dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from src.omnifed.hybrid.topology_builder import (
    DEFAULT_HYBRID_COMMUNICATORS,
    build_hybrid_topology,
)

__all__ = [
    "load_hybrid_cfg",
    "load_hybrid_cfg_for_engine",
    "hybrid_slurm_world_size_from_topology_yaml",
    "engine_has_runtime_hybrid_layout",
    "hybrid_slurm_world_size_from_engine_layout",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _topology_name_from_arg(config_arg: str) -> str:
    name = Path(config_arg).name
    return name[:-5] if name.endswith(".yaml") else name


def engine_has_runtime_hybrid_layout(engine_cfg_root: Any) -> bool:
    """
    True if ``engine.hybrid.layout`` is a non-empty dict with fields required by
    :func:`build_hybrid_topology` (at least ``num_facilities``, ``mpi_ranks_per_facility``).
    """
    node = OmegaConf.select(engine_cfg_root, "engine.hybrid.layout", default=None)
    if node is None:
        return False
    try:
        d = OmegaConf.to_container(node, resolve=True)
    except Exception:
        return False
    if not isinstance(d, dict) or not d:
        return False
    return "num_facilities" in d and "mpi_ranks_per_facility" in d


def _layout_kwargs_and_communicators(layout_node: Any) -> tuple[dict, dict | None]:
    kwargs = OmegaConf.to_container(layout_node, resolve=True)
    if not isinstance(kwargs, dict):
        raise TypeError(f"hybrid layout must be a mapping, got {type(kwargs)}")
    comm = kwargs.pop("communicators", None)
    if comm is not None and not isinstance(comm, dict):
        raise TypeError("layout communicators override must be a mapping")
    return kwargs, comm


def _ensure_topology_declares_communicators(topo_oc: Any) -> None:
    """Static YAML (no layout merge) gets default ``communicators`` for PR clarity."""
    if OmegaConf.select(topo_oc, "communicators", default=None) is not None:
        return
    with open_dict(topo_oc):
        topo_oc.communicators = OmegaConf.create(dict(DEFAULT_HYBRID_COMMUNICATORS))


def hybrid_slurm_world_size_from_engine_layout(engine_cfg_root: Any) -> int:
    """``world_size`` from ``engine.hybrid.layout`` (resolved via builder)."""
    layout = OmegaConf.select(engine_cfg_root, "engine.hybrid.layout")
    kwargs, comm = _layout_kwargs_and_communicators(layout)
    return int(build_hybrid_topology(**kwargs, communicators=comm)["world_size"])


def hybrid_slurm_world_size_from_topology_yaml(topology_config: str) -> int:
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
        return int(build_hybrid_topology(**kwargs, communicators=comm)["world_size"])

    if "layout" in raw:
        kwargs, comm = _layout_kwargs_and_communicators(raw.layout)
        return int(build_hybrid_topology(**kwargs, communicators=comm)["world_size"])

    raise ValueError(
        f"{topo_path} has no topology.world_size and no layout block; "
        "cannot derive Slurm world size."
    )


def _maybe_merge_layout_into_topology(cfg) -> None:
    """
    If ``cfg.topology.layout`` is set, replace it with the output of
    :func:`build_hybrid_topology` (Phase A: generated ranks / facilities).
    """
    if "topology" not in cfg or "layout" not in cfg.topology:
        return

    layout = cfg.topology.layout
    kwargs, comm = _layout_kwargs_and_communicators(layout)
    topo_dict = build_hybrid_topology(**kwargs, communicators=comm)
    built = OmegaConf.create(topo_dict)

    with open_dict(cfg.topology):
        del cfg.topology.layout
        for key in built:
            cfg.topology[key] = built[key]


def load_hybrid_cfg(config_arg: str):
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


def load_hybrid_cfg_for_engine(engine_cfg_root: Any) -> OmegaConf:
    """
    Build the hybrid OmegaConf blob used by :func:`run_hybrid_training`.

    * If ``engine.hybrid.layout`` is set (**runtime generation**): call
      :func:`build_hybrid_topology` with those kwargs (plus optional ``engine.hybrid.training`` copy).
    * Else: compose from ``engine.hybrid.topology_config`` (YAML under ``conf_hybrid/topology/``).

    When both are present, ``layout`` **wins**.
    """
    if engine_has_runtime_hybrid_layout(engine_cfg_root):
        layout = OmegaConf.select(engine_cfg_root, "engine.hybrid.layout")
        kwargs, comm = _layout_kwargs_and_communicators(layout)
        topo_dict = build_hybrid_topology(**kwargs, communicators=comm)
        out = OmegaConf.create({"topology": topo_dict})
        train_node = OmegaConf.select(engine_cfg_root, "engine.hybrid.training", default=None)
        if train_node is not None:
            train_dict = OmegaConf.to_container(train_node, resolve=True)
            if not isinstance(train_dict, dict):
                raise TypeError("engine.hybrid.training must be a mapping when provided")
            with open_dict(out):
                out.training = OmegaConf.create(train_dict)
        return out

    tc = OmegaConf.select(engine_cfg_root, "engine.hybrid.topology_config", default=None)
    if tc is None or str(tc).strip() == "":
        raise ValueError(
            "Hybrid Slurm requires either engine.hybrid.layout "
            "{num_facilities, mpi_ranks_per_facility, ...} "
            "or engine.hybrid.topology_config (e.g. built_symmetric_2x3.yaml)."
        )
    return load_hybrid_cfg(str(tc))
