"""Hydra compose for ``conf_hybrid`` (no torch dependency)."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from src.omnifed.hybrid.topology_builder import build_hybrid_topology

__all__ = ["load_hybrid_cfg", "hybrid_slurm_world_size_from_topology_yaml"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _topology_name_from_arg(config_arg: str) -> str:
    name = Path(config_arg).name
    return name[:-5] if name.endswith(".yaml") else name


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
        kwargs = OmegaConf.to_container(layout, resolve=True)
        return int(build_hybrid_topology(**kwargs)["world_size"])

    if "layout" in raw:
        kwargs = OmegaConf.to_container(raw.layout, resolve=True)
        return int(build_hybrid_topology(**kwargs)["world_size"])

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
    kwargs = OmegaConf.to_container(layout, resolve=True)
    topo_dict = build_hybrid_topology(**kwargs)
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
    return cfg
