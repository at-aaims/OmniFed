from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import open_dict


def _topology_name_from_arg(config_arg: str) -> str:
    # Accept either "try1_hybrid_topo.yaml" or "try1_hybrid_topo".
    name = Path(config_arg).name
    return name[:-5] if name.endswith(".yaml") else name


def load_hybrid_cfg(config_arg: str):
    repo_root = Path(__file__).resolve().parents[3]
    conf_dir = repo_root / "conf_hybrid"
    topology_name = _topology_name_from_arg(config_arg)

    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="base", overrides=[f"topology={topology_name}"])

    # Backward-compatible normalization for topology files that include:
    # topology:
    #   world_size: ...
    # training:
    #   dataset_total_clients: ...
    # and are loaded via defaults key `topology=...`.
    #
    # In that case Hydra places them under cfg.topology.*; unwrap so callers can use:
    # - cfg.topology.world_size
    # - cfg.training.dataset_total_clients
    with open_dict(cfg):
        nested_training = None
        if "topology" in cfg and "training" in cfg.topology:
            nested_training = cfg.topology.training

        if "topology" in cfg and "topology" in cfg.topology:
            cfg.topology = cfg.topology.topology

        if "training" not in cfg and nested_training is not None:
            cfg.training = nested_training
    return cfg
