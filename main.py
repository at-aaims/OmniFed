import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print
from rich.panel import Panel
from rich.pretty import Pretty

from src.flora import utils
from src.flora.Engine import Engine

# =============================================================================


@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg: DictConfig) -> None:
    import logging

    utils.setup_rich_logging()
    # NOTE: migrate to logging soon (this is currently unused)
    logger = logging.getLogger(__name__)

    utils.log_sep("FLORA Federated Learning Framework", color="blue")

    print(
        Panel(
            Pretty(
                OmegaConf.to_container(cfg, resolve=True),
                expand_all=True,
                indent_guides=True,
            ),
            title="Configuration",
        )
    )

    engine = Engine(
        topology_cfg=cfg.topology,
        node_defaults=cfg.node,
        global_rounds=cfg.global_rounds,
    )

    engine.run_experiment()


if __name__ == "__main__":
    main()
