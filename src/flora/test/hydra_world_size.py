import argparse

from src.flora.test.hydra_hybrid_config import load_hybrid_cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_hybrid_cfg(args.config)
    print(int(cfg.topology.world_size))
