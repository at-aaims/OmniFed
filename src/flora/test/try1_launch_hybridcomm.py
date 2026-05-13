# file: src/flora/test/launch_hybridcomm.py

import argparse
import yaml

import src.flora.helper as helper
from src.flora.test.try1_train_hybrid_comm import HybridTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=1234, help="seed value for result replication"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="/home/shruti/omnifed_data/flora_test/",
        help="dir where data is downloaded and/or saved",
    )
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--global-rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument(
        "--comm-freq",
        type=int,
        default=100,
        help="# iterations after which updates are synchronized",
    )
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--test-bsz", type=int, default=32)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--determinism", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--train-dir", type=str, default="~/")
    parser.add_argument("--test-dir", type=str, default="~/")

    #NEW: config file
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to hybrid communicator config.yaml",
    )

    args = parser.parse_args()

    #Load config
    with open(args.config, "r") as f:
        args.config_dict = yaml.safe_load(f)

    helper.set_seed(args.seed, determinism=False)

    HybridTrainer(args=args).start_training()

