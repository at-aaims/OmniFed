# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import src.flora.helper as helper
from src.flora.test.quantization_training import QuantizedCompressionTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=1234, help="seed value for result replication"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="/Users/ssq/Desktop/datasets/flora_test/",
        help="dir where data is downloaded and/or saved",
    )
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="28564")
    parser.add_argument("--backend", type=str, default="Gloo")
    parser.add_argument("--test-bsz", type=int, default=32)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--determinism", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--mobv3-lr-step-size", type=int, default=40)
    parser.add_argument("--mobv3-num-classes", type=int, default=257)
    parser.add_argument("--train-dir", type=str, default="~/")
    parser.add_argument("--test-dir", type=str, default="~/")
    parser.add_argument("--compression-type", type=str, default="QSGD")
    parser.add_argument("--quantized-bitwidth", type=float, default=8)

    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    if args.backend == "Gloo":
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0"

    helper.set_seed(args.seed, determinism=False)
    QuantizedCompressionTrainer(args=args).start()