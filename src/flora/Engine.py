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

import logging
import time

import ray
import rich.repr
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich import box
from rich.table import Table

from . import utils
from .topology.BaseTopology import Topology


@rich.repr.auto
class Engine:
    """
    Core orchestration engine for federated learning experiments.

    - Orchestrate distributed federated learning experiments
    - Manage node lifecycle and resource allocation
    - Collect, analyze, and report metrics and statistics.

    Integration:
    - Coordinates with Topology definitions for network structure
    - Manages Node actors running Algorithm implementations
    - Instantiated through Hydra configuration
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):
        utils.log_sep(f"{self.__class__.__name__} Init", color="blue")
        # Initialize Ray with more verbose logging and explicit namespace
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=True,  # NOTE: when false, Task and Actor logs are not copied to the driver stdout.
            logging_level=logging.INFO,  # TODO: Tie this to Hydra's logging level
            namespace="federated_learning",
        )
        # ---
        self.cfg: DictConfig = cfg

        self.topology: Topology = instantiate(self.cfg.topology)

        self.global_rounds: int = cfg.global_rounds

    def setup(self):
        utils.log_sep("FLORA Engine Setup", color="blue")

        self.topology.setup(
            comm_cfg=self.cfg.comm,
            algo_cfg=self.cfg.algo,
            model_cfg=self.cfg.model,
            data_cfg=self.cfg.data,
        )

    def start(self):
        """
        NOTE: stuffed everything in here for now, plan to refactor into smaller generalized methods.
        """
        try:
            # ----------------------------------------------------------------
            utils.log_sep("FLORA Engine Start", color="blue")

            summaries = []
            for round_idx in range(self.global_rounds):
                round_num = round_idx + 1
                utils.log_sep(f"Round {round_num}/{self.global_rounds}")
                _t_start_round = time.time()

                results_futures = []
                for node in self.topology:
                    future = node.execute_round.remote(
                        round_num,
                    )
                    results_futures.append(future)

                results = ray.get(results_futures)

                # ---
                _t_round = time.time() - _t_start_round

                _ct_total = len(results)
                _ct_success = len([r for r in results if r is not None])

                _success_rate = (_ct_success / _ct_total) * 100

                # ---
                summaries.append(
                    {
                        "round_num": round_num,
                        "duration": _t_round,
                        "total_count": _ct_total,
                        "success_count": _ct_success,
                        "success_rate": _success_rate,
                    }
                )
                print(f"Round Complete | {summaries[-1]}", flush=True)

            utils.log_sep("FL Rounds End", color="blue")

            # ----------------------------------------------------------------

            table = Table(
                title="Summary",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
            )

            for key in summaries[0].keys():
                table.add_column(
                    key.replace("_", " ").title(),
                    justify="center",
                    style="cyan",
                )

            for metrics in summaries:
                table.add_row(*[str(metrics[key]) for key in metrics.keys()])

            utils.console.print(table)

        finally:
            print("Engine shutting down")
            ray.shutdown()
