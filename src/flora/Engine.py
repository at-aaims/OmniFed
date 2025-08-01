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

import json
import os
import pickle
import time
import warnings
from dataclasses import asdict
from typing import List

import ray
import rich.repr
from hydra.conf import HydraConf
from tqdm.auto import tqdm
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich.pretty import pprint

from . import utils
from .mixins import RequiredSetup
from .Node import Node
from .topology.BaseTopology import BaseTopology
from .utils import print
from .utils.ExperimentDisplay import ExperimentResultsDisplay
from .utils.MetricFormatter import MetricFormatter

LOG_FLUSH_DELAY = 2.0


@rich.repr.auto
class Engine(RequiredSetup):
    """
    Main engine for federated learning experiments.

    Coordinates distributed Ray actors (nodes) to run FL algorithms across different topologies.
    Handles experiment setup, execution, and results collection with automatic
    GPU allocation and output management.

    Use this as the main entry point for running FL experiments with Hydra configurations.
    See working examples in the conf/ directory.
    """

    def __init__(
        self,
        flora_cfg: DictConfig,
    ):
        """
        Initialize the federated learning experiment engine.

        Args:
            flora_cfg: Complete FLORA configuration including topology, algorithm,
                      model, and datamodule specifications from Hydra
        """
        super().__init__()
        utils.print_rule()

        self.flora_cfg: DictConfig = flora_cfg
        self.hydra_cfg: HydraConf = HydraConfig.get()

        self.topology: BaseTopology = instantiate(
            self.flora_cfg.topology, _recursive_=False
        )
        self.global_rounds: int = flora_cfg.global_rounds

        self.output_dir: str = self.hydra_cfg.runtime.output_dir
        self.engine_dir: str = os.path.join(self.output_dir, "engine")
        self.results_dir: str = os.path.join(self.engine_dir, "node_results")

        self._metric_formatter: MetricFormatter = MetricFormatter()
        self._results_display: ExperimentResultsDisplay = ExperimentResultsDisplay()

        self._ray_actor_refs: List[Node] = []

    def _setup_output_directories(self) -> None:
        """
        Create and validate output directories for experiment data.

        Creates engine/ and node_results/ directories under Hydra's output path.
        Fails with RuntimeError if conflicting experiment files already exist
        (ignores Hydra standard files like main.log).
        """
        # Check for pre-existing files that could overwrite results (ignore Hydra standard files)
        if os.path.exists(self.output_dir):
            hydra_standard_files = {".hydra", "main.log", ".gitignore"}
            existing_files = [
                f
                for f in os.listdir(self.output_dir)
                if not f.startswith(".") and f not in hydra_standard_files
            ]
            if existing_files:
                raise RuntimeError(
                    f"Output directory contains existing files: {self.output_dir}\n"
                    f"Found: {existing_files[:5]}{'...' if len(existing_files) > 5 else ''}\n"
                    f"This could overwrite previous experiment results. "
                    f"Use a fresh Hydra output directory or clean the existing one."
                )

        # Create engine directory
        os.makedirs(self.engine_dir, exist_ok=True)

        # Check if engine directory is not empty (indicates conflicting experiment)
        if os.path.exists(self.engine_dir):
            existing_files = [
                f for f in os.listdir(self.engine_dir) if not f.startswith(".")
            ]
            if existing_files:
                raise RuntimeError(
                    f"Engine directory is not empty: {self.engine_dir}\n"
                    f"Found: {existing_files[:5]}{'...' if len(existing_files) > 5 else ''}\n"
                    f"This indicates a conflicting experiment setup."
                )

        print(f"Created engine directory: {self.engine_dir}")

    def _setup(self) -> None:
        """
        Initialize Ray cluster and launch distributed nodes.

        Sets up the distributed infrastructure including:
        - Output directory validation
        - Ray cluster initialization
        - Smart GPU allocation (fractional for single-node, full for multi-node)
        - Node actor creation and setup
        """
        utils.print_rule()

        # Setup directories first
        self._setup_output_directories()

        # Initialize Ray cluster
        ray.init(**self.flora_cfg.ray)

        # Smart GPU allocation: detect single-node vs multi-node scenarios
        ray_available_resources = ray.available_resources()
        print("ray.available_resources()")
        pprint(ray_available_resources)

        # Check if single node
        ray_nodes = ray.nodes()
        print("ray.nodes()")
        pprint(ray_nodes)

        # Save Ray cluster information to engine directory
        ray_resources_path = os.path.join(
            self.engine_dir, "ray_available_resources.json"
        )
        ray_nodes_path = os.path.join(self.engine_dir, "ray_nodes.json")

        with open(ray_resources_path, "w") as f:
            json.dump(ray_available_resources, f, indent=2, default=str)
            print(f"Saved Ray resources info to: {ray_resources_path}")

        with open(ray_nodes_path, "w") as f:
            json.dump(ray_nodes, f, indent=2, default=str)
            print(f"Saved Ray nodes info to: {ray_nodes_path}")

        # print(f"Saved Ray cluster info to: {self.engine_dir}")

        ray_nodes_alive = [node for node in ray_nodes if node["Alive"]]
        is_single_node = len(ray_nodes_alive) == 1

        available_gpus = ray_available_resources.get("GPU", 0)
        total_actors = len(list(self.topology))

        # Determine GPU allocation strategy
        use_fractional_gpu = (
            is_single_node and total_actors > available_gpus and available_gpus > 0
        )

        print(f"Launching {total_actors} Actors")
        for node_config in self.topology:
            # Set log directory
            node_config.log_dir_base = (
                node_config.log_dir_base or self.hydra_cfg.runtime.output_dir
            )

            # Configure GPU allocation
            ray_actor_options = asdict(node_config.ray_actor_options)
            if ray_actor_options.get("num_gpus") is None and available_gpus > 0:
                if use_fractional_gpu:
                    # Single-node with GPU shortage: use fractional allocation
                    ray_actor_options["num_gpus"] = available_gpus / total_actors
                else:
                    # Multi-node or sufficient GPUs: request 1 GPU per actor
                    ray_actor_options["num_gpus"] = 1

            pprint(node_config)

            node_actor = Node.options(**ray_actor_options).remote(
                **asdict(node_config),  # type: ignore - Ray's remote() typing doesn't understand dataclass unpacking
                algorithm=self.flora_cfg.algorithm,
                model=self.flora_cfg.model,
                datamodule=self.flora_cfg.datamodule,
            )
            self._ray_actor_refs.append(node_actor)

        print(f"Calling setup() on {len(self._ray_actor_refs)} Nodes")
        setup_futures = [node.setup.remote() for node in self._ray_actor_refs]
        ray.get(setup_futures)

    def _save_node_results(self, results: List) -> None:
        """
        Save individual node results as pickle files for debugging.

        Creates node_results/ directory and saves each node's results as
        node_000_results.pkl, node_001_results.pkl, etc.
        Non-fatal - logs errors but doesn't crash the experiment.

        Args:
            results: List of result dictionaries from each node
        """
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)

        print(f"Saving node results to: {self.results_dir}")

        # Save node results with progress bar
        for node_idx, node_result in tqdm(
            enumerate(results),
            desc="Saving node results",
            unit="file",
            total=len(results),
        ):
            filename = f"node_{node_idx:03d}_results.pkl"
            filepath = os.path.join(self.results_dir, filename)

            with open(filepath, "wb") as f:
                pickle.dump(node_result, f)

        print(
            f":heavy_check_mark: Saved {len(results)} node result files successfully!"
        )

    def run_experiment(self) -> None:
        """
        Run the federated learning experiment.

        Coordinates experiment execution across all nodes:
        - Launches experiment on all Ray actors
        - Waits for completion with progress feedback
        - Saves node results for debugging
        - Displays formatted experiment results
        - Handles Ray cluster shutdown
        """
        try:
            utils.print_rule()
            print(f"Starting Experiment with {len(self._ray_actor_refs)} Nodes")

            experiment_start_time = time.time()

            node_results_futures = []
            for node in self._ray_actor_refs:
                future = node.run_experiment.remote(self.global_rounds)
                node_results_futures.append(future)

            print(
                f"Waiting for {len(node_results_futures)} nodes to complete experiments...",
                flush=True,
            )

            # Wait for all nodes to complete
            results = ray.get(node_results_futures)

            print(
                f":heavy_check_mark: All {len(results)} nodes completed successfully!",
                flush=True,
            )

            # Save node results for debugging
            try:
                self._save_node_results(results)
            except Exception as e:
                warnings.warn(
                    f"Failed to save node results for debugging: {e}. "
                    f"Experiment results will still be displayed normally.",
                    UserWarning,
                )

            if results:
                print("=" * 80)
                print("DEBUG: First node's returned data structure:")
                print("=" * 80)

                print(json.dumps(results[0], indent=2, default=str))
                print("=" * 80)

            experiment_end_time = time.time()
            experiment_duration = experiment_end_time - experiment_start_time

            utils.print_rule()
            time.sleep(
                LOG_FLUSH_DELAY
            )  # Ensure async Ray logs complete before displaying results

            self._results_display.show_experiment_results(
                results,
                experiment_duration,
                self.global_rounds,
                len(self.topology),
            )

        finally:
            print("Shutting down...", flush=True)
            ray.shutdown()
