from __future__ import annotations

import json
import os
import pickle
import time
import warnings
from dataclasses import asdict, is_dataclass
from typing import Any, List

import ray
from omegaconf import OmegaConf
from rich.pretty import pprint
from tqdm.auto import tqdm

from ...node import Node, NodeConfig
from ...utils import ResultsDisplay, print, print_rule

LOG_FLUSH_DELAY = 2.0


class RayRuntime:
    """Ray launch path extracted from Engine (behavior freeze, untested in Phase 1)."""

    def __init__(
        self,
        *,
        cfg: Any,
        hydra_cfg: Any,
        topology: Any,
        results_display: ResultsDisplay,
        engine_dir: str,
        results_dir: str,
    ) -> None:
        self.cfg = cfg
        self.hydra_cfg = hydra_cfg
        self.topology = topology
        self._results_display = results_display
        self.ray_cfg = cfg.ray
        self.global_rounds = int(cfg.global_rounds)
        self.engine_dir = engine_dir
        self.results_dir = results_dir
        self.actor_refs: List[Node] = []

    def setup(self) -> None:
        if is_dataclass(self.ray_cfg):
            rcfg = asdict(self.ray_cfg)
        else:
            try:
                rcfg = OmegaConf.to_container(self.ray_cfg, resolve=True)
            except Exception:
                rcfg = dict(self.ray_cfg)

        addr = rcfg.get("address")

        if addr not in (None, "", "local"):
            for k in (
                "num_cpus",
                "num_gpus",
                "resources",
                "object_store_memory",
                "include_dashboard",
                "dashboard_host",
                "dashboard_port",
                "runtime_env",
            ):
                rcfg.pop(k, None)

        ray.init(**rcfg)

        ray_available_resources = ray.available_resources()
        print("ray.available_resources()")
        pprint(ray_available_resources)

        ray_nodes = ray.nodes()
        print("ray.nodes()")
        pprint(ray_nodes)

        _savepath_ray_resources = os.path.join(
            self.engine_dir, "ray_available_resources.json"
        )
        _savepath_ray_nodes = os.path.join(self.engine_dir, "ray_nodes.json")

        with open(_savepath_ray_resources, "w") as f:
            json.dump(ray_available_resources, f, indent=2, default=str)
            print(f"Saved Ray resources info to: {_savepath_ray_resources}")

        with open(_savepath_ray_nodes, "w") as f:
            json.dump(ray_nodes, f, indent=2, default=str)
            print(f"Saved Ray nodes info to: {_savepath_ray_nodes}")

        ray_nodes_alive = [node for node in ray_nodes if node["Alive"]]
        is_single_node = len(ray_nodes_alive) == 1

        available_gpus = ray_available_resources.get("GPU", 0)
        print(f"Available GPUs: {available_gpus}")

        total_actors = len(self.topology)

        use_fractional_gpu = (
            is_single_node and total_actors > available_gpus and available_gpus > 0
        )

        if available_gpus == 0:
            gpus_per_actor = 0.0
        elif total_actors <= available_gpus:
            gpus_per_actor = 1.0
        else:
            gpus_per_actor = max(available_gpus / total_actors, 0.25)

        print(f"Launching {len(list(self.topology))} Ray Actors")

        self.actor_refs = self._init_ray_actors(gpus_per_actor)

        print(f"Calling setup() on {len(self.actor_refs)} Nodes")
        setup_futures = [
            node.setup.remote(
                total_rounds=self.global_rounds,
            )
            for node in self.actor_refs
        ]
        ray.get(setup_futures)

    def _init_ray_actors(self, gpus_per_actor: float = 1.0) -> List[Node]:
        ray_actor_refs: List[Node] = []

        node_config: NodeConfig
        for node_config in self.topology:
            node_config.log_dir_base = (
                node_config.log_dir_base or self.hydra_cfg.runtime.output_dir
            )

            if node_config.ray_actor_options.num_gpus is None and gpus_per_actor >= 0:
                node_config.ray_actor_options.num_gpus = gpus_per_actor

            print_rule()
            pprint(node_config)

            node_actor = Node.options(**node_config.ray_actor_options).remote(
                **node_config,  # type: ignore[call-arg]
            )
            ray_actor_refs.append(node_actor)

        return ray_actor_refs

    def _save_node_results(self, results: List) -> None:
        os.makedirs(self.results_dir, exist_ok=True)

        print(f"Saving node results to: {self.results_dir}")

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
        try:
            print_rule()
            print(f"Starting Experiment with {len(self.actor_refs)} Nodes")

            experiment_start_time = time.time()

            node_results_futures = []
            for node in self.actor_refs:
                future = node.run_experiment.remote()
                node_results_futures.append(future)

            print(
                f"Waiting for {len(node_results_futures)} nodes to complete experiments...",
                flush=True,
            )

            results = ray.get(node_results_futures)

            print(
                f":heavy_check_mark: All {len(results)} nodes completed successfully!",
                flush=True,
            )

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

            print_rule()
            time.sleep(LOG_FLUSH_DELAY)

            self._results_display.show_experiment_results(
                results,
                experiment_duration,
                self.global_rounds,
                len(self.topology),
            )

        finally:
            print("Shutting down...", flush=True)
            ray.shutdown()
