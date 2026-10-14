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

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import rich.repr
from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf

from . import utils
from .algorithm import BaseAlgorithmConfig
from .data import DataModuleConfig
from .engine_communication import is_hierarchical_cfg
from .execution import uses_torchtitan, validate_execution_mode
from .execution.ray.runtime import RayRuntime
from .execution.slurm.config import SlurmConfig
from .execution.slurm.runtime import SlurmRuntime
from .model import ModelConfig
from .topology import BaseTopology, BaseTopologyConfig
from .utils import RequiredSetup, ResultsDisplay, print

from dataclasses import asdict, is_dataclass
from omegaconf import OmegaConf
import sys
import shlex


@dataclass
class RayConfig:
    """Ray cluster configuration for distributed federated learning."""

    # ─────────────────────────────────────────
    # Cluster Connection & Resource Allocation
    # ─────────────────────────────────────────

    # Cluster connection (null = auto-detect local cluster)
    # Use "ray://host:port" for remote clusters, "local" to force local
    address: Optional[str] = None

    # Resource allocation - CRITICAL for proper GPU/CPU distribution
    # null = auto-detect based on hardware, explicit numbers override detection
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None

    # Custom resources: {"accelerator_type": "V100", "high_memory": 2}
    resources: Optional[Dict[str, Any]] = None

    # ─────────────────────────────────────────
    # Memory & Performance
    # ─────────────────────────────────────────

    # Object store memory for large model sharing (null = 30% of system memory)
    object_store_memory: Optional[int] = None

    # ─────────────────────────────────────────
    # Monitoring & Development
    # ─────────────────────────────────────────

    # Essential for FL: forward all distributed node logs to main process
    log_to_driver: bool = True

    # Ray dashboard (null = auto-start if dependencies available)
    include_dashboard: Optional[bool] = None
    dashboard_host: str = "127.0.0.1"  # Use "0.0.0.0" for external access
    dashboard_port: Optional[int] = None  # null = auto-find port starting from 8265

    # Development convenience - allow multiple ray.init() calls without error
    ignore_reinit_error: bool = True

    # ─────────────────────────────────────────
    # Advanced Configuration
    # ─────────────────────────────────────────

    # Experiment isolation (null = anonymous namespace)
    namespace: Optional[str] = None

    # Runtime environment for distributed workers (empty = inherit from main process)
    # Example: {"pip": ["torch==1.12.0"], "env_vars": {"CUDA_VISIBLE_DEVICES": "0,1"}}
    runtime_env: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.resources is None:
            self.resources = {}
        if self.runtime_env is None:
            self.runtime_env = {}

@dataclass
class EngineConfig:
    """Main configuration for OmniFed federated learning experiments."""

    # Required experiment parameters
    global_rounds: int = MISSING

    # Optional experiment parameters
    overwrite: bool = False

    # Component configurations - these will be resolved by Hydra defaults
    topology: BaseTopologyConfig = MISSING
    algorithm: BaseAlgorithmConfig = MISSING
    model: ModelConfig = MISSING
    datamodule: DataModuleConfig = MISSING

    # Infrastructure configurations
    ray: RayConfig = field(default_factory=RayConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    # mode: ray | slurm. Topology selects hops. Titan is the client (torchtitan.module).
    engine: Dict[str, Any] = field(
        default_factory=lambda: {
            "mode": "ray",
        }
    )


# Register the config with Hydra's ConfigStore for structured configs
cs = ConfigStore.instance()
cs.store(name="base_config", node=EngineConfig)

@rich.repr.auto
class Engine(RequiredSetup):
    """
    Main engine for federated learning experiments.

    Phase 1: thin dispatcher. ``engine.mode=ray`` uses RayRuntime;
    ``engine.mode=slurm`` uses SlurmRuntime then ``slurm_worker``.
    """

    def __init__(
        self,
        cfg: EngineConfig,
    ):
        super().__init__()
        utils.print_rule()

        self.cfg = cfg
        self.hydra_cfg = HydraConfig.get()
        self.output_dir = str(self.hydra_cfg.runtime.output_dir)
        self.engine_dir = os.path.join(self.output_dir, "engine")
        self.results_dir = os.path.join(self.engine_dir, "node_results")

        self.uses_torchtitan: bool = uses_torchtitan(cfg)
        if cfg.topology is None:
            raise ValueError(
                "topology is required (centralized or decentralized). "
                "Titan is the client, not topology: null."
            )
        self.topology: BaseTopology = instantiate(cfg.topology, _recursive_=False)
        self.global_rounds: int = cfg.global_rounds
        self.overwrite: bool = cfg.overwrite

        self.ray_cfg: RayConfig = cfg.ray

        self.global_rounds = int(cfg.global_rounds)
        self.overwrite = bool(cfg.overwrite)

        self._results_display: ResultsDisplay = ResultsDisplay()
        self.execution_mode: str = validate_execution_mode(
            OmegaConf.select(self.cfg, "engine.mode", default="ray")
        )
        self.ray_runtime: Optional[RayRuntime] = None
        self.slurm_runtime: Optional[SlurmRuntime] = None

    def _setup_output_directories(self) -> None:
        """
        Create and validate output directories for experiment data.

        Creates engine/ and node_results/ directories under Hydra's output path.
        Issues warnings if conflicting experiment files already exist unless overwrite=True.
        """
        if os.path.exists(self.output_dir):
            hydra_standard_files = {".hydra", "main.log", ".gitignore"}
            existing_files = [
                f
                for f in os.listdir(self.output_dir)
                if not f.startswith(".") and f not in hydra_standard_files
            ]
            if existing_files:
                if self.overwrite:
                    warnings.warn(
                        f"Output directory contains existing files: {self.output_dir}\n"
                        f"Found: {existing_files[:5]}{'...' if len(existing_files) > 5 else ''}\n"
                        f"Proceeding with overwrite=True - previous experiment results may be overwritten.",
                        UserWarning,
                    )
                else:
                    raise RuntimeError(
                        f"Output directory contains existing files: {self.output_dir}\n"
                        f"Found: {existing_files[:5]}{'...' if len(existing_files) > 5 else ''}\n"
                        f"This could overwrite previous experiment results. "
                        f"Use a fresh Hydra output directory, clean the existing one, or set overwrite=true."
                    )

        os.makedirs(self.engine_dir, exist_ok=True)

        if os.path.exists(self.engine_dir):
            existing_files = [
                f for f in os.listdir(self.engine_dir) if not f.startswith(".")
            ]
            if existing_files:
                if self.overwrite:
                    warnings.warn(
                        f"Engine directory is not empty: {self.engine_dir}\n"
                        f"Found: {existing_files[:5]}{'...' if len(existing_files) > 5 else ''}\n"
                        f"Proceeding with overwrite=True - conflicting experiment files may be overwritten.",
                        UserWarning,
                    )
                else:
                    raise RuntimeError(
                        f"Engine directory is not empty: {self.engine_dir}\n"
                        f"Found: {existing_files[:5]}{'...' if len(existing_files) > 5 else ''}\n"
                        f"This indicates a conflicting experiment setup. Set overwrite=true to proceed anyway."
                    )

        print(f"Created engine directory: {self.engine_dir}")

    def _setup(self) -> None:
        utils.print_rule()

        self._setup_output_directories()

        mode = self.execution_mode
        if self.uses_torchtitan:
            if mode != "slurm":
                raise ValueError(
                    "A Titan client (torchtitan.module) is only valid with engine.mode=slurm."
                )
        else:
            self.topology.setup(
                default_algorithm_cfg=self.cfg.algorithm,
                default_model_cfg=self.cfg.model,
                default_datamodule_cfg=self.cfg.datamodule,
            )

        if is_hierarchical_cfg(self.cfg) and mode != "slurm":
            raise ValueError(
                "topology: hierarchical is only valid with engine.mode=slurm."
            )
        if is_hierarchical_cfg(self.cfg) and self.uses_torchtitan:
            raise ValueError(
                "TorchTitan is not wired to hierarchical yet. "
                "Use 1-GPU clients with topology: hierarchical."
            )

        if mode == "slurm":
            repo_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            self.slurm_runtime = SlurmRuntime(
                cfg=self.cfg,
                hydra_cfg=self.hydra_cfg,
                topology=self.topology,
                output_dir=self.output_dir,
                engine_dir=self.engine_dir,
                repo_root=repo_root,
            )
            self.slurm_runtime.setup()
            return

        self.ray_runtime = RayRuntime(
            cfg=self.cfg,
            hydra_cfg=self.hydra_cfg,
            topology=self.topology,
            results_display=self._results_display,
        )
        self.ray_runtime.setup()

    def run_experiment(self) -> None:
        if self.execution_mode != "ray":
            raise RuntimeError(
                "run_experiment() is only reached for Ray execution. "
                "Slurm submission should exit during setup()."
            )

        if self.ray_runtime is None:
            raise RuntimeError(
                "Ray runtime is not initialized. Call engine.setup() first."
            )

        self.ray_runtime.run_experiment()
