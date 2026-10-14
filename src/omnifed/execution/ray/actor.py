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
import time
import warnings
from typing import Any, Dict, Optional

import ray
import torch
from hydra.utils import instantiate
from torch import nn

from ...algorithm import (
    BaseAlgorithm,
    BaseAlgorithmConfig,
)
from ...communicator import (
    AggregationOp,
    BaseCommunicator,
    BaseCommunicatorConfig,
)
from ...data import (
    DataModule,
    DataModuleConfig,
    apply_federated_shard_env,
)
from ...model import ModelConfig
from ...summary import (
    instantiate_model_timed,
    move_model_to_device_timed,
    emit_model_startup,
)
from ...utils import RequiredSetup, print


@ray.remote
class RayActor(RequiredSetup):
    """
    Distributed federated learning participant (server or client).

    Ray actor that executes FL algorithms with local data and model state.
    Each node manages its own training loop, model updates, and communication with other nodes.

    Nodes execute FL rounds autonomously once Engine calls run_experiment().

    See conf/ directory for topology examples.
    """

    def __init__(
        self,
        name: str,
        local_comm: BaseCommunicatorConfig,
        global_comm: Optional[BaseCommunicatorConfig],
        # ---
        algorithm: BaseAlgorithmConfig,
        model: ModelConfig,
        datamodule: DataModuleConfig,
        # ---
        log_dir_base: str,
        device_hint: str,
        has_server: bool = True,
    ):
        """
        Initialize federated learning node with configs.

        Args:
            name: Unique node identifier (e.g., "0.1" for group 0, rank 1)
            local_comm: Communication config for intra-group coordination
            global_comm: Communication config for inter-group coordination (hierarchical only)
            algorithm: FL algorithm config
            model: Neural network model config
            datamodule: Data loading and preprocessing config
            device_hint: Device placement ("auto", "cpu", "cuda:X", etc.)
            exp_dir: Base experiment directory for logs and outputs
        """
        super().__init__()
        self.name: str = name
        os.environ["OMNIFED_NODE_NAME"] = name

        self.device_hint: str = device_hint
        self.log_dir: str = os.path.join(log_dir_base, name)

        # Store config for model (instantiated during setup)
        self.model_cfg: ModelConfig = model

        # Instantiate components with setup phases
        self.local_comm: BaseCommunicator = instantiate(local_comm)
        self.global_comm: Optional[BaseCommunicator] = (
            instantiate(global_comm) if global_comm else None
        )

        self.algorithm: BaseAlgorithm = instantiate(
            algorithm,
            log_dir=self.log_dir,
        )

        if os.environ.get("OMNIFED_FEDERATED_CLIENT_INDEX") in (None, ""):
            comm_rank = int(self.local_comm.rank)
            world = int(self.local_comm.world_size)
            n_trainers = max(world - 1, 1) if has_server else world
            apply_federated_shard_env(
                rank=comm_rank, num_trainers=n_trainers, has_server=has_server
            )

        self.datamodule: DataModule = instantiate(datamodule)
        # Deferred instantiation
        self.__device: Optional[torch.device] = None


    def get_runtime_identity(self) -> Dict[str, Any]:
        """
        Return the actor's TorchDist rank and its Ray-node IP address.

        This is called before communicator setup so the runtime can discover
        where local rank 0 was scheduled.
        """
        return {
            "name": self.name,
            "local_rank": int(self.local_comm.rank),
            "node_ip": ray.util.get_node_ip_address(),
        }

    def set_local_master_addr(self, master_addr: str) -> None:
        """
        Set the TorchDist rendezvous address before process-group setup.
        """
        if self.local_comm.is_ready:
            raise RuntimeError(
                "Cannot change local_comm.master_addr after communicator setup."
            )

        self.local_comm.master_addr = str(master_addr)

        print(
            f"Configured local TorchDist master: "
            f"rank={self.local_comm.rank}, "
            f"master_addr={self.local_comm.master_addr}, "
            f"master_port={self.local_comm.master_port}",
            flush=True,
        )

    def _setup(self, total_rounds: int) -> None:
        """
        Instantiate remaining components and establish connections.

        Called by Engine after all nodes are created but before experiment starts.
        Instantiates model, establishes communicator connections,
        and passes dependencies to algorithm.
        """
        model, model_load_s = instantiate_model_timed(self.model_cfg)

        # Establish communicator connections
        self.local_comm.setup()
        if self.global_comm:
            self.global_comm.setup()
        
        self.original_device = next(model.parameters()).device
        model, model_to_device_s = move_model_to_device_timed(model, self.device)
        emit_model_startup(
            rank=int(self.local_comm.rank),
            log_dir=self.log_dir,
            model_load_s=model_load_s,
            model_to_device_s=model_to_device_s,
            algorithm=self.algorithm,
        )

        # Standard federated learning setup: broadcast initial model from server
        # In hierarchical topologies: global comm first, then local comm
        _t_init_total_start = time.time()

        _t_bcast_total_start = time.time()
        _t_bcast_global_start = time.time()
        if self.global_comm:
            model = self.global_comm.broadcast(model)
        _t_bcast_global_end = time.time()

        _t_bcast_local_start = time.time()
        model = self.local_comm.broadcast(model)
        _t_bcast_local_end = time.time()
        _t_bcast_total_end = time.time()

        # Discover distributed training parameters for synchronized execution
        _t_agg_start = time.time()
        local_iters_per_epoch = (
            len(self.datamodule.train) if self.datamodule.train is not None else 0
        )

        # Find global maximum iterations and epochs (batched for efficiency)
        group_max_epochs_and_iters = self.local_comm.aggregate(
            dict(
                iters_per_epoch=torch.tensor(
                    local_iters_per_epoch,
                    dtype=torch.int,
                    device = self.device,
                ),
                epochs_per_round=torch.tensor(
                    self.algorithm.max_epochs_per_round,
                    dtype=torch.int,
                    device = self.device
                ),
            ),
            AggregationOp.MAX,
        )
        _t_agg_end = time.time()

        self.algorithm.setup(
            self.local_comm,
            self.global_comm,
            model,
            self.datamodule,
            int(group_max_epochs_and_iters["iters_per_epoch"].item()),
            int(group_max_epochs_and_iters["epochs_per_round"].item()),
            total_rounds,
        )
        if self.global_comm:
            self.global_comm.set_logger(self.algorithm)
        
        self.local_comm.set_logger(self.algorithm)

        if self.global_comm:
            self.global_comm.set_logger(self.algorithm)

        self.local_comm.set_logger(self.algorithm)

        _t_init_total_end = time.time()

        # Log initialization timing metrics individually (uses current context)
        self.algorithm.log_metric(
            "comm_time/bcast_global", _t_bcast_global_end - _t_bcast_global_start
        )
        self.algorithm.log_metric(
            "comm_time/bcast_local", _t_bcast_local_end - _t_bcast_local_start
        )
        self.algorithm.log_metric(
            "comm_time/bcast_total", _t_bcast_total_end - _t_bcast_total_start
        )
        self.algorithm.log_metric("comm_time/agg", _t_agg_end - _t_agg_start)
        self.algorithm.log_metric(
            "comm_time/total", _t_init_total_end - _t_init_total_start
        )

    def run_experiment(self) -> Dict[str, Any]:
        """
        Execute federated learning experiment autonomously.

        Runs the complete experiment lifecycle.
        Moves model to compute device, executes all FL rounds via the algorithm.
        Collects timeline data and restores model to original device afterward.

        Returns:
            Timeline data containing metrics with FL coordinates
        """
        if not self.is_ready:
            raise RuntimeError("Ray actor not ready - call setup() first")

        print(
            "Ray actor starting experiment",
            flush=True,
        )

        # Device management for experiment execution
        # original_device = next(self.algorithm.local_model.parameters()).device
        self.algorithm.local_model = self.algorithm.local_model.to(self.device)

        try:
            for round_idx in range(self.algorithm.max_rounds):
                self.algorithm.round_exec(round_idx, self.algorithm.max_rounds)

            print(
                "Ray actor completed experiment",
                flush=True,
            )

        finally:
            # Restore original device placement
            self.algorithm.local_model = self.algorithm.local_model.to(self.original_device)
            print(
                f"Model restored to original device: {self.original_device}",
                flush=True,
            )

        # Return experiment timeline data for display purposes
        return self.algorithm.get_experiment_data()

    def __repr__(self) -> str:
        """Ray actor string representation with name and timestamp."""
        _time = time.strftime("%H:%M:%S", time.gmtime())
        return f"{self.name} {_time}"

    @property
    def device(self) -> torch.device:
        """Compute device with automatic GPU assignment based on rank."""
        backend = getattr(self.local_comm, "backend", "gloo").lower()
        use_cuda = (backend == "nccl") and torch.cuda.is_available()
        if self.__device is None:
            if use_cuda:
                self.__device = self.__resolve_device(
                    self.device_hint, rank=self.local_comm.rank
                )
            else:
                self.__device = torch.device("cpu")

        return self.__device

    @staticmethod
    def __resolve_device(device_hint: str, rank: Optional[int] = None) -> torch.device:
        """
        Resolve device placement for this node.

        Args:
            device_hint: Device specification ("auto", "cpu", "cuda", "cuda:X", etc.)
            rank: Process rank for round-robin GPU assignment (optional)

        Returns:
            PyTorch device for computation
        """
        if device_hint != "auto":
            print(f"Explicit: {device_hint}")
            return torch.device(device_hint)

        # Auto-assignment with GPU detection
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print("Auto: CPU (no GPUs available)")
            return torch.device("cpu")

        # Round-robin GPU assignment
        effective_rank = rank if rank is not None else 0
        if rank is None:
            warnings.warn("No rank provided, defaulting to GPU 0")

        gpu_id = effective_rank % gpu_count
        device_str = f"cuda:{gpu_id}"
        print(f"Auto: {device_str} (rank {effective_rank}, {gpu_count} GPUs)")
        return torch.device(device_str)
