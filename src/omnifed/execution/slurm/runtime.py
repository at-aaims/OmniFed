from __future__ import annotations

import json
import os
from typing import Any, Optional

from omegaconf import OmegaConf

from ...engine_communication import is_hierarchical_cfg, resolve_slurm_ntasks
from ...execution.shared import uses_torchtitan
from src.omnifed.data.federated_shards import topology_has_server
from .slurm_launcher import (
    SlurmOnlyLauncher,
    allocation_slot_count,
    resolve_slurm_frozen_cfg_path,
    tasks_per_allocated_node,
)
from ...utils import print
from .config import SlurmConfig
from .environment import build_frontier_setup_lines
from .torchtitan_launcher import TorchTitanSlurmLauncher


def frontier_setup_lines() -> list[str]:
    """Site env previously hardcoded in Engine._setup (behavior freeze)."""
    return [
        "module load PrgEnv-gnu/8.6.0",
        "module load rocm/6.4.1",
        "module load rccl-net-plugin",
        "module load craype-accel-amd-gfx90a",
        "module load miniforge3/23.11.0-0",
        'export OMNIFED_DATA_DIR="/lustre/orion/gen150/scratch/shruti2395/omnifed_data"',
        'mkdir -p "$OMNIFED_DATA_DIR"',
        'echo "[setup] OMNIFED_DATA_DIR=$OMNIFED_DATA_DIR"',
        'export PYEXE="${PYEXE:-/ccs/home/shruti2395/.conda/envs/pytorch_rocm/bin/python}"',
        'echo "[setup] PYEXE=$PYEXE"',
        '"$PYEXE" -c "import torch; print(torch.__version__)"',
        "",
        "export MIOPEN_USER_DB_PATH=/tmp/${USER}/miopen-cache",
        "export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}",
        "export MIOPEN_FIND_MODE=1",
        'mkdir -p "$MIOPEN_USER_DB_PATH"',
    ]


class SlurmRuntime:
    """Freeze config, sbatch, and exit. Worker is still ``slurm_worker``."""

    def __init__(
        self,
        *,
        cfg: Any,
        hydra_cfg: Any,
        topology: Any,
        output_dir: str,
        engine_dir: str,
        repo_root: str,
    ) -> None:
        self.cfg = cfg
        self.hydra_cfg = hydra_cfg
        self.topology = topology
        self.output_dir = output_dir
        self.engine_dir = engine_dir
        self.repo_root = repo_root

    def setup(self) -> None:

        titan = uses_torchtitan(self.cfg)
        worker_name = "torchtitan_worker.py" if titan else "slurm_worker.py"
        if "SLURM_JOB_ID" in os.environ:
            print(f"[Engine] Inside Slurm allocation; {worker_name} handles execution.")
            raise SystemExit(0)

        cfg_json_path = resolve_slurm_frozen_cfg_path(self.output_dir)
        from src.omnifed.checkpoint.hybrid_round_checkpoint import (
            resolve_experiment_checkpoint_dir,
        )

        ckpt_dir = resolve_experiment_checkpoint_dir(self.cfg) or os.path.join(
            self.engine_dir, "ckpt"
        )

        frozen = {
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
            "hydra_output_dir": self.hydra_cfg.runtime.output_dir,
            "slurm_checkpoint_dir": ckpt_dir,
        }
        with open(cfg_json_path, "w") as f:
            json.dump(frozen, f, indent=2)

        slurm_dict = OmegaConf.to_container(self.cfg.slurm, resolve=True)
        sconf = SlurmConfig(**slurm_dict)

        sconf.work_dir = self.repo_root
        sconf.cfg_json_path = cfg_json_path
        sconf.stdout = os.path.join(self.hydra_cfg.runtime.output_dir, "slurm-%j.out")
        sconf.stderr = os.path.join(self.hydra_cfg.runtime.output_dir, "slurm-%j.err")
        sconf.pyexe = "python"
        # sconf.setup_lines = frontier_setup_lines()
        sconf.setup_lines = build_frontier_setup_lines()

        if titan:
            self._submit_torchtitan(sconf)
            return

        topo_nodes = len(list(self.topology))
        sconf.ntasks = resolve_slurm_ntasks(self.cfg, topo_nodes)

        if is_hierarchical_cfg(self.cfg):
            print(
                f"[Engine] topology=hierarchical: Slurm --ntasks={sconf.ntasks} "
                f"(len(topology)={topo_nodes}).",
                flush=True,
            )

        if sconf.ntasks_per_node and sconf.ntasks_per_node > 0:
            needed_nodes = (
                sconf.ntasks + sconf.ntasks_per_node - 1
            ) // sconf.ntasks_per_node
            prev_nodes = sconf.nodes
            sconf.nodes = max(sconf.nodes, needed_nodes)
            if sconf.nodes != prev_nodes:
                print(
                    f"[Engine] slurm.nodes raised {prev_nodes} -> {sconf.nodes} "
                    f"(need >= ceil(ntasks={sconf.ntasks}/ntasks_per_node="
                    f"{sconf.ntasks_per_node})={needed_nodes})",
                    flush=True,
                )

        placement = tasks_per_allocated_node(
            int(sconf.ntasks), int(sconf.nodes), int(sconf.ntasks_per_node)
        )
        alloc_n = allocation_slot_count(
            int(sconf.ntasks), int(sconf.nodes), int(sconf.ntasks_per_node)
        )
        if (
            sconf.gpus_per_node
            and int(sconf.gpus_per_node) > 0
            and int(sconf.ntasks_per_node) > int(sconf.gpus_per_node)
        ):
            prev_gpus = sconf.gpus_per_node
            sconf.gpus_per_node = int(sconf.ntasks_per_node)
            print(
                f"[Engine] slurm.gpus_per_node raised {prev_gpus} -> "
                f"{sconf.gpus_per_node} (need >= ntasks_per_node="
                f"{sconf.ntasks_per_node})",
                flush=True,
            )
        print(
            f"[Engine] worker_ntasks={sconf.ntasks} alloc_ntasks={alloc_n} "
            f"rank placement tasks/node={placement} "
            f"(rank 0 = gRPC server on first host)",
            flush=True,
        )

        SlurmOnlyLauncher.submit_or_exit(sconf)

    def _submit_torchtitan(self, sconf: SlurmConfig) -> None:
        subclusters = OmegaConf.select(self.cfg, "torchtitan.subclusters", default=None)
        if subclusters is None:
            raise ValueError("TorchTitan requires torchtitan.subclusters")
        if not bool(OmegaConf.select(subclusters, "enabled", default=False)):
            raise ValueError(
                "TorchTitan requires torchtitan.subclusters.enabled=true"
            )

        num_clients = int(self.topology.num_clients)
        has_server = topology_has_server(self.topology)
        nodes_per_client = int(subclusters.nodes_per_client)
        gpus_per_node = int(subclusters.gpus_per_node)

        extra_server = 1 if has_server else 0
        sconf.nodes = extra_server + num_clients * nodes_per_client
        sconf.ntasks = None
        sconf.ntasks_per_node = gpus_per_node
        sconf.gpus_per_node = gpus_per_node
        sconf.gpus_per_task = None
        sconf.gres = None

        if has_server:
            print(
                f"[Engine] Titan client: nodes={sconf.nodes} "
                f"(1 server + {num_clients} clients × {nodes_per_client} nodes), "
                f"gpus_per_node={gpus_per_node}",
                flush=True,
            )
        else:
            print(
                f"[Engine] Titan client: nodes={sconf.nodes} "
                f"(topology has no server, {num_clients} clients × {nodes_per_client} nodes), "
                f"gpus_per_node={gpus_per_node}",
                flush=True,
            )

        federated = OmegaConf.select(self.cfg, "torchtitan.federated", default=None)
        if federated is None:
            raise ValueError("TorchTitan requires torchtitan.federated")

        launcher_config = {
            "subclusters": OmegaConf.to_container(subclusters, resolve=True),
            "server_port": int(federated.server_port),
            "has_server": has_server,
        }
        launcher_config["subclusters"]["num_clients"] = num_clients
        TorchTitanSlurmLauncher.submit_or_exit(
            sconf=sconf,
            launcher_cfg=launcher_config,
        )
