from __future__ import annotations

import os
import shlex
import subprocess
from typing import Any

from .slurm_launcher import SlurmConfig

TORCHTITAN_WORKER_MODULE = "src.omnifed.execution.slurm.torchtitan_worker"


class TorchTitanSlurmLauncher:
    @staticmethod
    def submit_or_exit(
        sconf: SlurmConfig,
        launcher_cfg: dict[str, Any],
    ) -> None:
        if not sconf.work_dir:
            raise ValueError(
                "Slurm work_dir is required"
            )

        if not sconf.cfg_json_path:
            raise ValueError(
                "Frozen configuration path is required"
            )

        subclusters = launcher_cfg["subclusters"]

        num_clients = int(
            subclusters["num_clients"]
        )

        if num_clients <= 0:
            raise ValueError(
                "num_clients must be positive"
            )
        if "has_server" not in launcher_cfg:
            raise ValueError(
                "has_server is required (from topology.has_server)"
            )
        federated = bool(launcher_cfg["has_server"])

        nodes_per_client = int(
            subclusters["nodes_per_client"]
        )

        if nodes_per_client <= 0:
            raise ValueError(
                "nodes_per_client must be positive"
            )

        gpus_per_node = int(
            subclusters["gpus_per_node"]
        )

        if gpus_per_node <= 0:
            raise ValueError(
                "gpus_per_node must be positive"
            )

        leader_rank = int(
            subclusters["leader_rank"]
        )
        master_port_base = int(
            subclusters["master_port_base"]
        )
        checkpoint_root = str(
            subclusters["checkpoint_root"]
        )

        server_port = int(
            launcher_cfg["server_port"]
        )

        pyexe = (
            sconf.pyexe
            or os.getenv("PYEXE")
            or "python"
        )

        worker_entrypoint = (
            "bash -lc "
            + shlex.quote(
                'if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then '
                'export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"; '
                'fi; '
                'unset ROCR_VISIBLE_DEVICES; '
                'export OMNIFED_PROFILE_TORCH_COMM="${OMNIFED_PROFILE_TORCH_COMM:-0}"; '
                f'export HF_HOME="{checkpoint_root}/hf_cache"; '
                'export HF_DATASETS_CACHE="${HF_HOME}/datasets"; '
                'export TRANSFORMERS_CACHE="${HF_HOME}/transformers"; '
                'mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"; '
                'echo "[worker] hostname=$(hostname) HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"; '
                'echo "[worker] OMNIFED_PROFILE_TORCH_COMM=${OMNIFED_PROFILE_TORCH_COMM:-<unset>}"; '
                'echo "HF_HOME=$HF_HOME"; '
                'echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"; '
                'exec "$PYEXE" -u -m ' + TORCHTITAN_WORKER_MODULE + ' --cfg-json "$CFG_JSON"'
            )
        )

        lines = ["#!/bin/bash"]
        lines += sconf.sbatch_lines()
        lines += ["set -euo pipefail"]

        if sconf.setup_lines:
            lines += sconf.setup_lines + [""]

        lines += [
            f'export PYTHONPATH="{sconf.work_dir}:${{PYTHONPATH:-}}"',
            f'export PYEXE="${{PYEXE:-{pyexe}}}"',
            'mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")',
            f'export CFG_JSON="{sconf.cfg_json_path}"',
            f'CHECKPOINT_ROOT="{checkpoint_root}/job_${{SLURM_JOB_ID}}"',
            'mkdir -p "$CHECKPOINT_ROOT"',
            "",
        ]

        if federated:
            lines += [
                'SERVER_HOST="${HOSTS[0]}"',
                'srun --exclusive --nodes=1 --ntasks=1 '
                '--nodelist="$SERVER_HOST" '
                'env OMNIFED_ROLE=server '
                'FEDERATED_RANK=0 '
                'CHECKPOINT_ROOT="$CHECKPOINT_ROOT" '
                f'{worker_entrypoint} &',
                "",
                f'SERVER_PORT={server_port}',
                'SERVER_READY=0',
                'echo "[setup] waiting for gRPC server at ${SERVER_HOST}:${SERVER_PORT}"',
                'for attempt in $(seq 1 120); do',
                '    if timeout 2 bash -c "</dev/tcp/${SERVER_HOST}/${SERVER_PORT}" 2>/dev/null; then',
                '        SERVER_READY=1',
                '        echo "[setup] gRPC server is reachable"',
                '        break',
                '    fi',
                '    echo "[setup] server not ready: attempt ${attempt}/120"',
                '    sleep 5',
                'done',
                'if [ "$SERVER_READY" -ne 1 ]; then',
                '    echo "[setup] ERROR: gRPC server did not become ready"',
                '    exit 1',
                'fi',
                "",
            ]
        else:
            lines += [
                'echo "[setup] topology has no server: no federated gRPC server srun"',
                "",
            ]

        host0 = 1 if federated else 0
        for client_id in range(num_clients):
            first_node = host0 + client_id * nodes_per_client
            world_size = nodes_per_client * gpus_per_node
            server_env = (
                'SERVER_ADDR="$SERVER_HOST" '
                if federated
                else ""
            )

            lines += [
                f'CLIENT_{client_id}_NODES=$(IFS=,; echo "${{HOSTS[*]:{first_node}:{nodes_per_client}}}")',
                f'CLIENT_{client_id}_MASTER="${{HOSTS[{first_node}]}}"',
                (
                    f"srun --exclusive --nodes={nodes_per_client} "
                    f"--ntasks={world_size} --ntasks-per-node={gpus_per_node} "
                    f'--nodelist="$CLIENT_{client_id}_NODES" '
                    f'env OMNIFED_ROLE=client '
                    f'CLIENT_ID={client_id} '
                    f'FEDERATED_RANK={client_id + 1} '
                    f'CLIENT_LEADER_RANK={leader_rank} '
                    f'CLIENT_MASTER_ADDR="$CLIENT_{client_id}_MASTER" '
                    f'CLIENT_MASTER_PORT={master_port_base + client_id} '
                    f'{server_env}'
                    f'CLIENT_CHECKPOINT_ROOT="$CHECKPOINT_ROOT/client_{client_id}" '
                    f'{worker_entrypoint} &'
                ),
            ]

        lines += [
            "wait",
            'echo "All Torchtitan subclusters completed"',
        ]

        script_path = os.path.join(
            sconf.work_dir, "omnifed_torchtitan_slurm.sh"
        )

        with open(script_path, "w") as file:
            file.write("\n".join(lines) + "\n")

        os.chmod(script_path, 0o755)
        response = subprocess.check_output(
            ["sbatch", script_path], text=True
        ).strip()
        print(response)
        raise SystemExit(0)