from __future__ import annotations

import base64
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

# Frozen contract: one srun, 1 GPU = 1 client.
SLURM_WORKER_MODULE = "src.omnifed.execution.slurm.slurm_worker"


def _inside_slurm() -> bool:
    return "SLURM_JOB_ID" in os.environ


def resolve_slurm_frozen_cfg_path(hydra_output_dir: str) -> str:
    """Absolute path to per-run ``engine_frozen.json`` (under the Hydra run directory)."""
    return os.path.abspath(os.path.join(hydra_output_dir, "engine_frozen.json"))


def tasks_per_allocated_node(
    worker_ntasks: int, nodes: int, ntasks_per_node: int
) -> List[int]:
    """How many worker ranks sit on each allocated node, rank 0 first.

    Packing A (7×1): ``[1, 1, 1, 1, 1, 1, 1]``.
    Packing B (2 nodes, 6 GPUs on the second): ``[1, 6]`` so the gRPC
    **server** is alone on node 0 and trainers are ``SLURM_LOCALID`` 0..5
    on node 1. Not hardcoded 7: yaml ``nodes`` / ``ntasks_per_node`` plus
    Engine ``worker_ntasks = len(topology)``.
    """
    worker_ntasks = int(worker_ntasks)
    nodes = int(nodes)
    ntasks_per_node = int(ntasks_per_node)
    if worker_ntasks < 1 or nodes < 1 or ntasks_per_node < 1:
        raise ValueError(
            f"worker_ntasks={worker_ntasks}, nodes={nodes}, "
            f"ntasks_per_node={ntasks_per_node} must all be >= 1"
        )
    rest = nodes - 1
    first = worker_ntasks - rest * ntasks_per_node
    if first < 1:
        raise ValueError(
            "Rank 0 (gRPC server) needs at least one task on the first node. "
            f"Got worker_ntasks={worker_ntasks}, nodes={nodes}, "
            f"ntasks_per_node={ntasks_per_node}."
        )
    if first > ntasks_per_node:
        raise ValueError(
            f"First node would need {first} tasks but ntasks_per_node="
            f"{ntasks_per_node}. Raise slurm.nodes or slurm.ntasks_per_node."
        )
    counts = [first] + [ntasks_per_node] * rest
    if sum(counts) != worker_ntasks:
        raise ValueError(
            f"placement {counts} does not sum to worker_ntasks={worker_ntasks}"
        )
    return counts


def allocation_slot_count(worker_ntasks: int, nodes: int, ntasks_per_node: int) -> int:
    """SBATCH ``--ntasks`` so every node has ``ntasks_per_node`` slots.

    Packing B needs 6 slots on *both* nodes (12) even though only 7 worker
    ranks run; otherwise Slurm packs 6+1 with rank 0 on the GPU node.
    """
    tasks_per_allocated_node(worker_ntasks, nodes, ntasks_per_node)
    return int(nodes) * int(ntasks_per_node)


@dataclass
class SlurmConfig:
    enabled: bool = False
    account: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    time: str = "02:00:00"
    nodes: int = 2
    ntasks_per_node: int = 1
    cpus_per_task: int = 6

    gres: Optional[str] = None
    gpus_per_node: int = 0
    gpus_per_task: Optional[int] = None
    gpu_bind: str = "closest"

    job_name: str = "omnifed"
    exclusive: bool = False
    constraint: Optional[str] = None
    reservation: Optional[str] = None
    setup_lines: List[str] = field(default_factory=list)

    checkpoint_dir: Optional[str] = None
    experiment_id: Optional[str] = None
    resume: bool = False
    dependency_singleton: bool = False
    preempt_signal: str = "USR1"
    preempt_notice_sec: int = 180
    resume_from: Optional[str] = None

    ntasks: Optional[int] = None

    work_dir: Optional[str] = None
    cfg_json_path: Optional[str] = None
    pyexe: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

    def sbatch_lines(self) -> List[str]:
        lines = [
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --ntasks-per-node={self.ntasks_per_node}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --time={self.time}",
            f"#SBATCH --signal=B:{self.preempt_signal}@{self.preempt_notice_sec}",
        ]

        if self.ntasks:
            alloc = allocation_slot_count(
                int(self.ntasks), int(self.nodes), int(self.ntasks_per_node)
            )
            lines.append(f"#SBATCH --ntasks={alloc}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.exclusive:
            lines.append("#SBATCH --exclusive")
        if self.dependency_singleton:
            lines.append("#SBATCH -d singleton")
        if self.qos:
            lines.append(f"#SBATCH --qos={self.qos}")
        if self.constraint:
            lines.append(f"#SBATCH --constraint={self.constraint}")
        if self.reservation:
            lines.append(f"#SBATCH --reservation={self.reservation}")
        if self.stdout:
            lines.append(f"#SBATCH -o {self.stdout}")
        if self.stderr:
            lines.append(f"#SBATCH -e {self.stderr}")
        if self.work_dir:
            lines.append(f"#SBATCH --chdir={self.work_dir}")

        if self.gres:
            lines.append(f"#SBATCH --gres={self.gres}")
        elif self.gpus_per_node and self.gpus_per_node > 0:
            lines.append(f"#SBATCH --gpus-per-node={int(self.gpus_per_node)}")

        if self.gpus_per_task is not None:
            lines.append(f"#SBATCH --gpus-per-task={int(self.gpus_per_task)}")

        return lines


def build_sbatch_script(sconf: SlurmConfig, *, pyexe: str) -> str:
    """Build the sbatch body. Worker entrypoint is ``SLURM_WORKER_MODULE``."""
    assert sconf.work_dir and sconf.cfg_json_path, "work_dir and cfg_json_path must be set"

    with open(sconf.cfg_json_path, "rb") as f:
        payload_b64 = base64.b64encode(f.read()).decode("ascii")

    cfg_dir = os.path.dirname(sconf.cfg_json_path)
    repo_root = sconf.work_dir

    lines: List[str] = ["#!/bin/bash"]
    lines += sconf.sbatch_lines()
    lines += [
        "set -euo pipefail",
        f'export PYTHONPATH="${{PYTHONPATH:-}}:{repo_root}"',
        "export PYTHONUNBUFFERED=1",
        "export HYDRA_FULL_ERROR=1",
        "export OMNIFED_DEBUG=${OMNIFED_DEBUG:-0}",
        "",
    ]

    if sconf.setup_lines:
        lines += sconf.setup_lines + [""]

    lines += [
        'echo "SLURM_JOB_ID=$SLURM_JOB_ID"',
        'echo "SLURM_NODELIST=$SLURM_NODELIST"',
        'echo "Running on $(hostname)"',
        'echo "PYTHONPATH=$PYTHONPATH"',
        f'echo "Requested PYEXE={pyexe}"',
        "",
        f'srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -lc {shlex.quote(f"mkdir -p {cfg_dir}")}',
        "",
        f'export OMNIFED_CFG_B64="{payload_b64}"',
        "srun -N \"$SLURM_JOB_NUM_NODES\" -n \"$SLURM_JOB_NUM_NODES\" --ntasks-per-node=1 bash -lc "
        + shlex.quote(f'echo "$OMNIFED_CFG_B64" | base64 -d > {sconf.cfg_json_path}'),
        "srun -N \"$SLURM_JOB_NUM_NODES\" -n \"$SLURM_JOB_NUM_NODES\" --ntasks-per-node=1 bash -lc "
        + shlex.quote(
            f'echo "[$(hostname)] wrote {sconf.cfg_json_path}; size=$(stat -c%s {sconf.cfg_json_path}) bytes"'
        ),
        "",
    ]

    worker_cmd = (
        'if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then '
        'export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"; '
        "fi; "
        "unset ROCR_VISIBLE_DEVICES; "
        'echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"; '
        'echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"; '
        'echo "Using worker python: ${PYEXE:-' + pyexe + '}"; '
        'echo "SLURM_PROCID=${SLURM_PROCID:-?} SLURM_LOCALID=${SLURM_LOCALID:-?} '
        'SLURM_NTASKS=${SLURM_NTASKS:-?} host=$(hostname)"; '
        "${PYEXE:-" + pyexe + "} -u -m " + SLURM_WORKER_MODULE + " "
        "--cfg-json " + shlex.quote(sconf.cfg_json_path)
    )
    worker_srun = "srun --export=ALL bash -lc " + shlex.quote(worker_cmd)
    if sconf.ntasks:
        worker_n = int(sconf.ntasks)
        counts = tasks_per_allocated_node(
            worker_n, int(sconf.nodes), int(sconf.ntasks_per_node)
        )
        hostfile_path = os.path.join(cfg_dir, "slurm_rank_hosts")
        counts_bash = " ".join(str(c) for c in counts)
        lines += [
            f"WORKER_NTASKS={worker_n}",
            f"COUNTS=({counts_bash})",
            'mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_NODELIST")',
            'echo "SLURM hosts: ${HOSTS[*]}"',
            'if [ "${#HOSTS[@]}" -ne "${#COUNTS[@]}" ]; then',
            '  echo "[omnifed] host count ${#HOSTS[@]} != placement ${#COUNTS[@]}" >&2',
            '  exit 1',
            "fi",
            f"RANK_HOSTFILE={shlex.quote(hostfile_path)}",
            ': > "$RANK_HOSTFILE"',
            'for i in "${!COUNTS[@]}"; do',
            "  for ((k=0; k<COUNTS[i]; k++)); do",
            '    echo "${HOSTS[i]}" >> "$RANK_HOSTFILE"',
            "  done",
            "done",
            'echo "=== rank hostfile (rank 0 = gRPC server node) ==="',
            'nl -ba "$RANK_HOSTFILE"',

            "",
        ]
        # Frontier srun has no --hostfile. Slurm reads the list from SLURM_HOSTFILE
        # when --distribution=arbitrary (packing B: 1+6 ranks, not 7×1).
        worker_srun = (
            "export SLURM_HOSTFILE=\"$RANK_HOSTFILE\" && "
            f'srun --ntasks="{worker_n}" --distribution=arbitrary '
            "--export=ALL bash -lc "
            + shlex.quote(worker_cmd)
        )

    lines += [
        "set -x",
        worker_srun,
        "set +x",
    ]

    return "\n".join(lines) + "\n"


class SlurmOnlyLauncher:
    """One srun: each Slurm task is one OmniFed node (1 GPU per client)."""

    @staticmethod
    def submit_or_exit(sconf: SlurmConfig) -> None:
        assert not _inside_slurm(), "submit_or_exit() must be called outside Slurm."
        assert sconf.work_dir and sconf.cfg_json_path, "work_dir and cfg_json_path must be set"

        pyexe = sconf.pyexe or os.getenv("PYEXE") or "python"
        script = build_sbatch_script(sconf, pyexe=pyexe)

        path = os.path.join(sconf.work_dir, "omnifed_slurm_only.sh")
        with open(path, "w") as f:
            f.write(script)
        os.chmod(path, 0o755)

        print("\n===== Generated sbatch (Slurm-only) =====\n")
        print(script)
        print("===== end sbatch =====\n")

        out = subprocess.check_output(["sbatch", path], text=True).strip()
        print(f"[SlurmOnlyLauncher] sbatch response: {out}")
        raise SystemExit(0)
