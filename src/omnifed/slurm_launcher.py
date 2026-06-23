from __future__ import annotations

import base64
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional


def _inside_slurm() -> bool:
    return "SLURM_JOB_ID" in os.environ


@dataclass
class SlurmConfig:
    enabled: bool = False
    account: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    time: str = "02:00:00"
    nodes: int = 2
    ntasks_per_node: int = 1
    cpus_per_task: int = 8

    # GPU options
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

    # launcher will set this so Slurm world size == topology size
    ntasks: Optional[int] = None

    # runtime fields (set by Engine)
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
            lines.append(f"#SBATCH --ntasks={self.ntasks}")
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

        # GPU request section
        if self.gres:
            lines.append(f"#SBATCH --gres={self.gres}")
        elif self.gpus_per_node and self.gpus_per_node > 0:
            lines.append(f"#SBATCH --gpus-per-node={int(self.gpus_per_node)}")

        if self.gpus_per_task is not None:
            lines.append(f"#SBATCH --gpus-per-task={int(self.gpus_per_task)}")

        return lines


class SlurmOnlyLauncher:
    """
    Submit sbatch from outside Slurm; inside allocation the worker module drives the run.
    """

    @staticmethod
    def submit_or_exit(sconf: SlurmConfig) -> None:
        assert not _inside_slurm(), "submit_or_exit() must be called outside Slurm."
        assert sconf.work_dir and sconf.cfg_json_path, "work_dir and cfg_json_path must be set"

        pyexe = sconf.pyexe or os.getenv("PYEXE") or "python"

        # Read JSON and base64-encode so we can broadcast to all nodes without scp
        with open(sconf.cfg_json_path, "rb") as f:
            payload_b64 = base64.b64encode(f.read()).decode("ascii")

        cfg_dir = os.path.dirname(sconf.cfg_json_path)

        # Compute project root (directory that contains "src")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        lines: List[str] = ["#!/bin/bash"]
        lines += sconf.sbatch_lines()
        lines += [
            "set -euo pipefail",
            f'export PYTHONPATH="${{PYTHONPATH:-}}:{repo_root}"',
            'export PYTHONUNBUFFERED=1',
            'export HYDRA_FULL_ERROR=1',
            'export OMNIFED_DEBUG=${OMNIFED_DEBUG:-0}',
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
            # Ensure the JSON directory exists on every node
            f'srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -lc {shlex.quote(f"mkdir -p {cfg_dir}")}',
            "",
            # Write identical JSON file on each node by decoding embedded base64
            f'export OMNIFED_CFG_B64="{payload_b64}"',
            'srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -lc ' +
            shlex.quote(f'echo "$OMNIFED_CFG_B64" | base64 -d > {sconf.cfg_json_path}'),
            'srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -lc ' +
            shlex.quote(
                f'echo "[$(hostname)] wrote {sconf.cfg_json_path}; size=$(stat -c%s {sconf.cfg_json_path}) bytes"'
            ),
            "",
            # Frontier / AMD fix for Ray import path
            "set -x",
            "srun --export=ALL bash -lc " + shlex.quote(
                'if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then '
                'export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"; '
                'fi; '
                'unset ROCR_VISIBLE_DEVICES; '
                'echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"; '
                'echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"; '
                'echo "Using worker python: ${PYEXE:-' + pyexe + '}"; '
                '${PYEXE:-' + pyexe + '} -u -m src.omnifed.slurm_worker '
                '--cfg-json ' + shlex.quote(sconf.cfg_json_path)
            ),
            "set +x",
        ]

        script = "\n".join(lines) + "\n"

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
