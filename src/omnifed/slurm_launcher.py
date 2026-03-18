# src/omnifed/slurm_launcher.py
from __future__ import annotations
import base64
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    #gpus_per_node: int = 0
    # --- GPU options ---
    # Prefer to use gres if your site requires it, e.g. "gpu:1" or "gpu:a100:1"
    gres: Optional[str] = None
    gpus_per_node: int = 0
    gpus_per_task: Optional[int] = None   # if you want Slurm to bind GPUs per task
    gpu_bind: str = "closest"
    job_name: str = "omnifed"
    constraint: Optional[str] = None
    reservation: Optional[str] = None
    setup_lines: List[str] = field(default_factory=list)
    checkpoint_dir: Optional[str] = None
    preempt_signal: str = "USR1"
    preempt_notice_sec: int = 180
    resume_from: Optional[str] = None

    # launcher will set this so Slurm world size == topology size
    ntasks: Optional[int] = None

    # runtime fields (set by Engine)
    work_dir: Optional[str] = None           # if set, we use SBATCH --chdir
    cfg_json_path: Optional[str] = None      # absolute path for engine_frozen.json
    pyexe: Optional[str] = None              # absolute python path
    stdout: Optional[str] = None             # SBATCH -o
    stderr: Optional[str] = None             # SBATCH -e

    def sbatch_lines(self) -> List[str]:
        L = [
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --ntasks-per-node={self.ntasks_per_node}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --gpus-per-node={self.gpus_per_node}",
            f"#SBATCH --time={self.time}",
            f"#SBATCH --signal=B:{self.preempt_signal}@{self.preempt_notice_sec}",
        ]
        if self.ntasks:      L.append(f"#SBATCH --ntasks={self.ntasks}")
        if self.account:     L.append(f"#SBATCH --account={self.account}")
        if self.partition:   L.append(f"#SBATCH --partition={self.partition}")
        if self.qos:         L.append(f"#SBATCH --qos={self.qos}")
        if self.constraint:  L.append(f"#SBATCH --constraint={self.constraint}")
        if self.reservation: L.append(f"#SBATCH --reservation={self.reservation}")
        if self.stdout:      L.append(f"#SBATCH -o {self.stdout}")
        if self.stderr:      L.append(f"#SBATCH -e {self.stderr}")
        if self.work_dir:    L.append(f"#SBATCH --chdir={self.work_dir}")

        # ----- GPU request section -----
        # Prefer GRES if provided (typical on many clusters)
        if self.gres:
            L.append(f"#SBATCH --gres={self.gres}")
        else:
            if self.gpus_per_node and self.gpus_per_node > 0:
                L.append(f"#SBATCH --gpus-per-node={int(self.gpus_per_node)}")

        if self.gpus_per_task is not None:
            # Only add if user wants explicit per-task binding
            L.append(f"#SBATCH --gpus-per-task={int(self.gpus_per_task)}")
        # --------------------------------

        return L

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
        # We add it to PYTHONPATH so -m src.omnifed.slurm_worker always imports.
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
            'echo "SLURM_NODELIST=$SLURM_NODELIST on $(hostname)"',
            f'echo "Using PYEXE={pyexe}"',
            "",
            # Ensure the JSON directory exists on *every* node
            f'srun --ntasks-per-node=1 bash -lc {shlex.quote(f"mkdir -p {cfg_dir}")}',
            # Write identical JSON file on each node by decoding the embedded base64
            f'export OMNIFED_CFG_B64="{payload_b64}"',
            'srun --ntasks-per-node=1 bash -lc ' +
            shlex.quote(f'echo "$OMNIFED_CFG_B64" | base64 -d > {sconf.cfg_json_path}'),
            # Log a quick check
            'srun --ntasks-per-node=1 bash -lc ' +
            shlex.quote(f'echo "[$(hostname)] wrote {sconf.cfg_json_path}; size=$(stat -c%s {sconf.cfg_json_path}) bytes"'),
            "",
            # Finally launch one Python worker per task
            "set -x",
            "srun --export=ALL " +
            shlex.quote(pyexe) + " -u -m src.omnifed.slurm_worker " +
            "--cfg-json " + shlex.quote(sconf.cfg_json_path),
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
