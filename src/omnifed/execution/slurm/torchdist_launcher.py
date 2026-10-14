from __future__ import annotations

import base64
import os
import shlex
import subprocess

from .config import SlurmConfig


def _inside_slurm() -> bool:
    return "SLURM_JOB_ID" in os.environ



class TorchDistSlurmLauncher:
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
        # repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        repo_root = sconf.work_dir

        lines: list[str] = ["#!/bin/bash"]
        lines += sconf.sbatch_lines()
        lines += [
            "set -euo pipefail",
            # f'export PYTHONPATH="${{PYTHONPATH:-}}:{repo_root}"',
            f'export PYTHONPATH="{repo_root}:${{PYTHONPATH:-}}"',
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
                '${PYEXE:-' + pyexe + '} -u -m src.omnifed.execution.slurm.torchdist_worker '
                '--cfg-json ' + shlex.quote(sconf.cfg_json_path)
            ),
            "set +x",
        ]

        script = "\n".join(lines) + "\n"

        path = os.path.join(
            sconf.work_dir,
            "omnifed_torchdist_slurm.sh",
        )
        with open(path, "w") as f:
            f.write(script)
        os.chmod(path, 0o755)

        print("\n===== Generated sbatch (Slurm-only) =====\n")
        print(script)
        print("===== end sbatch =====\n")

        out = subprocess.check_output(["sbatch", path], text=True).strip()
        print(
            "[TorchDistSlurmLauncher] "
            f"sbatch response: {out}"
        )
        raise SystemExit(0)