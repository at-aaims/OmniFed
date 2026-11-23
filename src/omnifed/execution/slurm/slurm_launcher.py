from __future__ import annotations

import base64
import os
import shlex
import subprocess
from typing import List

from .config import SlurmConfig, allocation_slot_count, tasks_per_allocated_node

# Frozen contract: one srun, 1 GPU = 1 client.
SLURM_WORKER_MODULE = "src.omnifed.execution.slurm.slurm_worker"


def _inside_slurm() -> bool:
    return "SLURM_JOB_ID" in os.environ


def resolve_slurm_frozen_cfg_path(hydra_output_dir: str) -> str:
    """Absolute path to per-run ``engine_frozen.json`` (under the Hydra run directory)."""
    return os.path.abspath(os.path.join(hydra_output_dir, "engine_frozen.json"))


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
