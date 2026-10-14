from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

def tasks_per_allocated_node(
    worker_ntasks: int, nodes: int, ntasks_per_node: int
) -> list[int]:
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

    setup_lines: list[str] = field(
        default_factory=list
    )

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

    def sbatch_lines(self) -> list[str]:
        lines = [
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --nodes={self.nodes}",
            (
                "#SBATCH --ntasks-per-node="
                f"{self.ntasks_per_node}"
            ),
            (
                "#SBATCH --cpus-per-task="
                f"{self.cpus_per_task}"
            ),
            f"#SBATCH --time={self.time}",
            (
                "#SBATCH --signal=B:"
                f"{self.preempt_signal}"
                f"@{self.preempt_notice_sec}"
            ),
            "#SBATCH -C nvme",
            # "#SBATCH --exclusive",
        ]

        # if self.ntasks is not None:
        #     lines.append(
        #         f"#SBATCH --ntasks={self.ntasks}"
        #     )

        if self.ntasks:
            alloc = allocation_slot_count(
                int(self.ntasks), int(self.nodes), int(self.ntasks_per_node)
            )
            lines.append(f"#SBATCH --ntasks={alloc}")

        if self.account:
            lines.append(
                f"#SBATCH --account={self.account}"
            )

        if self.partition:
            lines.append(
                f"#SBATCH --partition={self.partition}"
            )

        if self.exclusive:
            lines.append("#SBATCH --exclusive")

        if self.qos:
            lines.append(
                f"#SBATCH --qos={self.qos}"
            )

        if self.constraint:
            lines.append(
                f"#SBATCH --constraint={self.constraint}"
            )

        if self.reservation:
            lines.append(
                f"#SBATCH --reservation={self.reservation}"
            )

        if self.stdout:
            lines.append(
                f"#SBATCH -o {self.stdout}"
            )

        if self.stderr:
            lines.append(
                f"#SBATCH -e {self.stderr}"
            )

        if self.work_dir:
            lines.append(
                f"#SBATCH --chdir={self.work_dir}"
            )

        if self.gres:
            lines.append(f"#SBATCH --gres={self.gres}")
        elif self.gpus_per_node and self.gpus_per_node > 0:
            lines.append(f"#SBATCH --gpus-per-node={int(self.gpus_per_node)}")

        if self.gpus_per_task is not None:
            lines.append(f"#SBATCH --gpus-per-task={int(self.gpus_per_task)}")

        return lines

