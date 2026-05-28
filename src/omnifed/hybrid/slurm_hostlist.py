"""Map ``SLURM_PROCID`` → hostname list for hybrid rendezvous (RPC + facility TCP)."""

from __future__ import annotations

import os
import subprocess

from omegaconf import open_dict

__all__ = ["slurm_job_hosts_ordered", "apply_hosts_to_hybrid_topology"]


def slurm_job_hosts_ordered() -> list[str]:
    nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not nodelist:
        raise RuntimeError("SLURM_JOB_NODELIST / SLURM_NODELIST is not set.")
    out = subprocess.check_output(
        ["scontrol", "show", "hostnames", nodelist],
        text=True,
    )
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("scontrol show hostnames returned no lines.")
    return [h.split(".")[0] for h in lines]


def apply_hosts_to_hybrid_topology(topo, hosts: list[str]) -> None:
    """Patch ``rpc.addr`` and each ``facility.mpi.addr`` (leader node's short hostname)."""
    ws = int(topo.world_size)
    if len(hosts) < ws:
        raise ValueError(
            f"Slurm hostlist has {len(hosts)} nodes; hybrid topology needs world_size={ws}. "
            "Use one node per task (e.g. --ntasks-per-node=1) and match #SBATCH --ntasks."
        )
    srv = int(topo.rpc.server_rank)
    with open_dict(topo.rpc):
        topo.rpc.addr = hosts[srv]
    for fac in topo.facilities:
        lr = int(fac.mpi.leader_rank)
        with open_dict(fac.mpi):
            fac.mpi.addr = hosts[lr]
