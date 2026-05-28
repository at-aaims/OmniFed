"""Optional env-based address overrides for multi-node hybrid (no torch dependency)."""

from __future__ import annotations

import os

from omegaconf import open_dict

__all__ = ["apply_hybrid_addr_env_overrides"]


def apply_hybrid_addr_env_overrides(cfg) -> None:
    """
    Multi-node: set ``OMNIFED_HYBRID_RPC_ADDR`` and
    ``OMNIFED_HYBRID_FACILITY_MPI_ADDRS`` (comma-separated, one host per
    facility, typically each facility leader's node).
    Optional ``OMNIFED_HYBRID_RPC_PORT`` (int).
    """
    rpc_addr = os.environ.get("OMNIFED_HYBRID_RPC_ADDR")
    rpc_port = os.environ.get("OMNIFED_HYBRID_RPC_PORT")
    fac_mpi = os.environ.get("OMNIFED_HYBRID_FACILITY_MPI_ADDRS")
    if not (rpc_addr or rpc_port or fac_mpi):
        return

    with open_dict(cfg.topology):
        if rpc_addr:
            cfg.topology.rpc.addr = rpc_addr
        if rpc_port:
            cfg.topology.rpc.port = int(rpc_port)

    if fac_mpi:
        parts = [x.strip() for x in fac_mpi.split(",") if x.strip()]
        nfs = len(cfg.topology.facilities)
        if len(parts) != nfs:
            raise ValueError(
                f"OMNIFED_HYBRID_FACILITY_MPI_ADDRS has {len(parts)} comma-separated "
                f"hosts; expected {nfs} (one rendezvous host per facility)."
            )
        for i, fac in enumerate(cfg.topology.facilities):
            with open_dict(fac.mpi):
                fac.mpi.addr = parts[i]
