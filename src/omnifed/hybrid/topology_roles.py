"""Map global hybrid rank -> facility and facility-local rank (OmegaConf topology)."""

from __future__ import annotations

from typing import Any, Optional


def find_facility_for_global_rank(cfg_topology: Any, global_rank: int) -> Optional[Any]:
    for fac in cfg_topology.facilities:
        members = [int(m) for m in fac.mpi.members]
        if int(global_rank) in members:
            return fac
    return None


def facility_local_rank(facility: Any, global_rank: int) -> int:
    members = [int(m) for m in facility.mpi.members]
    gr = int(global_rank)
    if gr not in members:
        raise RuntimeError(f"global_rank={gr} not in facility members={members}")
    return members.index(gr)


def hybrid_rank_to_centralized_node_index(
    hybrid_rank: int,
    *,
    rpc_server_rank: int,
    world_size: int,
    num_clients: int,
) -> int:
    """
    Map hybrid ``SLURM_PROCID`` to :class:`~src.omnifed.topology.CentralizedTopology` node list index.

    Centralized node ``i`` corresponds to overrides key ``topology.overrides.<i>`` (``i=0`` server,
    ``i=1..num_clients`` clients). Hybrid global ranks omit the RPC server rank from the trainer
    set; trainers are sorted in ascending hybrid rank order and assigned sequential client indices ``1``.

    Preconditions: ``world_size == num_clients + 1`` (one FL server slot + ``num_clients`` trainers).
    The FL server role runs on ``rpc_server_rank`` (typically also hybrid rank excluded from MPI facilities).
    """
    wr = int(world_size)
    hr = int(hybrid_rank)
    rs = int(rpc_server_rank)
    nc = int(num_clients)

    if wr != nc + 1:
        raise ValueError(
            f"Hybrid world_size={wr} must equal num_clients+1={nc + 1} for centralized FL mapping."
        )
    if not (0 <= hr < wr):
        raise ValueError(f"hybrid_rank={hr} out of range for world_size={wr}")
    if not (0 <= rs < wr):
        raise ValueError(f"rpc_server_rank={rs} out of range for world_size={wr}")

    if hr == rs:
        return 0

    trainers = sorted(r for r in range(wr) if r != rs)
    if len(trainers) != nc:
        raise ValueError(
            f"Expected {nc} trainer ranks excluding rpc_server_rank={rs}, got {len(trainers)}: {trainers}"
        )
    return 1 + trainers.index(hr)
