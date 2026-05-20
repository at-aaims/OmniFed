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
