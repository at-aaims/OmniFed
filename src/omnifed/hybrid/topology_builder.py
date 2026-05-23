"""
Pure builder for hybrid Slurm/engine layouts: one dedicated gRPC server rank,
multiple facilities, each with its own torch distributed (MPI-style) group.

Rank assignment (matches conf_hybrid/topology/try1_hybrid_topo.yaml for 2×3):
  - global 0                          → RPC server only (if dedicated_rpc_server)
  - globals 1 .. num_facilities       → facility leaders (also gRPC clients)
  - remaining ranks                   → workers, in facility order (fac0 fills
                                        first, then fac1, ...)
Within each facility, local_rank == members.index(global_rank); leader is local 0
and appears first in ``members``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

__all__ = [
    "DEFAULT_HYBRID_COMMUNICATORS",
    "build_hybrid_topology",
    "merge_hybrid_communicators",
    "validate_hybrid_topology_dict",
]

# Declarative (Hydra‑visible) communicator roles — actual wiring stays in hybrid runner.
DEFAULT_HYBRID_COMMUNICATORS: Dict[str, str] = {
    # Facility‑internal collectives (`TorchMPIAdapter` → process group).
    "intra_facility": "torch_mpi",
    # Cross‑facility federated averaging (`GrpcLeaderCommunicator` + Flora daemon).
    "global_aggregation": "grpc",
}


def merge_hybrid_communicators(
    overrides: Dict[str, Any] | None,
) -> Dict[str, str]:
    """Merge YAML overrides onto :data:`DEFAULT_HYBRID_COMMUNICATORS` (string roles)."""
    out: Dict[str, str] = dict(DEFAULT_HYBRID_COMMUNICATORS)
    if not overrides:
        return out
    for key, raw in overrides.items():
        sk = str(key)
        if raw is None:
            continue
        out[sk] = str(raw)
    return out


def _normalize_mpi_ranks_per_facility(
    mpi_ranks_per_facility: Union[int, Sequence[int]],
    num_facilities: int,
) -> List[int]:
    if isinstance(mpi_ranks_per_facility, int):
        if mpi_ranks_per_facility < 1:
            raise ValueError("mpi_ranks_per_facility must be >= 1 when int")
        return [mpi_ranks_per_facility] * num_facilities
    seq = list(mpi_ranks_per_facility)
    if len(seq) != num_facilities:
        raise ValueError(
            f"mpi_ranks_per_facility list length {len(seq)} must equal "
            f"num_facilities={num_facilities}"
        )
    for i, w in enumerate(seq):
        if w < 1:
            raise ValueError(f"mpi_ranks_per_facility[{i}]={w} must be >= 1")
    return seq


def build_hybrid_topology(
    *,
    num_facilities: int,
    mpi_ranks_per_facility: Union[int, Sequence[int]],
    dedicated_rpc_server: bool = True,
    rpc_addr: str = "127.0.0.1",
    rpc_port: int = 50051,
    facility_mpi_addr: str = "127.0.0.1",
    facility_mpi_base_port: int = 28250,
    facility_mpi_port_stride: int = 40,
    facility_name_prefix: str = "fac",
    communicators: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build a resolved ``topology`` dict (Hydra/OmegaConf-friendly) for hybrid runs.

    Parameters
    ----------
    num_facilities:
        Number of facilities (each with its own torch process group).
    mpi_ranks_per_facility:
        Either a single int (same size for every facility) or a length-
        ``num_facilities`` list for asymmetric groups.
    dedicated_rpc_server:
        If True, global rank 0 is reserved for gRPC only; world_size is
        ``1 + sum(mpi_ranks_per_facility)``. If False, not implemented yet.
    rpc_addr / rpc_port:
        Central gRPC server address and port (leaders connect as clients).
    facility_mpi_addr:
        TCP init_method host for each facility store (often overridden per run).
    facility_mpi_base_port / facility_mpi_port_stride:
        Facility ``f`` uses port ``facility_mpi_base_port + f * stride``.
    facility_name_prefix:
        Names are ``{prefix}{1-based index}`` (e.g. fac1, fac2).
    communicators:
        Optional override mapping merged onto :data:`DEFAULT_HYBRID_COMMUNICATORS`
        (declarative only; hybrid runner wiring is unchanged).
    """
    if not dedicated_rpc_server:
        raise NotImplementedError(
            "Only dedicated_rpc_server=True is supported for now (rank 0 = gRPC only)."
        )
    if num_facilities < 1:
        raise ValueError("num_facilities must be >= 1")

    sizes = _normalize_mpi_ranks_per_facility(mpi_ranks_per_facility, num_facilities)

    # Leaders consume ranks 1 .. num_facilities
    leader_ranks = list(range(1, num_facilities + 1))
    next_rank = num_facilities + 1
    facilities_out: List[Dict[str, Any]] = []

    for f, world_sz in enumerate(sizes):
        members = [leader_ranks[f]]
        n_workers = world_sz - 1
        for _ in range(n_workers):
            members.append(next_rank)
            next_rank += 1
        facilities_out.append(
            {
                "name": f"{facility_name_prefix}{f + 1}",
                "mpi": {
                    "addr": facility_mpi_addr,
                    "port": facility_mpi_base_port + f * facility_mpi_port_stride,
                    "world_size": world_sz,
                    "members": members,
                    "leader_rank": leader_ranks[f],
                },
            }
        )

    world_size = next_rank  # last assigned + 1 == next_rank after final increment
    if world_size != 1 + sum(sizes):
        raise RuntimeError("internal error: world_size mismatch")

    topology: Dict[str, Any] = {
        "world_size": world_size,
        "rpc": {
            "server_rank": 0,
            "addr": rpc_addr,
            "port": rpc_port,
            "client_ranks": list(leader_ranks),
        },
        "facilities": facilities_out,
        "communicators": merge_hybrid_communicators(communicators),
    }
    validate_hybrid_topology_dict(topology)
    return topology


def validate_hybrid_topology_dict(topology: Dict[str, Any]) -> None:
    """Light validation of a resolved topology dict (builder output or YAML)."""
    ws = int(topology["world_size"])
    if ws < 1:
        raise ValueError("world_size must be >= 1")

    rpc = topology["rpc"]
    srv = int(rpc["server_rank"])
    clients = [int(x) for x in rpc["client_ranks"]]
    facs = topology["facilities"]

    seen = {srv}
    for fac in facs:
        mpi = fac["mpi"]
        members = [int(m) for m in mpi["members"]]
        leader = int(mpi["leader_rank"])
        wsz = int(mpi["world_size"])

        if leader not in members:
            raise ValueError(f"leader_rank={leader} not in members={members}")
        if members.index(leader) != 0:
            raise ValueError(
                f"leader_rank={leader} must be first in members={members} "
                "so local_rank 0 is the facility leader"
            )
        if len(members) != wsz:
            raise ValueError(
                f"len(members)={len(members)} != mpi.world_size={wsz}"
            )
        for m in members:
            if m in seen:
                raise ValueError(f"duplicate global rank {m} across topology")
            seen.add(m)

        if leader not in clients:
            raise ValueError(
                f"facility leader {leader} must appear in rpc.client_ranks={clients}"
            )

    if len(clients) != len(facs):
        raise ValueError(
            f"expected one gRPC client per facility, got client_ranks={clients} "
            f"for {len(facs)} facilities"
        )

    if seen != set(range(ws)):
        missing = set(range(ws)) - seen
        extra = seen - set(range(ws))
        raise ValueError(
            f"ranks must be exactly 0..{ws - 1}; missing={sorted(missing)} "
            f"extra_or_dup={sorted(extra)}"
        )

    comm = topology.get("communicators")
    if comm is not None and not isinstance(comm, dict):
        raise ValueError("topology.communicators must be a mapping when present")
