# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
"""
gRPC rank map for the simulated hierarchical outer hop.

Classic ``GrpcCommunicator`` is a single world: rank 0 is the server and
``world_size`` includes that server. Facility workers are not in this world.
"""

from __future__ import annotations

from typing import Iterable

__all__ = ["outer_grpc_world_size", "outer_grpc_rank"]


def outer_grpc_world_size(client_ranks: Iterable[int]) -> int:
    """Server plus one gRPC client per facility leader."""
    return 1 + len({int(x) for x in client_ranks})


def outer_grpc_rank(
    slurm_rank: int,
    client_ranks: Iterable[int],
    *,
    server_rank: int = 0,
) -> int:
    """
    Map a Slurm rank onto the outer gRPC world.

    Server → 0. Leaders, in sorted Slurm rank order → 1, 2, ...
    """
    sr = int(slurm_rank)
    if sr == int(server_rank):
        return 0
    leaders = sorted(int(x) for x in client_ranks)
    try:
        return 1 + leaders.index(sr)
    except ValueError as exc:
        raise ValueError(
            f"Slurm rank {sr} is not the outer-gRPC server ({server_rank}) "
            f"or a facility leader {leaders}"
        ) from exc
