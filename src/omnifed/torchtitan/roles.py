from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TorchTitanRole:
    role: str
    federated_rank: int
    client_id: int | None
    torchtitan_rank: int | None
    torchtitan_world_size: int | None
    is_client_leader: bool

    @property
    def is_server(self) -> bool:
        return self.role == "server"

    @property
    def is_client(self) -> bool:
        return self.role == "client"


def resolve_torchtitan_role(
    cfg: Any,
) -> TorchTitanRole:
    role_name = os.environ["OMNIFED_ROLE"]

    if role_name == "server":
        return TorchTitanRole(
            role="server",
            federated_rank=0,
            client_id=None,
            torchtitan_rank=None,
            torchtitan_world_size=None,
            is_client_leader=False,
        )

    if role_name != "client":
        raise ValueError(
            "OMNIFED_ROLE must be 'server' or "
            f"'client'; got {role_name!r}"
        )

    client_id = int(os.environ["CLIENT_ID"])
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    leader_rank = int(
        cfg.torchtitan.subclusters.leader_rank
    )

    return TorchTitanRole(
        role="client",
        federated_rank=client_id + 1,
        client_id=client_id,
        torchtitan_rank=rank,
        torchtitan_world_size=world_size,
        is_client_leader=rank == leader_rank,
    )