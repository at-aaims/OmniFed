# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unique train shards for FL trainers (classic and hierarchical share this)."""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset

FEDERATED_CLIENT_INDEX_ENV = "OMNIFED_FEDERATED_CLIENT_INDEX"
NUM_FEDERATED_CLIENTS_ENV = "OMNIFED_NUM_FEDERATED_CLIENTS"

# Shared permutation so every rank slices the same shuffled order (disjoint shards).
_SHARD_PERM_SEED = 0


def topology_has_server(topology_cfg: Any) -> bool:
    """True for parameter-server layouts (rank 0 does not train)."""
    if topology_cfg is None:
        return True
    explicit = None
    if hasattr(topology_cfg, "get"):
        try:
            explicit = topology_cfg.get("has_server")
        except Exception:
            explicit = None
        target = str(topology_cfg.get("_target_", "") or "")
    else:
        explicit = getattr(topology_cfg, "has_server", None)
        target = str(
            getattr(topology_cfg, "_target_", "") or type(topology_cfg).__name__
        )
    if explicit is not None and not callable(explicit):
        return bool(explicit)
    return "DecentralizedTopology" not in target


def apply_federated_shard_env(
    *, rank: int, num_trainers: int, has_server: bool = True
) -> None:
    """
    Set shard env for this process.

    Parameter-server (``has_server=True``): rank 0 → index -1 (no train);
    trainers 1..N → indices 0..N-1.

    All-reduce (``has_server=False``): rank i → index i (every rank trains).

    ``num_trainers`` is ``topology.num_clients``, not Slurm node count when a
    server is present (then world = num_trainers + 1).
    """
    n = int(num_trainers)
    r = int(rank)
    if n < 1:
        raise ValueError(f"num_trainers must be >= 1, got {n}")
    os.environ[NUM_FEDERATED_CLIENTS_ENV] = str(n)
    if has_server:
        if r == 0:
            os.environ[FEDERATED_CLIENT_INDEX_ENV] = "-1"
            return
        idx = r - 1
        if idx >= n:
            raise ValueError(
                f"trainer rank={r} maps to client index {idx} but num_trainers={n}"
            )
        os.environ[FEDERATED_CLIENT_INDEX_ENV] = str(idx)
        return
    if not (0 <= r < n):
        raise ValueError(
            f"all-reduce rank={r} out of range for num_trainers={n}"
        )
    os.environ[FEDERATED_CLIENT_INDEX_ENV] = str(r)


def resolve_num_federated_clients(explicit: Optional[int] = None) -> int:
    if explicit is not None:
        n = int(explicit)
        if n < 1:
            raise ValueError(f"num_federated_clients must be >= 1, got {n}")
        return n
    raw = os.environ.get(NUM_FEDERATED_CLIENTS_ENV)
    if raw is None or raw == "":
        return 1
    n = int(raw)
    if n < 1:
        raise ValueError(f"{NUM_FEDERATED_CLIENTS_ENV}={raw!r} must be >= 1")
    return n


def resolve_federated_client_index(
    num_clients: int,
    *,
    env_var: str = FEDERATED_CLIENT_INDEX_ENV,
) -> Optional[int]:
    """
    Trainer shard index in ``0 .. num_clients-1``, or ``None`` for the server (no train).

    Missing env with ``num_clients > 1`` is an error (do not silently use shard 0).
    """
    n = int(num_clients)
    if n < 1:
        raise ValueError(f"num_federated_clients must be >= 1, got {n}")
    raw = os.environ.get(env_var)
    if raw is None or raw == "":
        if n <= 1:
            return 0
        raise RuntimeError(
            f"{env_var} must be set when num_federated_clients={n} "
            "(server=-1, trainers=0..N-1)"
        )
    idx = int(raw)
    if idx < 0:
        return None
    if idx >= n:
        raise ValueError(f"{env_var}={raw!r} out of range for num_federated_clients={n}")
    return idx


def unique_train_subset(
    dataset: Dataset,
    num_clients: int,
    client_idx: int,
    *,
    seed: int = _SHARD_PERM_SEED,
) -> Subset:
    """Deterministic disjoint subset. Last client receives the remainder."""
    n_cli = int(num_clients)
    cidx = int(client_idx)
    if not (0 <= cidx < n_cli):
        raise ValueError(f"client_idx={cidx} out of range for num_clients={n_cli}")
    n = len(dataset)  # type: ignore[arg-type]
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator).tolist()
    chunk = n // n_cli
    start = cidx * chunk
    end = n if cidx == n_cli - 1 else start + chunk
    return Subset(dataset, perm[start:end])


def rebuild_dataloader(loader: DataLoader, dataset: Dataset) -> DataLoader:
    shuffle = isinstance(loader.sampler, torch.utils.data.RandomSampler)
    kwargs: dict = {
        "batch_size": loader.batch_size,
        "shuffle": shuffle,
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "collate_fn": loader.collate_fn,
        "timeout": loader.timeout,
    }
    if int(loader.num_workers) > 0:
        kwargs["persistent_workers"] = loader.persistent_workers
    return DataLoader(dataset, **kwargs)
